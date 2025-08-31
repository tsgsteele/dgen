from __future__ import annotations

import ast
import datetime
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

import settings
import utility_functions as utilfunc
import agent_mutation

# ---- helper imports (kept local so you can drop-in without changing file imports)
from batt_dispatch_helpers import (
    configure_retail_rate_dispatch,
    dispatch_export_diags,
)

# DB helpers for batch inserts
import psycopg2.extras as pgx

# PySAM modules
import PySAM.Battery as battery
import PySAM.BatteryTools as battery_tools
import PySAM.Utilityrate5 as utility
import PySAM.Cashloan as cashloan
import PySAM.CustomGeneration as customgen
import PySAM.Pvsamv1 as pvsamv1
import PySAM.Pvwattsv8 as pvwattsv8

# Toggle: avoid DC assignments entirely unless explicitly needed.
SKIP_DEMAND_CHARGES = True

# Force net billing
FORCE_NET_BILLING = True

#==============================================================================
# Logger
#==============================================================================
logger = utilfunc.get_logger()

#==============================================================================
# ECON TABLES (NEW)
#==============================================================================

_tables_ready = False

def _ensure_econ_tables(cur):
    """
    Create the long-format hourly table and annual finance tables iff absent.
    Uses current connection's search_path (scenario schema).
    """
    global _tables_ready
    if _tables_ready:
        return

    # Hourly
    cur.execute("""
    CREATE TABLE IF NOT EXISTS agent_hourly_econ (
        agent_id           BIGINT,
        year               INT,
        scenario_case      TEXT,           -- 'pv_only' or 'pv_batt'
        hour_index         INT,            -- 1..8760
        variable           TEXT,           -- e.g. 'load_kwh', 'batt_to_load_kwh'
        value              DOUBLE PRECISION,
        units              TEXT,
        tou_period_id      INT,
        created_at         TIMESTAMPTZ DEFAULT now()
    );
    """)

    # Annual summary
    cur.execute("""
    CREATE TABLE IF NOT EXISTS agent_annual_finance (
        agent_id                         BIGINT,
        year                             INT,
        scenario_case                    TEXT,
        system_kw                        DOUBLE PRECISION,
        batt_kw                          DOUBLE PRECISION,
        batt_kwh                         DOUBLE PRECISION,
        npv_usd                          DOUBLE PRECISION,
        payback_yrs                      DOUBLE PRECISION,
        bill_wo_sys_year1_usd            DOUBLE PRECISION,
        bill_w_sys_year1_usd             DOUBLE PRECISION,
        energy_value_year1_usd           DOUBLE PRECISION,
        annual_import_kwh                DOUBLE PRECISION,
        annual_export_kwh                DOUBLE PRECISION,
        created_at                       TIMESTAMPTZ DEFAULT now()
    );
    """)

    # Detailed annual finance (long format)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS agent_annual_finance_detail (
        agent_id     BIGINT,
        year         INT,
        scenario_case TEXT,        -- 'pv_only' or 'pv_batt'
        metric       TEXT,         -- e.g. 'cf_debt_payment_interest'
        period_index INT,          -- 0-based year index in cash flow series
        value        DOUBLE PRECISION,
        created_at   TIMESTAMPTZ DEFAULT now()
    );
    """)

    _tables_ready = True


def _resolve_hourly_buy_prices(ER) -> np.ndarray:
    """
    Build an 8760 buy price vector using Utilityrate5's TOU energy matrix and
    weekday/weekend schedules. Price = first-tier price for the active period.
    """
    tou = np.asarray(getattr(ER, "ur_ec_tou_mat", []), dtype=float)  # [period,tier,max,unit,price,sell]
    wk  = np.asarray(getattr(ER, "ur_ec_sched_weekday", []), dtype=int)
    we  = np.asarray(getattr(ER, "ur_ec_sched_weekend", []), dtype=int)
    if tou.size == 0 or wk.size == 0 or we.size == 0:
        return np.zeros(8760, dtype=float)

    first_tier = tou[tou[:, 1] == 1]
    period_price = {}
    for per in np.unique(first_tier[:, 0].astype(int)):
        prices = first_tier[first_tier[:, 0] == per][:, 4]
        period_price[int(per)] = float(np.mean(prices)) if prices.size else 0.0

    # Build 12×24 schedule → 8760 (restart DOW each month for deterministic mapping)
    hours_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    sched_hours = []
    for mi, days in enumerate(hours_per_month):
        dow = 0
        for _ in range(days):
            row = np.asarray(wk[mi], dtype=int) if dow < 5 else np.asarray(we[mi], dtype=int)
            sched_hours.append(row)
            dow = (dow + 1) % 7
    sched_hours = np.vstack(sched_hours).ravel()

    prices = np.vectorize(lambda p: period_price.get(int(p), 0.0))(sched_hours)
    return prices.astype(float)

def _attach_hourly_prices(utilityrate, util_out: dict) -> dict:
    """
    Populate util_out with an hourly utility price series in USD/kWh,
    computed via PySAM.Utilityrateforecast (NREL example style).

    Notes:
    - Uses the *configured* Utilityrate5 object's ElectricityRates settings.
    - Sets analysis_period=1, steps_per_hour=1 (8760 points).
    - Follows the NREL "forecast" pattern: grid_power = -1 * load.
    - Fixed charges are not included in the price stream.
    """
    import numpy as np
    import PySAM.Utilityrateforecast as utility_rate_forecast

    # Build forecast object
    rf = utility_rate_forecast.new()

    # Lifetime / steps
    try:
        infl = float(getattr(utilityrate.Lifetime, "inflation_rate", 0.0))
    except Exception:
        infl = 0.0
    rf.value("analysis_period", 1)
    rf.value("inflation_rate", infl)   # percent (e.g., 2.5)
    rf.value("steps_per_hour", 1)

    # Copy relevant ElectricityRates fields over (best-effort)
    ER = utilityrate.ElectricityRates
    _keys = [
        "ur_metering_option",
        "ur_monthly_fixed_charge",
        "ur_ec_tou_mat", "ur_ec_sched_weekday", "ur_ec_sched_weekend",
        "ur_en_ts_sell_rate", "ur_ts_sell_rate",
        "ur_en_ts_buy_rate",  "ur_ts_buy_rate",
        "TOU_demand_single_peak",
        "ur_dc_enable", "ur_dc_sched_weekday", "ur_dc_sched_weekend",
        "ur_dc_flat_mat", "ur_dc_tou_mat",
    ]
    for k in _keys:
        try:
            rf.value(k, getattr(ER, k))
        except Exception:
            pass  # some fields may not exist / be unset; that's fine

    # Build time series inputs following the NREL example
    load = np.asarray(getattr(utilityrate.Load, "load", []), dtype=float).ravel()
    gen  = np.asarray(getattr(utilityrate.SystemOutput, "gen", []), dtype=float).ravel()
    if gen.size == 0:
        gen = np.zeros_like(load)

    # Forecast engine wants kW series; grid_power negative for import
    rf.value("gen", gen.tolist())                       # kW
    rf.value("load", load.tolist())                     # kW
    rf.value("grid_power", (-1.0 * load)[:8760].tolist())
    rf.value("idx", 0)

    # Execute and stash the hourly price series
    rf.setup()
    rf.execute()

    try:
        price_series = rf.export()["Outputs"]["ur_price_series"]
    except Exception:
        price_series = []

    if isinstance(price_series, (list, tuple)) and len(price_series) == 8760:
        # Store as USD/kWh to avoid any confusion with cents
        util_out["hourly_utility_price_usd_per_kwh"] = [float(x) for x in price_series]

    return util_out


def _pack_hourly_records(agent_id: int, year: int, case: str,
                         util_out: dict, batt_out: dict) -> Tuple[List[Tuple], int]:
    """
    Writes long-format hourly rows to agent_hourly_econ.

    Conditional sourcing:
      - With battery:   grid_import_kwh <- batt_out['grid_to_load']
                        grid_export_kwh <- batt_out['system_to_grid']
      - Without battery: grid_import_kwh <- util_out['year1_hourly_e_fromgrid']
                         grid_export_kwh <- util_out['year1_hourly_e_togrid']

    Also exports: system_to_load_kwh, batt_to_load_kwh, (optional) grid_to_batt/system_to_batt,
                  batt_to_grid_kwh, batt_soc_pct, batt_power_kw, utility_price_usd_per_kwh, load_kwh, etc.
    """
    import numpy as np

    rows: List[Tuple] = []

    # Helper to push a whole hourly series
    def push_series(name: str, seq, units: str = "", tou_seq=None):
        arr = np.asarray(seq, dtype=float).ravel()
        tou = np.asarray(tou_seq, dtype=int).ravel() if tou_seq is not None else np.full(arr.shape, -1, dtype=int)
        m = int(arr.size)
        for i in range(m):
            rows.append((int(agent_id), int(year), case, int(i + 1), name, float(arr[i]), units, int(tou[i])))

    # TOU schedules (optional)
    tou_ec = util_out.get("year1_hourly_ec_tou_schedule", None)
    tou_dc = util_out.get("year1_hourly_dc_tou_schedule", None)

    # ---- Base Utilityrate outputs (used for many series regardless of batt) ----
    s2l_util = np.asarray(util_out.get("year1_hourly_system_to_load", []), dtype=float)
    e_fromgrid_util = np.asarray(util_out.get("year1_hourly_e_fromgrid", []), dtype=float)
    e_togrid_util   = np.asarray(util_out.get("year1_hourly_e_togrid",   []), dtype=float)
    load_yr1        = np.asarray(util_out.get("load_kwh", []), dtype=float)

    # ---- Battery outputs (if present) ----
    has_batt = (
        case == 'pv_batt' and
        isinstance(batt_out, dict) and
        any(len(np.asarray(batt_out.get(k, []))) > 0 for k in ('grid_to_load','system_to_grid','batt_to_load'))
    )
    b2l = np.asarray(batt_out.get("batt_to_load",   []), dtype=float) if has_batt else np.array([])
    g2b = np.asarray(batt_out.get("grid_to_batt",   []), dtype=float) if has_batt else np.array([])
    s2b = np.asarray(batt_out.get("system_to_batt", []), dtype=float) if has_batt else np.array([])
    b2g = np.asarray(batt_out.get("batt_to_grid",   []), dtype=float) if has_batt else np.array([])
    g2l = np.asarray(batt_out.get("grid_to_load",   []), dtype=float) if has_batt else np.array([])
    s2g = np.asarray(batt_out.get("system_to_grid", []), dtype=float) if has_batt else np.array([])
    s2l = np.asarray(batt_out.get("system_to_load", []), dtype=float) if has_batt else np.array([])
    g2l_util = np.asarray(batt_out.get("year1_hourly_e_fromgrid", []), dtype=float) if has_batt else np.array([])

    # Determine unified length n
    lengths = [x.size for x in (s2l_util, e_fromgrid_util, e_togrid_util, load_yr1, b2l, g2b, s2b, b2g, g2l, s2g) if x.size]
    n = max(lengths) if lengths else 0
    if n == 0:
        return rows, 0

    # Resize/pad helper
    def fit(a):
        a = np.asarray(a, dtype=float).ravel()
        if a.size == 0:
            return np.zeros(n, dtype=float)
        if a.size == n:
            return a
        return np.resize(a, n)

    s2l_util        = fit(s2l_util)
    e_fromgrid_util = fit(e_fromgrid_util)
    e_togrid_util   = fit(e_togrid_util)
    load_yr1        = fit(load_yr1)
    b2l             = fit(b2l)
    g2b             = fit(g2b)
    s2b             = fit(s2b)
    b2g             = fit(b2g)
    g2l             = fit(g2l)
    s2g             = fit(s2g)
    s2l             = fit(s2l)
    g2l_util        = fit(g2l_util)

    # ---- Conditional sourcing per your instructions ----
    # grid_import_kwh
    grid_import = g2l if has_batt else e_fromgrid_util

    # grid_export_kwh
    grid_export = s2g if has_batt else e_togrid_util

    # ---- Push series ----
    # Load (if present)
    if load_yr1.any():
        push_series("load_kwh", load_yr1, "kWh", tou_ec)

    # Core flows
    push_series("system_to_load_kwh", s2l_util, "kWh", tou_ec)
    if has_batt:
        push_series("batt_to_load_kwh", b2l, "kWh", tou_ec)
        push_series("grid_to_load_util_kwh", g2l_util, "kWh", tou_ec)
    push_series("grid_import_kwh", grid_import, "kWh", tou_ec)
    push_series("grid_export_kwh", grid_export, "kWh", tou_ec)

    # Battery charge/discharge details (keep names distinct; no duplication of totals)
    if has_batt:
        if s2b.any(): push_series("system_to_batt_kwh", s2b, "kWh", tou_ec)
        if g2b.any(): push_series("grid_to_batt_kwh",   g2b, "kWh", tou_ec)
        if b2g.any(): push_series("batt_to_grid_kwh",   b2g, "kWh", tou_ec)

        if "batt_SOC" in batt_out:
            push_series("batt_soc_pct", batt_out["batt_SOC"], "%", tou_ec)
        if "batt_power" in batt_out:
            push_series("batt_power_kw", batt_out["batt_power"], "kW", tou_ec)

    # Prices & billing
    if "hourly_utility_price_usd_per_kwh" in util_out:
        push_series("utility_price_usd_per_kwh",
                    util_out["hourly_utility_price_usd_per_kwh"], "USD/kWh", tou_ec)
    if "hourly_export_price_usd_per_kwh" in util_out:
        push_series("export_price_usd_per_kwh",
                    util_out["hourly_export_price_usd_per_kwh"], "USD/kWh", tou_ec)
    if "year1_hourly_salespurchases_with_system" in util_out:
        push_series("sales_purchases_usd", util_out["year1_hourly_salespurchases_with_system"], "$", tou_ec)
    if "year1_hourly_ec_with_system" in util_out:
        push_series("energy_charge_with_sys_usd", util_out["year1_hourly_ec_with_system"], "$", tou_ec)
    if "year1_hourly_ec_without_system" in util_out:
        push_series("energy_charge_without_sys_usd", util_out["year1_hourly_ec_without_system"], "$", tou_ec)
    if "year1_hourly_dc_with_system" in util_out:
        push_series("demand_charge_with_sys_usd", util_out["year1_hourly_dc_with_system"], "$", tou_dc)
    if "year1_hourly_dc_without_system" in util_out:
        push_series("demand_charge_without_sys_usd", util_out["year1_hourly_dc_without_system"], "$", tou_dc)
    if "year1_hourly_p_tofromgrid" in util_out:
        push_series("p_tofromgrid_kw", util_out["year1_hourly_p_tofromgrid"], "kW", tou_dc)
    if "year1_hourly_p_system_to_load" in util_out:
        push_series("p_system_to_load_kw", util_out["year1_hourly_p_system_to_load"], "kW", tou_ec)
    if "pv_generation" in util_out:
        push_series("pv_generation_kw", util_out["pv_generation"], "kW", tou_ec)

    return rows, n


def _insert_hourly_records(conn, rows: List[Tuple]):
    """
    Bulk insert long-format hourly rows.
    rows: (agent_id, year, case, hour_index, variable, value, units, tou_period_id)
    """
    if not rows:
        return
    with conn.cursor() as cur:
        _ensure_econ_tables(cur)
        pgx.execute_values(
            cur,
            """
            INSERT INTO agent_hourly_econ
            (agent_id, year, scenario_case, hour_index, variable, value, units, tou_period_id)
            VALUES %s
            """,
            rows,
            page_size=5000
        )
    conn.commit()


def _insert_annual_finance(conn, rows: List[Tuple]):
    """
    Bulk insert annual summary rows.
    rows: (agent_id, year, case, system_kw, batt_kw, batt_kwh, npv_usd, payback_yrs,
           bill_wo_sys_year1_usd, bill_w_sys_year1_usd, energy_value_year1_usd,
           annual_import_kwh, annual_export_kwh)
    """
    if not rows:
        return
    with conn.cursor() as cur:
        _ensure_econ_tables(cur)
        pgx.execute_values(
            cur,
            """
            INSERT INTO agent_annual_finance
            (agent_id, year, scenario_case, system_kw, batt_kw, batt_kwh, npv_usd, payback_yrs,
             bill_wo_sys_year1_usd, bill_w_sys_year1_usd, energy_value_year1_usd,
             annual_import_kwh, annual_export_kwh)
            VALUES %s
            """,
            rows,
            page_size=1000
        )
    conn.commit()


def _insert_annual_finance_detail(conn, rows: List[Tuple]):
    """
    Bulk insert per-year cashflow rows (long format).
    rows: (agent_id, year, case, metric, period_index, value)
    """
    if not rows:
        return
    with conn.cursor() as cur:
        _ensure_econ_tables(cur)
        pgx.execute_values(
            cur,
            """
            INSERT INTO agent_annual_finance_detail
            (agent_id, year, scenario_case, metric, period_index, value)
            VALUES %s
            """,
            rows,
            page_size=5000
        )
    conn.commit()


def _pack_annual_finance_case(agent_id: int, year: int, case: str,
                              util_out: dict, loan_out: dict,
                              system_kw: float, batt_kw: float, batt_kwh: float) -> Tuple[Tuple, List[Tuple]]:
    """
    Build:
      • one summary row for agent_annual_finance (tuple)
      • many detail rows for agent_annual_finance_detail (list of tuples)
    from Utilityrate5.Outputs and Cashloan.Outputs **dicts**.
    """
    import numpy as np

    bill_w  = float(util_out.get("utility_bill_w_sys_year1", np.nan))
    bill_wo = float(util_out.get("utility_bill_wo_sys_year1", np.nan))
    bill_sav_y1 = float(util_out.get("savings_year1", np.nan))

    imp_y1 = float(np.nansum(np.asarray(util_out.get("year1_hourly_e_fromgrid", []), dtype=float))) if "year1_hourly_e_fromgrid" in util_out else float("nan")
    exp_y1 = float(np.nansum(np.asarray(util_out.get("year1_hourly_e_togrid", []),  dtype=float)))   if "year1_hourly_e_togrid"  in util_out else float("nan")

    npv_usd = float(loan_out.get("npv", float("nan")))
    payback = float(loan_out.get("payback", float("nan")))

    summary_row = (
        int(agent_id), int(year), case,
        float(system_kw), float(batt_kw), float(batt_kwh),
        npv_usd, payback,
        bill_wo, bill_w, bill_sav_y1,
        imp_y1, exp_y1
    )

    # detail long-format
    detail_rows: List[Tuple] = []
    def _push(metric: str):
        seq = loan_out.get(metric, [])
        arr = np.asarray(seq, dtype=float).ravel()
        for i, v in enumerate(arr):
            detail_rows.append((int(agent_id), int(year), case, metric, int(i), float(v)))

    for metric in [
        "cf_debt_payment_interest",
        "cf_debt_payment_principal",
        "cf_debt_payment_total",
        "cf_debt_balance",
        "cf_after_tax_cash_flow",
        "cf_utility_bill",
        "cf_energy_value",
        "cf_discounted_costs",
        "cf_payback_with_expenses",
        "cf_cumulative_payback_with_expenses",
        "cf_annual_cost_lcos",
        "cf_annual_discharge_lcos",
        "cf_charging_cost_grid",
        "cf_charging_cost_pv",
        "cf_battery_replacement_cost",
    ]:
        if metric in loan_out:
            _push(metric)

    return summary_row, detail_rows


#==============================================================================
# PySAM stack creation
#==============================================================================

def _init_pv_batt_stack(sector_abbr: str):
    """
    Build a PV+Battery+UtilityRate+Loan stack. Prefer configs that include the
    Battery compute module since we pass custom generation.

    Returns
    -------
    (driver_mod, batt, utilityrate, loan, market_flag)
      - market_flag: 0 for residential, 1 for non-residential
    """
    if sector_abbr == 'res':
        cands = [
            ("CustomGenerationBatteryResidential", customgen),
            ("PVWattsBatteryResidential",          pvwattsv8),
            ("PVBatteryResidential",               pvsamv1),
        ]
        market = 0
    else:
        cands = [
            ("CustomGenerationBatteryCommercial",  customgen),
            ("PVWattsBatteryCommercial",           pvwattsv8),
            ("PVBatteryCommercial",                pvsamv1),
        ]
        market = 1

    for cfg, drv_mod in cands:
        try:
            driver = drv_mod.default(cfg)
            batt   = battery.from_existing(driver,   cfg)
            util   = utility.from_existing(driver,   cfg)
            loanm  = cashloan.from_existing(driver,  cfg)
            return driver, batt, util, loanm, market
        except Exception:
            continue

    # Fallback (version-proof)
    driver = None
    batt   = battery.new()
    util   = utility.new()
    loanm  = cashloan.new()
    return driver, batt, util, loanm, market


#==============================================================================
# Core sizing objective
#==============================================================================

def calc_system_performance(
    kw: float,
    pv: Dict[str, np.ndarray],
    utilityrate,
    loan,
    batt,
    costs: Dict[str, float],
    agent: pd.Series,
    rate_switch_table: Optional[pd.DataFrame],
    en_batt: bool = True,
    batt_dispatch: str = 'price_signal_forecast',
    batt_kwh: Optional[float] = None,          # <---- size override
    batt_power_kw: Optional[float] = None      # <---- size override
):
    """
    Objective function for scalar search on PV kW (battery sized internally).

    Returns negative NPV (so we minimize) computed from Cashloan.
    """
    inv_eff = 0.96
    gen_hourly = pv['generation_hourly']
    load_hourly = pv['consumption_hourly']

    # PV production path (W -> kWh)
    dc = (gen_hourly * kw) * 1000.0
    ac = dc * inv_eff
    gen = ac / 1000.0

    # Hard reset key loan fields so they don't carry over between runs
    loan.SystemCosts.add_om_num_types = 0
    loan.SystemCosts.om_capacity = [0.0]
    loan.SystemCosts.om_batt_capacity_cost = [0.0]
    loan.SystemCosts.om_batt_replacement_cost = [0.0]
    loan.SystemCosts.om_batt_nameplate = 0.0
    loan.BatterySystem.battery_per_kWh = 0.0  # never let Cashloan auto-add battery capex

    if en_batt:
        # Minimal, robust battery setup sized to PV
        batt.BatterySystem.en_batt = 1
        batt.BatterySystem.batt_ac_or_dc = 1
        batt.BatteryCell.batt_chem = 1
        batt.BatterySystem.batt_meter_position = 0
        batt.Lifetime.system_use_lifetime_output = 0
        batt.BatterySystem.batt_replacement_option = 0
        batt.batt_minimum_SOC = 10

        # --- SIZE BATTERY ---
        _default_capacity_to_power_ratio = 2.0
        _default_kwh   = 8.0
        _kwh = float(batt_kwh) if batt_kwh is not None else _default_kwh
        if batt_power_kw is not None:
            _kw = float(batt_power_kw)
        else:
            _kw = _kwh / _default_capacity_to_power_ratio

        desired_voltage = 500 if agent.loc['sector_abbr'] != 'res' else 240
        battery_tools.battery_model_sizing(
            batt, _kw, _kwh, desired_voltage=desired_voltage, tol=1e38
        )

        batt.Load.load = load_hourly
        batt.SystemOutput.gen = gen
        batt.BatteryCell.batt_initial_SOC = 30

        # Dispatch: retail-rate aware, no grid-charging unless surplus
        configure_retail_rate_dispatch(
            batt,
            allow_export=True,
            allow_grid_charge=False,
            charge_only_when_surplus=False,
            lookahead_hours=6,
        )
        if not hasattr(batt.BatteryDispatch, 'batt_look_ahead_hours'):
            batt.BatteryDispatch.batt_look_ahead_hours = 6

        batt.execute()

        # Rate switch based on installed storage
        if batt.BatterySystem.batt_computed_bank_capacity > 0.0:
            agent, one_time_charge = agent_mutation.elec.apply_rate_switch(
                rate_switch_table,
                agent,
                batt.BatterySystem.batt_computed_bank_capacity,
                tech='storage'
            )
        else:
            one_time_charge = 0.0

        # Build Utilityrate5 from switched tariff
        net_billing_sell_rate = 0.0 
        ts_sell = np.asarray(agent.loc['wholesale_prices'], dtype=float).ravel() * agent.loc['elec_price_multiplier']
        td_norm = normalize_tariff(agent.loc['tariff_dict'], net_sell_rate_scalar=net_billing_sell_rate)
        process_tariff(utilityrate, td_norm, net_billing_sell_rate, ts_sell_rate=ts_sell)

        # Hand gen to the rate engine
        utilityrate.SystemOutput.gen = batt.SystemOutput.gen

        # Wire up loan pieces for battery case
        loan.BatterySystem.en_batt = 1
        loan.BatterySystem.batt_computed_bank_capacity = batt.Outputs.batt_bank_installed_capacity
        loan.BatterySystem.batt_bank_replacement = batt.Outputs.batt_bank_replacement

        loan.SystemCosts.add_om_num_types = 0
        if kw > 0:
            loan.SystemCosts.om_batt_capacity_cost = [0.0]
            loan.SystemCosts.om_batt_variable_cost = [0.0]
            loan.SystemCosts.om_batt_replacement_cost = [0.0]
            system_costs = costs['system_capex_per_kw_combined'] * kw
        else:
            loan.SystemCosts.om_batt_capacity_cost = [0.0]
            loan.SystemCosts.om_batt_variable_cost = [0.0]
            loan.SystemCosts.om_batt_replacement_cost = [0.0]
            system_costs = costs['system_capex_per_kw'] * kw

        batt_costs = costs['batt_capex_per_kwh_combined'] * batt.Outputs.batt_bank_installed_capacity
        value_of_resiliency = agent.loc['value_of_resiliency_usd']

    else:
        # No battery
        batt.BatterySystem.en_batt = 0
        loan.BatterySystem.en_batt = 0
        loan.LCOS.batt_annual_charge_energy = [0]
        loan.LCOS.batt_annual_charge_from_system = [0]
        loan.LCOS.batt_annual_discharge_energy = [0]
        loan.LCOS.batt_capacity_percent = [0]
        loan.LCOS.batt_salvage_percentage = 0
        loan.LCOS.battery_total_cost_lcos = 0

        # Rate switch for solar-only case
        if kw > 0:
            agent, one_time_charge = agent_mutation.elec.apply_rate_switch(
                rate_switch_table, agent, kw, tech='solar'
            )
        else:
            one_time_charge = 0.0

        # Rebuild Utilityrate5 FROM the switched tariff (TOU + schedules)
        net_billing_sell_rate = 0
        ts_sell = np.asarray(agent.loc['wholesale_prices'], dtype=float).ravel() * agent.loc['elec_price_multiplier']
        td_norm = normalize_tariff(agent.loc['tariff_dict'], net_sell_rate_scalar=net_billing_sell_rate)
        process_tariff(utilityrate, td_norm, net_billing_sell_rate, ts_sell_rate=ts_sell)

        utilityrate.SystemOutput.gen = gen
        loan.SystemCosts.add_om_num_types = 0
        loan.SystemCosts.om_batt_replacement_cost = [0.0]
        loan.SystemCosts.om_batt_nameplate = 0
        system_costs = costs['system_capex_per_kw'] * kw
        batt_costs = 0.0
        value_of_resiliency = 0.0

    # ---- Rate engine ----
    utilityrate.Load.load = load_hourly
    utilityrate.execute()


    # ---- Financials ----
    loan.FinancialParameters.system_capacity = kw

    aev = list(utilityrate.Outputs.annual_energy_value)
    annual_energy_value = [aev[0] + value_of_resiliency] + [x + value_of_resiliency for x in aev[1:]]

    loan.SystemOutput.annual_energy_value = annual_energy_value
    loan.SystemOutput.gen = utilityrate.SystemOutput.gen

    direct_costs = (system_costs + batt_costs) * costs['cap_cost_multiplier']
    sales_tax = 0.0
    loan.SystemCosts.total_installed_cost = direct_costs + sales_tax + one_time_charge

    #  No ITC for batteries
    loan.TaxCreditIncentives.itc_fed_percent = [0.0]

    loan.execute()
    return -loan.Outputs.npv

def calc_system_size_and_performance(con, agent: pd.Series, sectors, rate_switch_table=None):
    """
    Same behavior as before, but stashes Outputs as dicts:
      _util_out_pv_only, _batt_out_pv_only, _loan_out_pv_only
      _util_out_pv_batt, _batt_out_pv_batt, _loan_out_pv_batt
    and per-case sizes:
      _pv_only_system_kw, _pv_batt_system_kw, _pv_batt_batt_kw, _pv_batt_batt_kwh
    """
    import numpy as np
    from scipy import optimize
    import PySAM.Battery as battery
    import PySAM.Utilityrate5 as utility
    import PySAM.Cashloan as cashloan
    import PySAM.CustomGeneration as customgen
    import PySAM.Pvsamv1 as pvsamv1
    import PySAM.Pvwattsv8 as pvwattsv8
    import agent_mutation

    cur = con.cursor()
    agent = agent.copy()
    if 'agent_id' not in agent.index:
        agent.loc['agent_id'] = agent.name

    # --- Load & resource
    lp = agent_mutation.elec.get_and_apply_agent_load_profiles(con, agent)
    cons = np.asarray(lp['consumption_hourly'].iloc[0], dtype=float)
    norm = agent_mutation.elec.get_and_apply_normalized_hourly_resource_solar(con, agent)
    gen_per_kw = np.asarray(norm['solar_cf_profile'].iloc[0], dtype=float) / 1e6
    agent.loc['naep'] = float(gen_per_kw.sum())
    pv = {'consumption_hourly': cons, 'generation_hourly': gen_per_kw}

    # --- Build modules (same as before)
    def _init_pv_batt_stack(sector_abbr: str):
        if sector_abbr == 'res':
            cands = [
                ("CustomGenerationBatteryResidential", customgen),
                ("PVWattsBatteryResidential",          pvwattsv8),
                ("PVBatteryResidential",               pvsamv1),
            ]
            market = 0
        else:
            cands = [
                ("CustomGenerationBatteryCommercial",  customgen),
                ("PVWattsBatteryCommercial",           pvwattsv8),
                ("PVBatteryCommercial",                pvsamv1),
            ]
            market = 1
        for cfg, drv_mod in cands:
            try:
                driver = drv_mod.default(cfg)
                batt   = battery.from_existing(driver,   cfg)
                util   = utility.from_existing(driver,   cfg)
                loanm  = cashloan.from_existing(driver,  cfg)
                return driver, batt, util, loanm, market
            except Exception:
                continue
        driver = None
        batt   = battery.new()
        util   = utility.new()
        loanm  = cashloan.new()
        return driver, batt, util, loanm, market

    driver_mod, batt, utilityrate, loan, market_flag = _init_pv_batt_stack(agent.loc['sector_abbr'])
    loan.FinancialParameters.market = market_flag

    # --- Configure Utilityrate/Cashloan (same as your current version)
    utilityrate.Lifetime.inflation_rate = agent.loc['inflation_rate'] * 100
    utilityrate.Lifetime.analysis_period = agent.loc['economic_lifetime_yrs']
    utilityrate.Lifetime.system_use_lifetime_output = 0
    utilityrate.SystemOutput.degradation = [agent.loc['pv_degradation_factor'] * 100]
    utilityrate.ElectricityRates.rate_escalation = [agent.loc['elec_price_escalator'] * 100]
    utilityrate.ElectricityRates.ur_nm_yearend_sell_rate = 0
    utilityrate.ElectricityRates.ur_sell_eq_buy = 0
    utilityrate.ElectricityRates.TOU_demand_single_peak = 0
    utilityrate.ElectricityRates.en_electricity_rates = 1

    ts_sell = np.asarray(agent.loc['wholesale_prices'], dtype=float).ravel() * agent.loc['elec_price_multiplier']
    tariff_dict = normalize_tariff(agent.loc['tariff_dict'], net_sell_rate_scalar=0)
    process_tariff(utilityrate, tariff_dict, 0, ts_sell_rate=ts_sell)

    loan.FinancialParameters.analysis_period = agent.loc['economic_lifetime_yrs']
    loan.FinancialParameters.debt_fraction = 100 - (agent.loc['down_payment_fraction'] * 100)
    loan.FinancialParameters.federal_tax_rate = [(agent.loc['tax_rate'] * 100) * 0.7]
    loan.FinancialParameters.inflation_rate = agent.loc['inflation_rate'] * 100
    loan.FinancialParameters.insurance_rate = 0
    loan.FinancialParameters.loan_rate = agent.loc['loan_interest_rate'] * 100
    loan.FinancialParameters.loan_term = agent.loc['loan_term_yrs']
    loan.FinancialParameters.mortgage = 0
    loan.FinancialParameters.prop_tax_assessed_decline = 5
    loan.FinancialParameters.prop_tax_cost_assessed_percent = 95
    loan.FinancialParameters.property_tax_rate = 0
    loan.FinancialParameters.real_discount_rate = agent.loc['real_discount_rate'] * 100
    loan.FinancialParameters.salvage_percentage = 0
    loan.FinancialParameters.state_tax_rate = [(agent.loc['tax_rate'] * 100) * 0.3]
    loan.FinancialParameters.system_heat_rate = 0

    # costs bundle (same as before)
    sc = {
        'system_capex_per_kw':                 agent.loc['system_capex_per_kw'],
        'system_om_per_kw':                    agent.loc['system_om_per_kw'],
        'system_variable_om_per_kw':           agent.loc['system_variable_om_per_kw'],
        'cap_cost_multiplier':                 agent.loc['cap_cost_multiplier'],
        'batt_capex_per_kw':                   agent.loc['batt_capex_per_kw'],
        'batt_capex_per_kwh':                  agent.loc['batt_capex_per_kwh'],
        'batt_om_per_kw':                      agent.loc['batt_om_per_kw'],
        'batt_om_per_kwh':                     agent.loc['batt_om_per_kwh'],
        'linear_constant':                     agent.loc['linear_constant'],
        'system_capex_per_kw_combined':        agent.loc['system_capex_per_kw_combined'],
        'system_om_per_kw_combined':           agent.loc['system_om_per_kw'],
        'system_variable_om_per_kw_combined':  agent.loc['system_variable_om_per_kw'],
        'batt_capex_per_kw_combined':          agent.loc['batt_capex_per_kw_combined'],
        'batt_capex_per_kwh_combined':         agent.loc['batt_capex_per_kwh_combined'],
        'batt_om_per_kw_combined':             agent.loc['batt_om_per_kw_combined'],
        'batt_om_per_kwh_combined':            agent.loc['batt_om_per_kwh_combined'],
        'linear_constant_combined':            agent.loc['linear_constant_combined']
    }

    if agent.loc['sector_abbr'] == 'res':
        loan.Depreciation.depr_fed_type = 0
        loan.Depreciation.depr_sta_type = 0
    else:
        loan.Depreciation.depr_fed_type = 1
        loan.Depreciation.depr_sta_type = 0

    loan.TaxCreditIncentives.itc_fed_percent             = [agent.loc['itc_fraction_of_capex'] * 100]
    loan.BatterySystem.batt_replacement_option           = 2
    loan.BatterySystem.batt_replacement_schedule_percent = [0] * (agent.loc['batt_lifetime_yrs'] - 1) + [1]
    loan.SystemOutput.degradation                        = [agent.loc['pv_degradation_factor'] * 100]
    loan.Lifetime.system_use_lifetime_output             = 0

    # --- Objective wrappers (use your existing calc_system_performance)
    max_load   = agent.loc['load_kwh_per_customer_in_bin'] / agent.loc['naep']
    max_system = max_load
    tol        = min(0.25 * max_system, 0.25)
    low        = max_system * 0.8
    high       = max_system * 1.25
    batt_disp  = 'peak_shaving' if agent.loc['sector_abbr'] != 'res' else 'price_signal_forecast'

    def perf_no_batt(x):
        return calc_system_performance(
            x, pv, utilityrate, loan, batt, sc, agent, rate_switch_table, False, 0
        )

    def perf_with_batt(x, _batt_kwh=None, _batt_kw=None):
        return calc_system_performance(
            x, pv, utilityrate, loan, batt, sc, agent, rate_switch_table,
            True, batt_disp, batt_kwh=_batt_kwh, batt_power_kw=_batt_kw
        )

    # ---- PV only optimize ----
    res_n = optimize.minimize_scalar(
        perf_no_batt,
        bounds=(low, high),
        method='bounded',
        options={'xatol': max(2, tol)}
    )
    # Snapshot pv_only **Outputs dicts**
    util_out_pv_only = utilityrate.Outputs.export()
    util_out_pv_only = _attach_hourly_prices(utilityrate, util_out_pv_only) 
    util_out_pv_only['load_kwh'] = list(utilityrate.Load.load)
    util_out_pv_only['year1_hourly_e_fromgrid'] = list(utilityrate.Outputs.year1_hourly_e_fromgrid)
    util_out_pv_only['year1_hourly_e_togrid']   = list(utilityrate.Outputs.year1_hourly_e_togrid)
    util_out_pv_only['pv_generation'] = list(utilityrate.SystemOutput.gen)
    batt_out_pv_only = None
    loan_out_pv_only = loan.Outputs.export()
    pv_only_kw       = float(res_n.x)

    out_n_loan   = loan_out_pv_only
    out_n_util   = util_out_pv_only
    gen_n_annual = float(np.nansum(np.asarray(getattr(utilityrate.SystemOutput, "gen", []), dtype=float)))
    load_n_ts    = np.asarray(getattr(utilityrate.Load, "load", []), dtype=float)
    _pv_n_raw    = util_out_pv_only.get("year1_hourly_system_to_load", None)
    if _pv_n_raw is None or len(_pv_n_raw) == 0:
        _pv_n_raw = getattr(utilityrate.SystemOutput, "gen", [])
    pv_n_ts  = np.asarray(_pv_n_raw, dtype=float)
    npv_n    = float(out_n_loan.get("npv", float("-inf")))

    # ---- Battery sweep (at pv*)
    pv_star = float(res_n.x)
    hours_candidates = [2.0, 3.0, 4.0]
    c_rate_hours     = 2.0
    best_batt = {'npv': float('-inf'), 'kwh': 0.0, 'kw': 0.0}
    for h in hours_candidates:
        kwh = h * pv_star
        kwp = kwh / c_rate_hours
        neg_npv = perf_with_batt(pv_star, _batt_kwh=kwh, _batt_kw=kwp)
        if -neg_npv > best_batt['npv']:
            best_batt.update({'npv': -neg_npv, 'kwh': kwh, 'kw': kwp})

    util_out_pv_batt = None; batt_out_pv_batt = None; loan_out_pv_batt = None
    if best_batt['kwh'] > 0:
        _ = perf_with_batt(pv_star, _batt_kwh=best_batt['kwh'], _batt_kw=best_batt['kw'])
        # Snapshot pv_batt **Outputs dicts**
        util_out_pv_batt = utilityrate.Outputs.export()
        util_out_pv_batt = _attach_hourly_prices(utilityrate, util_out_pv_batt)
        util_out_pv_batt['load_kwh'] = list(utilityrate.Load.load)
        util_out_pv_batt['grid_to_load'] = list(batt.Outputs.grid_to_load)
        util_out_pv_batt['system_to_grid'] = list(batt.Outputs.system_to_grid)
        util_out_pv_batt['system_to_load'] = list(batt.Outputs.system_to_load)
        util_out_pv_batt['pv_generation'] = list(utilityrate.SystemOutput.gen)
        util_out_pv_batt['year1_hourly_e_fromgrid'] = list(utilityrate.Outputs.year1_hourly_e_fromgrid)
        batt_out_pv_batt = batt.Outputs.export()
        loan_out_pv_batt = loan.Outputs.export()

        out_b_loan = loan_out_pv_batt
        out_b_util = util_out_pv_batt
        gen_w_annual = float(np.nansum(np.asarray(getattr(utilityrate.SystemOutput, "gen", []), dtype=float)))
        kw_w   = float(batt_out_pv_batt.get("batt_bank_installed_capacity", 0.0))  # note: Outputs gives capacity (kWh); power may need another field
        # Try max discharge power if present
        batt_kw_field = "batt_power"
        if "batt_to_load" in batt_out_pv_batt:
            # estimate peak AC discharge kW
            kw_w_est = float(np.nanmax(np.asarray(batt_out_pv_batt["batt_to_load"], dtype=float))) if len(batt_out_pv_batt["batt_to_load"]) else 0.0
        else:
            kw_w_est = 0.0
        load_w_ts = np.asarray(getattr(utilityrate.Load, "load", []), dtype=float)
        _pv_w_raw = out_b_util.get("year1_hourly_system_to_load", None)
        if _pv_w_raw is None or len(_pv_w_raw) == 0:
            _pv_w_raw = getattr(utilityrate.SystemOutput, "gen", [])
        pv_w_ts  = np.asarray(_pv_w_raw, dtype=float)
        npv_w    = float(out_b_loan.get("npv", float("-inf")))
        btl_w_ts = np.asarray(batt_out_pv_batt.get("batt_to_load", []), dtype=float)
    else:
        npv_w = float('-inf')

    # Winner for agent scalars (unchanged)
    def _align_to_n(*arrays):
        sizes = [a.size for a in arrays if isinstance(a, np.ndarray)]
        n = max(sizes) if sizes else 0
        out = []
        for a in arrays:
            a = np.asarray(a, dtype=float).ravel()
            if a.size == 0:
                out.append(np.zeros(n, dtype=float))
            elif a.size < n:
                out.append(np.pad(a, (0, n - a.size)))
            else:
                out.append(a[:n])
        return out

    if npv_w >= npv_n:
        system_kw     = float(pv_star)
        annual_kwh    = gen_w_annual
        first_with    = float(out_b_util.get('utility_bill_w_sys_year1', 0.0))
        first_without = float(out_b_util.get('utility_bill_wo_sys_year1', 0.0))
        npv_final     = npv_w
        cash_flow     = list(out_b_loan.get('cf_payback_with_expenses', []) or [])
        payback       = float(out_b_loan.get('payback', np.nan))
        batt_kw       = best_batt['kw']
        batt_kwh      = float(batt_out_pv_batt.get("batt_bank_installed_capacity", 0.0))
        load_ts, pv_ts, btl_ts = _align_to_n(load_w_ts, pv_w_ts, btl_w_ts)
        case = "pv_batt"
    else:
        system_kw     = float(res_n.x)
        annual_kwh    = gen_n_annual
        first_with    = float(out_n_util.get('utility_bill_w_sys_year1', 0.0))
        first_without = float(out_n_util.get('utility_bill_wo_sys_year1', 0.0))
        npv_final     = npv_n
        cash_flow     = list(out_n_loan.get('cf_payback_with_expenses', []) or [])
        payback       = float(out_n_loan.get('payback', np.nan))
        batt_kw       = 0.0
        batt_kwh      = 0.0
        load_ts, pv_ts = _align_to_n(load_n_ts, pv_n_ts)
        btl_ts = np.zeros_like(load_ts)
        case = "pv_only"

    if not first_without or first_without == 0:
        first_without = 1.0

    agent.loc['baseline_net_hourly'] = np.asarray(cons, dtype=float)[:load_ts.size].tolist()
    adopter_load_ts  = load_ts
    adopter_pv_ts    = pv_ts
    adopter_batt_ts  = btl_ts
    adopter_net_ts   = np.maximum(adopter_load_ts - adopter_pv_ts - adopter_batt_ts, 0.0)

    agent.loc['adopter_load_hourly'] = adopter_load_ts.tolist()
    agent.loc['adopter_pv_hourly']   = adopter_pv_ts.tolist()
    agent.loc['adopter_batt_hourly'] = adopter_batt_ts.tolist()
    agent.loc['adopter_net_hourly']  = adopter_net_ts.tolist()

    naep_final   = annual_kwh / max(system_kw, 1e-9)
    savings      = first_without - first_with
    savings_frac = savings / first_without
    avg_price    = first_without / agent.loc['load_kwh_per_customer_in_bin']

    agent.loc['system_kw']                           = system_kw
    agent.loc['batt_kw']                             = best_batt['kw']
    agent.loc['batt_kwh']                            = best_batt['kwh']
    agent.loc['npv']                                 = npv_final
    agent.loc['payback_period']                      = float(np.round(payback if np.isfinite(payback) else 30.1, 1))
    agent.loc['cash_flow']                           = cash_flow
    agent.loc['annual_energy_production_kwh']        = annual_kwh
    agent.loc['naep']                                = naep_final
    agent.loc['capacity_factor']                     = naep_final / 8760.0
    agent.loc['first_year_elec_bill_with_system']    = first_with
    agent.loc['first_year_elec_bill_savings']        = savings
    agent.loc['first_year_elec_bill_savings_frac']   = savings_frac
    agent.loc['max_system_kw']                       = max_system
    agent.loc['first_year_elec_bill_without_system'] = first_without
    agent.loc['avg_elec_price_cents_per_kwh']        = avg_price
    agent.loc['pv_per_kw_hourly']                    = gen_per_kw.tolist()

    agent.loc['_util_out_pv_only'] = util_out_pv_only
    agent.loc['_batt_out_pv_only'] = batt_out_pv_only
    agent.loc['_loan_out_pv_only'] = loan_out_pv_only
    agent.loc['_pv_only_system_kw'] = pv_only_kw

    agent.loc['_util_out_pv_batt'] = util_out_pv_batt
    agent.loc['_batt_out_pv_batt'] = batt_out_pv_batt
    agent.loc['_loan_out_pv_batt'] = loan_out_pv_batt
    agent.loc['_pv_batt_system_kw'] = float(pv_star) if util_out_pv_batt is not None else 0.0

    # Use the best battery candidate, not the winner-branch locals
    agent.loc['_pv_batt_batt_kw']  = float(best_batt['kw'])  if util_out_pv_batt is not None else 0.0
    agent.loc['_pv_batt_batt_kwh'] = float(best_batt['kwh']) if util_out_pv_batt is not None else 0.0

    agent.loc['_case_for_hourly']   = case  # winner (unchanged)

    cur.close()
    return agent


#==============================================================================
# UtilityRate5 processing
#==============================================================================

def process_tariff(utilityrate, tariff_dict, net_billing_sell_rate, ts_sell_rate=None, ts_buy_rate=None):
    """
    Apply tariff safely for Utilityrate5.

    Policy:
      • Force Net Billing (mo=2) when FORCE_NET_BILLING=True.
      • Preserve TOU energy (buy) via ur_ec_tou_mat + schedules.
      • Allow 8760 SELL under Net Billing. (TS BUY optional; TOU buy is typical.)
      • Skip demand-charge arrays unless explicitly enabled & not globally skipped.
    """
    # ---- Metering option ----
    mo_in = int(tariff_dict.get('ur_metering_option', 0))
    mo = 2 if FORCE_NET_BILLING else mo_in 
    utilityrate.ElectricityRates.ur_metering_option = mo

    # ---- Fixed charges ----
    ER = utilityrate.ElectricityRates
    ER.ur_monthly_fixed_charge = float(tariff_dict.get('ur_monthly_fixed_charge', 0.0))
    ER.ur_annual_min_charge = 0.0
    ER.ur_monthly_min_charge = 0.0

    # ---- Demand charges (skipped unless truly present and you opt-in) ----
    dc_flat_raw = tariff_dict.get('ur_dc_flat_mat')
    dc_tou_raw  = tariff_dict.get('ur_dc_tou_mat')
    dc_flag     = bool(tariff_dict.get('ur_dc_enable', 0)) or bool(dc_flat_raw) or bool(dc_tou_raw)

    if SKIP_DEMAND_CHARGES or not dc_flag:
        ER.ur_dc_enable = 0
        ER.ur_enable_billing_demand = 0
    else:
        dc_wkday = _sched_12x24(tariff_dict.get('ur_dc_sched_weekday'))
        dc_wkend = _sched_12x24(tariff_dict.get('ur_dc_sched_weekend'))
        dc_flat  = _mat2d(dc_flat_raw)
        dc_tou   = _mat2d(dc_tou_raw)
        dc_enable = bool(dc_flat) or bool(dc_tou)
        ER.ur_dc_sched_weekday = dc_wkday
        ER.ur_dc_sched_weekend = dc_wkend
        ER.ur_dc_flat_mat = dc_flat if dc_enable else []
        ER.ur_dc_tou_mat  = dc_tou  if dc_enable else []
        ER.ur_enable_billing_demand = 0
        ER.ur_dc_enable = int(dc_enable)

    # ---- Energy charges (TOU BUY) ----
    ec_tou = tariff_dict.get('ur_ec_tou_mat')
    if ec_tou:
        ER.ur_ec_tou_mat = _mat2d(ec_tou)
        ER.ur_ec_sched_weekday = _sched_12x24(tariff_dict.get('ur_ec_sched_weekday'))
        ER.ur_ec_sched_weekend = _sched_12x24(tariff_dict.get('ur_ec_sched_weekend'))

    # ---- Time-series SELL / BUY under Net Billing ----
    if mo == 2:
        sell = _list1d_8760(ts_sell_rate)
        if sell is not None:
            ER.ur_en_ts_sell_rate = 1
            ER.ur_ts_sell_rate = sell
        else:
            ER.ur_en_ts_sell_rate = 0
            ER.ur_ts_sell_rate = [0.0]

        buy = _list1d_8760(ts_buy_rate)
        if buy is not None:
            ER.ur_en_ts_buy_rate = 1
            ER.ur_ts_buy_rate = buy
        else:
            ER.ur_en_ts_buy_rate = 0
            # TOU buy remains active from ur_ec_tou_mat
    else:
        ER.ur_en_ts_sell_rate = 0
        ER.ur_ts_sell_rate    = [0.0]
        ER.ur_en_ts_buy_rate  = 0

    return utilityrate


#==============================================================================
# Tariff normalization helpers
#==============================================================================

def _parse_tariff_dict(raw) -> Dict[str, Any]:
    """Accept dict or string; coerce to dict, tolerating 'nan'/None/None-like."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    s = raw
    s = s.replace("'", '"')
    s = re.sub(r'\b(nan|none|null)\b', 'null', s, flags=re.IGNORECASE)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return {}

def _num(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip().lower() in {"", "nan", "none", "null"}:
            return default
        return float(x)
    except Exception:
        return default

def _coerce_bool(x, default=False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(default)

def _sched_12x24_single_period() -> List[List[int]]:
    return [[1] * 24 for _ in range(12)]

def _plus1_sched(mat) -> List[List[int]]:
    if not mat:
        return _sched_12x24_single_period()
    out: List[List[int]] = []
    for r in range(12):
        row = mat[r] if r < len(mat) else []
        fixed = []
        for c in range(24):
            v = row[c] if c < len(row) else 0
            try:
                fixed.append(int(v) + 1)
            except Exception:
                fixed.append(1)
        out.append(fixed)
    return out

def _sched_12x24(x) -> List[List[int]]:
    if x is None:
        return [[0]*24 for _ in range(12)]
    a = np.asarray(x)
    if a.ndim != 2:
        return [[0]*24 for _ in range(12)]
    r, c = a.shape[0], a.shape[1] if a.ndim == 2 else (0, 0)
    if r == 12 and c == 24:
        return a.astype(np.int32, copy=False).tolist()
    if r >= 12 and c >= 24:
        return a[:12, :24].astype(np.int32, copy=False).tolist()
    if r <= 12 and c <= 24:
        z = np.zeros((12, 24), dtype=np.int32)
        z[:r, :c] = np.asarray(a, dtype=np.int32)
        return z.tolist()
    return [[0]*24 for _ in range(12)]

def _mat2d(x, size_limit: int = 4096) -> List[List[float]]:
    if x is None or x == []:
        return []
    a = np.asarray(x)
    if a.ndim != 2 or a.size == 0 or a.size > size_limit:
        return []
    if not np.isfinite(a.astype(np.float64, copy=False)).all():
        return []
    return a.astype(np.float32, copy=False).tolist()

def _list1d_8760(x: Optional[Iterable[float]]) -> Optional[List[float]]:
    if x is None:
        return None
    try:
        a = np.asarray(x, dtype=np.float32).ravel()
    except Exception:
        return None
    if a.size != 8760 or not np.isfinite(a).all():
        return None
    return a.tolist()

def _build_ur_ec_from_e_parts(td: Dict[str, Any], net_sell_rate_scalar=0.0) -> List[List[float]]:
    prices = td.get('e_prices') or []
    if not prices:
        return []
    levels = td.get('e_levels') or []
    n_tiers = len(prices)
    n_periods = len(prices[0]) if n_tiers else 0
    BIG = 1e38
    if (not levels) or (len(levels) != n_tiers) or any(len(L) != n_periods for L in levels):
        levels = [[BIG] * n_periods for _ in range(n_tiers)]
    unit_map = {'kWh': 0, 'kWh/kW': 1, 'kWh daily': 2, 'kWh/kW daily': 3}
    ucode = unit_map.get(str(td.get('energy_rate_unit', 'kWh')), 0)

    rows: List[List[float]] = []
    for p in range(n_periods):
        for t in range(n_tiers):
            rows.append([
                float(p + 1), float(t + 1),
                float(levels[t][p]),
                float(ucode),
                float(prices[t][p]),
                float(net_sell_rate_scalar),
            ])
    return rows

def _build_ur_dc_from_d_parts(td: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    out = {'ur_dc_flat_mat': [], 'ur_dc_tou_mat': []}

    dfl, dfr = td.get('d_flat_levels') or [], td.get('d_flat_prices') or []
    if dfl and dfr:
        n_tiers = len(dfl); n_periods = len(dfl[0]) if n_tiers else 0
        flat = []
        for p in range(n_periods):
            for t in range(n_tiers):
                flat.append([p + 1, t + 1, float(dfl[t][p]), float(dfr[t][p])])
        out['ur_dc_flat_mat'] = flat

    dtl, dtr = td.get('d_tou_levels') or [], td.get('d_tou_prices') or []
    if dtl and dtr:
        n_tiers = len(dtl); n_periods = len(dtl[0]) if n_tiers else 0
        tou = []
        for p in range(n_periods):
            for t in range(n_tiers):
                tou.append([p + 1, t + 1, float(dtl[t][p]), float(dtr[t][p])])
        out['ur_dc_tou_mat'] = tou

    out['ur_dc_sched_weekday'] = _plus1_sched(td.get('ur_dc_sched_weekday') or td.get('d_wkday_12by24'))
    out['ur_dc_sched_weekend'] = _plus1_sched(td.get('ur_dc_sched_weekend') or td.get('d_wkend_12by24'))

    dc_enable = 1 if (out['ur_dc_flat_mat'] or out['ur_dc_tou_mat'] or _coerce_bool(td.get('d_flat_exists')) or _coerce_bool(td.get('d_tou_exists'))) else 0
    return out, dc_enable

def _reconcile_periods_and_equalize_tiers(ec_tou_mat, wk_sched, we_sched):
    BIG = 1e38

    tou = np.asarray(ec_tou_mat or [], dtype=float)
    wk  = np.asarray(wk_sched or [], dtype=int)
    we  = np.asarray(we_sched or [], dtype=int)

    if tou.size == 0:
        wk = np.asarray(_sched_12x24(wk.tolist()), dtype=int)
        we = np.asarray(_sched_12x24(we.tolist()), dtype=int)
        wk[wk < 1] = 1
        we[we < 1] = 1
        return [], wk.tolist(), we.tolist()

    per_ids = np.unique(tou[:, 0].astype(int))
    if per_ids.size == 0:
        tou[:, 0] = 1
        per_ids = np.array([1], dtype=int)

    new_ids = np.arange(1, per_ids.size + 1, dtype=int)
    remap = {int(old): int(new) for old, new in zip(per_ids.tolist(), new_ids.tolist())}
    tou[:, 0] = np.vectorize(lambda x: remap.get(int(x), 1))(tou[:, 0])

    max_tiers = 0
    tiers_by_period: Dict[int, List[np.ndarray]] = {}
    for p in np.unique(tou[:, 0].astype(int)):
        rows_p = tou[tou[:, 0] == p]
        rows_p = rows_p[np.argsort(rows_p[:, 1])]
        tiers_by_period[p] = [rows_p[rows_p[:, 1] == t] for t in np.unique(rows_p[:, 1].astype(int))]
        max_tiers = max(max_tiers, len(tiers_by_period[p]))

    fixed_rows = []
    for p in sorted(tiers_by_period.keys()):
        rows_p = tou[tou[:, 0] == p]
        rows_p = rows_p[np.argsort(rows_p[:, 1])]
        have_tiers = [int(t) for t in np.unique(rows_p[:, 1].astype(int))]
        if len(have_tiers) == max_tiers:
            for i, row in enumerate(rows_p):
                row[1] = float(i + 1)
            fixed_rows.append(rows_p)
            continue

        last = rows_p[-1].copy()
        unit_code = last[3] if rows_p.shape[1] >= 4 else 0.0
        price     = last[4] if rows_p.shape[1] >= 5 else 0.0
        nsell     = last[5] if rows_p.shape[1] >= 6 else 0.0

        padded = [r.copy() for r in rows_p]
        for add_t in range(len(have_tiers) + 1, max_tiers + 1):
            padded.append(np.array([float(p), float(add_t), float(BIG), float(unit_code), float(price), float(nsell)], dtype=float))
        padded = np.vstack(padded)
        padded = padded[np.argsort(padded[:, 1])]
        for i in range(padded.shape[0]):
            padded[i, 1] = float(i + 1)
        fixed_rows.append(padded)

    tou_fixed = np.vstack(fixed_rows)

    wk = np.asarray(_sched_12x24(wk.tolist()), dtype=int)
    we = np.asarray(_sched_12x24(we.tolist()), dtype=int)
    wk[wk < 1] = 1
    we[we < 1] = 1
    P = int(np.max(tou_fixed[:, 0]).astype(int))
    wk[wk > P] = 1
    we[we > P] = 1

    tou_fixed = tou_fixed[np.lexsort((tou_fixed[:, 1], tou_fixed[:, 0]))]
    return tou_fixed.astype(np.float32).tolist(), wk.tolist(), we.tolist()

def _harmonize_tier_caps_and_units(ec_tou_mat: List[List[float]]) -> List[List[float]]:
    if not ec_tou_mat:
        return []

    tou = np.asarray(ec_tou_mat, dtype=float)  # [period, tier, max_usage, unit_code, price, net_sell]
    if tou.ndim != 2 or tou.shape[1] < 6:
        return []

    tiers = np.unique(tou[:, 1].astype(int))
    BIG = 1e38
    BIG_THRESH = 1e37

    unit_codes = tou[:, 3].astype(int)
    min_uc = unit_codes.min()
    shift = 0 if min_uc >= 0 else -min_uc
    counts = np.bincount((unit_codes + shift).astype(int))
    unit_code_mode = int(np.argmax(counts) - shift)

    for t in tiers:
        rows_t = (tou[:, 1].astype(int) == t)
        caps_t = tou[rows_t, 2]
        finite_caps = caps_t[(caps_t > 0) & (caps_t < BIG_THRESH)]
        cap = float(np.min(finite_caps)) if finite_caps.size else float(BIG)
        tou[rows_t, 2] = cap

    tou[:, 3] = float(unit_code_mode)
    tou = tou[np.lexsort((tou[:, 1], tou[:, 0]))]
    return tou.astype(np.float32).tolist()

def normalize_tariff(raw, net_sell_rate_scalar=0.0, debug=False):
    td = _parse_tariff_dict(raw)
    out = {}

    # Always enable electricity rates
    out['en_electricity_rates'] = int(td.get('en_electricity_rates', 1))

    # Force Net Billing if configured, else honor input (0=NM, 1=Net Billing, 2=BA/SA)
    mo_in = int(td.get('ur_metering_option', 0))
    out['ur_metering_option'] = 1 if FORCE_NET_BILLING else mo_in

    # Fixed charge
    fc = td.get('ur_monthly_fixed_charge', td.get('fixed_charge', 0.0))
    out['ur_monthly_fixed_charge'] = _num(fc, 0.0)

    # Energy structure (BUY)
    ec_tou = td.get('ur_ec_tou_mat') or _build_ur_ec_from_e_parts(td, net_sell_rate_scalar)
    wkday  = td.get('ur_ec_sched_weekday') or _plus1_sched(td.get('e_wkday_12by24')) or _sched_12x24_single_period()
    wkend  = td.get('ur_ec_sched_weekend') or _plus1_sched(td.get('e_wkend_12by24')) or _sched_12x24_single_period()

    ec_tou, wkday, wkend = _reconcile_periods_and_equalize_tiers(ec_tou, wkday, wkend)
    ec_tou = _harmonize_tier_caps_and_units(ec_tou)

    out['ur_ec_tou_mat']        = ec_tou
    out['ur_ec_sched_weekday']  = wkday
    out['ur_ec_sched_weekend']  = wkend

    # Demand charge structure
    dc_mats, dc_enable_guess = _build_ur_dc_from_d_parts(td)
    out['ur_dc_flat_mat']        = td.get('ur_dc_flat_mat') or dc_mats['ur_dc_flat_mat'] or []
    out['ur_dc_tou_mat']         = td.get('ur_dc_tou_mat')  or dc_mats['ur_dc_tou_mat']  or []
    out['ur_dc_sched_weekday']   = td.get('ur_dc_sched_weekday') or dc_mats['ur_dc_sched_weekday'] or _sched_12x24_single_period()
    out['ur_dc_sched_weekend']   = td.get('ur_dc_sched_weekend') or dc_mats['ur_dc_sched_weekend'] or _sched_12x24_single_period()
    out['ur_dc_enable']          = int(td.get('ur_dc_enable', dc_enable_guess))
    out['ur_enable_billing_demand'] = bool(td.get('ur_enable_billing_demand', False))

    if debug or os.environ.get("DGEN_DEBUG"):
        need = ("ur_monthly_fixed_charge","ur_ec_tou_mat","ur_ec_sched_weekday","ur_ec_sched_weekend")
        miss = [k for k in need if k not in out or out[k] in (None, [], "", "nan")]
        if miss:
            print(f"[DEBUG] normalize_tariff filled/left defaults: {miss}")

    return out


#==============================================================================
# Incentives & post-processing helpers
#==============================================================================

def process_incentives(loan, kw, batt_kw, batt_kwh, generation_hourly, agent):
    """
    Apply state incentives (CBI/PBI/IBI) from agent['state_incentives'] to the Cashloan object.
    Leaves fields untouched if incentives are not present.
    """
    incentive_df = agent.loc['state_incentives']

    if not isinstance(incentive_df, pd.DataFrame):
        return loan

    # Fill NaNs with conservative defaults
    incentive_df = incentive_df.fillna(value={'incentive_duration_yrs': 5, 'max_incentive_usd': 10000})

    # ---- Capacity-based incentives (CBI) ----
    cbi_df = (
        incentive_df.loc[pd.notnull(incentive_df['cbi_usd_p_w'])]
        .sort_values(['cbi_usd_p_w'], ascending=False)
        .reset_index(drop=True)
    )
    if len(cbi_df) == 1:
        loan.PaymentIncentives.cbi_sta_amount = cbi_df['cbi_usd_p_w'].iloc[0]
        loan.PaymentIncentives.cbi_sta_deprbas_fed = 0
        loan.PaymentIncentives.cbi_sta_deprbas_sta = 0
        loan.PaymentIncentives.cbi_sta_maxvalue = cbi_df['max_incentive_usd'].iloc[0]
        loan.PaymentIncentives.cbi_sta_tax_fed = 0
        loan.PaymentIncentives.cbi_sta_tax_sta = 0
    elif len(cbi_df) >= 2:
        loan.PaymentIncentives.cbi_sta_amount = cbi_df['cbi_usd_p_w'].iloc[0]
        loan.PaymentIncentives.cbi_sta_deprbas_fed = 0
        loan.PaymentIncentives.cbi_sta_deprbas_sta = 0
        loan.PaymentIncentives.cbi_sta_maxvalue = cbi_df['max_incentive_usd'].iloc[0]
        loan.PaymentIncentives.cbi_sta_tax_fed = 1
        loan.PaymentIncentives.cbi_sta_tax_sta = 1

        loan.PaymentIncentives.cbi_oth_amount = cbi_df['cbi_usd_p_w'].iloc[1]
        loan.PaymentIncentives.cbi_oth_deprbas_fed = 0
        loan.PaymentIncentives.cbi_oth_deprbas_sta = 0
        loan.PaymentIncentives.cbi_oth_maxvalue = cbi_df['max_incentive_usd'].iloc[1]
        loan.PaymentIncentives.cbi_oth_tax_fed = 1
        loan.PaymentIncentives.cbi_oth_tax_sta = 1

    # ---- Production-based incentives (PBI) ----
    pbi_df = (
        incentive_df.loc[pd.notnull(incentive_df['pbi_usd_p_kwh'])]
        .sort_values(['pbi_usd_p_kwh'], ascending=False)
        .reset_index(drop=True)
    )
    agent.loc['timesteps_per_year'] = 1
    pv_kwh_by_year = np.array([np.sum(generation_hourly)], dtype=float)
    pv_kwh_by_year = np.concatenate([
        (pv_kwh_by_year - (pv_kwh_by_year * agent.loc['pv_degradation_factor'] * i))
        for i in range(1, agent.loc['economic_lifetime_yrs'] + 1)
    ])
    kwh_by_timestep = kw * pv_kwh_by_year

    if len(pbi_df) == 1:
        amt = float(pbi_df['pbi_usd_p_kwh'].iloc[0])
        dur = int(pbi_df['incentive_duration_yrs'].iloc[0])
        loan.PaymentIncentives.pbi_sta_amount = [amt] * dur
        loan.PaymentIncentives.pbi_sta_escal = 0.0
        loan.PaymentIncentives.pbi_sta_tax_fed = 1
        loan.PaymentIncentives.pbi_sta_tax_sta = 1
        loan.PaymentIncentives.pbi_sta_term = dur
    elif len(pbi_df) >= 2:
        amt0 = float(pbi_df['pbi_usd_p_kwh'].iloc[0]); dur0 = int(pbi_df['incentive_duration_yrs'].iloc[0])
        amt1 = float(pbi_df['pbi_usd_p_kwh'].iloc[1]); dur1 = int(pbi_df['incentive_duration_yrs'].iloc[1])
        loan.PaymentIncentives.pbi_sta_amount = [amt0] * dur0
        loan.PaymentIncentives.pbi_sta_escal = 0.0
        loan.PaymentIncentives.pbi_sta_tax_fed = 1
        loan.PaymentIncentives.pbi_sta_tax_sta = 1
        loan.PaymentIncentives.pbi_sta_term = dur0

        loan.PaymentIncentives.pbi_oth_amount = [amt1] * dur1
        loan.PaymentIncentives.pbi_oth_escal = 0.0
        loan.PaymentIncentives.pbi_oth_tax_fed = 1
        loan.PaymentIncentives.pbi_oth_tax_sta = 1
        loan.PaymentIncentives.pbi_oth_term = dur1

    # ---- Investment-based incentives (IBI) ----
    ibi_df = (
        incentive_df.loc[pd.notnull(incentive_df['ibi_pct'])]
        .sort_values(['ibi_pct'], ascending=False)
        .reset_index(drop=True)
    )
    if len(ibi_df) == 1:
        loan.PaymentIncentives.ibi_sta_percent = ibi_df['ibi_pct'].iloc[0]
        loan.PaymentIncentives.ibi_sta_percent_deprbas_fed = 0
        loan.PaymentIncentives.ibi_sta_percent_deprbas_sta = 0
        loan.PaymentIncentives.ibi_sta_percent_maxvalue = ibi_df['max_incentive_usd'].iloc[0]
        loan.PaymentIncentives.ibi_sta_percent_tax_fed = 1
        loan.PaymentIncentives.ibi_sta_percent_tax_sta = 1
    elif len(ibi_df) >= 2:
        loan.PaymentIncentives.ibi_sta_percent = ibi_df['ibi_pct'].iloc[0]
        loan.PaymentIncentives.ibi_sta_percent_deprbas_fed = 0
        loan.PaymentIncentives.ibi_sta_percent_deprbas_sta = 0
        loan.PaymentIncentives.ibi_sta_percent_maxvalue = ibi_df['max_incentive_usd'].iloc[0]
        loan.PaymentIncentives.ibi_sta_percent_tax_fed = 1
        loan.PaymentIncentives.ibi_sta_percent_tax_sta = 1

        loan.PaymentIncentives.ibi_oth_percent = ibi_df['ibi_pct'].iloc[1]
        loan.PaymentIncentives.ibi_oth_percent_deprbas_fed = 0
        loan.PaymentIncentives.ibi_oth_percent_deprbas_sta = 0
        loan.PaymentIncentives.ibi_oth_percent_maxvalue = ibi_df['max_incentive_usd'].iloc[1]
        loan.PaymentIncentives.ibi_oth_percent_tax_fed = 1
        loan.PaymentIncentives.ibi_oth_percent_tax_sta = 1

    return loan


#==============================================================================
# Chunked execution helpers
#==============================================================================

_worker_conn = None

def _init_worker(dsn, role, schema):
    global _worker_conn
    _worker_conn, _ = utilfunc.make_con(dsn, role)
    with _worker_conn.cursor() as cur:
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        cur.execute(f'SET search_path TO "{schema}"')
        _ensure_econ_tables(cur)
    _worker_conn.commit()

def size_chunk(static_agents_df: pd.DataFrame, sectors, rate_switch_table, mode="simple"):
    """
    Aggregation + write hourly & annual finance for BOTH pv_only and pv_batt.
    Uses Outputs dicts (no PySAM assign/cloning).
    """
    global _worker_conn
    results = []

    n_hours = None
    net_sum = None

    hourly_rows_batch: List[Tuple] = []
    annual_recs_batch: List[Tuple] = []
    annual_detail_rows_batch: List[Tuple] = []

    for aid, row in static_agents_df.iterrows():
        agent = row.copy()
        agent.name = aid

        sized = calc_system_size_and_performance(
            _worker_conn, agent, sectors, rate_switch_table
        )
        year = int(sized.get("year", 0))

        # ---- PV-only case ----
        util_pvo = sized.get("_util_out_pv_only")
        batt_pvo = sized.get("_batt_out_pv_only")
        loan_pvo = sized.get("_loan_out_pv_only")
        if isinstance(util_pvo, dict) and isinstance(loan_pvo, dict) and year:
            rows, _ = _pack_hourly_records(int(aid), year, "pv_only", util_pvo, batt_pvo or {})
            hourly_rows_batch.extend(rows)

            system_kw = float(sized.get("_pv_only_system_kw", 0.0))
            summary_row, detail_rows = _pack_annual_finance_case(
                int(aid), year, "pv_only",
                util_pvo, loan_pvo,
                system_kw=system_kw, batt_kw=0.0, batt_kwh=0.0
            )
            annual_recs_batch.append(summary_row)
            annual_detail_rows_batch.extend(detail_rows)

        # ---- PV+batt case: always log the best candidate
        best_kwh = float(sized.get("_pv_batt_batt_kwh", 0.0) or 0.0)
        best_kw  = float(sized.get("_pv_batt_batt_kw",  0.0) or 0.0)
        best_sys = float(sized.get("_pv_batt_system_kw", 0.0) or 0.0)
        util_pvb = sized.get("_util_out_pv_batt")
        loan_pvb = sized.get("_loan_out_pv_batt")

        if isinstance(util_pvb, dict) and isinstance(loan_pvb, dict) and year and best_kwh > 0:
            # hourly econ
            rows, _ = _pack_hourly_records(int(aid), year, "pv_batt", util_pvb, sized.get("_batt_out_pv_batt") or {})
            hourly_rows_batch.extend(rows)

            # annual finance + detailed cashflows
            summary_row, detail_rows = _pack_annual_finance_case(
                int(aid), year, "pv_batt",
                util_pvb, loan_pvb,
                system_kw=best_sys, batt_kw=best_kw, batt_kwh=best_kwh
            )
            annual_recs_batch.append(summary_row)
            annual_detail_rows_batch.extend(detail_rows)


        # ---- Grid aggregation (unchanged)
        import numpy as np
        base = np.asarray(sized.get("baseline_net_hourly", []), dtype=float)
        adop = np.asarray(sized.get("adopter_net_hourly",  []), dtype=float)

        n_cust  = float(sized.get("customers_in_bin", 0.0)) if "customers_in_bin" in sized else 0.0
        n_adopt = float(sized.get("number_of_adopters", 0.0)) if "number_of_adopters" in sized else 0.0
        n_non   = max(n_cust - n_adopt, 0.0)

        if adop.size or base.size:
            m = adop.size if base.size == 0 else (base.size if adop.size == 0 else min(adop.size, base.size))
            adop = adop[:m] if adop.size >= m else np.resize(adop, m)
            base = base[:m] if base.size >= m else np.resize(base, m)

            if n_hours is None:
                n_hours = m
                net_sum = np.zeros(n_hours, dtype=float)
            elif m != n_hours:
                m = min(m, n_hours)
                adop = adop[:m]; base = base[:m]

            net_sum[:m] += adop[:m] * n_adopt + base[:m] * n_non

        # drop heavy stashed objects before returning
        for c in (
            "adopter_load_hourly","adopter_pv_hourly","adopter_batt_hourly",
            "pv_per_kw_hourly","consumption_hourly","generation_hourly",
            "batt_dispatch_profile","net_hourly",
            "_util_out_pv_only","_batt_out_pv_only","_loan_out_pv_only",
            "_util_out_pv_batt","_batt_out_pv_batt","_loan_out_pv_batt",
            "_case_for_hourly","_pv_only_system_kw","_pv_batt_system_kw","_pv_batt_batt_kw","_pv_batt_batt_kwh",
        ):
            if c in sized.index:
                sized = sized.drop(labels=[c])

        results.append(sized)

    # Bulk inserts
    _insert_hourly_records(_worker_conn, hourly_rows_batch)
    _insert_annual_finance(_worker_conn, annual_recs_batch)
    _insert_annual_finance_detail(_worker_conn, annual_detail_rows_batch)

    df_out = pd.DataFrame(results)
    agg = {
        "mode": "simple",
        "n_hours": int(n_hours or 0),
        "net_sum_kw": (net_sum.tolist() if net_sum is not None else []),
        "hourly_rows_inserted": len(hourly_rows_batch),
        "annual_rows_inserted": len(annual_recs_batch),
        "annual_detail_rows_inserted": len(annual_detail_rows_batch),
    }
    return df_out, agg


#==============================================================================
# Financial post-processing (payback, market share)
#==============================================================================

def calc_financial_performance(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate payback period from 'cash_flow' and join back to the agent DataFrame.
    """
    dataframe = dataframe.reset_index()
    cfs = np.vstack(dataframe['cash_flow']).astype(float)

    tech_lifetime = cfs.shape[1] - 1
    payback = calc_payback_vectorized(cfs, tech_lifetime)

    dataframe['payback_period'] = payback
    dataframe = dataframe.set_index('agent_id')
    return dataframe


def calc_payback_vectorized(cfs: np.ndarray, tech_lifetime: int) -> np.ndarray:
    """
    Payback period in years: first year where cumulative cash flows turn positive.
    If never pays back, returns 30.1.
    """
    years = np.array([np.arange(0, tech_lifetime)] * cfs.shape[0])

    cum_cfs = cfs.cumsum(axis=1)
    no_payback = np.logical_or(cum_cfs[:, -1] <= 0, np.all(cum_cfs <= 0, axis=1))
    instant_payback = np.all(cum_cfs > 0, axis=1)
    neg_to_pos_years = np.diff(np.sign(cum_cfs)) > 0
    base_years = np.amax(np.where(neg_to_pos_years, years, -1), axis=1)
    base_years_fix = np.where(base_years == -1, tech_lifetime - 1, base_years)
    base_year_mask = years == base_years_fix[:, np.newaxis]
    base_year_values = cum_cfs[:, :-1][base_year_mask]
    next_year_values = cum_cfs[:, 1:][base_year_mask]
    frac_years = base_year_values / (base_year_values - next_year_values + 1e-9)
    pp_year = base_years_fix + frac_years
    pp_precise = np.where(no_payback, 30.1, np.where(instant_payback, 0, pp_year))

    return np.array(pp_precise).round(1)


def calc_max_market_share(dataframe: pd.DataFrame, max_market_share_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach `max_market_share` to each agent row using a lookup table keyed by
    (sector_abbr, business_model, metric, payback_period_as_factor).
    """
    in_cols = list(dataframe.columns)
    dataframe = dataframe.reset_index()

    # set up metric
    dataframe['business_model'] = 'host_owned'
    dataframe['metric'] = 'payback_period'

    # bounds from lookup
    max_pb = max_market_share_df.loc[
        max_market_share_df.metric=='payback_period','payback_period'
    ].max()
    min_pb = max_market_share_df.loc[
        max_market_share_df.metric=='payback_period','payback_period'
    ].min()

    # clip and discretize payback to match lookup
    pb = dataframe['payback_period'].copy()
    pb = pb.where(pb >= min_pb, min_pb)
    pb = pb.where(pb <= max_pb, max_pb)
    dataframe['payback_period_bounded'] = pb.round(1)

    factor = (dataframe['payback_period_bounded'] * 100).round()
    factor = factor.replace([np.inf, -np.inf], np.nan)
    dataframe['payback_period_as_factor'] = factor.astype('Int64')

    max_market_share_df = max_market_share_df.copy()
    mms_factor = (max_market_share_df['payback_period'] * 100).round()
    mms_factor = mms_factor.replace([np.inf, -np.inf], np.nan)
    max_market_share_df['payback_period_as_factor'] = mms_factor.astype('Int64')

    merged = pd.merge(
        dataframe,
        max_market_share_df[[
            'sector_abbr','business_model','metric',
            'payback_period_as_factor','max_market_share'
        ]],
        how='left',
        on=['sector_abbr','business_model','metric','payback_period_as_factor']
    )

    out_cols = in_cols + ['max_market_share','metric']
    return merged[out_cols]


#==============================================================================
# Misc helpers (kept for API compatibility)
#==============================================================================

def check_incentive_constraints(incentive_data, incentive_value, system_cost):
    """Clamp incentives to min/max dollar and percent-of-cost limits."""
    if not pd.isnull(incentive_data['max_incentive_usd']):
        incentive_value = min(incentive_value, incentive_data['max_incentive_usd'])

    if not pd.isnull(incentive_data['max_incentive_pct']):
        incentive_value = min(incentive_value, system_cost * incentive_data['max_incentive_pct'])

    if not pd.isnull(incentive_data['min_incentive_usd']):
        incentive_value *= int(incentive_value > incentive_data['min_incentive_usd'])

    return incentive_value


def check_minmax(value, min_, max_):
    """
    Returns a boolean mask indicating whether `value` is within [min_, max_].
    Accepts scalars or NumPy arrays/Series.
    """
    output = True
    if isinstance(min_, float) and not np.isnan(min_):
        output = output * (value >= min_)
    if isinstance(max_, float) and not np.isnan(max_):
        output = output * (value <= max_)
    return output


def get_expiration(end_date, current_year, timesteps_per_year):
    """Compute timestep index for an end_date within current_year."""
    return float(((end_date - datetime.date(current_year, 1, 1)).days / 365.0) * timesteps_per_year)


def eqn_builder(method, incentive_info, info_params, default_params, additional_data):
    """
    Build an equation to scale timestep values for incentive schedules.

    method: 'linear_decay' | 'flat_rate'
    """
    # Fill defaults
    for i, r in enumerate(info_params):
        try:
            if np.isnan(incentive_info[r]):
                incentive_info[r] = default_params[i]
        except Exception:
            if incentive_info[r] is None:
                incentive_info[r] = default_params[i]

    pbi_usd_p_kwh = float(incentive_info[info_params[0]])
    years = float(incentive_info[info_params[1]])
    end_date = incentive_info[info_params[2]]

    current_year = int(additional_data[0])
    timesteps_per_year = float(additional_data[1])

    # Expiration timestep
    try:
        expiration = get_expiration(end_date, current_year, timesteps_per_year)
    except Exception:
        expiration = years * timesteps_per_year

    expiration = min(years * timesteps_per_year, expiration)

    if method == 'linear_decay':
        def function(ts):
            if ts > expiration:
                return 0.0
            fraction = (expiration - ts) if (expiration - ts) < 1 else 1
            return fraction * (pbi_usd_p_kwh + ((-1 * (pbi_usd_p_kwh / expiration) * ts)))
        return function

    if method == 'flat_rate':
        def function(ts):
            if ts > expiration:
                return 0.0
            fraction = (expiration - ts) if (expiration - ts) < 1 else 1
            return fraction * pbi_usd_p_kwh
        return function


def eqn_linear_decay_to_zero(incentive_info, info_params, default_params, additional_params):
    return eqn_builder('linear_decay', incentive_info, info_params, default_params, additional_params)


def eqn_flat_rate(incentive_info, info_params, default_params, additional_params):
    return eqn_builder('flat_rate', incentive_info, info_params, default_params, additional_params)


def dump_rate_brief(utilityrate, title="[rate]"):
    ER = utilityrate.ElectricityRates

    mo = int(getattr(ER, "ur_metering_option", -1))  # 0=NM,1=NB,2=BA/SA
    ts_sell_en = int(getattr(ER, "ur_en_ts_sell_rate", 0))
    ts_buy_en  = int(getattr(ER, "ur_en_ts_buy_rate", 0))

    tou = np.asarray(getattr(ER, "ur_ec_tou_mat", []), dtype=float)
    wk  = np.asarray(getattr(ER, "ur_ec_sched_weekday", []), dtype=int)
    we  = np.asarray(getattr(ER, "ur_ec_sched_weekend", []), dtype=int)

    print(f"{title} mo={mo} (0=NM,1=NB,2=BA/SA)  ec_rows={tou.shape}  "
          f"wk_unique={np.unique(wk) if wk.size else []}  "
          f"we_unique={np.unique(we) if we.size else []}  "
          f"ts_sell_en={ts_sell_en} ts_buy_en={ts_buy_en}",
          flush=True)

    if tou.size:
        # First-tier buy prices by period
        first_tier = tou[tou[:, 1] == 1]
        period_prices = {}
        for per in np.unique(first_tier[:, 0].astype(int)):
            prices_this_period = first_tier[first_tier[:, 0] == per][:, 4]
            if prices_this_period.size:
                period_prices[per] = float(np.mean(prices_this_period))
        for per in sorted(period_prices):
            print(f"{title}  period {per}: buy ${period_prices[per]:.4f}/kWh", flush=True)

        def _hist(label, sched):
            if sched.size:
                vals, counts = np.unique(sched, return_counts=True)
                pairs = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts))
                print(f"{title}  {label} sched histogram -> {pairs}", flush=True)
        _hist("wkday", wk)
        _hist("wkend", we)

    if ts_sell_en:
        sell = np.asarray(getattr(ER, "ur_ts_sell_rate", []), dtype=float)
        if sell.size:
            print(f"{title}  ts_sell len={sell.size}  "
                  f"min={sell.min():.4f} mean={sell.mean():.4f} max={sell.max():.4f}",
                  flush=True)
