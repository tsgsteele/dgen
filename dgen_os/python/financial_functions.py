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
FORCE_NET_BILLING = False

#==============================================================================
# Logger
#==============================================================================
logger = utilfunc.get_logger()

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
    batt_dispatch: str = 'price_signal_forecast'
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

    # Hard reset all loan assignments so they don't carry over between runs
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

        pv_to_batt_ratio = 0.8
        batt_capacity_to_power_ratio = 2.0
        desired_size  = kw / pv_to_batt_ratio
        desired_power = desired_size / batt_capacity_to_power_ratio
        desired_voltage = 500 if agent.loc['sector_abbr'] != 'res' else 240
        battery_tools.battery_model_sizing(
            batt, desired_power, desired_size, desired_voltage=desired_voltage, tol=1e38
        )

        batt.Load.load = load_hourly
        batt.SystemOutput.gen = gen
        batt.BatteryCell.batt_initial_SOC = 30

        # Dispatch: retail-rate aware, no grid-charging unless surplus
        configure_retail_rate_dispatch(
            batt,
            allow_export=True,
            allow_grid_charge=False,
            charge_only_when_surplus=True,
            lookahead_hours=24,
        )
        if not hasattr(batt.BatteryDispatch, 'batt_look_ahead_hours'):
            batt.BatteryDispatch.batt_look_ahead_hours = 24

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
        # [NEM3] Only CA: disable TS sell so TOU sell column is used
        is_ca = str(agent.get('state_abbr', '')).upper() == 'CA'  # [NEM3]
        ts_sell = None if is_ca else (                            # [NEM3]
            np.asarray(agent.loc['wholesale_prices'], dtype=float).ravel() * agent.loc['elec_price_multiplier']
        )
        td_norm = normalize_tariff(agent.loc['tariff_dict'], net_sell_rate_scalar=net_billing_sell_rate)
        # [NEM3] Only CA: set TOU sell (col 6) = 0.25 * buy (col 5)
        if is_ca and td_norm.get('ur_ec_tou_mat'):               # [NEM3]
            _tou = np.asarray(td_norm['ur_ec_tou_mat'], dtype=float)
            if _tou.ndim == 2 and _tou.shape[1] >= 6:
                _tou[:, 5] = _tou[:, 4] * 0.25
                td_norm['ur_ec_tou_mat'] = _tou.tolist()
        if is_ca: td_norm['ur_metering_option'] = 2
        process_tariff(utilityrate, td_norm, net_billing_sell_rate, ts_sell_rate=ts_sell)

        # Hand gen to the rate engine
        utilityrate.SystemOutput.gen = batt.SystemOutput.gen

        # Wire up loan pieces for battery case
        loan.BatterySystem.en_batt = 1
        loan.BatterySystem.batt_computed_bank_capacity = batt.Outputs.batt_bank_installed_capacity
        loan.BatterySystem.batt_bank_replacement = batt.Outputs.batt_bank_replacement

        loan.SystemCosts.add_om_num_types = 0
        if kw > 0:
            #loan.SystemCosts.om_capacity = [costs['system_om_per_kw_combined'] + costs['system_variable_om_per_kw_combined']]
            loan.SystemCosts.om_batt_capacity_cost = [0.0]
            loan.SystemCosts.om_batt_variable_cost = [0.0]
            loan.SystemCosts.om_batt_replacement_cost = [0.0]
            #loan.SystemCosts.om_batt_nameplate = batt.BatterySystem.batt_power_discharge_max_kwdc
            system_costs = costs['system_capex_per_kw_combined'] * kw
        else:
            #loan.SystemCosts.om_capacity = [costs['system_om_per_kw'] + costs['system_variable_om_per_kw']]
            loan.SystemCosts.om_batt_capacity_cost = [0.0]
            loan.SystemCosts.om_batt_variable_cost = [0.0]
            loan.SystemCosts.om_batt_replacement_cost = [0.0]
            #loan.SystemCosts.om_batt_nameplate = batt.BatterySystem.batt_power_discharge_max_kwdc
            system_costs = costs['system_capex_per_kw'] * kw

        #loan.SystemCosts.om_production1_values = batt.Outputs.batt_annual_discharge_energy
        batt_costs = costs['batt_capex_per_kwh_combined'] * batt.Outputs.batt_bank_installed_capacity * .7 # For the investment tax credit
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
        # [NEM3] Only CA: disable TS sell so TOU sell column is used
        is_ca = str(agent.get('state_abbr', '')).upper() == 'CA'  # [NEM3]
        ts_sell = None if is_ca else (                            # [NEM3]
            np.asarray(agent.loc['wholesale_prices'], dtype=float).ravel() * agent.loc['elec_price_multiplier']
        )
        td_norm = normalize_tariff(agent.loc['tariff_dict'], net_sell_rate_scalar=net_billing_sell_rate)
        # [NEM3] Only CA: set TOU sell (col 6) = 0.25 * buy (col 5)
        if is_ca and td_norm.get('ur_ec_tou_mat'):               # [NEM3]
            _tou = np.asarray(td_norm['ur_ec_tou_mat'], dtype=float)
            if _tou.ndim == 2 and _tou.shape[1] >= 6:
                _tou[:, 5] = _tou[:, 4] * 0.25
                td_norm['ur_ec_tou_mat'] = _tou.tolist()
        if is_ca: td_norm['ur_metering_option'] = 2
        process_tariff(utilityrate, td_norm, net_billing_sell_rate, ts_sell_rate=ts_sell)

        utilityrate.SystemOutput.gen = gen
        loan.SystemCosts.add_om_num_types = 0
        #loan.SystemCosts.om_capacity = [costs['system_om_per_kw'] + costs['system_variable_om_per_kw']]
        loan.SystemCosts.om_batt_replacement_cost = [0.0]
        loan.SystemCosts.om_batt_nameplate = 0
        system_costs = costs['system_capex_per_kw'] * kw
        batt_costs = 0.0
        linear_constant = 0.0
        value_of_resiliency = 0.0

    # ---- Rate engine ----
    utilityrate.Load.load = load_hourly
    utilityrate.execute()

    loan.FinancialParameters.system_capacity = kw

    aev = list(utilityrate.Outputs.annual_energy_value)
    annual_energy_value = [aev[0] + value_of_resiliency] + [x + value_of_resiliency for x in aev[1:]]

    loan.SystemOutput.annual_energy_value = annual_energy_value
    loan.SystemOutput.gen = utilityrate.SystemOutput.gen

    direct_costs = (system_costs + batt_costs) * costs['cap_cost_multiplier']
    sales_tax = 0.0
    loan.SystemCosts.total_installed_cost = direct_costs + sales_tax + one_time_charge

    # ITC
    loan.TaxCreditIncentives.itc_fed_percent = [agent.loc['itc_fraction_of_capex']]

    loan.execute()
    return -loan.Outputs.npv


def calc_system_size_and_performance(con, agent: pd.Series, sectors, rate_switch_table=None):
    """
    Size PV once (PV-only), then bolt on a battery at a fixed ratio, and record **both**
    PV-only and PV+Battery results on the agent row.

    Purpose
    -------
    - Optimize PV capacity (kW_dc) **without** a battery to get the economically optimal PV size.
    - With PV fixed at that optimum, run a **single** forward simulation with a battery sized
      as a fixed ratio to PV (no PV re-optimization).
    - Store *both* sets of hourly arrays and scalars:
        * PV-only: adopter_net_hourly_pvonly, first_year_elec_bill_with_system_pvonly, etc.
        * PV+Batt: adopter_net_hourly_with_batt, first_year_elec_bill_with_system_with_batt, batt kw/kWh, etc.
      For backward compatibility, adopter_net_hourly and first_year_elec_bill_with_system are set to PV-only.

    Inputs
    ------
    con : DB-API connection
        Read-only connection used by agent_mutation (load/resource).
    agent : pd.Series
        Agent row with tariff, incentives, costs, and runtime fields (mutated copy returned).
    sectors : Any
        Unused here (kept for signature compatibility).
    rate_switch_table : pd.DataFrame or None
        Table used by agent_mutation.elec.apply_rate_switch.

    Returns
    -------
    agent : pd.Series
        The input row with additional fields populated:
        - 'baseline_net_hourly' (list[float])
        - 'adopter_net_hourly_pvonly', 'adopter_net_hourly_with_batt' (list[float])
        - 'adopter_net_hourly' (PV-only, for backward compatibility)
        - 'adopter_load_hourly', 'adopter_pv_hourly' (PV-only components)
        - PV-only scalars: 'system_kw', 'annual_energy_production_kwh', 'naep', 'capacity_factor',
                           'first_year_elec_bill_with_system' (PV-only), 'cash_flow', 'npv', 'payback_period'
        - Battery scalars at fixed ratio: 'batt_kw', 'batt_kwh',
          and with-battery economics: 'first_year_elec_bill_with_system_with_batt'
        - 'pv_per_kw_hourly' (per-kW PV profile used for NAEP and scaling)
    """
    cur = con.cursor()

    agent = agent.copy()
    if 'agent_id' not in agent.index:
        agent.loc['agent_id'] = agent.name

    # -----------------------------
    # 1) Load profile (hourly, kW)
    # -----------------------------
    t0 = time.time()
    lp = agent_mutation.elec.get_and_apply_agent_load_profiles(con, agent)
    cons = np.asarray(lp['consumption_hourly'].iloc[0], dtype=float)
    del lp
    load_profiles_total_time = time.time() - t0

    # ----------------------------------------------------
    # 2) Solar resource per kW_dc (PV generation per kW)
    # ----------------------------------------------------
    t0 = time.time()
    norm = agent_mutation.elec.get_and_apply_normalized_hourly_resource_solar(con, agent)
    gen_per_kw = np.asarray(norm['solar_cf_profile'].iloc[0], dtype=float) / 1e6
    agent.loc['naep'] = float(gen_per_kw.sum())
    del norm
    solar_resource_total_time = time.time() - t0

    pv = {'consumption_hourly': cons, 'generation_hourly': gen_per_kw}

    # --------------------------
    # 3) PySAM object scaffolds
    # --------------------------
    driver_mod, batt, utilityrate, loan, market_flag = _init_pv_batt_stack(agent.loc['sector_abbr'])
    loan.FinancialParameters.market = market_flag

    # Lifetime/fin params
    utilityrate.Lifetime.inflation_rate = agent.loc['inflation_rate'] * 100
    utilityrate.Lifetime.analysis_period = agent.loc['economic_lifetime_yrs']
    utilityrate.Lifetime.system_use_lifetime_output = 0
    utilityrate.SystemOutput.degradation = [agent.loc['pv_degradation_factor'] * 100]
    utilityrate.ElectricityRates.rate_escalation = [agent.loc['elec_price_escalator'] * 100]

    # Initial tariff (pre-switch)
    is_ca = str(agent.get('state_abbr', '')).upper() == 'CA'
    ts_sell = None if is_ca else (np.asarray(agent.loc['wholesale_prices'], dtype=float).ravel()
                                  * agent.loc['elec_price_multiplier'])
    tariff_dict = normalize_tariff(agent.loc['tariff_dict'], net_sell_rate_scalar=0.0)
    # California NEM3 adjustment
    if is_ca and tariff_dict.get('ur_ec_tou_mat'):
        _tou = np.asarray(tariff_dict['ur_ec_tou_mat'], dtype=float)
        if _tou.ndim == 2 and _tou.shape[1] >= 6:
            _tou[:, 5] = _tou[:, 4] * 0.25
            tariff_dict['ur_ec_tou_mat'] = _tou.tolist()
    if is_ca: tariff_dict['ur_metering_option'] = 2
    utilityrate = process_tariff(utilityrate, tariff_dict, 0.0, ts_sell_rate=ts_sell)

    # Loan params
    loan.FinancialParameters.analysis_period = agent.loc['economic_lifetime_yrs']
    loan.FinancialParameters.debt_fraction = 100 - (agent.loc['down_payment_fraction'] * 100)
    loan.FinancialParameters.federal_tax_rate = [(agent.loc['tax_rate'] * 100) * 0.7]
    loan.FinancialParameters.inflation_rate = agent.loc['inflation_rate'] * 100
    loan.FinancialParameters.loan_term = agent.loc['loan_term_yrs']
    loan.FinancialParameters.property_tax_rate = 0
    loan.FinancialParameters.real_discount_rate = agent.loc['real_discount_rate'] * 100
    loan.FinancialParameters.salvage_percentage = 0
    loan.FinancialParameters.state_tax_rate = [(agent.loc['tax_rate'] * 100) * 0.3]
    loan.FinancialParameters.system_heat_rate = 0

    sc = {
        'system_capex_per_kw':                 agent.loc['system_capex_per_kw'],
        'system_om_per_kw':                    agent.loc['system_om_per_kw'],
        'system_variable_om_per_kw':           agent.loc['system_variable_om_per_kw'],
        'cap_cost_multiplier':                 agent.loc['cap_cost_multiplier'],
        'batt_capex_per_kw':                   agent.loc['batt_capex_per_kw'],
        'batt_capex_per_kwh':                  agent.loc['batt_capex_per_kwh'],
        'batt_om_per_kw':                      agent.loc['batt_om_per_kw'],
        'batt_om_per_kwh':                     agent.loc['batt_om_per_kwh'],
        'system_capex_per_kw_combined':        agent.loc['system_capex_per_kw_combined'],
        'system_om_per_kw_combined':           agent.loc['system_om_per_kw'],
        'system_variable_om_per_kw_combined':  agent.loc['system_variable_om_per_kw'],
        'batt_capex_per_kw_combined':          agent.loc['batt_capex_per_kw_combined'],
        'batt_capex_per_kwh_combined':         agent.loc['batt_capex_per_kwh_combined'],
        'batt_om_per_kw_combined':             agent.loc['batt_om_per_kw_combined'],
        'batt_om_per_kwh_combined':            agent.loc['batt_om_per_kwh_combined'],
        'linear_constant_combined':            agent.loc['linear_constant_combined'],
    }

    # Depreciation switches
    if agent.loc['sector_abbr'] == 'res':
        loan.Depreciation.depr_fed_type = 0
        loan.Depreciation.depr_sta_type = 0
    else:
        loan.Depreciation.depr_fed_type = 2
        loan.Depreciation.depr_sta_type = 2

    # Battery dispatch selection
    batt_disp = 0 # for peak shaving

    # Objective handles (shared)
    def perf_with_batt(x):
        return calc_system_performance(
            x, pv, utilityrate, loan, batt, sc, agent, rate_switch_table, True, batt_disp
        )

    def perf_no_batt(x):
        return calc_system_performance(
            x, pv, utilityrate, loan, batt, sc, agent, rate_switch_table, False, 0
        )

    # ==========================================
    # 4) Optimize PV-only to get kW*
    # ==========================================
    max_load   = agent.loc['load_kwh_per_customer_in_bin'] / agent.loc['naep']
    max_system = max_load    
    low        = max_system * 0.8
    high       = max_system * 1.25
    tol = max(2, int(max(1, high - low) * 1e-3))
    res_n = optimize.minimize_scalar(
        perf_no_batt, bounds=(low, high), method='bounded', options={'xatol': tol}
    )   

    # --- PV-only outputs at kW* (objective call already set state) ---
    out_n_loan = loan.Outputs.export()
    out_n_util = utilityrate.Outputs.export()
    gen_n_annual = float(np.nansum(np.asarray(getattr(utilityrate.SystemOutput, "gen", []), dtype=float)))
    kw_star = float(res_n.x)

    load_n_ts = np.asarray(getattr(utilityrate.Load, "load", []), dtype=float)
    _pv_n_raw = getattr(utilityrate.Outputs, "year1_hourly_system_to_load", None)  # PV→load in PV-only
    if _pv_n_raw is None or (hasattr(_pv_n_raw, "__len__") and len(_pv_n_raw) == 0):
        _pv_n_raw = getattr(utilityrate.SystemOutput, "gen", [])
    pv_n_ts  = np.asarray(_pv_n_raw, dtype=float)

    # *** Net import from grid (PV-only) — authoritative ***
    e_fromgrid_n = np.asarray(getattr(utilityrate.Outputs, "year1_hourly_e_fromgrid", []), dtype=float)

    first_with_n    = out_n_util.get('utility_bill_w_sys_year1', 0.0)
    first_without_n = out_n_util.get('utility_bill_wo_sys_year1', 0.0)
    npv_n           = out_n_loan.get("npv", float("-inf"))
    cash_flow_n     = list(out_n_loan.get('cf_payback_with_expenses', []))
    payback_n       = out_n_loan.get('payback', None)

    # ==========================================
    # 5) Single forward run with battery at kW*
    # ==========================================
    _ = perf_with_batt(kw_star)
    out_w_loan = loan.Outputs.export()
    out_w_util = utilityrate.Outputs.export()
    gen_w_annual = float(np.nansum(np.asarray(getattr(utilityrate.SystemOutput, "gen", []), dtype=float)))
    kw_w         = getattr(batt.BatterySystem, "batt_power_discharge_max_kwdc", 0.0)
    kwh_w        = getattr(batt.Outputs, "batt_bank_installed_capacity", 0.0)

    load_w_ts = np.asarray(getattr(utilityrate.Load, "load", []), dtype=float)
    _sys_w_raw = getattr(batt.Outputs, "system_to_load", None)  # PV+Batt → load
    if _sys_w_raw is None or (hasattr(_sys_w_raw, "__len__") and len(_sys_w_raw) == 0):
        _sys_w_raw = getattr(utilityrate.SystemOutput, "gen", [])
    sys_to_load_w  = np.asarray(_sys_w_raw, dtype=float)

    # *** Net import from grid (PV+Batt) — authoritative ***
    e_fromgrid_w = np.asarray(getattr(batt.Outputs, "grid_to_load", []), dtype=float)

    first_with_w    = out_w_util.get('utility_bill_w_sys_year1', 0.0)
    first_without_w = out_w_util.get('utility_bill_wo_sys_year1', 0.0)

    # -----------------------------
    # 6) Align arrays and set fields
    # -----------------------------
    def _align_to_n(*arrays):
        sizes = [np.asarray(a, dtype=float).size for a in arrays]
        n = max([s for s in sizes if s is not None] or [0])
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

    # Baseline (non-adopters): just the original load
    load_ts = np.asarray(cons, dtype=float)
    agent.loc['baseline_net_hourly'] = load_ts.tolist()

    # PV-only arrays
    load_n_ts, pv_n_ts, e_fromgrid_n = _align_to_n(load_n_ts, pv_n_ts, e_fromgrid_n)
    adopter_load_ts = load_n_ts
    adopter_pv_ts   = pv_n_ts  # in PV-only run this is PV→load
    adopter_net_pvo = e_fromgrid_n  # authoritative net import

    # With-battery arrays (net from grid)
    load_w_ts, sys_to_load_w, e_fromgrid_w = _align_to_n(load_w_ts, sys_to_load_w, e_fromgrid_w)
    adopter_net_wb  = e_fromgrid_w  # already includes PV + battery to load

    # Store arrays
    agent.loc['adopter_load_hourly']            = adopter_load_ts.tolist()
    agent.loc['adopter_pv_hourly']              = adopter_pv_ts.tolist()
    agent.loc['adopter_net_hourly_pvonly']      = adopter_net_pvo.tolist()
    agent.loc['adopter_net_hourly_with_batt']   = adopter_net_wb.tolist()
    # Back-compat (PV-only)
    agent.loc['adopter_net_hourly']             = adopter_net_pvo.tolist()

    # PV-only scalars
    system_kw     = kw_star
    annual_kwh    = gen_n_annual
    naep_final    = annual_kwh / max(system_kw, 1e-9)
    first_with    = first_with_n
    first_without = first_without_n
    savings       = first_without - first_with
    savings_frac  = savings / first_without if first_without != 0 else 0.0
    avg_price     = first_without / agent.loc['load_kwh_per_customer_in_bin']

    agent.loc['system_kw']                        = system_kw
    agent.loc['annual_energy_production_kwh']     = annual_kwh
    agent.loc['naep']                             = naep_final
    agent.loc['capacity_factor']                  = naep_final / 8760.0
    agent.loc['first_year_elec_bill_with_system'] = first_with  # PV-only (back-compat)
    agent.loc['first_year_elec_bill_wo_system']   = first_without
    agent.loc['first_year_elec_bill_savings']     = savings
    agent.loc['first_year_elec_bill_savings_frac']= savings_frac
    agent.loc['price_per_kwh']                    = avg_price
    agent.loc['npv']                              = npv_n
    agent.loc['payback_period']                   = float(np.round(payback_n if np.isfinite(payback_n) else 30.1, 1))
    agent.loc['cash_flow']                        = cash_flow_n

    # With-battery scalars (at fixed ratio)
    agent.loc['batt_kw']                                        = float(kw_w)
    agent.loc['batt_kwh']                                       = float(kwh_w)
    agent.loc['first_year_elec_bill_with_system_with_batt']     = first_with_w
    agent.loc['first_year_elec_bill_wo_system_with_batt']       = first_without_w
    agent.loc['first_year_elec_bill_savings_with_batt']         = first_without_w - first_with_w

    # Keep per-kW PV profile for downstream use/debug
    agent.loc['pv_per_kw_hourly'] = gen_per_kw.tolist()

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

    # ---- Time‑series SELL / BUY under Net Billing ----
    # Under NM (mo=0), SAM forbids TS rates. We’re forcing mo=2, so enable SELL if provided.
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
        # Fallback: if you ever let mo!=1, keep SAM happy
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
    # normalize common non-values and single quotes
    s = s.replace("'", '"')
    s = re.sub(r'\b(nan|none|null)\b', 'null', s, flags=re.IGNORECASE)
    try:
        return json.loads(s)  # prefer strict JSON
    except json.JSONDecodeError:
        try:
            # Literal eval fallback for legacy Python-dict-ish strings
            return ast.literal_eval(raw)
        except Exception:
            return {}

def _num(x, default: float = 0.0) -> float:
    """Coerce scalar to float with safe null/empty handling."""
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip().lower() in {"", "nan", "none", "null"}:
            return default
        return float(x)
    except Exception:
        return default

def _coerce_bool(x, default=False) -> bool:
    """Robust bool coercion from ints/strings."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(default)

def _sched_12x24_single_period() -> List[List[int]]:
    """12 months × 24 hours, all period=1 (PySAM uses 1-based periods)."""
    return [[1] * 24 for _ in range(12)]

def _plus1_sched(mat) -> List[List[int]]:
    """
    Convert 0/1/2 → 1/2/3; default to all-1s if missing.
    Guard ragged rows by filling/truncating to 24 columns.
    """
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
    """Return strict 12×24 schedule (ints); else zeros matrix of that shape."""
    if x is None:
        return [[0]*24 for _ in range(12)]
    a = np.asarray(x)
    if a.ndim != 2:
        return [[0]*24 for _ in range(12)]
    # pad/trim to (12,24) if close; else replace with zeros
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
    """
    Return small 2D float list (<= size_limit, finite); else [].
    """
    if x is None or x == []:
        return []
    a = np.asarray(x)
    if a.ndim != 2 or a.size == 0 or a.size > size_limit:
        return []
    if not np.isfinite(a.astype(np.float64, copy=False)).all():
        return []
    return a.astype(np.float32, copy=False).tolist()

def _list1d_8760(x: Optional[Iterable[float]]) -> Optional[List[float]]:
    """Return 8760-length float list for TS rates; else None."""
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
    """
    Build ur_ec_tou_mat from legacy energy fields.
    Row format per PySAM: [period(1..P), tier(1..T), max_usage, usage_units_code, price, net_sell_rate]
    """
    prices = td.get('e_prices') or []
    if not prices:
        return []
    levels = td.get('e_levels') or []
    n_tiers = len(prices)
    n_periods = len(prices[0]) if n_tiers else 0
    BIG = 1e38
    # If levels missing or mismatched, assume non-tiered cap (VERY BIG)
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
    """
    Build demand-charge mats from legacy fields.
    Row format: [period(1..P), tier(1..T), max_kW, price]
    """
    out = {'ur_dc_flat_mat': [], 'ur_dc_tou_mat': []}

    # Flat DC
    dfl, dfr = td.get('d_flat_levels') or [], td.get('d_flat_prices') or []
    if dfl and dfr:
        n_tiers = len(dfl)
        n_periods = len(dfl[0]) if n_tiers else 0
        flat = []
        for p in range(n_periods):
            for t in range(n_tiers):
                flat.append([p + 1, t + 1, float(dfl[t][p]), float(dfr[t][p])])
        out['ur_dc_flat_mat'] = flat

    # TOU DC
    dtl, dtr = td.get('d_tou_levels') or [], td.get('d_tou_prices') or []
    if dtl and dtr:
        n_tiers = len(dtl)
        n_periods = len(dtl[0]) if n_tiers else 0
        tou = []
        for p in range(n_periods):
            for t in range(n_tiers):
                tou.append([p + 1, t + 1, float(dtl[t][p]), float(dtr[t][p])])
        out['ur_dc_tou_mat'] = tou

    # Schedules: prefer UR fields else legacy 12×24, then normalize to 1..P
    out['ur_dc_sched_weekday'] = _plus1_sched(td.get('ur_dc_sched_weekday') or td.get('d_wkday_12by24'))
    out['ur_dc_sched_weekend'] = _plus1_sched(td.get('ur_dc_sched_weekend') or td.get('d_wkend_12by24'))

    # Enable if any DC structure exists or legacy flags indicate it
    dc_enable = 1 if (out['ur_dc_flat_mat'] or out['ur_dc_tou_mat'] or _coerce_bool(td.get('d_flat_exists')) or _coerce_bool(td.get('d_tou_exists'))) else 0
    return out, dc_enable

def _reconcile_periods_and_equalize_tiers(ec_tou_mat, wk_sched, we_sched):
    """
    1) Remap TOU period ids to contiguous 1..P.
    2) Make every period have the SAME number of tiers.
       If a period has fewer tiers than the max, pad with clones of its last tier
       using a very large max_usage (acts like "no upper bound").
    3) Clamp/normalize weekday/weekend schedules to 12x24 with ids in 1..P.
    Returns: (ec_tou_mat_fixed, wk_sched_fixed, we_sched_fixed)
    """
    BIG = 1e38

    tou = np.asarray(ec_tou_mat or [], dtype=float)
    wk  = np.asarray(wk_sched or [], dtype=int)
    we  = np.asarray(we_sched or [], dtype=int)

    # If no TOU rows, just return normalized schedules
    if tou.size == 0:
        wk = np.asarray(_sched_12x24(wk.tolist()), dtype=int)
        we = np.asarray(_sched_12x24(we.tolist()), dtype=int)
        wk[wk < 1] = 1
        we[we < 1] = 1
        return [], wk.tolist(), we.tolist()

    # Current unique period ids
    per_ids = np.unique(tou[:, 0].astype(int))
    if per_ids.size == 0:
        tou[:, 0] = 1
        per_ids = np.array([1], dtype=int)

    # Remap to contiguous 1..P
    new_ids = np.arange(1, per_ids.size + 1, dtype=int)
    remap = {int(old): int(new) for old, new in zip(per_ids.tolist(), new_ids.tolist())}
    tou[:, 0] = np.vectorize(lambda x: remap.get(int(x), 1))(tou[:, 0])

    # Equalize tiers across periods
    # Count tiers per (period)
    max_tiers = 0
    tiers_by_period: Dict[int, List[np.ndarray]] = {}
    for p in np.unique(tou[:, 0].astype(int)):
        rows_p = tou[tou[:, 0] == p]
        # sort by tier col (1-based)
        rows_p = rows_p[np.argsort(rows_p[:, 1])]
        tiers_by_period[p] = [rows_p[rows_p[:, 1] == t] for t in np.unique(rows_p[:, 1].astype(int))]
        max_tiers = max(max_tiers, len(tiers_by_period[p]))

    fixed_rows = []
    for p in sorted(tiers_by_period.keys()):
        rows_p = tou[tou[:, 0] == p]
        rows_p = rows_p[np.argsort(rows_p[:, 1])]
        have_tiers = [int(t) for t in np.unique(rows_p[:, 1].astype(int))]
        # If already at max_tiers, just normalize tier labels to 1..T and keep
        if len(have_tiers) == max_tiers:
            # ensure tiers are exactly 1..T
            for i, row in enumerate(rows_p):
                row[1] = float(i + 1)
            fixed_rows.append(rows_p)
            continue

        # Need to pad: clone last tier until count == max_tiers
        last = rows_p[-1].copy()  # [period, tier, max_usage, unit_code, price, net_sell_rate]
        unit_code = last[3] if rows_p.shape[1] >= 4 else 0.0
        price     = last[4] if rows_p.shape[1] >= 5 else 0.0
        nsell     = last[5] if rows_p.shape[1] >= 6 else 0.0

        padded = [r.copy() for r in rows_p]
        for add_t in range(len(have_tiers) + 1, max_tiers + 1):
            padded.append(np.array([float(p), float(add_t), float(BIG), float(unit_code), float(price), float(nsell)], dtype=float))
        padded = np.vstack(padded)
        # Renumber tiers 1..T cleanly
        padded = padded[np.argsort(padded[:, 1])]
        for i in range(padded.shape[0]):
            padded[i, 1] = float(i + 1)
        fixed_rows.append(padded)

    tou_fixed = np.vstack(fixed_rows)

    # Normalize schedules to 12x24 and clamp to 1..P
    wk = np.asarray(_sched_12x24(wk.tolist()), dtype=int)
    we = np.asarray(_sched_12x24(we.tolist()), dtype=int)
    wk[wk < 1] = 1
    we[we < 1] = 1
    P = int(np.max(tou_fixed[:, 0]).astype(int))
    wk[wk > P] = 1
    we[we > P] = 1

    # Sort rows by (period, tier) for readability
    tou_fixed = tou_fixed[np.lexsort((tou_fixed[:, 1], tou_fixed[:, 0]))]
    return tou_fixed.astype(np.float32).tolist(), wk.tolist(), we.tolist()

def _harmonize_tier_caps_and_units(ec_tou_mat: List[List[float]]) -> List[List[float]]:
    """
    Ensure each tier has a single, consistent Max. Usage across all periods
    (PySAM requirement for TOU tiers). If any finite caps exist for a tier,
    use the smallest finite cap across periods; otherwise use the BIG cap
    (i.e., effectively "no tiering"). Also normalize usage_units_code to the
    most common value across the matrix.
    """
    if not ec_tou_mat:
        return []

    tou = np.asarray(ec_tou_mat, dtype=float)  # cols: [period, tier, max_usage, unit_code, price, net_sell]
    if tou.ndim != 2 or tou.shape[1] < 6:
        return []

    # Identify tiers and a BIG sentinel threshold
    tiers = np.unique(tou[:, 1].astype(int))
    BIG = 1e38
    BIG_THRESH = 1e37  # treat >= this as "infinite" cap

    # Choose a single unit_code: the mode (most frequent) among rows
    unit_codes = tou[:, 3].astype(int)
    # mode without scipy: bincount over shifted nonnegative ints
    min_uc = unit_codes.min()
    shift = 0 if min_uc >= 0 else -min_uc
    counts = np.bincount((unit_codes + shift).astype(int))
    unit_code_mode = int(np.argmax(counts) - shift)

    # For each tier, set a single cap across all periods
    for t in tiers:
        rows_t = (tou[:, 1].astype(int) == t)
        caps_t = tou[rows_t, 2]
        finite_caps = caps_t[(caps_t > 0) & (caps_t < BIG_THRESH)]
        cap = float(np.min(finite_caps)) if finite_caps.size else float(BIG)
        tou[rows_t, 2] = cap

    # Normalize unit codes
    tou[:, 3] = float(unit_code_mode)

    # Sort rows by (period, tier) for readability/consistency
    tou = tou[np.lexsort((tou[:, 1], tou[:, 0]))]
    return tou.astype(np.float32).tolist()

def normalize_tariff(raw, net_sell_rate_scalar=0.0, debug=False):
    td = _parse_tariff_dict(raw)
    out = {}

    # Always enable electricity rates
    out['en_electricity_rates'] = int(td.get('en_electricity_rates', 1))

    # Force Net Billing if configured, else honor input (0=NM, 1=Net Billing, 2=BA/SA)
    mo_in = int(td.get('ur_metering_option', 0))
    out['ur_metering_option'] = 2 if FORCE_NET_BILLING else mo_in

    # Fixed charge (accept legacy key too)
    fc = td.get('ur_monthly_fixed_charge', td.get('fixed_charge', 0.0))
    out['ur_monthly_fixed_charge'] = _num(fc, 0.0)

    # Energy structure (BUY side) — prefer explicit UR fields; else build from legacy
    ec_tou = td.get('ur_ec_tou_mat') or _build_ur_ec_from_e_parts(td, net_sell_rate_scalar)
    wkday  = td.get('ur_ec_sched_weekday') or _plus1_sched(td.get('e_wkday_12by24')) or _sched_12x24_single_period()
    wkend  = td.get('ur_ec_sched_weekend') or _plus1_sched(td.get('e_wkend_12by24')) or _sched_12x24_single_period()

    # ✨ Reconcile period ids & schedules here
    ec_tou, wkday, wkend = _reconcile_periods_and_equalize_tiers(ec_tou, wkday, wkend)

    # ✨ NEW: make tier caps & units consistent across periods
    ec_tou = _harmonize_tier_caps_and_units(ec_tou)

    out['ur_ec_tou_mat']        = ec_tou
    out['ur_ec_sched_weekday']  = wkday
    out['ur_ec_sched_weekend']  = wkend

    # Demand charge structure — parsed but only enabled if present (and not globally skipped elsewhere)
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

def _init_worker(dsn, role):
    """
    Pool initializer: open a fresh DB connection in this worker.
    """
    global _worker_conn
    _worker_conn, _ = utilfunc.make_con(dsn, role)

def size_chunk(static_agents_df: pd.DataFrame, sectors, rate_switch_table, mode="simple"):
    """
    Simple aggregation:
      net_sum_kw[h] = Σ_agents [ adopter_net_hourly[h] * number_of_adopters
                                 + baseline_net_hourly[h] * (customers_in_bin - number_of_adopters) ]
    Drops hourly arrays before returning df_chunk.
    """
    global _worker_conn
    results = []

    n_hours = None
    net_sum = None  # kW

    for aid, row in static_agents_df.iterrows():
        agent = row.copy()
        agent.name = aid

        sized = calc_system_size_and_performance(
            _worker_conn, agent, sectors, rate_switch_table
        )

        # Pull arrays (per-customer kW)
        base = np.asarray(sized.get("baseline_net_hourly", []), dtype=float)
        adop = np.asarray(sized.get("adopter_net_hourly",  []), dtype=float)

        # Scalars
        try:
            n_cust = float(sized.get("customers_in_bin", 0.0))
        except Exception:
            n_cust = 0.0
        try:
            n_adopt = float(sized.get("number_of_adopters", 0.0))  # cumulative
        except Exception:
            n_adopt = 0.0

        n_non = max(n_cust - n_adopt, 0.0)

        # Align and accumulate
        if adop.size == 0 and base.size == 0:
            pass
        else:
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

        # Drop heavy arrays so they don't hit agent_outputs
        drop_cols = (
            "baseline_net_hourly",
            "adopter_net_hourly",                # <-- REMOVE this line
            "adopter_load_hourly",
            "adopter_pv_hourly",
            "adopter_batt_to_load_hourly",
            "adopter_grid_to_batt_hourly",
            "pv_per_kw_hourly",
            "consumption_hourly",
            "generation_hourly",
            "batt_dispatch_profile",
            "net_hourly",
        )
        # keep baseline/adopter net so we can aggregate *after* diffusion
        drop_cols = tuple(c for c in drop_cols if c not in ("baseline_net_hourly","adopter_net_hourly"))
        for c in drop_cols:
            if c in sized.index:
                sized = sized.drop(labels=[c])

        results.append(sized)

    df_out = pd.DataFrame(results)
    agg = {
        "mode": "simple",
        "n_hours": int(n_hours or 0),
        "net_sum_kw": (net_sum.tolist() if net_sum is not None else []),
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
    # np.float is deprecated; use float
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

    # Core toggles
    mo = int(getattr(ER, "ur_metering_option", -1))  # 0=NM, 1=Net Billing, 2=Buy-All/Sell-All
    ts_sell_en = int(getattr(ER, "ur_en_ts_sell_rate", 0))
    ts_buy_en  = int(getattr(ER, "ur_en_ts_buy_rate", 0))

    # TOU matrix (period, tier, max_usage, unit_code, price, net_sell_rate)
    tou = np.asarray(getattr(ER, "ur_ec_tou_mat", []), dtype=float)
    wk  = np.asarray(getattr(ER, "ur_ec_sched_weekday", []), dtype=int)
    we  = np.asarray(getattr(ER, "ur_ec_sched_weekend", []), dtype=int)

    print(f"{title} mo={mo} (0=NM,1=NB,2=BA/SA)  ec_rows={tou.shape}  "
          f"wk_unique={np.unique(wk) if wk.size else []}  "
          f"we_unique={np.unique(we) if we.size else []}  "
          f"ts_sell_en={ts_sell_en} ts_buy_en={ts_buy_en}",
          flush=True)

    if tou.size:
        # First-tier (tier==1) buy prices by period (1-based periods)
        first_tier = tou[tou[:, 1] == 1]
        period_prices = {}
        for per in np.unique(first_tier[:, 0].astype(int)):
            prices_this_period = first_tier[first_tier[:, 0] == per][:, 4]
            if prices_this_period.size:
                period_prices[per] = float(np.mean(prices_this_period))
        for per in sorted(period_prices):
            print(f"{title}  period {per}: buy ${period_prices[per]:.4f}/kWh", flush=True)

        # Small schedule histograms (helps detect “all 1s” flattening)
        def _hist(label, sched):
            if sched.size:
                vals, counts = np.unique(sched, return_counts=True)
                pairs = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts))
                print(f"{title}  {label} sched histogram -> {pairs}", flush=True)
        _hist("wkday", wk)
        _hist("wkend", we)

    # If TS sell is on, show quick stats
    if ts_sell_en:
        sell = np.asarray(getattr(ER, "ur_ts_sell_rate", []), dtype=float)
        if sell.size:
            print(f"{title}  ts_sell len={sell.size}  "
                  f"min={sell.min():.4f} mean={sell.mean():.4f} max={sell.max():.4f}",
                  flush=True)
