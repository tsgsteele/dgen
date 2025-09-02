# battery_analysis.py
"""
Agent-level battery value decomposition and plotting, SQL-first (no 12M-row pulls).

Tables used (schema-qualified):
- agent_hourly_econ (long, hourly): columns
  agent_id, year, scenario_case ('pv_only'|'pv_batt'), hour_index, variable, value
- agent_annual_finance (annual summary): columns
  agent_id, year, scenario_case, system_kw, batt_kw, batt_kwh, npv_usd, payback_yrs,
  bill_wo_sys_year1_usd, bill_w_sys_year1_usd, energy_value_year1_usd,
  annual_import_kwh, annual_export_kwh

Key idea:
We compute, per agent, the incremental battery value as the change in *with-system* charges:
    ΔEnergyCharge   = Σ energy_charge_with_sys_usd (pv_only) - Σ energy_charge_with_sys_usd (pv_batt)
    ΔDemandCharge   = Σ demand_charge_with_sys_usd (pv_only) - Σ demand_charge_with_sys_usd (pv_batt)
    ΔSalesPurchases = Σ sales_purchases_usd       (pv_batt) - Σ sales_purchases_usd       (pv_only)
These reconcile (first-year) to bill deltas and help explain NPV differences.

Usage:
    from battery_analysis import get_engine, load_annual, aggregate_hourly_components, reconcile_summary, \
                                 plot_scatter_value_vs_size, plot_hist_value, plot_agent_day

    engine, schema = get_engine()  # or get_engine(schema="myschema")
    annual = load_annual(engine, schema, year=2026)
    hourly = aggregate_hourly_components(engine, schema, year=2026)
    view   = reconcile_summary(annual, hourly)

    # Example plots
    plot_scatter_value_vs_size(view)
    plot_hist_value(view, column="delta_energy_charge_usd + delta_demand_charge_usd + delta_sales_purchases_usd")

    # Typical-day plot for a specific agent:
    plot_agent_day(engine, schema, agent_id=483994, year=2026)
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple, List, Dict

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Optional: prefer project-native settings/utilfunc if available
try:
    import settings  # your project module
    import utility_functions as utilfunc  # your project module
    _HAS_INTERNAL = True
except Exception:
    _HAS_INTERNAL = False


# -----------------------
# Connection / bootstrap
# -----------------------

def get_engine(schema: Optional[str] = None) -> Tuple[Engine, str]:
    """
    Build an Engine (SQLAlchemy). Prefers internal project settings/utilfunc, else env var.

    Env fallback:
      DGEN_ENGINE_URL = postgresql+psycopg2://user:pass@host:5432/db
      DGEN_SCHEMA     = <schema>   (default 'public')

    Returns: (engine, schema)
    """
    if _HAS_INTERNAL:
        ms = settings.init_model_settings()
        engine = utilfunc.make_engine(ms.pg_engine_string)
        target_schema = schema or getattr(ms, "schema", None) or getattr(ms, "pg_schema", None) or "public"
        return engine, target_schema

    url = os.getenv("DGEN_ENGINE_URL")
    if not url:
        raise RuntimeError("Set DGEN_ENGINE_URL (or use project settings/utilfunc).")
    engine = create_engine(url, pool_pre_ping=True, future=True)
    target_schema = schema or os.getenv("DGEN_SCHEMA", "public")
    return engine, target_schema


def _qname(schema: str, table: str) -> str:
    """Return a quoted, schema-qualified table name."""
    return f'"{schema}".{table}'


# ----------------------------------------
# Aggregations (all heavy work is in SQL)
# ----------------------------------------

_VARS_HOURLY = (
    # $ components
    "energy_charge_with_sys_usd",
    "demand_charge_with_sys_usd",
    "sales_purchases_usd",
    # kWh context (optional for plots / sanity)
    "grid_import_kwh",
    "grid_export_kwh",
    "batt_to_load_kwh",
    "system_to_batt_kwh",
    "batt_to_grid_kwh",
)

def aggregate_hourly_components(
    engine: Engine, schema: str, year: int, agent_ids: Optional[Iterable[int]] = None
) -> pd.DataFrame:
    """
    SQL-first aggregation to agent-level sums for the variables above, *per case*.
    Returns a wide DataFrame with pv_only_* and pv_batt_* columns and deltas prefixed 'delta_'.

    We intentionally compare the *with-system* series across cases:
      pv_only.energy_charge_with_sys_usd  vs  pv_batt.energy_charge_with_sys_usd
    so we isolate incremental battery value, not PV-vs-baseline.
    """
    tbl = _qname(schema, "agent_hourly_econ")
    vars_list = ", ".join([f"'{v}'" for v in _VARS_HOURLY])

    where_agents = ""
    params: Dict[str, object] = {"year": int(year)}

    if agent_ids:
        ids = list(set(int(a) for a in agent_ids))
        ids_param = ", ".join([str(i) for i in ids])
        where_agents = f"AND agent_id IN ({ids_param})"

    sql = f"""
        WITH base AS (
            SELECT agent_id,
                   scenario_case,
                   variable,
                   SUM(value) AS total_val
            FROM {tbl}
            WHERE year = :year
              AND variable IN ({vars_list})
              {where_agents}
            GROUP BY agent_id, scenario_case, variable
        ),
        pivot AS (
            SELECT agent_id,
                   scenario_case,
                   MAX(CASE WHEN variable = 'energy_charge_with_sys_usd' THEN total_val END) AS energy_charge_with_sys_usd,
                   MAX(CASE WHEN variable = 'demand_charge_with_sys_usd' THEN total_val END) AS demand_charge_with_sys_usd,
                   MAX(CASE WHEN variable = 'sales_purchases_usd'       THEN total_val END) AS sales_purchases_usd,
                   MAX(CASE WHEN variable = 'grid_import_kwh'           THEN total_val END) AS grid_import_kwh,
                   MAX(CASE WHEN variable = 'grid_export_kwh'           THEN total_val END) AS grid_export_kwh,
                   MAX(CASE WHEN variable = 'batt_to_load_kwh'          THEN total_val END) AS batt_to_load_kwh,
                   MAX(CASE WHEN variable = 'system_to_batt_kwh'        THEN total_val END) AS system_to_batt_kwh,
                   MAX(CASE WHEN variable = 'grid_to_batt_kwh'          THEN total_val END) AS grid_to_batt_kwh,
                   MAX(CASE WHEN variable = 'batt_to_grid_kwh'          THEN total_val END) AS batt_to_grid_kwh
            FROM base
            GROUP BY agent_id, scenario_case
        )
        SELECT * FROM pivot
        ORDER BY agent_id, scenario_case;
    """

    df_long = pd.read_sql(text(sql), engine, params=params)

    if df_long.empty:
        return df_long

    # pivot cases to columns: pv_only_* and pv_batt_*
    df_wide = df_long.pivot(index="agent_id", columns="scenario_case")
    # flatten MultiIndex columns
    df_wide.columns = [f"{c2}_{c1}" for (c1, c2) in df_wide.columns]
    df_wide = df_wide.reset_index()

    # deltas (pv_batt minus pv_only where appropriate)
    def _pick(col: str, case: str, frame: pd.DataFrame):
        a = f"{case}_{col}"
        b = f"{col}_{case}"
        if a in frame.columns: return frame[a]
        if b in frame.columns: return frame[b]
        return None

    def _delta(col: str, reverse: bool = False):
        a = _pick(col, "pv_batt", df_wide)
        b = _pick(col, "pv_only", df_wide)
        if a is None or b is None:
            return np.nan
        return (a - b) if not reverse else (b - a)

    df_wide["delta_energy_charge_usd"]   = _delta("energy_charge_with_sys_usd", reverse=True)  # avoided cost
    df_wide["delta_demand_charge_usd"]   = _delta("demand_charge_with_sys_usd", reverse=True)  # avoided demand
    df_wide["delta_sales_purchases_usd"] = _delta("sales_purchases_usd",       reverse=False)  # net change in export/purchase $
    df_wide["delta_import_kwh"]          = _delta("grid_import_kwh",           reverse=True)   # avoided imports (kWh)
    df_wide["delta_export_kwh"]          = _delta("grid_export_kwh",           reverse=False)  # extra exports (kWh)
    df_wide["batt_to_load_kwh_pv_batt"]  = df_wide.get("batt_to_load_kwh_pv_batt", np.nan)
    df_wide["batt_to_grid_kwh_pv_batt"]  = df_wide.get("batt_to_grid_kwh_pv_batt", np.nan)

    # summed $ components (first-year dollar view)
    for col in ("delta_energy_charge_usd", "delta_demand_charge_usd", "delta_sales_purchases_usd"):
        if col not in df_wide:
            df_wide[col] = np.nan
    df_wide["delta_total_energy_plus_demand_plus_sales_usd"] = (
        df_wide["delta_energy_charge_usd"].fillna(0.0)
        + df_wide["delta_demand_charge_usd"].fillna(0.0)
        + df_wide["delta_sales_purchases_usd"].fillna(0.0)
    )

    return df_wide


def load_annual(engine: Engine, schema: str, year: int, agent_ids: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """
    Load annual summary rows for the requested year; pivot pv_only/pv_batt to wide and compute deltas.
    Works with either column order: 'pv_batt_npv_usd' or 'npv_usd_pv_batt', etc.
    """
    tbl = _qname(schema, "agent_annual_finance")

    where_agents = ""
    params: Dict[str, object] = {"year": int(year)}
    if agent_ids:
        ids = list(set(int(a) for a in agent_ids))
        ids_param = ", ".join([str(i) for i in ids])
        where_agents = f"AND agent_id IN ({ids_param})"

    sql = f"""
        SELECT agent_id, year, scenario_case,
               system_kw, batt_kw, batt_kwh,
               npv_usd, payback_yrs,
               bill_wo_sys_year1_usd, bill_w_sys_year1_usd,
               energy_value_year1_usd,
               annual_import_kwh, annual_export_kwh
        FROM {tbl}
        WHERE year = :year
          {where_agents}
        ORDER BY agent_id, scenario_case;
    """
    df = pd.read_sql(text(sql), engine, params=params)
    if df.empty:
        return df

    wide = df.pivot(index=["agent_id", "year"], columns="scenario_case")
    # Flatten to case_first_metric: 'pv_batt_npv_usd'
    wide.columns = [f"{c2}_{c1}" for (c1, c2) in wide.columns]
    wide = wide.reset_index()

    def pick(frame: pd.DataFrame, metric: str, case: str):
        """Return Series for either 'case_metric' or 'metric_case' if present."""
        a = f"{case}_{metric}"
        b = f"{metric}_{case}"
        if a in frame.columns: return frame[a]
        if b in frame.columns: return frame[b]
        return None

    def delta(metric: str):
        a = pick(wide, metric, "pv_batt")
        b = pick(wide, metric, "pv_only")
        if a is None or b is None: return pd.Series(np.nan, index=wide.index)
        return a - b

    # Main deltas
    wide["delta_npv_usd"]     = delta("npv_usd")
    wide["delta_payback_yrs"] = delta("payback_yrs")

    # Design sizes from pv_batt side
    bkwh = pick(wide, "batt_kwh", "pv_batt")
    bkw  = pick(wide, "batt_kw",  "pv_batt")
    wide["batt_kwh_design"] = bkwh if bkwh is not None else np.nan
    wide["batt_kw_design"]  = bkw  if bkw  is not None else np.nan

    return wide

def compute_incremental_value_and_cost_from_detail(
    engine: Engine, schema: str, year: int = 2026, n_periods: int = 25,
    agent_ids: Optional[Iterable[int]] = None
) -> pd.DataFrame:
    """
    Return a tidy frame with per-agent incremental metrics computed from
    agent_annual_finance_detail (SQL-first):
      - delta_energy_value_usd = energy_value_usd_pv_batt - energy_value_usd_pv_only
      - delta_first_cost_usd   = first_cost_usd_pv_batt   - first_cost_usd_pv_only

    Notes
    -----
    • Uses load_energy_value_and_cost_from_detail(...) which already aggregates
      SUM(cf_energy_value) and SUM(cf_discounted_costs) by (agent, case).
    • Drops agents missing either case.
    """
    dfw = load_energy_value_and_cost_from_detail(
        engine, schema, year=year, n_periods=n_periods, agent_ids=agent_ids
    )
    if dfw.empty:
        return dfw

    # Helper to tolerate either 'case_metric' or 'metric_case'
    def pick(frame: pd.DataFrame, metric: str, case: str):
        a = f"{case}_{metric}"
        b = f"{metric}_{case}"
        if a in frame.columns: return frame[a]
        if b in frame.columns: return frame[b]
        return None

    ev_only = pick(dfw, "energy_value_usd", "pv_only")
    ev_batt = pick(dfw, "energy_value_usd", "pv_batt")
    c_only  = pick(dfw, "first_cost_usd",  "pv_only")
    c_batt  = pick(dfw, "first_cost_usd",  "pv_batt")

    # Build result, keeping agents with both scenarios present
    out = pd.DataFrame({"agent_id": dfw["agent_id"]})
    out["delta_energy_value_usd"] = (ev_batt - ev_only).astype(float)
    out["delta_first_cost_usd"]   = (c_batt - c_only).astype(float)

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta_energy_value_usd", "delta_first_cost_usd"])
    return out


def plot_incremental_value_vs_cost_from_detail(
    engine: Engine, schema: str, year: int = 2026, n_periods: int = 25,
    sample: Optional[int] = None, show_breakeven: bool = True
) -> None:
    """
    Scatter where:
      x = incremental energy value (pv_batt − pv_only)
      y = incremental first cost  (pv_batt − pv_only)

    A point on the 45° line (y=x) implies rough "annualized breakeven" if you
    interpret x as a level annual value and y as up-front cost (ignoring discounting timing).
    """
    df = compute_incremental_value_and_cost_from_detail(
        engine, schema, year=year, n_periods=n_periods
    )
    df = df[(df['delta_energy_value_usd'] <= 10000) & 
            (df['delta_energy_value_usd'] >= -10000) &
            (df['delta_first_cost_usd'] <= 10000) &
            (df['delta_first_cost_usd'] >= -10000)]
    if df.empty:
        print("No incremental rows to plot.")
        return

    if sample is not None and len(df) > sample:
        df = df.sample(sample, random_state=11)

    x = df["delta_energy_value_usd"].values
    y = df["delta_first_cost_usd"].values

    plt = _matplotlib_import()
    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, alpha=0.75)

    # Axes at zero for quick quadrant read:
    plt.axhline(0, linewidth=1, color="k")
    plt.axvline(0, linewidth=1, color="k")

    # Optional y=x breakeven guide
    if show_breakeven:
        lim = max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y)))
        lim = float(lim if np.isfinite(lim) and lim > 0 else 1.0)
        grid = np.linspace(-lim, lim, 100)
        plt.plot(grid, grid, linestyle="--", linewidth=1, label="y = x (breakeven)")

    plt.xlabel("Incremental energy value (pv_batt − pv_only) [$ / yr]")
    plt.ylabel("Incremental first cost (pv_batt − pv_only) [$]")
    plt.title(f"Battery incremental value vs cost — year {year}")
    if show_breakeven:
        plt.legend()
    plt.tight_layout()
    plt.show()


def load_energy_value_and_cost_from_detail(
    engine: Engine, schema: str, year: int = 2026, n_periods: int = 25, agent_ids: Optional[Iterable[int]] = None
) -> pd.DataFrame:
    """
    Aggregate agent_annual_finance_detail into per-agent, per-case totals:
      - first_cost_usd: SUM(cf_discounted_costs) over periods 1..n_periods
      - energy_value_usd: SUM(cf_energy_value) over periods 1..n_periods
    Returns a wide frame with pv_only_* and pv_batt_* columns.
    """
    tbl = _qname(schema, "agent_annual_finance_detail")

    where_agents = ""
    if agent_ids:
        ids = list(set(int(a) for a in agent_ids))
        ids_param = ", ".join([str(i) for i in ids])
        where_agents = f" AND agent_id IN ({ids_param}) "

    sql = f"""
        WITH base AS (
            SELECT agent_id,
                   scenario_case,
                   metric,
                   SUM(value) AS total_val
            FROM {tbl}
            WHERE year = :year
              AND scenario_case IN ('pv_only','pv_batt')
              AND metric IN ('cf_discounted_costs','cf_energy_value')
              {where_agents}
            GROUP BY agent_id, scenario_case, metric
        ),
        agg AS (
            SELECT agent_id,
                   scenario_case,
                   MAX(CASE WHEN metric = 'cf_discounted_costs' THEN total_val END) AS first_cost_usd,
                   MAX(CASE WHEN metric = 'cf_energy_value'      THEN total_val END) AS energy_value_usd
            FROM base
            GROUP BY agent_id, scenario_case
        )
        SELECT * FROM agg
        ORDER BY agent_id, scenario_case;
    """
    df = pd.read_sql(
        text(sql), engine, params={"year": int(year), "nper": int(n_periods)}
    )
    if df.empty:
        return df

    # pivot to wide; accept 'case_metric' order
    wide = df.pivot(index="agent_id", columns="scenario_case")
    wide.columns = [f"{c2}_{c1}" for (c1, c2) in wide.columns]
    wide = wide.reset_index()
    return wide

def plot_energy_value_vs_cost_from_detail(
    engine: Engine, schema: str, year: int = 2026, n_periods: int = 25,
    payback_lines: List[int] = [5, 10, 20], sample: Optional[int] = None
) -> None:
    """
    Pulls cost/value from agent_annual_finance_detail and plots Annual Energy Value vs First Cost
    for pv_only and pv_batt, connected per agent, with iso-payback lines (y = x / years).
    """
    dfw = load_energy_value_and_cost_from_detail(engine, schema, year=year, n_periods=n_periods)
    if dfw.empty:
        print("No aggregated detail data found.")
        return

    # Helper to pick either 'case_metric' or 'metric_case' (if you later change naming)
    def pick(frame: pd.DataFrame, metric: str, case: str):
        a = f"{case}_{metric}"
        b = f"{metric}_{case}"
        if a in frame.columns: return frame[a]
        if b in frame.columns: return frame[b]
        return None

    ev_only = pick(dfw, "energy_value_usd", "pv_only")
    ev_batt = pick(dfw, "energy_value_usd", "pv_batt")
    c_only  = pick(dfw, "first_cost_usd",  "pv_only")
    c_batt  = pick(dfw, "first_cost_usd",  "pv_batt")

    missing = []
    if ev_only is None: missing.append("pv_only.energy_value_usd")
    if ev_batt is None: missing.append("pv_batt.energy_value_usd")
    if c_only  is None: missing.append("pv_only.first_cost_usd")
    if c_batt  is None: missing.append("pv_batt.first_cost_usd")
    if missing:
        print("Missing columns:", ", ".join(missing))
        return

    dfp = pd.DataFrame({
        "agent_id": dfw["agent_id"],
        "ev_only":  ev_only.astype(float),
        "ev_batt":  ev_batt.astype(float),
        "c_only":   c_only.astype(float),
        "c_batt":   c_batt.astype(float),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if dfp.empty:
        print("No finite rows to plot.")
        return

    if sample is not None and len(dfp) > sample:
        dfp = dfp.sample(sample, random_state=11)

    plt = _matplotlib_import()
    plt.figure(figsize=(8, 6))

    # Scatter for pv_only (hollow) and pv_batt (filled)
    plt.scatter(dfp["c_only"], dfp["ev_only"], label="pv_only", alpha=0.7, facecolors='none', edgecolors='tab:gray')
    plt.scatter(dfp["c_batt"], dfp["ev_batt"], label="pv_batt", alpha=0.85, color='tab:blue')

    # Connect per-agent points
    for _, r in dfp.iterrows():
        plt.plot([r["c_only"], r["c_batt"]], [r["ev_only"], r["ev_batt"]],
                 color='tab:blue', alpha=0.25, linewidth=1)

    # Iso-payback lines: y = x / years
    xmax = float(max(dfp["c_only"].max(), dfp["c_batt"].max()))
    ymax = float(max(dfp["ev_only"].max(), dfp["ev_batt"].max()))
    x = np.linspace(0, max(xmax, 1.0), 100)
    for yrs in payback_lines:
        y = x / float(yrs)
        plt.plot(x, y, linestyle='--', linewidth=1, label=f"{yrs}-yr payback")

    plt.xlim(left=0)
    # ensure room for shortest payback line
    ytop = max(ymax, (xmax / min(payback_lines)) * 1.1)
    plt.ylim(bottom=0, top=ytop)

    plt.xlabel("First cost from cf_discounted_costs (sum over periods) [$]")
    plt.ylabel("Annual energy value (sum of cf_energy_value) [$ / yr]")
    plt.title(f"Energy value vs cost — pv_only ↔ pv_batt (year {year}, N={n_periods} periods)")
    plt.legend()
    plt.tight_layout()
    plt.show()



def reconcile_summary(annual_wide: pd.DataFrame, hourly_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Join annual and hourly views; compute first-year bill delta and a break-even annual value.
    Robust to both column-name orders: 'pv_batt_bill_w_sys_year1_usd' or 'bill_w_sys_year1_usd_pv_batt'.
    """
    if annual_wide.empty or hourly_wide.empty:
        return pd.DataFrame()

    view = annual_wide.merge(hourly_wide, on="agent_id", how="inner")

    def pick(frame: pd.DataFrame, metric: str, case: str):
        a = f"{case}_{metric}"
        b = f"{metric}_{case}"
        if a in frame.columns: return frame[a]
        if b in frame.columns: return frame[b]
        return None

    # First-year bill delta (pv_batt - pv_only): negative = savings
    bill_only = pick(view, "bill_w_sys_year1_usd", "pv_only")
    bill_batt = pick(view, "bill_w_sys_year1_usd", "pv_batt")
    if bill_only is not None and bill_batt is not None:
        view["delta_bill_y1_usd"] = bill_batt - bill_only

    # Keep reconciliation against component sum when available
    if "delta_total_energy_plus_demand_plus_sales_usd" in view and "delta_bill_y1_usd" in view:
        view["recon_error_usd"] = (
            view["delta_bill_y1_usd"].fillna(0.0)
            - view["delta_total_energy_plus_demand_plus_sales_usd"].fillna(0.0)
        )

    # Value intensity using bill delta
    if "batt_kwh_design" in view:
        denom = view["batt_kwh_design"].replace(0, np.nan)
        view["bill_delta_per_kwh_usd_per_yr"] = view.get("delta_bill_y1_usd", np.nan) / denom

    # Break-even: annual savings needed to make ΔNPV >= 0
    dn_pvb = pick(view, "npv_usd", "pv_batt")
    dn_pvo = pick(view, "npv_usd", "pv_only")
    if dn_pvb is None or dn_pvo is None:
        view["annual_savings_needed_usd"] = np.nan
        view["annual_needed_per_kwh_usd"] = np.nan
        return view
    view["delta_npv_usd"] = dn_pvb - dn_pvo

    # r and N: try to read; else defaults
    r = None
    for cand in ("pv_batt_nominal_discount_rate", "nominal_discount_rate_pv_batt",
                 "pv_only_nominal_discount_rate", "nominal_discount_rate_pv_only"):
        if cand in view.columns:
            r = view[cand].astype(float)
            break
    if r is None: r = pd.Series(0.06, index=view.index)

    N = None
    for cand in ("pv_batt_cf_length", "cf_length_pv_batt",
                 "pv_only_cf_length", "cf_length_pv_only"):
        if cand in view.columns:
            N = view[cand].astype(float)
            break
    if N is None: N = pd.Series(25.0, index=view.index)

    pvf = (1 - (1 + r).pow(-N)) / r.replace(0, np.nan)
    pvf = pvf.fillna(N)

    view["annual_savings_needed_usd"] = np.maximum(0.0, (-view["delta_npv_usd"].fillna(0.0)) / pvf)

    denom = view.get("batt_kwh_design", pd.Series(np.nan, index=view.index)).replace(0, np.nan)
    view["annual_needed_per_kwh_usd"] = view["annual_savings_needed_usd"] / denom

    return view


# ----------------
# Plotting helpers
# ----------------

def _matplotlib_import():
    import matplotlib.pyplot as plt
    return plt


def plot_scatter_value_vs_size(view: pd.DataFrame, x="batt_kwh_design",
                               y="delta_bill_y1_usd", annotate_frac: float = 0.0) -> None:
    """
    Scatter: battery size (kWh design) vs first-year bill delta ($).
    Negative Y means battery lowered the bill.
    """
    if view.empty:
        print("No data to plot.")
        return
    plt = _matplotlib_import()

    sub = view[[x, y]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[sub[x] > 0]  # must have a battery

    if sub.empty:
        print("No finite points to plot.")
        return

    xs = sub[x].values
    ys = sub[y].values
    plt.figure()
    plt.axhline(0, linewidth=1, color="k")
    plt.scatter(xs, ys, alpha=0.6)
    plt.xlabel("Battery size (kWh, design)")
    plt.ylabel("First-year bill delta ($)  [pv_batt − pv_only]")
    plt.title("Battery value vs size (agent-level)")
    plt.show()


def plot_topN_component_bars(view: pd.DataFrame, N: int = 12) -> None:
    """
    Stacked bars of first-year components explaining bill delta for top-|Δbill| agents.
    Requires the hourly-component deltas if present; falls back gracefully.
    """
    if view.empty:
        print("No data to plot.")
        return
    needed = ["delta_bill_y1_usd"]
    if not all(c in view.columns for c in needed):
        print("Missing delta_bill_y1_usd.")
        return

    # choose top-N by absolute bill delta
    sub = view.copy()
    sub = sub.replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna(subset=["delta_bill_y1_usd"])
    sub = sub.sort_values("delta_bill_y1_usd", key=lambda s: s.abs(), ascending=False).head(N)

    # components (default to 0 if missing)
    comp_cols = {
        "delta_energy_charge_usd": "Energy charges",
        "delta_demand_charge_usd": "Demand charges",
        "delta_sales_purchases_usd": "Sales/Purchases",
    }
    comps = {label: sub.get(col, pd.Series(0.0, index=sub.index)).fillna(0.0).values
             for col, label in comp_cols.items()}

    ind = np.arange(len(sub))
    plt = _matplotlib_import()
    plt.figure(figsize=(max(8, N), 5))
    bottom = np.zeros(len(sub))
    for label, vals in comps.items():
        plt.bar(ind, vals, bottom=bottom, label=label, alpha=0.8)
        bottom += vals
    plt.axhline(0, linewidth=1, color="k")
    plt.xticks(ind, sub["agent_id"].astype(str).tolist(), rotation=45, ha="right")
    plt.ylabel("First-year $")
    plt.title("Battery bill impact decomposition — top |Δbill| agents")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_break_even_hist(view: pd.DataFrame, per_kwh: bool = True, bins: int = 40) -> None:
    """
    Histogram of required level annual savings to make ΔNPV >= 0.
    If per_kwh=True, normalizes by battery size (needed $/kWh-yr).
    """
    if view.empty:
        print("No data to plot.")
        return
    plt = _matplotlib_import()

    col = "annual_needed_per_kwh_usd" if per_kwh else "annual_savings_needed_usd"
    if col not in view.columns:
        print(f"Column '{col}' missing.")
        return

    vals = view[col].replace([np.inf, -np.inf], np.nan).dropna()
    if per_kwh:
        vals = vals[vals > 0]

    if vals.empty:
        print("No data to plot.")
        return

    plt.figure()
    plt.hist(vals.values, bins=bins)
    plt.xlabel("$ per kWh-year needed to break even" if per_kwh else "Annual $ needed to break even")
    plt.ylabel("Count of agents")
    plt.title("Battery break-even requirement (level annual)")
    plt.show()


def plot_hist_value(view: pd.DataFrame,
                    column="delta_total_energy_plus_demand_plus_sales_usd",
                    bins: int = 40) -> None:
    """
    Histogram of incremental first-year value across agents.
    """
    if view.empty:
        print("No data to plot.")
        return
    plt = _matplotlib_import()
    vals = view[column].dropna().values
    plt.figure()
    plt.hist(vals, bins=bins)
    plt.xlabel("Incremental first-year value ($)")
    plt.ylabel("Count of agents")
    plt.title("Distribution of battery value")
    plt.show()


def plot_agent_day(engine: Engine, schema: str, agent_id: int, year: int,
                   day_index: int = 200) -> None:
    """
    Typical-day plot for one agent/day.

    Left y-axis: kWh flows (PV→load, Batt→load, Grid import/export, PV→batt, Grid→batt, Batt→grid)
    Second right y-axis (offset): Utility BUY price (¢/kWh) from utility_price_usd_per_kwh
    """
    import numpy as np
    import pandas as pd
    from sqlalchemy import text

    tbl = _qname(schema, "agent_hourly_econ")
    h0 = int(day_index) * 24 + 1
    h1 = h0 + 23

    # variables to pull
    vars_needed = (
        "system_to_load_kwh",
        "batt_to_load_kwh",
        "grid_import_kwh",
        "grid_export_kwh",
        "system_to_batt_kwh",
        "utility_price_usd_per_kwh"
        # "batt_soc_pct",
    )
    vars_list = ", ".join([f"'{v}'" for v in vars_needed])

    sql = f"""
        SELECT scenario_case, hour_index, variable, value
        FROM {tbl}
        WHERE agent_id = :aid
          AND year = :year
          AND variable IN ({vars_list})
          AND hour_index BETWEEN :h0 AND :h1
        ORDER BY scenario_case, hour_index, variable;
    """
    df = pd.read_sql(
        text(sql), engine,
        params={"aid": int(agent_id), "year": int(year), "h0": h0, "h1": h1}
    )
    if df.empty:
        print("No data for that agent/day.")
        return

    # Pivot per case
    cases = {}
    for case in ("pv_only", "pv_batt"):
        sub = df[df["scenario_case"] == case]
        if not sub.empty:
            pvt = sub.pivot(index="hour_index", columns="variable", values="value").sort_index()
            if case == "pv_batt" and "system_to_load_kwh" in pvt and "batt_to_load_kwh" in pvt:
                pvt["system_to_load_kwh"] = pvt["system_to_load_kwh"] - pvt["batt_to_load_kwh"].fillna(0.0)
            cases[case] = pvt
    if not cases:
        print("No matching case data.")
        return

    plt = _matplotlib_import()
    style_map = {
        "system_to_load_kwh": ("PV→load",         "tab:orange", "-"),
        "batt_to_load_kwh":   ("Batt→load",       "tab:blue",   "-"),
        "grid_import_kwh":    ("Grid import",     "tab:red",    ":"),
        "grid_export_kwh":    ("Grid export",     "tab:orange",  "-."),
        "system_to_batt_kwh": ("PV→batt (charge)","tab:blue", "-."),
    }
    hours = list(range(24))
    xticks = [0, 6, 12, 18, 23]
    xticklabels = ["12am", "6am", "12pm", "6pm", "11pm"]

    # Battery kW cap from annual table (pv_batt)
    ann_tbl = _qname(schema, "agent_annual_finance")
    ann = pd.read_sql(
        text(f"SELECT batt_kw FROM {ann_tbl} WHERE agent_id=:aid AND year=:year AND scenario_case='pv_batt' LIMIT 1;"),
        engine, params={"aid": int(agent_id), "year": int(year)}
    )
    batt_kw = float(ann.loc[0, "batt_kw"]) if not ann.empty and pd.notnull(ann.loc[0, "batt_kw"]) else None

    for case, pivot in cases.items():
        fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')

        # Left axis: energy flows (your styles preserved)
        for var, (label, color, ls) in style_map.items():
            if var in pivot.columns and len(pivot[var]) == 24:
                ax.plot(hours, pivot[var].to_numpy(), label=label, color=color, linestyle=ls)

        # # --- Second right axis: BUY price (¢/kWh) ---
        # price_ax = ax.twinx()
        # price_ax.spines["right"].set_position(("axes", 1.08))  # offset so it doesn't overlap
        # price_ax.set_frame_on(True)
        # price_ax.patch.set_visible(False)

        # if "utility_price_usd_per_kwh" in pivot.columns and len(pivot["utility_price_usd_per_kwh"]) == 24:
        #     price_c_per_kwh = pivot["utility_price_usd_per_kwh"].astype(float).to_numpy() * 100.0
        #     if np.isfinite(price_c_per_kwh).any():
        #         price_ax.plot(
        #             hours, price_c_per_kwh,
        #             linestyle="--", color="black", alpha=0.65, linewidth=1.5,
        #             label="BUY price"
        #         )
        #         price_ax.set_ylabel("Utility BUY price (¢/kWh)")

        # Labels, ticks, legend
        ax.set_xticks(xticks); ax.set_xticklabels(xticklabels)
        ax.set_xlabel("Hour of day"); ax.set_ylabel("kWh")
        ax.set_title(f"Agent {agent_id} — {case} — day {day_index}")

        # Unified legend (flows + price)
        h1, l1 = ax.get_legend_handles_labels()
        # h2, l2 = price_ax.get_legend_handles_labels()
        # ax.legend(h1 + h2, l1 + l2, loc='upper right', ncol=2)
        ax.legend(h1, l1, loc='upper right', ncol=2) # If no buy price

        plt.show()
    return df


# --------------------------
# Simple CLI (optional use)
# --------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Battery value decomposition (SQL-first).")
    p.add_argument("--year", type=int, required=True, help="Simulation year (e.g., 2026)")
    p.add_argument("--schema", type=str, default=None, help="DB schema (defaults to project/env)")
    p.add_argument("--sample", type=int, default=0, help="Optional sample size of agents")
    args = p.parse_args()

    engine, schema = get_engine(schema=args.schema)
    annual = load_annual(engine, schema, year=args.year)

    agent_ids = None
    if args.sample and not annual.empty:
        agent_ids = annual["agent_id"].sample(min(args.sample, len(annual)), random_state=1).tolist()

    hourly = aggregate_hourly_components(engine, schema, year=args.year, agent_ids=agent_ids)
    view = reconcile_summary(annual, hourly)

    if view.empty:
        print("No data.")
        return

    # Print a compact summary
    cols = [
        "agent_id",
        "batt_kwh_design",
        "delta_total_energy_plus_demand_plus_sales_usd",
        "delta_npv_usd",
        "delta_payback_yrs",
        "recon_error_usd",
    ]
    existing = [c for c in cols if c in view.columns]
    print(view[existing].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
