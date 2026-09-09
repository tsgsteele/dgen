
"""
analysis_functions.py
=====================

Utilities for loading, aggregating, and visualizing **dGen** model outputs across U.S. states,
with **single-pass I/O** and **production-grade documentation**.

Key design principles
---------------------
1. **Load once, use everywhere.** The :class:`DataWarehouse` class reads all per‑state CSVs
   (baseline & policy) and hourly files (state & RTO) exactly once. All downstream analysis
   and plotting functions accept dataframes and/or a warehouse instance so they **do not
   touch the filesystem** again.
2. **Assume the schema.** These helpers are purpose-built for the schema shown in the
   example files (e.g., ``baseline.csv``, ``policy.csv``, ``*_state_hourly.csv``, ``*_rto_hourly.csv``).
   Columns are treated as present with expected dtypes; we *do not* add defensive checks,
   try/except wrappers, or fallbacks unless strictly necessary.
3. **Clear, typed docstrings.** Every function includes inputs/outputs and assumptions.

Coincident-peak methodology
---------------------------
Historically, coincident peak reduction was measured as the **difference at the single
baseline peak hour**. That metric is noisy: a one-hour spike can dominate results.
This module adds a **top‑N averaging** option (default: N=10):

- ``method="baseline_topN_avg"`` *(default)* — Find the **top N hours** in the **baseline** series.
  Return ``mean(baseline[topN]) - mean(policy[topN])``. This retains a common timestamp
  basis (true *coincidence*) while reducing noise and better representing capacity value.
- ``method="single_hour"`` — Legacy behavior: use the single **max** baseline hour.
- ``method="separate_topN_means"`` — Compare each scenario’s own top‑N mean
  (``mean(baseline.topN) - mean(policy.topN)``). This is *not* strictly coincident, but
  can be informative for sensitivity.
- ``method="percentile_mean"`` — Average across baseline hours at/above a given percentile
  (e.g., 99.9th). Set ``percentile=0.999``.

The coincident functions expose ``top_n`` and ``percentile`` parameters to switch strategies.

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import geopandas as gpd  # optional, only needed for choropleths
except Exception:  # pragma: no cover - optional at runtime
    gpd = None


# -----------------------------------------------------------------------------
# Constants (schema)
# -----------------------------------------------------------------------------

AGENT_USECOLS: Sequence[str] = (
    "agent_id",
    # identity / keys
    "state_abbr", "scenario", "year", "rto",
    # adoption cohorts & stocks
    "new_adopters", "number_of_adopters",
    "customers_in_bin", "batt_adopters_added_this_year",
    # tech sizes (per-agent & cumulative)
    "system_kw", "new_system_kw", "system_kw_cum",
    "batt_kwh", "batt_kwh_cum",
    # prices / loads
    "price_per_kwh", "load_kwh_per_customer_in_bin_initial",
    # arrays (25-year per-adopter cashflows & bills)
    "cf_debt_payment_total_pv_only", "cf_debt_payment_total_pv_batt",
    "cf_energy_value_pv_only", "cf_energy_value_pv_batt",
    "utility_bill_w_sys_pv_only", "utility_bill_w_sys_pv_batt",
    "utility_bill_wo_sys_pv_only", "utility_bill_wo_sys_pv_batt",
    # market share (for plotting)
    "max_market_share",
    # payback period
    "payback_period",
    "system_capex_per_kw_combined", 
    "batt_capex_per_kwh_combined",
    "county_id",
    "pct_of_bldgs_developable"
)

STATE_HOURLY_USECOLS: Sequence[str] = ("scenario", "year", "net_sum_text")
RTO_HOURLY_USECOLS:   Sequence[str] = ("scenario", "rto", "year", "net_sum_text")


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _parse_array(s: object) -> List[float]:
    """
    Parse Postgres-like arrays (e.g. "{1,2,3}") or JSON lists ("[1,2,3]") into a list of floats.
    Returns [] if input is empty/NULL-like.
    """
    import json

    if s is None:
        return []
    if isinstance(s, (list, tuple, np.ndarray)):
        return [float(x) for x in s]

    t = str(s).strip()
    if not t or t.upper() in {"NULL", "NAN"}:
        return []

    # Fast path for Postgres-brace style: {1,2,3}
    if t.startswith("{") and t.endswith("}"):
        inner = t[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",")]
        out: List[float] = []
        for p in parts:
            if p:  # ignore empty tokens
                out.append(float(p))
        return out

    # JSON-like fallback: [1,2,3]
    if t.startswith("[") and t.endswith("]"):
        try:
            arr = json.loads(t)
            if isinstance(arr, list):
                return [float(x) for x in arr]
        except Exception:
            pass

    # As a last resort, try to split on commas without brackets
    try:
        parts = [p.strip() for p in t.split(",")]
        return [float(p) for p in parts if p]
    except Exception:
        return []



def _arr25(s: object) -> List[float]:
    """Parse + clip/pad to length 25 (life-year arrays)."""
    a = _parse_array(s)[:25]
    return a + [0.0] * (25 - len(a))


def _list_state_dirs(root_dir: str, states: Optional[Iterable[str]] = None) -> List[str]:
    """
    List subdirectories (two-letter state codes) under ``root_dir``.
    If ``states`` is provided, filter to that set (case-insensitive).
    """
    dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and len(d) == 2]
    if states:
        want = {s.strip().upper() for s in states if s}
        dirs = [d for d in dirs if os.path.basename(d).upper() in want]
    return sorted(dirs)


# -----------------------------------------------------------------------------
# Data warehouse (single-pass I/O)
# -----------------------------------------------------------------------------

@dataclass
class DataWarehouse:
    """
    Canonical in-memory representation of a full dGen run.

    Attributes
    ----------
    agents : pandas.DataFrame
        Per-state baseline & policy agent outputs (concatenated).
    state_hourly : pandas.DataFrame
        State-level hourly net load arrays by scenario/year.
        Columns: ``['state_abbr','scenario','year','net_load']``.
    rto_hourly : pandas.DataFrame
        RTO-level hourly net load arrays by scenario/year.
        Columns: ``['rto','scenario','year','net_load']``.
    """
    agents: pd.DataFrame
    state_hourly: pd.DataFrame
    rto_hourly: pd.DataFrame

    @staticmethod
    def _read_csv_intersect_cols(path: str, desired_cols: list[str]):
        """Read only desired columns that exist in the file; return None if zero overlap."""
        import os, warnings
        import pandas as pd

        if not os.path.exists(path):
            return None
        # Peek header to see what's actually there
        try:
            hdr = pd.read_csv(path, nrows=0, compression="infer")
        except pd.errors.EmptyDataError:
            warnings.warn(f"Skipping CSV with no readable header: {path}")
            return None

        available = [str(c).strip() for c in hdr.columns]
        wanted    = [str(c).strip() for c in desired_cols]
        cols = [c for c in wanted if c in available]
        if not cols:
            warnings.warn(
                "Skipping CSV with zero column overlap:\n"
                f"  file: {path}\n"
                f"  wanted (first 12): {wanted[:12]}{' ...' if len(wanted)>12 else ''}\n"
                f"  available (first 12): {available[:12]}{' ...' if len(available)>12 else ''}"
            )
            return None

        df = pd.read_csv(path, usecols=cols, compression="infer")
        # Drop duplicate-named columns if any (keep first)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
        return df

    @classmethod
    def from_disk(
        cls,
        root_dir: str,
        run_id: Optional[str] = None,
        states: Optional[Iterable[str]] = None,
    ) -> "DataWarehouse":
        """
        Load all states once (agents + hourly) into memory.

        Expectations
        ------------
        For each state folder ``<ROOT>/<STATE>[/<RUN_ID>]`` the following files exist:

        - ``baseline.csv`` and ``policy.csv``
        - ``baseline_state_hourly.csv`` and ``policy_state_hourly.csv``
        - ``baseline_rto_hourly.csv`` and ``policy_rto_hourly.csv``

        Parameters
        ----------
        root_dir : str
            Directory whose subfolders are two-letter state codes.
        run_id : str, optional
            Optional subfolder name to select a specific run.
        states : iterable[str], optional
            Subset of states to load (e.g., ``["CA", "NY"]``).

        Returns
        -------
        DataWarehouse
        """
        agent_frames: List[pd.DataFrame] = []
        state_hourly_rows: List[Tuple[str, str, int, List[float]]] = []
        rto_hourly_rows:   List[Tuple[str, str, int, List[float]]] = []

        for state_dir in _list_state_dirs(root_dir, states=states):
            state = os.path.basename(state_dir).upper()
            basedir = os.path.join(state_dir, str(run_id)) if run_id else state_dir

            # -- Agents (baseline + policy)
            for scen in ("baseline", "policy"):
                p = os.path.join(basedir, f"{scen}.csv")
                if not os.path.exists(p):
                    continue  # keep loader simple: skip missing files
                usecols = [c for c in AGENT_USECOLS if c != "scenario"]
                df = cls._read_csv_intersect_cols(p, usecols)
                if df is None or df.empty:
                    continue
                df["scenario"] = scen
                df["state_abbr"] = state.upper()
                agent_frames.append(df)

            # -- State hourly
            for scen in ("baseline", "policy"):
                p = os.path.join(basedir, f"{scen}_state_hourly.csv")
                if not os.path.exists(p):
                    continue
                h = pd.read_csv(p, usecols=STATE_HOURLY_USECOLS)
                # The file may or may not include scenario; we know which one we read
                h["scenario"] = scen
                for r in h.itertuples(index=False):
                    state_hourly_rows.append((state, scen, int(r.year), _parse_array(r.net_sum_text)))

            # -- RTO hourly
            for scen in ("baseline", "policy"):
                p = os.path.join(basedir, f"{scen}_rto_hourly.csv")
                if not os.path.exists(p):
                    continue
                h = pd.read_csv(p, usecols=RTO_HOURLY_USECOLS)
                h["scenario"] = scen
                for r in h.itertuples(index=False):
                    rto_hourly_rows.append((str(r.rto), scen, int(r.year), _parse_array(r.net_sum_text)))

        agents = pd.concat(agent_frames, ignore_index=True) if agent_frames else pd.DataFrame(columns=AGENT_USECOLS)
        state_hourly = pd.DataFrame(state_hourly_rows, columns=["state_abbr","scenario","year","net_load"])
        rto_hourly   = pd.DataFrame(rto_hourly_rows,   columns=["rto","scenario","year","net_load"])

        # Basic numeric coercions for key columns used downstream
        if not agents.empty:
            num_cols = ["year","new_adopters","number_of_adopters","customers_in_bin",
                        "batt_adopters_added_this_year","system_kw","new_system_kw",
                        "system_kw_cum","batt_kwh","batt_kwh_cum","price_per_kwh",
                        "load_kwh_per_customer_in_bin_initial","max_market_share"]
            for c in num_cols:
                if c in agents.columns:
                    agents[c] = pd.to_numeric(agents[c], errors="coerce")
            agents["state_abbr"] = agents["state_abbr"].astype(str).str.upper()
            agents["scenario"] = agents["scenario"].astype(str).str.lower()

        for df in (state_hourly, rto_hourly):
            if not df.empty:
                df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
                df["scenario"] = df["scenario"].astype(str).str.lower()

        return cls(agents=agents, state_hourly=state_hourly, rto_hourly=rto_hourly)


# -----------------------------------------------------------------------------
# Savings, totals, and market-share aggregations (agents)
# -----------------------------------------------------------------------------

@dataclass
class SavingsConfig:
    """
    Configuration for portfolio bill-savings rollups.

    Parameters
    ----------
    lifetime_years : int, default 25
        Credited lifetime of savings arrays.
    cap_to_horizon : bool, default False
        If True, only credit life-years that fall within the modeled calendar horizon
        (min(input years) .. max(input years)).
    """
    lifetime_years: int = 25
    cap_to_horizon: bool = False


def compute_portfolio_and_cumulative_savings(
    df: pd.DataFrame,
    cfg: SavingsConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build (1) annual portfolio bill savings by rolling each cohort's 25-yr array forward,
    (2) cumulative bill savings (cumsum of annual), (3) lifetime totals per state/scenario,
    (4) median lifetime savings per adopter to date (weighted by cohort size),
    (5) median upfront cost per adopter to date (weighted by cohort size),
    (6) median lifetime savings per adopter *this year* (weighted), and
    (7) median upfront cost per adopter *this year* (weighted).

    EXTENSIONS:
      - After-debt (net) metrics using cf_debt_payment_total_*.
      - NEW column `lifetime_net_after_upfront_total` = lifetime_net_savings_total − upfront_cost_total.
        (This represents lifetime savings net of upfront costs AND financing/debt.)
    """
    import numpy as np
    import pandas as pd

    def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        """Weighted median over 1D arrays; returns 0.0 if empty or total weight == 0."""
        if values is None:
            return 0.0
        v = np.asarray(values, dtype=float)
        if v.size == 0:
            return 0.0
        w = np.asarray(weights, dtype=float)
        mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not np.any(mask):
            return 0.0
        v = v[mask]; w = w[mask]
        order = np.argsort(v, kind="mergesort")
        v = v[order]; w = w[order]
        cw = np.cumsum(w)
        idx = np.searchsorted(cw, 0.5 * cw[-1], side="left")
        if idx >= len(v):
            idx = len(v) - 1
        return float(v[idx])

    # Empty scaffolding (now includes net & net-after-upfront columns)
    empty_cols_annual = [
        "state_abbr","scenario","year",
        "portfolio_annual_savings","portfolio_annual_net_savings",
        "portfolio_annual_upfront_cost","portfolio_annual_upfront_cost_pv","portfolio_annual_upfront_cost_batt",
        "portfolio_annual_financed_cost",
        "portfolio_annual_down_payment_cost","portfolio_annual_out_of_pocket_cost",
        "lifetime_savings_total","lifetime_net_savings_total","lifetime_net_after_upfront_total",
        "median_lifetime_savings_per_adopter_to_date",
        "median_lifetime_net_savings_per_adopter_to_date",
        "median_upfront_cost_per_adopter_to_date",
        "median_lifetime_savings_per_adopter_this_year",
        "median_lifetime_net_savings_per_adopter_this_year",
        "median_upfront_cost_per_adopter_this_year",
    ]
    empty_cols_cum = [
        "state_abbr","scenario","year",
        "cumulative_bill_savings","cumulative_net_savings",
        "lifetime_savings_total","lifetime_net_savings_total","lifetime_net_after_upfront_total",
        "median_lifetime_savings_per_adopter_to_date",
        "median_lifetime_net_savings_per_adopter_to_date",
        "median_upfront_cost_per_adopter_to_date",
        "median_lifetime_savings_per_adopter_this_year",
        "median_lifetime_net_savings_per_adopter_this_year",
        "median_upfront_cost_per_adopter_this_year",
    ]

    if df.empty:
        return (pd.DataFrame(columns=empty_cols_annual),
                pd.DataFrame(columns=empty_cols_cum),
                pd.DataFrame(columns=["state_abbr","scenario","lifetime_savings_total","lifetime_net_savings_total","lifetime_net_after_upfront_total"]))

    x = df.copy()

    # Cohort sizes
    x["new_adopters"] = pd.to_numeric(x.get("new_adopters", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    x["batt_adopters_added_this_year"] = pd.to_numeric(x.get("batt_adopters_added_this_year", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    x["pv_batt_n"] = np.minimum(x["batt_adopters_added_this_year"], x["new_adopters"])
    x["pv_only_n"] = (x["new_adopters"] - x["pv_batt_n"]).clip(lower=0.0)
    x["cohort_n"]  = x["pv_only_n"] + x["pv_batt_n"]

    # Arrays (energy value)
    x["cf_pv_only"] = x.get("cf_energy_value_pv_only", np.nan).apply(_arr25) if "cf_energy_value_pv_only" in x.columns else [[]]*len(x)
    x["cf_pv_batt"] = x.get("cf_energy_value_pv_batt", np.nan).apply(_arr25) if "cf_energy_value_pv_batt" in x.columns else [[]]*len(x)

    # Arrays (debt service)
    x["cf_debt_pv_only"] = x.get("cf_debt_payment_total_pv_only", np.nan).apply(_arr25) if "cf_debt_payment_total_pv_only" in x.columns else [[]]*len(x)
    x["cf_debt_pv_batt"] = x.get("cf_debt_payment_total_pv_batt", np.nan).apply(_arr25) if "cf_debt_payment_total_pv_batt" in x.columns else [[]]*len(x)

    # Net arrays = energy_value - debt_payment (per adopter, year-by-year)
    def _net_arr(e, d):
        e = list(e or [])
        d = list(d or [])
        if not e and not d:
            return []
        if not d:
            return [float(ee) for ee in e]
        n = min(len(e), len(d))
        return [float(e[i]) - float(d[i]) for i in range(n)]

    x["cf_net_pv_only"] = [ _net_arr(e, d) for e, d in zip(x["cf_pv_only"], x["cf_debt_pv_only"]) ]
    x["cf_net_pv_batt"] = [ _net_arr(e, d) for e, d in zip(x["cf_pv_batt"], x["cf_debt_pv_batt"]) ]

    # Horizon
    x["year"] = pd.to_numeric(x.get("year", np.nan), errors="coerce")
    years = x["year"].dropna()
    if years.empty:
        return (pd.DataFrame(columns=empty_cols_annual),
                pd.DataFrame(columns=empty_cols_cum),
                pd.DataFrame(columns=["state_abbr","scenario","lifetime_savings_total","lifetime_net_savings_total","lifetime_net_after_upfront_total"]))
    y_min, y_max = int(years.min()), int(years.max())
    L = int(getattr(cfg, "lifetime_years", 25) or 25)
    cap_to_horizon = bool(getattr(cfg, "cap_to_horizon", False))

    # ----------------------------
    # (1) Annual portfolio savings by rolling arrays forward (total & net)
    # ----------------------------
    contrib = []      # total energy-value savings
    contrib_net = []  # net after debt
    contrib_debt = [] # total debt service = ongoing FINANCED cost to consumers (Synapse)
    for r in x.itertuples(index=False):
        y0 = int(r.year) if not pd.isna(r.year) else None
        if y0 is None or (r.pv_only_n <= 0 and r.pv_batt_n <= 0):
            continue
        a_only, a_batt = list(r.cf_pv_only or []), list(r.cf_pv_batt or [])
        n_only, n_batt = list(r.cf_net_pv_only or []), list(r.cf_net_pv_batt or [])
        d_only, d_batt = list(r.cf_debt_pv_only or []), list(r.cf_debt_pv_batt or [])
        for k in range(25):
            y = y0 + k
            if cap_to_horizon and y > y_max:
                break
            if y < y_min or y > y_max:
                continue
            v_total = 0.0
            v_net   = 0.0
            v_debt  = 0.0
            if k < len(a_only) and r.pv_only_n > 0:
                v_total += a_only[k] * r.pv_only_n
            if k < len(a_batt) and r.pv_batt_n > 0:
                v_total += a_batt[k] * r.pv_batt_n
            if k < len(n_only) and r.pv_only_n > 0:
                v_net   += n_only[k] * r.pv_only_n
            if k < len(n_batt) and r.pv_batt_n > 0:
                v_net   += n_batt[k] * r.pv_batt_n
            if k < len(d_only) and r.pv_only_n > 0:
                v_debt  += d_only[k] * r.pv_only_n
            if k < len(d_batt) and r.pv_batt_n > 0:
                v_debt  += d_batt[k] * r.pv_batt_n
            if v_total != 0.0:
                contrib.append((r.state_abbr, r.scenario, y, v_total))
            if v_net != 0.0:
                contrib_net.append((r.state_abbr, r.scenario, y, v_net))
            if v_debt != 0.0:
                contrib_debt.append((r.state_abbr, r.scenario, y, v_debt))

    annual = (
        pd.DataFrame(contrib, columns=["state_abbr","scenario","year","portfolio_annual_savings"])
          .groupby(["state_abbr","scenario","year"], as_index=False)["portfolio_annual_savings"].sum()
    ) if contrib else pd.DataFrame(columns=["state_abbr","scenario","year","portfolio_annual_savings"])
    annual_net = (
        pd.DataFrame(contrib_net, columns=["state_abbr","scenario","year","portfolio_annual_net_savings"])
          .groupby(["state_abbr","scenario","year"], as_index=False)["portfolio_annual_net_savings"].sum()
    ) if contrib_net else pd.DataFrame(columns=["state_abbr","scenario","year","portfolio_annual_net_savings"])
    # Ongoing FINANCED cost to consumers = annual debt service, rolled forward by calendar year.
    annual_debt = (
        pd.DataFrame(contrib_debt, columns=["state_abbr","scenario","year","portfolio_annual_financed_cost"])
          .groupby(["state_abbr","scenario","year"], as_index=False)["portfolio_annual_financed_cost"].sum()
    ) if contrib_debt else pd.DataFrame(columns=["state_abbr","scenario","year","portfolio_annual_financed_cost"])

    if not annual.empty or not annual_net.empty:
        annual = (annual.merge(annual_net, on=["state_abbr","scenario","year"], how="outer")
                        .merge(annual_debt, on=["state_abbr","scenario","year"], how="outer")
                        .fillna(0.0))
        annual["portfolio_annual_savings"] = pd.to_numeric(annual["portfolio_annual_savings"], errors="coerce").fillna(0.0)
        annual["portfolio_annual_net_savings"] = pd.to_numeric(annual["portfolio_annual_net_savings"], errors="coerce").fillna(0.0)
        annual["portfolio_annual_financed_cost"] = pd.to_numeric(annual["portfolio_annual_financed_cost"], errors="coerce").fillna(0.0)

    # ----------------------------
    # (2) Lifetime totals per state/scenario (total, net, and net-after-upfront)
    # ----------------------------
    life_rows_total = []
    life_rows_net   = []
    # for medians
    cohort_rows_life_total = []
    cohort_rows_life_net   = []
    cohort_rows_cost       = []
    upfront_annual_rows    = []   # GROSS upfront deployment cost by install-year (Synapse: jobs)
    down_payment_annual_rows = []  # 30% cash down payment by install-year (Synapse: out-of-pocket stream)

    # Ensure numeric inputs for PV + Battery capex components
    x["system_capex_per_kw_combined"]   = pd.to_numeric(x.get("system_capex_per_kw_combined", np.nan), errors="coerce")
    x["system_kw"]                      = pd.to_numeric(x.get("system_kw", np.nan), errors="coerce")
    x["batt_capex_per_kwh_combined"]    = pd.to_numeric(x.get("batt_capex_per_kwh_combined", np.nan), errors="coerce")
    x["batt_kwh"]                       = pd.to_numeric(x.get("batt_kwh", np.nan), errors="coerce")

    CASH_FRACTION = 0.30  # subtract only the 30% cash/down-payment portion

    # Customer out-of-pocket down payment fraction (Synapse: stream 2, out-of-pocket cost).
    # Source: diffusion_shared.financing_atb_FY23.down_payment_fraction = 0.7 (debt) =>
    # SAM loan.FinancialParameters.debt_fraction = 70 => remaining 30% is cash paid at install.
    DOWN_PAYMENT_FRACTION = 0.30

    for r in x.itertuples(index=False):
        y0 = int(r.year) if not pd.isna(r.year) else None
        if y0 is None:
            continue
        credited = L if not cap_to_horizon else max(0, min(L, y_max - y0 + 1))
        if credited <= 0:
            continue

        # Per-adopter lifetime totals (dollars)
        per_only_tot = sum(r.cf_pv_only[:credited]) if (r.pv_only_n > 0 and r.cf_pv_only) else 0.0
        per_batt_tot = sum(r.cf_pv_batt[:credited]) if (r.pv_batt_n > 0 and r.cf_pv_batt) else 0.0
        per_only_net = sum(r.cf_net_pv_only[:credited]) if (r.pv_only_n > 0 and r.cf_net_pv_only) else 0.0
        per_batt_net = sum(r.cf_net_pv_batt[:credited]) if (r.pv_batt_n > 0 and r.cf_net_pv_batt) else 0.0

        # Aggregate lifetime across cohorts for totals (sum over adopters)
        cohort_total_tot = per_only_tot * r.pv_only_n + per_batt_tot * r.pv_batt_n
        cohort_total_net = per_only_net * r.pv_only_n + per_batt_net * r.pv_batt_n
        if cohort_total_tot != 0.0:
            life_rows_total.append((r.state_abbr, r.scenario, cohort_total_tot))
        if cohort_total_net != 0.0:
            life_rows_net.append((r.state_abbr, r.scenario, cohort_total_net))

        # ----- Upfront per adopter (PV + Battery), subtract only 30% cash portion -----
        pv_upfront = 0.0
        if np.isfinite(r.system_capex_per_kw_combined) and np.isfinite(r.system_kw):
            pv_upfront = float(r.system_capex_per_kw_combined) * float(r.system_kw)

        batt_upfront = 0.0
        if np.isfinite(getattr(r, "batt_capex_per_kwh_combined", np.nan)) and np.isfinite(getattr(r, "batt_kwh", np.nan)):
            batt_upfront = float(r.batt_capex_per_kwh_combined) * float(r.batt_kwh)

        per_adopter_upfront_total = pv_upfront + batt_upfront
        per_adopter_upfront_cash  = CASH_FRACTION * per_adopter_upfront_total  # loans include principal; only cash gets subtracted here

        if np.isfinite(per_adopter_upfront_cash) and (r.cohort_n > 0) and (per_adopter_upfront_cash > 0.0):
            cohort_rows_cost.append((r.state_abbr, r.scenario, y0, float(r.cohort_n), per_adopter_upfront_cash))

        # ----- GROSS upfront deployment cost this install-year (Synapse jobs input) -----
        # Full installed capex (NOT net of ITC / cash): all adopters get PV; only battery
        # adopters add battery. One-time in the install year (not rolled forward).
        up_pv   = pv_upfront   * float(r.cohort_n)   if r.cohort_n   > 0 else 0.0
        up_batt = batt_upfront * float(r.pv_batt_n)  if r.pv_batt_n  > 0 else 0.0
        if (up_pv + up_batt) > 0.0:
            upfront_annual_rows.append((r.state_abbr, r.scenario, y0, up_pv, up_batt))
            # ----- Customer out-of-pocket down payment (Synapse stream 2), booked in install year -----
            down_payment_annual_rows.append(
                (r.state_abbr, r.scenario, y0, DOWN_PAYMENT_FRACTION * (up_pv + up_batt))
            )

        # For medians (unchanged)
        if r.pv_only_n > 0:
            cohort_rows_life_total.append((r.state_abbr, r.scenario, y0, float(r.pv_only_n), float(per_only_tot)))
            cohort_rows_life_net.append((r.state_abbr, r.scenario, y0, float(r.pv_only_n), float(per_only_net)))
        if r.pv_batt_n > 0:
            cohort_rows_life_total.append((r.state_abbr, r.scenario, y0, float(r.pv_batt_n), float(per_batt_tot)))
            cohort_rows_life_net.append((r.state_abbr, r.scenario, y0, float(r.pv_batt_n), float(per_batt_net)))

    lifetime_total = (
        pd.DataFrame(life_rows_total, columns=["state_abbr","scenario","lifetime_savings_for_cohort"])
        .groupby(["state_abbr","scenario"], as_index=False)["lifetime_savings_for_cohort"].sum()
        .rename(columns={"lifetime_savings_for_cohort": "lifetime_savings_total"})
    ) if life_rows_total else pd.DataFrame(columns=["state_abbr","scenario","lifetime_savings_total"])

    lifetime_net = (
        pd.DataFrame(life_rows_net, columns=["state_abbr","scenario","lifetime_net_savings_for_cohort"])
        .groupby(["state_abbr","scenario"], as_index=False)["lifetime_net_savings_for_cohort"].sum()
        .rename(columns={"lifetime_net_savings_for_cohort": "lifetime_net_savings_total"})
    ) if life_rows_net else pd.DataFrame(columns=["state_abbr","scenario","lifetime_net_savings_total"])

    # Upfront totals (to compute net-after-upfront at the portfolio level)
    if cohort_rows_cost:
        upfront_df = (pd.DataFrame(cohort_rows_cost, columns=[
                        "state_abbr","scenario","cohort_year","cohort_n","per_adopter_upfront_$"
                    ])
                    .assign(upfront_total=lambda d: d["cohort_n"] * d["per_adopter_upfront_$"])
                    .groupby(["state_abbr","scenario"], as_index=False)["upfront_total"].sum())
    else:
        upfront_df = pd.DataFrame(columns=["state_abbr","scenario","upfront_total"])

    lifetime = lifetime_total.merge(lifetime_net, on=["state_abbr","scenario"], how="outer").merge(
        upfront_df, on=["state_abbr","scenario"], how="left"
    ).fillna(0.0)

    # NEW: lifetime net after upfront (portfolio level; loans already in debt arrays)
    lifetime["lifetime_net_after_upfront_total"] = (
        pd.to_numeric(lifetime.get("lifetime_net_savings_total", 0.0), errors="coerce").fillna(0.0)
        - pd.to_numeric(lifetime.get("upfront_total", 0.0), errors="coerce").fillna(0.0)
    )

    # ----------------------------
    # (3) Per-year medians (total & net) and upfront medians (unchanged)
    # ----------------------------
    # Lifetime medians - TOTAL
    if cohort_rows_life_total:
        c_life_tot = (pd.DataFrame(cohort_rows_life_total, columns=[
                    "state_abbr","scenario","cohort_year","cohort_n","per_adopter_lifetime_$"
                 ])
                 .groupby(["state_abbr","scenario","cohort_year","per_adopter_lifetime_$"], as_index=False)["cohort_n"]
                 .sum()
                 .rename(columns={"cohort_year":"year"}))
        med_life_to_date = []
        for (s, scn), g in c_life_tot.groupby(["state_abbr","scenario"], sort=False):
            g = g.sort_values(["year"])
            for y in sorted(g["year"].unique()):
                sub = g[g["year"] <= y]
                med = _weighted_median(sub["per_adopter_lifetime_$"].to_numpy(),
                                       sub["cohort_n"].to_numpy())
                med_life_to_date.append((s, scn, int(y), float(med)))
        med_life_to_date = pd.DataFrame(med_life_to_date, columns=[
            "state_abbr","scenario","year","median_lifetime_savings_per_adopter_to_date"
        ])
        med_life_this_rows = (c_life_tot
            .groupby(["state_abbr","scenario","year"], as_index=False)
            .apply(lambda g: pd.Series({
                "median_lifetime_savings_per_adopter_this_year":
                    _weighted_median(g["per_adopter_lifetime_$"].to_numpy(), g["cohort_n"].to_numpy())
            }))
            .reset_index(drop=True))
    else:
        med_life_to_date = pd.DataFrame(columns=[
            "state_abbr","scenario","year","median_lifetime_savings_per_adopter_to_date"
        ])
        med_life_this_rows = pd.DataFrame(columns=[
            "state_abbr","scenario","year","median_lifetime_savings_per_adopter_this_year"
        ])

    # Lifetime medians - NET
    if cohort_rows_life_net:
        c_life_net = (pd.DataFrame(cohort_rows_life_net, columns=[
                    "state_abbr","scenario","cohort_year","cohort_n","per_adopter_lifetime_net_$"
                 ])
                 .groupby(["state_abbr","scenario","cohort_year","per_adopter_lifetime_net_$"], as_index=False)["cohort_n"]
                 .sum()
                 .rename(columns={"cohort_year":"year"}))
        med_life_net_to_date = []
        for (s, scn), g in c_life_net.groupby(["state_abbr","scenario"], sort=False):
            g = g.sort_values(["year"])
            for y in sorted(g["year"].unique()):
                sub = g[g["year"] <= y]
                med = _weighted_median(sub["per_adopter_lifetime_net_$"].to_numpy(),
                                       sub["cohort_n"].to_numpy())
                med_life_net_to_date.append((s, scn, int(y), float(med)))
        med_life_net_to_date = pd.DataFrame(med_life_net_to_date, columns=[
            "state_abbr","scenario","year","median_lifetime_net_savings_per_adopter_to_date"
        ])
        med_life_net_this_rows = (c_life_net
            .groupby(["state_abbr","scenario","year"], as_index=False)
            .apply(lambda g: pd.Series({
                "median_lifetime_net_savings_per_adopter_this_year":
                    _weighted_median(g["per_adopter_lifetime_net_$"].to_numpy(), g["cohort_n"].to_numpy())
            }))
            .reset_index(drop=True))
    else:
        med_life_net_to_date = pd.DataFrame(columns=[
            "state_abbr","scenario","year","median_lifetime_net_savings_per_adopter_to_date"
        ])
        med_life_net_this_rows = pd.DataFrame(columns=[
            "state_abbr","scenario","year","median_lifetime_net_savings_per_adopter_this_year"
        ])

    # Upfront cost medians (per-adopter)
    if cohort_rows_cost:
        c_cost = (pd.DataFrame(cohort_rows_cost, columns=[
                    "state_abbr","scenario","cohort_year","cohort_n","per_adopter_upfront_$"
                 ])
                 .groupby(["state_abbr","scenario","cohort_year","per_adopter_upfront_$"], as_index=False)["cohort_n"]
                 .sum()
                 .rename(columns={"cohort_year":"year"}))
        med_cost_rows = []
        for (s, scn), g in c_cost.groupby(["state_abbr","scenario"], sort=False):
            g = g.sort_values(["year"])
            for y in sorted(g["year"].unique()):
                sub = g[g["year"] <= y]
                med = _weighted_median(sub["per_adopter_upfront_$"].to_numpy(),
                                       sub["cohort_n"].to_numpy())
                med_cost_rows.append((s, scn, int(y), float(med)))
        med_cost_to_date = pd.DataFrame(med_cost_rows, columns=[
            "state_abbr","scenario","year","median_upfront_cost_per_adopter_to_date"
        ])
        med_cost_this_rows = (c_cost
            .groupby(["state_abbr","scenario","year"], as_index=False)
            .apply(lambda g: pd.Series({
                "median_upfront_cost_per_adopter_this_year":
                    _weighted_median(g["per_adopter_upfront_$"].to_numpy(), g["cohort_n"].to_numpy())
            }))
            .reset_index(drop=True))
    else:
        med_cost_to_date = pd.DataFrame(columns=[
            "state_abbr","scenario","year","median_upfront_cost_per_adopter_to_date"
        ])
        med_cost_this_rows = pd.DataFrame(columns=[
            "state_abbr","scenario","year","median_upfront_cost_per_adopter_this_year"
        ])

    # ----------------------------
    # (4) Cumulative portfolio savings (cumsum) — total & net
    # ----------------------------
    if not annual.empty:
        annual = annual.sort_values(["state_abbr","scenario","year"])
        cumulative = annual[["state_abbr","scenario","year"]].copy()
        cumulative["cumulative_bill_savings"] = (
            annual.groupby(["state_abbr","scenario"], observed=True)["portfolio_annual_savings"].cumsum()
        )
        cumulative["cumulative_net_savings"] = (
            annual.groupby(["state_abbr","scenario"], observed=True)["portfolio_annual_net_savings"].cumsum()
        )
    else:
        cumulative = pd.DataFrame(columns=["state_abbr","scenario","year","cumulative_bill_savings","cumulative_net_savings"])

    # ----------------------------
    # (5) Attach lifetime totals + median columns (to-date and this-year)
    # ----------------------------
    if not annual.empty:
        annual = annual.merge(lifetime[["state_abbr","scenario","lifetime_savings_total","lifetime_net_savings_total","lifetime_net_after_upfront_total"]],
                              on=["state_abbr","scenario"], how="left")
    if not cumulative.empty:
        cumulative = cumulative.merge(lifetime[["state_abbr","scenario","lifetime_savings_total","lifetime_net_savings_total","lifetime_net_after_upfront_total"]],
                                      on=["state_abbr","scenario"], how="left")

    # To-date medians (total & net) + upfront medians
    # Use outer join so cohort years with $0 portfolio savings (e.g. installation year 2026) still appear
    for df_med in (med_life_to_date, med_life_net_to_date, med_cost_to_date):
        if not df_med.empty:
            annual    = annual.merge(df_med, on=["state_abbr","scenario","year"], how="outer") if not annual.empty else annual
            cumulative= cumulative.merge(df_med, on=["state_abbr","scenario","year"], how="left") if not cumulative.empty else cumulative

    # This-year medians
    for df_med_this in (med_life_this_rows, med_life_net_this_rows, med_cost_this_rows):
        if not df_med_this.empty:
            annual    = annual.merge(df_med_this, on=["state_abbr","scenario","year"], how="left") if not annual.empty else annual
            cumulative= cumulative.merge(df_med_this, on=["state_abbr","scenario","year"], how="left") if not cumulative.empty else cumulative

    # Annual GROSS upfront deployment cost (Synapse jobs input): total + PV/battery split, by install-year.
    if upfront_annual_rows and not annual.empty:
        upfront_annual = (
            pd.DataFrame(upfront_annual_rows, columns=["state_abbr","scenario","year",
                          "portfolio_annual_upfront_cost_pv","portfolio_annual_upfront_cost_batt"])
              .groupby(["state_abbr","scenario","year"], as_index=False).sum()
        )
        upfront_annual["portfolio_annual_upfront_cost"] = (
            upfront_annual["portfolio_annual_upfront_cost_pv"] + upfront_annual["portfolio_annual_upfront_cost_batt"]
        )
        annual = annual.merge(upfront_annual, on=["state_abbr","scenario","year"], how="left")

    # Annual customer out-of-pocket down payment (Synapse stream 2), by install-year.
    if down_payment_annual_rows and not annual.empty:
        down_payment_annual = (
            pd.DataFrame(down_payment_annual_rows, columns=["state_abbr","scenario","year",
                          "portfolio_annual_down_payment_cost"])
              .groupby(["state_abbr","scenario","year"], as_index=False)["portfolio_annual_down_payment_cost"].sum()
        )
        annual = annual.merge(down_payment_annual, on=["state_abbr","scenario","year"], how="left")

    # Ensure numeric and no NaNs
    num_cols = [
        "portfolio_annual_savings","portfolio_annual_net_savings",
        "portfolio_annual_financed_cost",
        "portfolio_annual_upfront_cost","portfolio_annual_upfront_cost_pv","portfolio_annual_upfront_cost_batt",
        "portfolio_annual_down_payment_cost",
        "cumulative_bill_savings","cumulative_net_savings",
        "lifetime_savings_total","lifetime_net_savings_total","lifetime_net_after_upfront_total",
        "median_lifetime_savings_per_adopter_to_date",
        "median_lifetime_net_savings_per_adopter_to_date",
        "median_upfront_cost_per_adopter_to_date",
        "median_lifetime_savings_per_adopter_this_year",
        "median_lifetime_net_savings_per_adopter_this_year",
        "median_upfront_cost_per_adopter_this_year",
    ]
    for col in num_cols:
        if col in annual.columns:
            annual[col] = pd.to_numeric(annual[col], errors="coerce").fillna(0.0)
        if col in cumulative.columns:
            cumulative[col] = pd.to_numeric(cumulative[col], errors="coerce").fillna(0.0)

    # Headline Synapse column: customer out-of-pocket cost = down payment (install year) +
    # financed loan payments (every year the loan is outstanding). Computed AFTER both
    # components are aggregated by (state_abbr, scenario, year) and NaN-coerced above, so a
    # year with both a new cohort's down payment and older cohorts' debt service sums both.
    if not annual.empty:
        down_payment_col = annual["portfolio_annual_down_payment_cost"] if "portfolio_annual_down_payment_cost" in annual.columns else 0.0
        financed_col = annual["portfolio_annual_financed_cost"] if "portfolio_annual_financed_cost" in annual.columns else 0.0
        if "portfolio_annual_down_payment_cost" not in annual.columns:
            annual["portfolio_annual_down_payment_cost"] = 0.0
        annual["portfolio_annual_out_of_pocket_cost"] = down_payment_col + financed_col

    for col in ["lifetime_savings_total","lifetime_net_savings_total","lifetime_net_after_upfront_total"]:
        if col in lifetime.columns:
            lifetime[col] = pd.to_numeric(lifetime[col], errors="coerce").fillna(0.0)

    return annual, cumulative, lifetime


def aggregate_state_metrics(agents: pd.DataFrame, cfg: SavingsConfig) -> Dict[str, pd.DataFrame]:
    """
    Aggregate per-state metrics for plotting and exports.

    Parameters
    ----------
    agents : pandas.DataFrame
        Agent-level state outputs (concatenated baseline & policy) as returned by
        :attr:`DataWarehouse.agents`.
    cfg : SavingsConfig
        Savings horizon/credit configuration.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys:
          - ``median_system_kw``      — state×year×scenario median PV size (kW) (adopters-only, weighted)
          - ``median_storage_kwh``    — state×year×scenario median storage size (kWh) (adopters-only, weighted)
          - ``totals``                — state×year×scenario totals: new adopters, cumulative kW/kWh, etc.
          - ``tech_2040``             — % of technical potential (2040 only)
          - ``portfolio_annual_savings``
          - ``cumulative_bill_savings``
          - ``lifetime_totals``
          - ``avg_price_2026_model``  — adopter-weighted average price in 2026 baseline
          - ``market_share_reached``
    """
    x = agents.copy()
    x["state_abbr"] = x["state_abbr"].astype(str).str.upper()
    x["scenario"]   = x["scenario"].astype(str).str.lower()

    for c in ("year","new_adopters","number_of_adopters","customers_in_bin","max_market_share",
              "system_kw","new_system_kw","system_kw_cum","batt_kwh","batt_kwh_cum",
              "price_per_kwh","batt_adopters_added_this_year","load_kwh_per_customer_in_bin_initial"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)

    # Weighted medians (adopters-only)
    def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
        v = pd.to_numeric(values, errors="coerce")
        w = pd.to_numeric(weights, errors="coerce").clip(lower=0)
        m = v.notna() & (w > 0)
        if not m.any():
            return float("nan")
        v, w = v[m].to_numpy(), w[m].to_numpy()
        order = np.argsort(v)
        v, w = v[order], w[order]
        cw = np.cumsum(w)
        return float(v[np.searchsorted(cw, 0.5 * w.sum(), side="left")])

    adopters = x[x["new_adopters"] > 0].copy()

    _grp = adopters.groupby(["state_abbr","year","scenario"], observed=True)
    median_kw = (
        (_grp["system_kw_cum"].sum() / _grp["number_of_adopters"].sum())
        .rename("median_system_kw").reset_index()
    )

    if "batt_kwh" in adopters.columns:
        has_storage = adopters[adopters["batt_kwh"] > 0]
        median_storage = (
            has_storage.groupby(["state_abbr","year","scenario"], observed=True)
                       .apply(lambda g: pd.Series({"median_batt_kwh": _weighted_median(g["batt_kwh"], g["new_adopters"])}), include_groups=False)
                       .reset_index()
        )
    else:
        median_storage = pd.DataFrame(columns=["state_abbr","year","scenario","median_batt_kwh"])

    # New battery storage added this year (energy, kWh) = battery adopters x per-system battery size.
    # Mirrors solar's new_system_kw. Defensive .get so a missing battery column can't hard-fail.
    x["new_batt_kwh"] = (
        pd.to_numeric(x.get("batt_adopters_added_this_year", 0.0), errors="coerce").fillna(0.0)
        * pd.to_numeric(x.get("batt_kwh", 0.0), errors="coerce").fillna(0.0)
    )

    # Totals
    totals = (
        x.groupby(["state_abbr","year","scenario"], observed=True)
         .agg(new_adopters=("new_adopters","sum"),
              new_system_kw=("new_system_kw","sum"),
              new_batt_adopters=("batt_adopters_added_this_year","sum"),
              new_batt_kwh=("new_batt_kwh","sum"),
              number_of_adopters=("number_of_adopters","sum"),
              system_kw_cum=("system_kw_cum","sum"),
              batt_kwh_cum=("batt_kwh_cum","sum"))
         .reset_index()
    )

    # Tech potential in 2040
    tech_2040 = x.loc[x["year"] == 2040, ["state_abbr","scenario","number_of_adopters","customers_in_bin"]].groupby(
                    ["state_abbr","scenario"], observed=True, as_index=False).sum()
    tech_2040["percent_tech_potential"] = np.where(
        tech_2040["customers_in_bin"] > 0,
        100.0 * tech_2040["number_of_adopters"] / tech_2040["customers_in_bin"],
        np.nan,
    )

    # Savings
    portfolio_annual, cumulative_savings, lifetime_totals = compute_portfolio_and_cumulative_savings(x, cfg)

    # 2026 host price (weighted by customers_in_bin, baseline only)
    price_2026 = x[(x["year"] == 2026) & (x["scenario"] == "baseline")][["state_abbr","customers_in_bin","price_per_kwh"]]
    def _wavg(g: pd.DataFrame) -> float:
        return float(np.average(g["price_per_kwh"], weights=g["customers_in_bin"])) if g["customers_in_bin"].sum() > 0 else np.nan
    avg_price_2026_model = price_2026.groupby("state_abbr", as_index=False).apply(_wavg).rename(columns={None:"price_per_kwh"})

    # Market share
    x["market_potential"] = x["customers_in_bin"] * x["max_market_share"]
    market_share = (
        x.groupby(["state_abbr","year","scenario"], observed=True)
         .agg(market_potential=("market_potential","sum"),
              market_reached=("number_of_adopters","sum"))
         .reset_index()
    )
    market_share["market_share_reached"] = np.where(
        market_share["market_potential"] > 0,
        market_share["market_reached"] / market_share["market_potential"],
        np.nan,
    )

    eabs_states = build_eabs_calendar_timeseries(agents=agents)

    # Median year-1 bill savings for the 2026 adoption cohort
    cohort_2026 = x[x["year"] == 2026].copy()
    cohort_2026["yr1_bill_savings"] = cohort_2026.apply(
        lambda r: (
            (_arr25(r.get("utility_bill_wo_sys_pv_only", []))[1] -
             _arr25(r.get("utility_bill_w_sys_pv_only",  []))[1])
            if r.get("utility_bill_wo_sys_pv_only") is not None else np.nan
        ), axis=1
    )
    med_yr1 = (
        cohort_2026.groupby(["state_abbr", "scenario"], observed=True)
        .apply(lambda g: pd.Series({
            "median_yr1_bill_savings_2026_cohort": _weighted_median(
                g["yr1_bill_savings"], g["new_adopters"]
            )
        }), include_groups=False)
        .reset_index()
    )

    return {
        "median_system_kw": median_kw,
        "median_storage_kwh": median_storage,
        "totals": totals,
        "tech_2040": tech_2040,
        "portfolio_annual_savings": portfolio_annual,
        "cumulative_bill_savings": cumulative_savings,
        "lifetime_totals": lifetime_totals,
        "avg_price_2026_model": avg_price_2026_model,
        "market_share_reached": market_share,
        "eabs_states": eabs_states,
        "median_yr1_bill_savings_2026_cohort": med_yr1,
    }

def build_us_cumulative_timeseries(
    annual_df: "pd.DataFrame",
    lifetime_df: "pd.DataFrame | None" = None,
    *,
    xticks_full: bool = False
) -> "pd.DataFrame":
    """
    National (U.S.) cumulative savings time series by scenario.

    Inputs
    ------
    annual_df : DataFrame
        Output[0] from `compute_portfolio_and_cumulative_savings` (the 'annual' frame),
        containing columns:
          ['state_abbr','scenario','year',
           'portfolio_annual_savings','portfolio_annual_net_savings']
        (It can also already include the lifetime_* columns merged in; that's fine.)
    lifetime_df : DataFrame, optional
        Output[2] from `compute_portfolio_and_cumulative_savings` (the 'lifetime' frame),
        with columns:
          ['state_abbr','scenario',
           'lifetime_savings_total','lifetime_net_savings_total','lifetime_net_after_upfront_total']

    Returns
    -------
    DataFrame
        ['year','scenario',
         'cumulative_bill_savings_us','cumulative_net_savings_us',
         'lifetime_net_after_upfront_total_us' (if lifetime_df provided)]
    """
    import numpy as np
    import pandas as pd

    if annual_df is None or annual_df.empty:
        cols = ["year","scenario","cumulative_bill_savings_us","cumulative_net_savings_us"]
        return pd.DataFrame(columns=cols)

    d = annual_df.copy()

    # Keep only what we need, ensure numeric
    keep_cols = ["state_abbr","scenario","year",
                 "portfolio_annual_savings","portfolio_annual_net_savings"]
    for c in keep_cols:
        if c not in d.columns:
            # tolerate missing net column by creating 0s (won't break gross)
            if c == "portfolio_annual_net_savings":
                d[c] = 0.0
            else:
                raise ValueError(f"build_us_cumulative_timeseries: missing required column '{c}'")

    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d[d["year"].notna()].copy()
    if d.empty:
        cols = ["year","scenario","cumulative_bill_savings_us","cumulative_net_savings_us"]
        return pd.DataFrame(columns=cols)

    for c in ("portfolio_annual_savings","portfolio_annual_net_savings"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    # Sum across states to national annual series
    nat_annual = (d.groupby(["scenario","year"], observed=True, as_index=False)
                    .agg(portfolio_annual_savings=("portfolio_annual_savings","sum"),
                         portfolio_annual_net_savings=("portfolio_annual_net_savings","sum"))
                  )

    # Build full year index per scenario (optional; ensures clean cumsums if gaps)
    if xticks_full:
        out_rows = []
        for scen, g in nat_annual.groupby("scenario", observed=True):
            y_min, y_max = int(g["year"].min()), int(g["year"].max())
            full = (pd.DataFrame({"year": np.arange(y_min, y_max+1, dtype=int)})
                      .merge(g, on="year", how="left")
                      .fillna({"portfolio_annual_savings":0.0, "portfolio_annual_net_savings":0.0}))
            full["scenario"] = scen
            out_rows.append(full)
        nat_annual = pd.concat(out_rows, ignore_index=True)
    else:
        nat_annual = nat_annual.sort_values(["scenario","year"])

    # Cumulative by scenario
    nat = nat_annual.copy()
    nat["cumulative_bill_savings_us"] = (nat.groupby("scenario", observed=True)["portfolio_annual_savings"]
                                           .cumsum())
    nat["cumulative_net_savings_us"] = (nat.groupby("scenario", observed=True)["portfolio_annual_net_savings"]
                                          .cumsum())

    # Keep tidy columns
    nat = nat[["year","scenario","cumulative_bill_savings_us","cumulative_net_savings_us"]]

    # Optionally attach national lifetime net-after-upfront totals (per scenario)
    if lifetime_df is not None and not lifetime_df.empty:
        L = lifetime_df.copy()
        need = ["state_abbr","scenario","lifetime_net_after_upfront_total"]
        for c in need:
            if c not in L.columns:
                # tolerate missing by skipping attachment
                L = None
                break
        if L is not None:
            L["lifetime_net_after_upfront_total"] = pd.to_numeric(
                L["lifetime_net_after_upfront_total"], errors="coerce"
            ).fillna(0.0)
            L_nat = (L.groupby("scenario", observed=True, as_index=False)
                       ["lifetime_net_after_upfront_total"].sum()
                       .rename(columns={"lifetime_net_after_upfront_total":
                                        "lifetime_net_after_upfront_total_us"}))
            nat = nat.merge(L_nat, on="scenario", how="left")

    return nat.sort_values(["scenario","year"]).reset_index(drop=True)

def summarize_us_cumulative_for_year(nat_ts: "pd.DataFrame", year: int) -> "pd.DataFrame":
    """
    Slice the national cumulative series to a single year.
    Returns ['scenario','year','cumulative_bill_savings_us','cumulative_net_savings_us',
             'lifetime_net_after_upfront_total_us' (if present)].
    """
    keep = ["scenario","year","cumulative_bill_savings_us","cumulative_net_savings_us"]
    if "lifetime_net_after_upfront_total_us" in nat_ts.columns:
        keep.append("lifetime_net_after_upfront_total_us")
    out = nat_ts[nat_ts["year"] == int(year)][keep].sort_values("scenario").reset_index(drop=True)
    return out



# -----------------------------------------------------------------------------
# Peaks & coincident reductions (hourly)
# -----------------------------------------------------------------------------

def compute_peaks_by_state(state_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute state×year×scenario peak (MW) from hourly arrays.

    Parameters
    ----------
    state_hourly : pandas.DataFrame
        Columns: ['state_abbr','scenario','year','net_load'] (list[float] per row).

    Returns
    -------
    pandas.DataFrame
        ['state_abbr','scenario','year','peak_mw']
    """
    rows: List[Tuple[str,str,int,float]] = []
    for r in state_hourly.itertuples(index=False):
        peak = float(np.max(r.net_load)) if r.net_load else np.nan
        rows.append((r.state_abbr, r.scenario, int(r.year), peak))
    return pd.DataFrame(rows, columns=["state_abbr","scenario","year","peak_mw"])


def _coincident_scalar(
    baseline: Sequence[float],
    policy: Sequence[float],
    method: str = "baseline_topN_avg",
    top_n: int = 10,
    percentile: Optional[float] = None,
) -> float:
    """
    Scalar coincident reduction from two hourly arrays.

    Parameters
    ----------
    baseline, policy : sequence of float
        Hourly net load for the same geography and year.
    method : {"baseline_topN_avg","single_hour","separate_topN_means","percentile_mean"}, default "baseline_topN_avg"
        Strategy for defining the peak slice.
    top_n : int, default 10
        Used for "baseline_topN_avg" and "separate_topN_means".
    percentile : float, optional
        Percentile for "percentile_mean" (e.g., 0.999 = top 0.1%).

    Returns
    -------
    float
        Coincident reduction in MW.
    """
    b = np.asarray(baseline, dtype=float)
    p = np.asarray(policy, dtype=float)
    if method == "single_hour":
        i = int(np.argmax(b))
        return float(b[i] - p[i])
    if method == "baseline_topN_avg":
        idx = np.argpartition(b, -top_n)[-top_n:]
        b_mean = float(b[idx].mean())
        p_mean = float(p[idx].mean())
        return b_mean - p_mean
    if method == "separate_topN_means":
        bi = np.argpartition(b, -top_n)[-top_n:]
        pi = np.argpartition(p, -top_n)[-top_n:]
        return float(b[bi].mean() - p[pi].mean())
    if method == "percentile_mean":
        assert percentile is not None and 0.0 < percentile < 1.0, "percentile must be (0,1)"
        thresh = np.quantile(b, percentile)
        mask = b >= thresh
        return float(b[mask].mean() - p[mask].mean())
    raise ValueError(f"Unknown method: {method}")


def compute_state_coincident_reduction(
    wh: DataWarehouse,
    method: str = "baseline_topN_avg",
    top_n: int = 10,
    percentile: Optional[float] = None,
) -> pd.DataFrame:
    """
    Per‑state coincident peak reduction for each year.

    Parameters
    ----------
    wh : DataWarehouse
        Loaded warehouse with ``state_hourly``.
    method : str, default "baseline_topN_avg"
        See :func:`_coincident_scalar` for options.
    top_n : int, default 10
        See :func:`_coincident_scalar`.
    percentile : float, optional
        See :func:`_coincident_scalar`.

    Returns
    -------
    pandas.DataFrame
        ['state_abbr','year','coincident_reduction_mw']
    """
    rows: List[Tuple[str,int,float]] = []
    g = wh.state_hourly.groupby(["state_abbr","year"], observed=True)
    for (state, year), sub in g:
        base = sub[sub["scenario"] == "baseline"]
        pol  = sub[sub["scenario"] == "policy"]
        if base.empty or pol.empty:
            continue
        b = base.iloc[0]["net_load"]
        p = pol.iloc[0]["net_load"]
        rows.append((state, int(year), _coincident_scalar(b, p, method=method, top_n=top_n, percentile=percentile)))
    return pd.DataFrame(rows, columns=["state_abbr","year","coincident_reduction_mw"]).sort_values(["state_abbr","year"]).reset_index(drop=True)


def compute_rto_coincident_reduction(
    wh: DataWarehouse,
    method: str = "baseline_topN_avg",
    top_n: int = 10,
    percentile: Optional[float] = None,
    return_by_rto: bool = False,
) -> pd.DataFrame:
    """
    RTO coincident peak reduction aggregated to national totals by year.

    Steps per RTO×year:
      1) Sum hourly net load across all contributing states for baseline and policy.
      2) Compute coincident reduction with the chosen method.
      3) (Optional) return each RTO series; otherwise sum across RTOs to national.

    Parameters
    ----------
    wh : DataWarehouse
        Loaded warehouse with ``rto_hourly``.
    method : str, default "baseline_topN_avg"
        See :func:`_coincident_scalar` for options.
    top_n : int, default 10
        See :func:`_coincident_scalar`.
    percentile : float, optional
        See :func:`_coincident_scalar`.
    return_by_rto : bool, default False
        If True, return a per‑RTO series; otherwise, return national totals.

    Returns
    -------
    pandas.DataFrame
        If ``return_by_rto=False``: ['year','coincident_reduction_mw']
        If ``return_by_rto=True`` : ['rto','year','coincident_reduction_mw']
    """
    # 1) Collapse all state rows → one array per (rto, scenario, year)
    def _sum_arrays(lst: List[List[float]]) -> List[float]:
        L = max(len(a) for a in lst)
        out = np.zeros(L, dtype=float)
        for a in lst:
            a = np.asarray(a, dtype=float)
            out[: len(a)] += a[:L]
        return out.tolist()

    sums = (
        wh.rto_hourly.groupby(["rto","scenario","year"], observed=True)["net_load"]
                     .apply(lambda s: _sum_arrays(list(s)))
                     .reset_index()
    )

    # 2) Compute coincident reduction per RTO×year
    rows_rto: List[Tuple[str,int,float]] = []
    for (rto, year), sub in sums.groupby(["rto","year"], observed=True):
        base = sub[sub["scenario"] == "baseline"]
        pol  = sub[sub["scenario"] == "policy"]
        if base.empty or pol.empty:
            continue
        b = base.iloc[0]["net_load"]
        p = pol.iloc[0]["net_load"]
        val = _coincident_scalar(b, p, method=method, top_n=top_n, percentile=percentile)
        rows_rto.append((str(rto), int(year), float(val)))

    df_rto = pd.DataFrame(rows_rto, columns=["rto","year","coincident_reduction_mw"]).sort_values(["rto","year"])

    if return_by_rto:
        return df_rto.reset_index(drop=True)

    # 3) National totals (sum across RTOs)
    nat = df_rto.groupby("year", as_index=False)["coincident_reduction_mw"].sum()
    return nat.sort_values("year").reset_index(drop=True)


# -----------------------------------------------------------------------------
# National totals / deltas and faceted plots
# -----------------------------------------------------------------------------

def build_national_totals(
    outputs: Dict[str, pd.DataFrame],
    peaks_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build U.S. totals per metric for plotting lines by scenario.

    Returns
    -------
    pandas.DataFrame
        ['year','scenario','metric','value']
    """
    import pandas as pd
    from typing import List

    def _sum_metric(df: pd.DataFrame, col: str, cumulative: bool) -> pd.DataFrame:
        # Guard: if required columns missing, return empty frame
        need = {"state_abbr", "year", "scenario", col}
        if not need.issubset(df.columns):
            return pd.DataFrame(columns=["year", "scenario", col])

        # Ensure numeric year for consistent reindex
        df = df.copy()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["year"])
        if df.empty:
            return pd.DataFrame(columns=["year", "scenario", col])

        years = pd.Index(range(int(df["year"].min()), int(df["year"].max()) + 1), name="year")

        filled_pieces = []
        for (state, scen), g in df.groupby(["state_abbr", "scenario"], observed=True):
            s = g.set_index("year")[[col]].reindex(years)
            s = (s.ffill() if cumulative else s.fillna(0.0)).fillna(0.0).reset_index()
            s["state_abbr"] = state
            s["scenario"]   = scen
            filled_pieces.append(s)
        filled = pd.concat(filled_pieces, ignore_index=True) if filled_pieces else pd.DataFrame(columns=["year", col, "state_abbr", "scenario"])
        nat = (filled.groupby(["year", "scenario"], observed=True, as_index=False)[col]
               .sum()
               .rename(columns={col: "value"}))
        return nat

    pieces: List[pd.DataFrame] = []
    totals = outputs["totals"]

    # Core cumulative totals
    for col in ("number_of_adopters", "system_kw_cum", "batt_kwh_cum"):
        s = _sum_metric(totals[["state_abbr","year","scenario",col]].copy(), col, cumulative=True)
        if not s.empty:
            s["metric"] = col
            pieces.append(s)

    # Annual new battery adopters / capacity — NEW (optional)
    for col in ("new_batt_adopters", "new_batt_kwh"):
        if col in totals.columns:
            s = _sum_metric(totals[["state_abbr","year","scenario",col]].copy(), col, cumulative=False)
            if not s.empty:
                s["metric"] = col
                pieces.append(s)

    # Annual portfolio bill savings (total) — existing
    if "portfolio_annual_savings" in outputs.get("portfolio_annual_savings", pd.DataFrame()).columns or \
       "portfolio_annual_savings" in (outputs.get("portfolio_annual_savings", pd.DataFrame()).columns if "portfolio_annual_savings" in outputs else []):
        pas = outputs["portfolio_annual_savings"]
        if not pas.empty and "portfolio_annual_savings" in pas.columns:
            s = _sum_metric(pas[["state_abbr","year","scenario","portfolio_annual_savings"]].copy(),
                            "portfolio_annual_savings", cumulative=False)
            if not s.empty:
                s["metric"] = "portfolio_annual_savings"
                pieces.append(s)

    # Annual portfolio net savings — NEW (optional)
    if "portfolio_annual_net_savings" in outputs.get("portfolio_annual_savings", pd.DataFrame()).columns:
        pan = outputs["portfolio_annual_savings"]
        s = _sum_metric(pan[["state_abbr","year","scenario","portfolio_annual_net_savings"]].copy(),
                        "portfolio_annual_net_savings", cumulative=False)
        if not s.empty:
            s["metric"] = "portfolio_annual_net_savings"
            pieces.append(s)

    # Annual portfolio upfront/financed costs — NEW (optional)
    for col in ("portfolio_annual_upfront_cost", "portfolio_annual_upfront_cost_pv",
                "portfolio_annual_upfront_cost_batt", "portfolio_annual_financed_cost",
                "portfolio_annual_down_payment_cost", "portfolio_annual_out_of_pocket_cost"):
        if col in outputs.get("portfolio_annual_savings", pd.DataFrame()).columns:
            pac = outputs["portfolio_annual_savings"]
            s = _sum_metric(pac[["state_abbr","year","scenario",col]].copy(), col, cumulative=False)
            if not s.empty:
                s["metric"] = col
                pieces.append(s)

    # Cumulative bill savings (total) — existing
    if "cumulative_bill_savings" in outputs.get("cumulative_bill_savings", pd.DataFrame()).columns:
        cbs = outputs["cumulative_bill_savings"]
        s = _sum_metric(cbs[["state_abbr","year","scenario","cumulative_bill_savings"]].copy(),
                        "cumulative_bill_savings", cumulative=True)
        if not s.empty:
            s["metric"] = "cumulative_bill_savings"
            pieces.append(s)

    # Cumulative net savings — NEW (optional)
    if "cumulative_net_savings" in outputs.get("cumulative_bill_savings", pd.DataFrame()).columns:
        cns = outputs["cumulative_bill_savings"]
        s = _sum_metric(cns[["state_abbr","year","scenario","cumulative_net_savings"]].copy(),
                        "cumulative_net_savings", cumulative=True)
        if not s.empty:
            s["metric"] = "cumulative_net_savings"
            pieces.append(s)

    # Optional: national peak MW panel
    if peaks_df is not None and not peaks_df.empty and "peak_mw" in peaks_df.columns:
        s = _sum_metric(peaks_df[["state_abbr","year","scenario","peak_mw"]].copy(), "peak_mw", cumulative=False)
        if not s.empty:
            s["metric"] = "peak_mw"
            pieces.append(s)

    if not pieces:
        return pd.DataFrame(columns=["year","scenario","metric","value"])

    return pd.concat(pieces, ignore_index=True)


def facet_lines_national_totals(
    outputs: Dict[str, pd.DataFrame],
    peaks_df: Optional[pd.DataFrame] = None,
    coincident_df: Optional[pd.DataFrame] = None,
    metrics: Optional[Iterable[str]] = (
        "number_of_adopters",
        "system_kw_cum",
        "batt_kwh_cum",
        "cumulative_bill_savings",
        "cumulative_net_savings",         
        "portfolio_annual_savings",
        "portfolio_annual_net_savings",    
        "peak_mw",
        "coincident_reduction_mw",
    ),
    xticks: Iterable[int] = (2026, 2030, 2035, 2040),
    title: str = "U.S. Totals — Baseline vs Policy",
    ncols: int = 3,
) -> None:
    """
    Faceted national line charts by metric. Optionally includes a single-series panel
    for ``coincident_reduction_mw`` (delta series, not per-scenario).

    Parameters
    ----------
    outputs : dict[str, pandas.DataFrame]
        Result of :func:`aggregate_state_metrics`.
    peaks_df : pandas.DataFrame, optional
        State peaks for inclusion (metric="peak_mw").
    coincident_df : pandas.DataFrame, optional
        National coincident reductions: ['year','coincident_reduction_mw'].
    metrics : iterable[str], optional
        Subset of metrics to plot.
    """
    nat = build_national_totals(outputs, peaks_df=peaks_df)

    # Optional coincident reduction single-series panel
    if coincident_df is not None and not coincident_df.empty:
        co = (coincident_df.groupby("year", as_index=False)["coincident_reduction_mw"]
              .sum()
              .rename(columns={"coincident_reduction_mw": "value"}))
        co["scenario"] = "coincident Δ"
        co["metric"]   = "coincident_reduction_mw"
        nat = pd.concat([nat, co], ignore_index=True, sort=False)

    # Filter to requested metrics
    if metrics is not None:
        nat = nat[nat["metric"].isin(set(metrics))]

    # Pretty titles (add net-savings labels)
    nice = {
        "number_of_adopters": "Cumulative Adopters",
        "system_kw_cum": "Cumulative PV (kW)",
        "batt_kwh_cum": "Cumulative Storage (kWh)",
        "cumulative_bill_savings": "Cumulative Bill Savings ($)",
        "cumulative_net_savings": "Cumulative Net Savings ($, after debt)",          # NEW
        "portfolio_annual_savings": "Portfolio Bill Savings ($/yr)",
        "portfolio_annual_net_savings": "Portfolio Net Savings ($/yr, after debt)", # NEW
        "peak_mw": "U.S. Peak Demand (MW)",
        "coincident_reduction_mw": "National Coincident Peak Reduction (MW)",
    }

    # Value formatter with unit-aware $ handling for new metrics
    def _fmt(metric: str, v: float) -> str:
        if metric == "number_of_adopters":             return f"{v/1e6:.1f}M"
        if metric == "system_kw_cum":                   return f"{v/1e6:.1f} GW"
        if metric == "batt_kwh_cum":                    return f"{v/1e6:.1f} GWh"
        if metric in {"cumulative_bill_savings", "cumulative_net_savings"}:
            return f"${v/1e9:.1f}B"
        if metric in {"portfolio_annual_savings", "portfolio_annual_net_savings"}:
            return f"${v/1e9:.1f}B/yr" if abs(v) >= 5e8 else f"${v/1e6:.1f}M/yr"
        if metric in {"peak_mw","coincident_reduction_mw"}:
            return f"{v/1e3:.1f} GW"
        return f"{v:.2g}"

    metric_list = list(nat["metric"].unique())
    n = len(metric_list)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.4*nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for i, m in enumerate(metric_list):
        ax = axes[i]
        d = nat[nat["metric"] == m].sort_values("year")
        for scen, g in d.groupby("scenario", observed=True):
            ax.plot(g["year"], g["value"], marker="o", label=scen.capitalize())
            # annotate last point for each series
            end_year = 2040 if 2040 in set(g["year"]) else int(g["year"].max())
            end = g[g["year"] == end_year]
            if not end.empty:
                y_end = float(end["value"].iloc[-1])
                x_end = float(end["year"].iloc[-1])
                ax.annotate(_fmt(m, y_end), xy=(x_end, y_end), xytext=(6, 0),
                            textcoords="offset points", ha="left", va="center",
                            fontsize=10, fontweight="bold")
        ax.set_xticks(list(xticks))
        ax.set_xlabel("Year")
        ax.set_title(nice.get(m, m))
        ax.legend(frameon=False, loc="best")

    # Hide any unused facets
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, y=1.02)
    plt.show()


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

def export_compiled_results_to_excel(
    outputs: Dict[str, pd.DataFrame],
    run_id: str,
    out_dir: str,
    peaks_df: Optional[pd.DataFrame] = None,
    coincident_df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Write a tidy Excel workbook with all aggregated tables and optional national series.

    Parameters
    ----------
    outputs : dict[str, pandas.DataFrame]
        Result dict from :func:`aggregate_state_metrics`.
    run_id : str
        Label to include in the filename.
    out_dir : str
        Destination directory (will be created if missing).
    peaks_df : pandas.DataFrame, optional
        State peaks table for a 'peaks' sheet.
    coincident_df : pandas.DataFrame, optional
        National coincident reductions for a 'coincident' sheet.

    Returns
    -------
    str
        Full path to the written .xlsx file.
    """
    import datetime as _dt
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.xlsx")

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as xw:
        meta = pd.DataFrame({
            "run_id": [run_id],
            "generated_at": [_dt.datetime.now().isoformat(timespec="seconds")],
            "tables_included": [", ".join(sorted([k for k, v in outputs.items() if isinstance(v, pd.DataFrame) and not v.empty]))]
        })
        meta.to_excel(xw, index=False, sheet_name="README")

        for key, df in outputs.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(xw, index=False, sheet_name=key[:31])

        if peaks_df is not None and not peaks_df.empty:
            peaks_df.to_excel(xw, index=False, sheet_name="peaks")

        if coincident_df is not None and not coincident_df.empty:
            coincident_df.to_excel(xw, index=False, sheet_name="coincident")

        nat_totals = build_national_totals(outputs, peaks_df=peaks_df)
        if coincident_df is not None and not coincident_df.empty:
            co = coincident_df.groupby("year", as_index=False)["coincident_reduction_mw"].sum().rename(columns={"coincident_reduction_mw":"value"})
            co["scenario"] = "coincident Δ"
            co["metric"]   = "coincident_reduction_mw"
            nat_totals = pd.concat([nat_totals, co], ignore_index=True)
        if not nat_totals.empty:
            nat_totals.to_excel(xw, index=False, sheet_name="national_totals")

    return out_path


# -----------------------------------------------------------------------------
# Thin wrappers for backward compatibility with older notebooks
# -----------------------------------------------------------------------------

def process_all_states(
    root_dir: str | None = None,
    run_id: str | None = None,
    cfg: SavingsConfig | None = None,
    n_jobs: int = 1,
    states: Optional[Iterable[str]] = None,
    warehouse: DataWarehouse | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Backward-compatible wrapper.
    If a ``warehouse`` is provided, reuse it; otherwise load from ``root_dir``/``run_id``.
    """
    if warehouse is None:
        assert root_dir is not None, "root_dir is required when warehouse is not provided"
        warehouse = DataWarehouse.from_disk(root_dir, run_id=run_id, states=states)
    return aggregate_state_metrics(warehouse.agents, cfg or SavingsConfig())


def process_all_states_peaks(
    root_dir: str | None = None,
    run_id: str | None = None,
    n_jobs: int = 1,
    states: Optional[Iterable[str]] = None,
    warehouse: DataWarehouse | None = None,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper for state peaks.
    """
    if warehouse is None:
        assert root_dir is not None, "root_dir is required when warehouse is not provided"
        warehouse = DataWarehouse.from_disk(root_dir, run_id=run_id, states=states)
    return compute_peaks_by_state(warehouse.state_hourly)


def compute_state_coincident_reduction_legacy(
    root_dir: str | None = None,
    run_id: str | None = None,
    states: Optional[Iterable[str]] = None,
    warehouse: DataWarehouse | None = None,
    method: str = "baseline_topN_avg",
    top_n: int = 10,
    percentile: Optional[float] = None,
) -> pd.DataFrame:
    """
    Wrapper that mirrors the old signature (no method in the original). The default here
    is top‑N averaging; set method="single_hour" to replicate the older behavior.
    """
    if warehouse is None:
        assert root_dir is not None, "root_dir is required when warehouse is not provided"
        warehouse = DataWarehouse.from_disk(root_dir, run_id=run_id, states=states)
    return compute_state_coincident_reduction(warehouse, method=method, top_n=top_n, percentile=percentile)


def compute_rto_coincident_reduction_legacy(
    root_dir: str | None = None,
    run_id: str | None = None,
    states: Optional[Iterable[str]] = None,
    warehouse: DataWarehouse | None = None,
    method: str = "baseline_topN_avg",
    top_n: int = 10,
    percentile: Optional[float] = None,
) -> pd.DataFrame:
    """
    Wrapper that mirrors the old signature (no method in the original). The default here
    is top‑N averaging; set method="single_hour" to replicate the older behavior.
    """
    if warehouse is None:
        assert root_dir is not None, "root_dir is required when warehouse is not provided"
        warehouse = DataWarehouse.from_disk(root_dir, run_id=run_id, states=states)
    return compute_rto_coincident_reduction(warehouse, method=method, top_n=top_n, percentile=percentile, return_by_rto=False)


def plot_us_cum_adopters_grouped(
    outputs: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Grouped bar plot of national cumulative adopters by year, comparing
    baseline vs policy. Returns the tidy table used for plotting.

    Parameters
    ----------
    outputs : dict[str, pandas.DataFrame]
        Aggregations returned by :func:`aggregate_state_metrics`.
    Returns
    -------
    pandas.DataFrame
        ['year','scenario','value'] where 'value' is U.S. total cumulative adopters.
    """

    nat = build_national_totals(outputs)
    if nat.empty:
        raise ValueError("No national totals available. Run process_all_states(...) first.")

    d = nat[nat["metric"] == "number_of_adopters"].copy()
    d['value'] = pd.to_numeric(d['value'])
    d['year'] = pd.to_numeric(d['year'], errors='coerce').fillna(0).astype(int)
    if d.empty:
        raise ValueError("National totals lack 'number_of_adopters' metric.")

    # Cosmetic scenario labels for the chart
    rename_map = {"baseline": "Business-as-usual", "policy": "$1/Watt"}
    d["scenario"] = d["scenario"].map(rename_map).fillna(d["scenario"])

    # Plot
    plt.rcParams["font.family"] = "Cabin"
    plt.figure(figsize=(12, 5), constrained_layout=True)
    ax = sns.barplot(
        data=d, x="year", y="value", hue="scenario",
        errorbar=None, palette=["#a2e0fc", "#1bb3ef"]
    )
    ax.set_xlabel("")
    ax.set_ylabel("Solar Installations (millions)")

    # Format y-axis in millions and annotate bars
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}"))
    for c in ax.containers:
        ax.bar_label(
            c,
            labels=[f"{v/1e6:.1f}M" if np.isfinite(v) else "" for v in c.datavalues],
            padding=2, fontsize=10.5,
        )

    plt.legend(title=None, frameon=False, fontsize=12)
    ax.grid(False)
    plt.show()


def build_payback_timeseries(
    root_dir: str | None = None,
    run_id: str | None = None,
    strict_run_id: bool = True,  # kept for signature compatibility; unused
    level: str = "state",        # "state" or "US"
    *,
    agents: pd.DataFrame | None = None,
    warehouse: "DataWarehouse" | None = None,
) -> pd.DataFrame:
    """
    Adoption‑weighted average payback by year for baseline vs policy.

    Notes
    -----
    For backward compatibility, the output column is named
    ``'payback_weighted_median'`` even though this is a weighted *mean*.

    Parameters
    ----------
    root_dir, run_id, strict_run_id : see :meth:`DataWarehouse.from_disk`
        If ``agents`` and ``warehouse`` are None, data will be loaded once via
        :class:`DataWarehouse`.
    level : {"state","US"}, default "state"
        Grouping level to return.
    agents : pandas.DataFrame, optional
        Preloaded agent table (preferred to avoid I/O).
    warehouse : DataWarehouse, optional
        Preloaded warehouse (preferred to avoid I/O).

    Returns
    -------
    pandas.DataFrame
        ['geo','scenario','year','payback_weighted_median']
    """
    # Resolve agent data without re-reading more than once
    if agents is None:
        if warehouse is None:
            if root_dir is None:
                return pd.DataFrame(columns=["geo", "scenario", "year", "payback_weighted_median"])
            warehouse = DataWarehouse.from_disk(root_dir, run_id=run_id)
        agents = warehouse.agents

    if agents.empty or "payback_period" not in agents.columns:
        return pd.DataFrame(columns=["geo", "scenario", "year", "payback_weighted_median"])

    x = agents.copy()
    x["year"] = pd.to_numeric(x.get("year"), errors="coerce")
    x["new_adopters"] = pd.to_numeric(x.get("new_adopters"), errors="coerce").fillna(0.0).clip(lower=0.0)
    x["payback_period"] = pd.to_numeric(x.get("payback_period"), errors="coerce")

    # Filter to rows that contribute to a weighted mean
    x = x[(x["new_adopters"] > 0) & x["year"].notna() & x["payback_period"].notna()].copy()
    if x.empty:
        return pd.DataFrame(columns=["geo", "scenario", "year", "payback_weighted_median"])

    if level.lower() == "state":
        grp = ["state_abbr", "scenario", "year"]
        geo_col = "state_abbr"
    else:
        x["geo"] = "US"
        grp = ["geo", "scenario", "year"]
        geo_col = "geo"

    # Weighted mean
    wsum = x.groupby(grp, observed=True).apply(
        lambda g: pd.Series({"payback_weighted_median": float(np.average(g["payback_period"], weights=g["new_adopters"]))
                  if g["new_adopters"].sum() > 0 else np.nan}),
        include_groups=False,
    ).reset_index()

    out = wsum.rename(columns={geo_col: "geo"}).sort_values(["geo", "scenario", "year"])
    return out.reset_index(drop=True)


def facet_choropleth_payback_continuous(
    root_dir: str | None = None,
    shapefile_path: str | None = None,
    run_id: str | None = None,
    strict_run_id: bool = True,     # kept for signature compatibility; unused
    year: int = 2040,
    cmap: str = "Blues_r",          # darker = lower payback
    *,
    agents: pd.DataFrame | None = None,
    warehouse: "DataWarehouse" | None = None,
    shape_gdf: "gpd.GeoDataFrame" | None = None,
) -> pd.DataFrame:
    """
    Vertically faceted choropleth of adoption‑weighted average payback (years) in `year`.
    Top: Baseline. Bottom: Policy. A shared color scale is used across panels.

    Parameters
    ----------
    root_dir, run_id, strict_run_id : see :meth:`DataWarehouse.from_disk`
        Used only if ``agents`` / ``warehouse`` are not provided.
    shapefile_path : str, optional
        Path to a state shapefile; ignored if ``shape_gdf`` is provided.
        Must include a two‑letter state code column (e.g., STUSPS).
    year : int, default 2040
        Calendar year to map.
    cmap : str, default "Blues_r"
        Matplotlib colormap.
    agents, warehouse : see :func:`build_payback_timeseries`
    shape_gdf : geopandas.GeoDataFrame, optional
        Preloaded shape with state boundaries to avoid I/O.

    Returns
    -------
    pandas.DataFrame
        ['state_abbr','scenario','payback_years'] used to paint the map.
    """
    import matplotlib.pyplot as plt

    if gpd is None:
        raise ImportError("geopandas is required for choropleths.")

    pb = build_payback_timeseries(
        root_dir=root_dir, run_id=run_id, strict_run_id=strict_run_id,
        level="state", agents=agents, warehouse=warehouse
    )
    if pb.empty:
        raise ValueError("No payback data found. Ensure 'payback_period' and 'new_adopters' are present.")

    d = pb.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d[(d["year"] == int(year)) & d["geo"].notna() & d["scenario"].notna()].copy()
    d = d.rename(columns={"geo": "state_abbr", "payback_weighted_median": "payback_years"})
    d["state_abbr"] = d["state_abbr"].astype(str).str.upper()
    d["scenario"] = d["scenario"].astype(str).str.lower().str.strip()
    d = d[d["scenario"].isin(["baseline", "policy"])]
    if d.empty:
        raise ValueError(f"No baseline/policy payback rows for year {year}.")

    # Shapefile
    gdf = shape_gdf.copy() if shape_gdf is not None else gpd.read_file(shapefile_path).to_crs("EPSG:5070")
    if "STUSPS" not in gdf.columns:
        for c in ("STUSPS", "stusps", "STATE_ABBR", "STATE", "STATEFP", "STATEFP20"):
            if c in gdf.columns:
                gdf["STUSPS"] = gdf[c].astype(str).str.upper()
                break
    if "STUSPS" not in gdf.columns:
        raise ValueError("Shapefile must include a two-letter state code (e.g., STUSPS).")
    gdf["STUSPS"] = gdf["STUSPS"].astype(str).str.upper()
    gdf = gdf[~gdf["STUSPS"].isin({"AK","HI","PR","GU","VI","AS","MP","DC"})].copy()

    vmin = float(np.nanmin(d["payback_years"].values))
    vmax = float(np.nanmax(d["payback_years"].values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Payback values are not finite; check inputs.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
    plt.rcParams["font.family"] = "Cabin"

    panels = [("baseline", f"Business-as-usual: Payback (Years) in {year}"),
              ("policy",   f"$1 per Watt: Payback (Years) in {year}")]

    for ax, (scen, _title) in zip(axes, panels):
        sub = d[d["scenario"] == scen][["state_abbr","payback_years"]]
        m = gdf.merge(sub.rename(columns={"state_abbr":"STUSPS"}), on="STUSPS", how="left")
        m.plot(
            column="payback_years",
            cmap=cmap, vmin=vmin, vmax=vmax,
            linewidth=0.6, edgecolor="grey",
            legend=(scen == "policy"),
            legend_kwds={"label": "Payback (years)",
                         "orientation": "horizontal",
                         "shrink": 0.8, "pad": 0.02},
            ax=ax,
            missing_kwds={"color": "#f5f5f5"},
        )
        ax.axis("off")

    plt.show()


def build_eabs_calendar_timeseries(
    root_dir: str | None = None,
    run_id: str | None = None,
    strict_run_id: bool = True,  # kept for signature compatibility; unused
    level: str = "state",
    *,
    agents: pd.DataFrame | None = None,
    warehouse: "DataWarehouse" | None = None,
    lifetime_years: int = 25,              # arrays horizon to roll forward
) -> pd.DataFrame:
    """
    Calendar-year, adoption-weighted bill metrics from 25-yr cohorts that adopt in prior years.

    NEW: adds a pure net-cashflow view using *only* arrays:
      - avg_debt : adoption-weighted average debt service paid this year (USD/yr per adopter)
      - eabs_net : net cashflow = (avg_bill_wo - avg_bill_with) - avg_debt
      - pct_net_vs_wo : eabs_net / avg_bill_wo

    Output columns per (geo × scenario × year):
      - adopters_active
      - avg_bill_with, avg_bill_wo, eabs, pct_savings  (unchanged)
      - avg_debt, eabs_net, pct_net_vs_wo              (new)
    """
    import numpy as np
    import pandas as pd

    if agents is None:
        if warehouse is None:
            if root_dir is None:
                return pd.DataFrame(columns=[
                    "geo","scenario","year","adopters_active",
                    "avg_bill_with","avg_bill_wo","eabs","pct_savings",
                    "avg_debt","eabs_net","pct_net_vs_wo"
                ])
            warehouse = DataWarehouse.from_disk(root_dir, run_id=run_id)
        agents = warehouse.agents

    # Required columns for classic EABS
    cols_needed = {
        "state_abbr","year","scenario","new_adopters","batt_adopters_added_this_year",
        "utility_bill_w_sys_pv_only","utility_bill_w_sys_pv_batt",
        "utility_bill_wo_sys_pv_only","utility_bill_wo_sys_pv_batt",
    }
    # Optional for NET: debt arrays (if missing, treated as zero)
    opt_cols = {"cf_debt_payment_total_pv_only","cf_debt_payment_total_pv_batt"}

    missing = cols_needed - set(agents.columns)
    if missing:
        return pd.DataFrame(columns=[
            "geo","scenario","year","adopters_active",
            "avg_bill_with","avg_bill_wo","eabs","pct_savings",
            "avg_debt","eabs_net","pct_net_vs_wo"
        ])

    x = agents.copy()

    # Cohort sizes
    x["year"] = pd.to_numeric(x["year"], errors="coerce")
    x["new_adopters"] = pd.to_numeric(x["new_adopters"], errors="coerce").fillna(0.0).clip(lower=0.0)
    x["batt_adopters_added_this_year"] = pd.to_numeric(x["batt_adopters_added_this_year"], errors="coerce").fillna(0.0).clip(lower=0.0)
    x["pv_batt_n"] = np.minimum(x["batt_adopters_added_this_year"], x["new_adopters"])
    x["pv_only_n"] = (x["new_adopters"] - x["pv_batt_n"]).clip(lower=0.0)

    # Parse 25-yr arrays (bills)
    x["bw_only"] = x["utility_bill_w_sys_pv_only"].apply(_arr25)
    x["bw_batt"] = x["utility_bill_w_sys_pv_batt"].apply(_arr25)
    x["bo_only"] = x["utility_bill_wo_sys_pv_only"].apply(_arr25)
    x["bo_batt"] = x["utility_bill_wo_sys_pv_batt"].apply(_arr25)

    # Parse 25-yr arrays (debt payments); treat missing as zeros
    x["db_only"] = x["cf_debt_payment_total_pv_only"].apply(_arr25) if "cf_debt_payment_total_pv_only" in x.columns else [[]]*len(x)
    x["db_batt"] = x["cf_debt_payment_total_pv_batt"].apply(_arr25) if "cf_debt_payment_total_pv_batt" in x.columns else [[]]*len(x)

    years = x["year"].dropna()
    if years.empty:
        return pd.DataFrame(columns=[
            "geo","scenario","year","adopters_active",
            "avg_bill_with","avg_bill_wo","eabs","pct_savings",
            "avg_debt","eabs_net","pct_net_vs_wo"
        ])

    y_min, y_max = int(years.min()), int(years.max())
    L = int(lifetime_years)

    # bucket: key -> (sum_with, sum_wo, adopters, sum_debt)
    bucket: dict[tuple[str,str,int], tuple[float,float,float,float]] = {}

    for r in x.itertuples(index=False):
        if pd.isna(r.year) or not r.scenario:
            continue
        y0 = int(r.year)
        geos = [r.state_abbr] if level.lower() == "state" else ["US"]

        pw_only, pw_batt = float(r.pv_only_n or 0.0), float(r.pv_batt_n or 0.0)
        a_bw_only, a_bw_batt = list(r.bw_only or []), list(r.bw_batt or [])
        a_bo_only, a_bo_batt = list(r.bo_only or []), list(r.bo_batt or [])
        a_db_only, a_db_batt = list(r.db_only or []), list(r.db_batt or [])

        for k in range(L):
            y = y0 + k
            if y < y_min or y > y_max:
                continue

            # Bills and adopters (same as before)
            sum_with = (a_bw_only[k]*pw_only if k < len(a_bw_only) and pw_only>0 else 0.0) + \
                       (a_bw_batt[k]*pw_batt if k < len(a_bw_batt) and pw_batt>0 else 0.0)
            sum_wo   = (a_bo_only[k]*pw_only if k < len(a_bo_only) and pw_only>0 else 0.0) + \
                       (a_bo_batt[k]*pw_batt if k < len(a_bo_batt) and pw_batt>0 else 0.0)
            adopters = (pw_only if (k < len(a_bo_only) or k < len(a_bw_only)) else 0.0) + \
                       (pw_batt if (k < len(a_bo_batt) or k < len(a_bw_batt)) else 0.0)

            # Debt service this calendar year
            sum_debt = (a_db_only[k]*pw_only if k < len(a_db_only) and pw_only>0 else 0.0) + \
                       (a_db_batt[k]*pw_batt if k < len(a_db_batt) and pw_batt>0 else 0.0)

            if (sum_with != 0.0) or (sum_wo != 0.0) or (adopters > 0.0) or (sum_debt != 0.0):
                for geo in geos:
                    key = (geo, r.scenario, y)
                    sw, so, na, sd = bucket.get(key, (0.0, 0.0, 0.0, 0.0))
                    bucket[key] = (sw + sum_with, so + sum_wo, na + adopters, sd + sum_debt)

    if not bucket:
        return pd.DataFrame(columns=[
            "geo","scenario","year","adopters_active",
            "avg_bill_with","avg_bill_wo","eabs","pct_savings",
            "avg_debt","eabs_net","pct_net_vs_wo"
        ])

    rows = []
    for (geo, scen, y), (sum_with, sum_wo, adopters, sum_debt) in bucket.items():
        if adopters > 0:
            avg_with = sum_with / adopters
            avg_wo   = sum_wo   / adopters
            eabs     = avg_wo - avg_with
            avg_debt = sum_debt / adopters
            eabs_net = eabs - avg_debt
            pct      = (eabs / avg_wo) if avg_wo > 0 else 0.0
            pct_net  = (eabs_net / avg_wo) if avg_wo > 0 else 0.0
        else:
            avg_with = avg_wo = eabs = avg_debt = eabs_net = pct = pct_net = 0.0

        rows.append((geo, scen, int(y), float(adopters),
                     float(avg_with), float(avg_wo), float(eabs), float(pct),
                     float(avg_debt), float(eabs_net), float(pct_net)))

    out = pd.DataFrame(
        rows,
        columns=["geo","scenario","year","adopters_active",
                 "avg_bill_with","avg_bill_wo","eabs","pct_savings",
                 "avg_debt","eabs_net","pct_net_vs_wo"]
    ).sort_values(["geo","scenario","year"]).reset_index(drop=True)
    return out



def summarize_us_eabs_for_year(eabs_ts: pd.DataFrame, year: int = 2040) -> pd.DataFrame:
    """
    Compact U.S. summary for a single year (now includes net-cashflow fields).
    Returns:
      ['scenario','year','adopters_active',
       'avg_bill_wo','avg_bill_with','eabs','pct_savings',
       'avg_debt','eabs_net','pct_net_vs_wo']
    """
    d = eabs_ts[(eabs_ts["geo"] == "US") & (eabs_ts["year"] == int(year))].copy()
    keep = ["scenario","year","adopters_active",
            "avg_bill_wo","avg_bill_with","eabs","pct_savings",
            "avg_debt","eabs_net","pct_net_vs_wo"]
    for c in keep:
        if c not in d.columns:
            d[c] = 0.0
    return d[keep].sort_values("scenario").reset_index(drop=True)



def choropleth_state_coincident_reduction(
    root_dir: str | None = None,
    shapefile_path: str | None = None,
    run_id: str | None = None,
    year: int = 2040,
    k_bins: int = 6,
    states: Optional[Iterable[str]] = None,
    *,
    warehouse: "DataWarehouse" | None = None,
    shape_gdf: "gpd.GeoDataFrame" | None = None,
    method: str = "baseline_topN_avg",
    top_n: int = 10,
    percentile: float | None = None,
) -> pd.DataFrame:
    """
    Choropleth of state‑level coincident peak reduction (GW) in `year`.

    Coincident reduction is computed using :func:`compute_state_coincident_reduction`
    with the provided method (default: baseline top‑N averaging).

    Parameters
    ----------
    root_dir, run_id : see :meth:`DataWarehouse.from_disk`
        Used only if ``warehouse`` is not provided.
    shapefile_path : str, optional
        Path to state shapefile; ignored if ``shape_gdf`` provided.
    year : int, default 2040
    k_bins : int, default 6
        Number of legend classes (Fisher‑Jenks if available; else quantiles).
    states : iterable[str], optional
        Optional subset of state codes for loading (if ``warehouse`` is None).
    warehouse : DataWarehouse, optional
        Preloaded warehouse to avoid I/O.
    shape_gdf : geopandas.GeoDataFrame, optional
        Preloaded shape.
    method, top_n, percentile : see :func:`_coincident_scalar`

    Returns
    -------
    pandas.DataFrame
        Tidy table used to paint the map: ['state_abbr','coincident_reduction_gw']
    """
    import matplotlib.pyplot as plt

    if gpd is None:
        raise ImportError("geopandas is required for choropleths.")

    if warehouse is None:
        if root_dir is None:
            raise ValueError("Provide either a preloaded `warehouse` or a `root_dir`.")
        warehouse = DataWarehouse.from_disk(root_dir, run_id=run_id, states=states)

    co = compute_state_coincident_reduction(
        warehouse, method=method, top_n=top_n, percentile=percentile
    )
    if co.empty:
        raise ValueError("No coincident reduction rows found; ensure hourly data exist for baseline/policy.")

    d = co.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d[(d["year"] == int(year)) & d["state_abbr"].notna()].copy()
    if d.empty:
        raise ValueError(f"No coincident reduction rows found for year {year}.")
    d["state_abbr"] = d["state_abbr"].astype(str).str.upper()
    d["coincident_reduction_gw"] = pd.to_numeric(d["coincident_reduction_mw"], errors="coerce") / 1000.0
    d = d[["state_abbr", "coincident_reduction_gw"]].copy()

    # Shapefile
    gdf = shape_gdf.copy() if shape_gdf is not None else gpd.read_file(shapefile_path).to_crs("EPSG:5070")
    if "STUSPS" not in gdf.columns:
        for c in ("STUSPS", "stusps", "STATE_ABBR", "STATE", "STATEFP", "STATEFP20"):
            if c in gdf.columns:
                gdf["STUSPS"] = gdf[c].astype(str).str.upper()
                break
    if "STUSPS" not in gdf.columns:
        raise ValueError("Shapefile must include a two-letter state code (e.g., STUSPS).")
    gdf["STUSPS"] = gdf["STUSPS"].astype(str).str.upper()
    gdf = gdf[~gdf["STUSPS"].isin({"AK","HI","PR","GU","VI","AS","MP","DC"})].copy()

    plot_df = gdf.merge(d.rename(columns={"state_abbr": "STUSPS"}), on="STUSPS", how="left")

    # Classing
    try:
        import mapclassify  # noqa: F401
        scheme = "fisher_jenks"; extra = dict(k=int(k_bins))
    except Exception:
        scheme = "quantiles";    extra = dict(k=int(k_bins))

    # Plot
    plt.rcParams["font.family"] = "Cabin"
    fig, ax = plt.subplots(1, 1, figsize=(13.5, 7))
    plot_df.plot(
        column="coincident_reduction_gw",
        cmap="Blues",
        linewidth=0.6, edgecolor="grey",
        legend=True,
        scheme=scheme,
        legend_kwds={
            "title": "Coincident Peak Reduction (GW)",
            "ncols": 2,
            "fmt": "{:.1f}",
            "loc": "center left",
            "bbox_to_anchor": (0.02, 0.02),
            "borderaxespad": 0.0,
            "frameon": False,
        },
        ax=ax,
        missing_kwds={"color": "lightgray"},
        **extra,
    )
    plt.tight_layout(rect=(0, 0, 0.86, 1))
    leg = ax.get_legend()
    if leg and hasattr(leg, "texts") and leg.get_texts():
        last = leg.get_texts()[-1]
        last.set_text(f" > {last.get_text().split(',')[0].strip()}")
    ax.axis("off")
    plt.show()

def choropleth_pct_households_with_solar(
    outputs: Dict[str, pd.DataFrame],
    shapefile_path: str | None = None,
    year: int = 2040,
    scenario: str = "policy",
    k_bins: int = 6,
    *,
    shape_gdf: "gpd.GeoDataFrame" | None = None,
    hh_col: str = "hh",
) -> pd.DataFrame:
    """
    Choropleth of % of households with rooftop solar by state.

    For each state:
        pct_households_with_solar = number_of_adopters(year, scenario) / households

    Parameters
    ----------
    outputs : dict[str, pandas.DataFrame]
        Result of :func:`aggregate_state_metrics` (uses outputs["totals"]).
    shapefile_path : str, optional
        State shapefile path; ignored if ``shape_gdf`` is provided. Must include
        a households column (default 'hh').
    year : int, default 2040
    scenario : {"baseline","policy"}, default "policy"
    k_bins : int, default 6
        Number of legend classes (Fisher‑Jenks if available; else quantiles).
    shape_gdf : geopandas.GeoDataFrame, optional
        Preloaded shape to avoid I/O.
    hh_col : str, default "hh"
        Name of the households column in the shapefile.

    Returns
    -------
    pandas.DataFrame
        ['state_abbr','number_of_adopters','hh','pct_households_with_solar']
    """
    import matplotlib.pyplot as plt

    if gpd is None:
        raise ImportError("geopandas is required for choropleths.")

    totals = outputs.get("totals", pd.DataFrame())
    if totals.empty:
        raise ValueError("outputs['totals'] is empty; run process_all_states(...) first.")

    need = {"state_abbr", "year", "scenario", "number_of_adopters"}
    missing = need - set(totals.columns)
    if missing:
        raise ValueError(f"outputs['totals'] missing columns: {sorted(missing)}")

    s = totals.loc[
        (totals["year"] == int(year)) &
        (totals["scenario"].astype(str).str.lower() == scenario.lower()),
        ["state_abbr", "number_of_adopters"]
    ].copy()
    if s.empty:
        raise ValueError(f"No rows for year={year}, scenario={scenario!r} in outputs['totals'].")

    s["state_abbr"] = s["state_abbr"].astype(str).str.upper()
    adopters = s.groupby("state_abbr", as_index=False)["number_of_adopters"].sum()

    # Shapefile
    gdf = shape_gdf.copy() if shape_gdf is not None else gpd.read_file(shapefile_path).to_crs("EPSG:5070")
    if "STUSPS" not in gdf.columns:
        for c in ("STUSPS", "stusps", "STATE_ABBR", "STATE", "STATEFP", "STATEFP20"):
            if c in gdf.columns:
                gdf["STUSPS"] = gdf[c].astype(str).str.upper()
                break
    if "STUSPS" not in gdf.columns:
        raise ValueError("Shapefile must include a two-letter state code (e.g., STUSPS).")
    if hh_col not in gdf.columns:
        raise ValueError(f"Shapefile must include a '{hh_col}' column for total households.")

    gdf["STUSPS"] = gdf["STUSPS"].astype(str).str.upper()
    gdf = gdf[~gdf["STUSPS"].isin({"AK","HI","PR","GU","VI","AS","MP","DC"})].copy()

    plot_df = gdf.merge(adopters.rename(columns={"state_abbr": "STUSPS"}), on="STUSPS", how="left")
    plot_df["hh"] = pd.to_numeric(plot_df[hh_col], errors="coerce")
    plot_df["number_of_adopters"] = pd.to_numeric(plot_df["number_of_adopters"], errors="coerce")
    plot_df["pct_households_with_solar"] = np.where(
        (plot_df["hh"] > 0) & plot_df["number_of_adopters"].notna(),
        plot_df["number_of_adopters"] / plot_df["hh"],
        np.nan
    )

    out = (
        plot_df[["STUSPS", "number_of_adopters", "hh", "pct_households_with_solar"]]
        .rename(columns={"STUSPS": "state_abbr"})
        .copy()
    )

    # Classing
    try:
        import mapclassify  # noqa: F401
        scheme = "fisher_jenks"; extra = dict(k=int(k_bins))
    except Exception:
        scheme = "quantiles";    extra = dict(k=int(k_bins))

    # Plot percentage *100 for intuitive legend tick formatting
    plt.rcParams["font.family"] = "Cabin"
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    plot_df["pct_times_100"] = plot_df["pct_households_with_solar"] * 100.0
    plot_df.plot(
        column="pct_times_100",
        cmap="Blues",
        linewidth=0.6, edgecolor="grey",
        legend=True,
        scheme=scheme,
        legend_kwds={
            "title": f"Households with Solar ({year}, {scenario})",
            "ncols": 2,
            "fmt": "{:.0f}%",
            "loc": "lower left",
            "bbox_to_anchor": (0.02, 0.02),
        },
        ax=ax,
        missing_kwds={"color": "lightgray"},
        **extra,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def plot_us_payback_customers_weighted_over_time(
    root_dir: str | None = None,
    run_id: str | None = None,
    *,
    agents: pd.DataFrame | None = None,
    warehouse: "DataWarehouse" | None = None,
    title: str = "Payback Period — Business-as-usual vs. $1/Watt",
    xticks: Iterable[int] = (2026, 2030, 2035, 2040),
) -> pd.DataFrame:
    """
    Plot the U.S. customers_in_bin-weighted average payback (years) over time
    for baseline vs policy scenarios, using Cabin font and the house colors.

    Returns the tidy DataFrame used for plotting:
        ['year','scenario','payback_years_us_weighted']
    """
    # Resolve agent data without extra I/O
    if agents is None:
        if warehouse is None:
            if root_dir is None:
                raise ValueError("Provide `agents` or (`root_dir`, `run_id`) or `warehouse`.")
            warehouse = DataWarehouse.from_disk(root_dir, run_id=run_id)
        agents = warehouse.agents

    if agents.empty or "payback_period" not in agents.columns or "customers_in_bin" not in agents.columns:
        raise ValueError("Agents must include 'payback_period' and 'customers_in_bin' columns.")

    x = agents.copy()
    # Coerce numerics
    x["year"] = pd.to_numeric(x.get("year"), errors="coerce")
    x["payback_period"] = pd.to_numeric(x.get("payback_period"), errors="coerce")
    x["customers_in_bin"] = pd.to_numeric(x.get("customers_in_bin"), errors="coerce").fillna(0.0).clip(lower=0.0)
    x["scenario"] = x["scenario"].astype(str).str.lower().str.strip()

    # Keep rows that contribute to a weighted mean
    x = x[(x["year"].notna()) & (x["payback_period"].notna()) & (x["customers_in_bin"] > 0)].copy()
    if x.empty:
        raise ValueError("No valid rows to compute customers_in_bin-weighted payback.")

    # National (US) weighted mean by year × scenario
    d = (
        x.groupby(["scenario", "year"], observed=True)
        .apply(lambda g: pd.Series({"payback_years_us_weighted": float(np.average(g["payback_period"], weights=g["customers_in_bin"]))
                if g["customers_in_bin"].sum() > 0 else np.nan}),
               include_groups=False)
        .reset_index()
        .dropna(subset=["payback_years_us_weighted"])
    )

    # Cosmetic labels & ordering
    rename_map = {"baseline": "Business-as-usual", "policy": "$1/watt"}
    d["scenario"] = d["scenario"].map(rename_map).fillna(d["scenario"])
    d["year"] = d["year"].astype(int)
    d = d.sort_values(["scenario", "year"])

    # Plot
    plt.rcParams["font.family"] = "Cabin"
    plt.figure(figsize=(10, 5), constrained_layout=True)
    ax = sns.lineplot(
        data=d, x="year", y="payback_years_us_weighted",
        hue="scenario", marker="o",
        palette=["#a2e0fc", "#1bb3ef"]
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Weighted Average Payback (years)")
    ax.set_xticks(list(xticks))
    ax.grid(False)
    plt.legend(title=None, frameon=False)

    plt.title(title)
    plt.show()


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def _parse_arr(v, L=25):
    """Parse JSON '[..]' or Postgres '{..}' into np.array(float) length<=L."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.asarray(v, dtype=float)[:L]
    if isinstance(v, str):
        s = v.strip()
        try:
            if s.startswith("{") and s.endswith("}"):
                s = "[" + s[1:-1] + "]"
            return np.asarray(json.loads(s), dtype=float)[:L]
        except Exception:
            pass
    return np.array([], dtype=float)

def _wmedian(values, weights):
    v = np.asarray(values, float); w = np.asarray(weights, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m): return np.nan
    v, w = v[m], w[m]
    o = np.argsort(v, kind="mergesort"); v, w = v[o], w[o]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, 0.5*cw[-1], side="left")])

def plot_us_net_savings_median_to_date_from_agents(
    warehouse=None, *, agents=None,
    lifetime_years=25,
    title="Net Savings per Household — Business-as-usual vs. $1/Watt"
):
    """
    For each calendar year Y and scenario, pool all adopters with adoption_year <= Y.
    Earnings credited only up to year Y (min(lifetime_years, Y - adopt_year + 1)).
    Net savings per adopter (to date) = median_to_date(earnings) - median_to_date(upfront).
    Weighted medians (weights = cohort sizes).

    Returns tidy DataFrame with columns:
      ['year','scenario','med_earn_to_date','med_upfront','net_savings_per_adopter_median_to_date']
    """
    if agents is None:
        if warehouse is None or not hasattr(warehouse, "agents"):
            raise ValueError("Provide `agents` or a `warehouse` with `.agents`.")
        agents = warehouse.agents
    x = agents.copy()

    # Basic fields
    x["scenario"] = x["scenario"].astype(str).str.lower().str.strip()
    x["year"] = pd.to_numeric(x["year"], errors="coerce")
    if not x["year"].notna().any():
        raise ValueError("No valid adoption years in agents.")

    # Cohort sizes (PV-only vs PV+batt)
    x["new_adopters"] = pd.to_numeric(x.get("new_adopters"), errors="coerce").fillna(0.0).clip(lower=0.0)
    x["batt_adopters_added_this_year"] = pd.to_numeric(
        x.get("batt_adopters_added_this_year"), errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    x["pv_batt_n"] = np.minimum(x["batt_adopters_added_this_year"], x["new_adopters"])
    x["pv_only_n"] = (x["new_adopters"] - x["pv_batt_n"]).clip(lower=0.0)
    x["cohort_n"]  = x["pv_only_n"] + x["pv_batt_n"]

    # Per-adopter upfront
    x["system_capex_per_kw_combined"] = pd.to_numeric(x.get("system_capex_per_kw_combined"), errors="coerce")
    x["system_kw"] = pd.to_numeric(x.get("system_kw"), errors="coerce")
    x["per_upfront"] = x["system_capex_per_kw_combined"] * x["system_kw"] * .3

    # Cashflow arrays
    has_only = "cf_energy_value_pv_only" in x.columns
    has_batt = "cf_energy_value_pv_batt" in x.columns
    if not (has_only or has_batt):
        raise ValueError("Missing cf_energy_value_pv_only / cf_energy_value_pv_batt columns.")
    x["cf_pv_only"] = x["cf_energy_value_pv_only"].apply(lambda v: _parse_arr(v, lifetime_years)) if has_only else [[]]*len(x)
    x["cf_pv_batt"] = x["cf_energy_value_pv_batt"].apply(lambda v: _parse_arr(v, lifetime_years)) if has_batt else [[]]*len(x)

    y_min, y_max = int(x["year"].min()), int(x["year"].max())

    rows = []
    for scn, gscn in x.groupby("scenario", observed=True):
        for Y in range(y_min, y_max + 1):
            # pool adopters up to Y
            pool = gscn[gscn["year"] <= Y]
            if pool.empty: 
                continue

            # Build per-adopter earnings-to-date + weights for pooled cohorts
            earn_vals, earn_wts = [], []
            up_vals, up_wts = [], []

            for r in pool.itertuples(index=False):
                credit = lifetime_years
                # PV-only subcohort
                if r.pv_only_n > 0 and len(r.cf_pv_only):
                    earn_vals.append(float(np.nansum(r.cf_pv_only[:credit])))
                    earn_wts.append(float(r.pv_only_n))
                # PV+batt subcohort
                if r.pv_batt_n > 0 and len(r.cf_pv_batt):
                    earn_vals.append(float(np.nansum(r.cf_pv_batt[:credit])))
                    earn_wts.append(float(r.pv_batt_n))
                # Upfront (entire cohort)
                if np.isfinite(r.per_upfront) and r.cohort_n > 0:
                    up_vals.append(float(r.per_upfront))
                    up_wts.append(float(r.cohort_n))

            med_earn = _wmedian(earn_vals, earn_wts) if len(earn_vals) else np.nan
            med_up   = _wmedian(up_vals, up_wts)     if len(up_vals)   else np.nan

            rows.append((Y, scn, med_earn, med_up))

    if not rows:
        raise ValueError("No plottable data — check cashflow parsing and cohort sizes.")
    out = pd.DataFrame(rows, columns=["year","scenario","med_earn_to_date","med_upfront"])
    out["net_savings_per_adopter_median_to_date"] = out["med_earn_to_date"] - out["med_upfront"]

    # Labels/order & plot
    out["scenario"] = out["scenario"].map({"baseline":"Business-as-usual","policy":"$1/watt"}).fillna(out["scenario"])
    out = out[out["scenario"].isin(["Business-as-usual","$1/watt"])].copy()
    out["year"] = out["year"].astype(int)
    out = out.sort_values(["year","scenario"]).reset_index(drop=True)

    plt.rcParams["font.family"] = "Cabin"
    plt.figure(figsize=(10, 5), constrained_layout=True)
    ax = sns.barplot(
        data=out, x="year", y="net_savings_per_adopter_median_to_date",
        hue="scenario", hue_order=["Business-as-usual","$1/watt"],
        palette=["#a2e0fc","#1bb3ef"]
    )
    ax.set_xlabel("")
    ax.set_ylabel("USD")
    ax.grid(False)
    plt.legend(title=None, frameon=False)
    plt.title(title)
    plt.show()

    return out

def bar_us_net_savings_median_2040_from_agents(
    warehouse=None, *, agents=None,
    year: int | None = None,             # if None -> use max year in data
    lifetime_years: int = 25,
    title: str = "",
    cash_portion_fraction: float = 0.30,  # 30% cash, 70% debt
) -> "pd.DataFrame":
    """
    Plot weighted *average* per-adopter net savings (Baseline vs Policy) for all cohorts
    with year <= `year` (cumulative up to that point). Net is computed per row first,
    then averaged across rows with weights = `new_adopters`.

    Per-row subcohort nets (L = lifetime_years):
      PV-only:
        net_only = sum(cf_energy_value_pv_only[:L])
                   - sum(cf_debt_payment_total_pv_only[:L])
                   - cash_portion_fraction * (system_kw * system_capex_per_kw_combined)
      PV+batt:
        net_batt = sum(cf_energy_value_pv_batt[:L])
                   - sum(cf_debt_payment_total_pv_batt[:L])
                   - cash_portion_fraction * (system_kw * system_capex_per_kw_combined)
                   - cash_portion_fraction * (batt_kwh * batt_capex_per_kwh_combined)

    Row blended per-adopter net:
      pv_batt_n = min(batt_adopters_added_this_year, new_adopters)
      pv_only_n = max(new_adopters - pv_batt_n, 0)
      cohort_n  = pv_only_n + pv_batt_n  (== new_adopters, clipped at 0)

      row_net_per_adopter =
          (pv_only_n*net_only + pv_batt_n*net_batt) / cohort_n   (if cohort_n>0, else 0)

    Scenario average (cumulative to `year`):
      avg_net_per_adopter =
          sum(row_net_per_adopter * new_adopters) / sum(new_adopters)

    Returns tidy DataFrame:
      ['scenario','avg_net_savings_per_adopter','total_adopters_used']
    """
    import numpy as np, pandas as pd, json
    import matplotlib.pyplot as plt

    # --- Resolve agents ---
    if agents is None:
        if warehouse is None or not hasattr(warehouse, "agents"):
            raise ValueError("Provide `agents` or a `warehouse` with `.agents`.")
        agents = warehouse.agents
    x = agents.copy()

    # --- Parse / filter years ---
    x["scenario"] = x["scenario"].astype(str).str.lower().str.strip()
    x["year"] = pd.to_numeric(x.get("year"), errors="coerce")
    x = x[x["year"].notna()].copy()
    if x.empty:
        raise ValueError("No valid adoption years in agents.")
    if year is None:
        year = int(x["year"].max())
    x = x[x["year"] <= int(year)].copy()
    if x.empty:
        raise ValueError(f"No adopters at or before {year}.")

    # --- Cohort sizes & weights ---
    x["new_adopters"] = pd.to_numeric(x.get("new_adopters"), errors="coerce").fillna(0.0).clip(lower=0.0)
    x["batt_adopters_added_this_year"] = pd.to_numeric(x.get("batt_adopters_added_this_year"), errors="coerce").fillna(0.0).clip(lower=0.0)
    x["pv_batt_n"] = np.minimum(x["batt_adopters_added_this_year"], x["new_adopters"])
    x["pv_only_n"] = (x["new_adopters"] - x["pv_batt_n"]).clip(lower=0.0)
    x["cohort_n"]  = (x["pv_only_n"] + x["pv_batt_n"]).astype(float)

    # drop rows with no adopters
    x = x[x["cohort_n"] > 0].copy()
    if x.empty:
        raise ValueError("All rows have zero adopters after filtering.")

    # --- Capex (PV & Battery) for cash portions ---
    x["system_capex_per_kw_combined"] = pd.to_numeric(x.get("system_capex_per_kw_combined"), errors="coerce")
    x["system_kw"] = pd.to_numeric(x.get("system_kw"), errors="coerce")
    x["batt_kwh"]  = pd.to_numeric(x.get("batt_kwh"), errors="coerce")
    x["batt_capex_per_kwh_combined"] = pd.to_numeric(x.get("batt_capex_per_kwh_combined"), errors="coerce")

    pv_capex   = (x["system_capex_per_kw_combined"] * x["system_kw"]).fillna(0.0)
    batt_capex = (x["batt_kwh"] * x["batt_capex_per_kwh_combined"]).fillna(0.0)

    pv_cash   = cash_portion_fraction * pv_capex
    batt_cash = cash_portion_fraction * batt_capex

    # --- Helpers for arrays ---
    def _arr(v, L):
        if isinstance(v, np.ndarray): return v.astype(float)[:L]
        if isinstance(v, (list, tuple)): return np.asarray(v, float)[:L]
        if isinstance(v, str):
            s = v.strip()
            try:
                if s.startswith("{") and s.endswith("}"): s = "[" + s[1:-1] + "]"
                return np.asarray(json.loads(s), float)[:L]
            except Exception:
                return np.array([], float)
        return np.array([], float)

    L = int(lifetime_years)

    # Sums of arrays (treat missing as 0)
    ev_only = x.get("cf_energy_value_pv_only", pd.Series([[]]*len(x))).apply(lambda v: float(np.nansum(_arr(v, L))))
    ev_batt = x.get("cf_energy_value_pv_batt", pd.Series([[]]*len(x))).apply(lambda v: float(np.nansum(_arr(v, L))))
    db_only = x.get("cf_debt_payment_total_pv_only", pd.Series([[]]*len(x))).apply(lambda v: float(np.nansum(_arr(v, L))))
    db_batt = x.get("cf_debt_payment_total_pv_batt", pd.Series([[]]*len(x))).apply(lambda v: float(np.nansum(_arr(v, L))))

    # Per-row subcohort nets
    net_only = ev_only - db_only - pv_cash
    net_batt = ev_batt - db_batt - pv_cash - batt_cash

    # Per-row blended per-adopter net (PV-only vs PV+batt)
    blended = (x["pv_only_n"] * net_only + x["pv_batt_n"] * net_batt) / x["cohort_n"]
    x["row_net_per_adopter"] = blended.fillna(0.0)

    # Scenario weighted average across all rows up to `year` (weights: new_adopters == cohort_n)
    def _wavg(g: pd.DataFrame) -> pd.Series:
        w = g["cohort_n"].to_numpy(float)
        v = g["row_net_per_adopter"].to_numpy(float)
        W = float(w.sum())
        avg = float((w * v).sum() / W) if W > 0 else 0.0
        return pd.Series({"avg_net_savings_per_adopter": avg, "total_adopters_used": W})
    
    # --- Weighted median helper ---
    def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        values = np.asarray(values, float)
        weights = np.asarray(weights, float)
        m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        values = values[m]
        weights = weights[m]
        if values.size == 0:
            return 0.0
        order = np.argsort(values)
        v = values[order]
        w = weights[order]
        cw = np.cumsum(w)
        cutoff = 0.5 * cw[-1]
        idx = int(np.searchsorted(cw, cutoff, side="left"))
        return float(v[min(idx, v.size - 1)])

    # Scenario weighted median across all rows up to `year` (weights: cohort_n)
    def _wmed(g: "pd.DataFrame") -> "pd.Series":
        w = g["cohort_n"].to_numpy(float)
        v = g["row_net_per_adopter"].to_numpy(float)
        W = float(w.sum())
        med = _weighted_median(v, w) if W > 0 else 0.0
        return pd.Series({"median_net_savings_per_adopter": med, "total_adopters_used": W})

    out = (x.groupby("scenario", observed=True)
             .apply(_wmed)
             .reset_index())

    # Relabel scenarios & order bars
    label_map = {"baseline": "Business-as-usual", "policy": "$1/watt"}
    out["scenario"] = out["scenario"].map(label_map).fillna(out["scenario"])
    out = out[out["scenario"].isin(["Business-as-usual", "$1/watt"])].copy()
    out = out.set_index("scenario").loc[["Business-as-usual", "$1/watt"]].reset_index()

    # --- Plot ---
    plt.rcParams["font.family"] = "Cabin"
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    vals = out["median_net_savings_per_adopter"].to_numpy(float)
    ax.bar(out["scenario"].tolist(), vals, color=["#a2e0fc", "#1bb3ef"], width=0.6)
    ax.set_ylabel("USD")
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.annotate(f"${v:,.0f}", xy=(i, v), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
    plt.show()

    return out[["scenario","median_net_savings_per_adopter","total_adopters_used"]]




