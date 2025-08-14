# analysis_functions.py
from __future__ import annotations

import os
import re
import glob
import math
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =============================================================================
# Discovery & I/O
# =============================================================================

# Columns expected in per-state CSVs. Missing columns are added as NaN/0 where appropriate.
NEEDED_COLS = [
    "state_abbr",
    "scenario",
    "year",

    # Cohort/cumulative adopters:
    "new_adopters",           # new adopters in that year (cohort size)  <-- used for savings math
    "number_of_adopters",     # cumulative adopters in that year         <-- kept for plots

    # Savings and pricing:
    "first_year_elec_bill_savings",
    "avg_elec_price_cents_per_kwh",

    # Tech potential / market share:
    "customers_in_bin",
    "max_market_share",

    # PV / storage:
    "system_kw",
    "system_kw_cum",
    "batt_kwh_cum",
]

SCHEMA_RE = re.compile(r"^diffusion_results_(baseline|policy)_([a-z]{2})_", re.IGNORECASE)


def discover_state_dirs(root_dir: str) -> List[str]:
    """
    Return absolute paths of immediate subdirectories under `root_dir`.
    Each subdirectory is expected to be a state folder (e.g., 'CA', 'ny').
    """
    return sorted(
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )


def find_state_files(state_dir: str, run_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Locate per-state CSVs for baseline and policy.

    Search order:
      1) baseline_{run_id}.csv / policy_{run_id}.csv (if run_id is provided)
      2) any baseline*.csv / policy*.csv

    Returns:
        (baseline_csv_path or None, policy_csv_path or None)
    """
    if run_id:
        b = glob.glob(os.path.join(state_dir, f"baseline_{run_id}.csv"))
        p = glob.glob(os.path.join(state_dir, f"policy_{run_id}.csv"))
        if b and p:
            return b[0], p[0]

    # Fallback: any baseline*/policy* file
    b = glob.glob(os.path.join(state_dir, "baseline*.csv"))
    p = glob.glob(os.path.join(state_dir, "policy*.csv"))
    return (b[0] if b else None), (p[0] if p else None)


def _read_with_selected_cols(path: str) -> pd.DataFrame:
    """
    Read a CSV keeping only columns present from NEEDED_COLS; missing columns are added.
    If `scenario` or `state_abbr` is missing, they are inferred from file and directory names.
    """
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=NEEDED_COLS)

    header = pd.read_csv(path, nrows=0)
    present = [c for c in NEEDED_COLS if c in header.columns]
    df = pd.read_csv(path, usecols=present)

    # Add missing columns
    for c in NEEDED_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Infer scenario/state if absent
    filename = os.path.basename(path).lower()
    if df["scenario"].isna().all():
        df["scenario"] = "baseline" if "baseline" in filename else ("policy" if "policy" in filename else np.nan)

    if df["state_abbr"].isna().any():
        # Directory name is used as state; normalized to upper-case
        state = os.path.basename(os.path.dirname(path)).upper()
        df.loc[df["state_abbr"].isna(), "state_abbr"] = state

    return df


def load_state_df(state_dir: str, run_id: Optional[str] = None) -> pd.DataFrame:
    """
    Load baseline and policy CSVs for a single state and concatenate them.

    Returns:
        DataFrame with (at minimum) NEEDED_COLS; missing columns are filled with NaN/0.
    """
    b_csv, p_csv = find_state_files(state_dir, run_id)
    parts: List[pd.DataFrame] = []
    for p in (b_csv, p_csv):
        if p:
            parts.append(_read_with_selected_cols(p))
    if not parts:
        return pd.DataFrame(columns=NEEDED_COLS)

    df = pd.concat(parts, ignore_index=True)

    # Basic typing
    for col in ("year", "new_adopters", "number_of_adopters",
                "first_year_elec_bill_savings", "system_kw",
                "system_kw_cum", "batt_kwh_cum",
                "customers_in_bin", "max_market_share",
                "avg_elec_price_cents_per_kwh"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill non-negative quantities
    for col in ("new_adopters", "number_of_adopters", "first_year_elec_bill_savings",
                "customers_in_bin", "max_market_share",
                "system_kw_cum", "batt_kwh_cum"):
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df


def stream_combine_two_csvs(baseline_csv: Optional[str], policy_csv: Optional[str],
                            out_csv: str, chunksize: int = 200_000) -> None:
    """
    Stream-append baseline and policy CSVs into `out_csv`. Creates directories as needed.
    No attempt is made to de-duplicate records.
    """
    if not baseline_csv and not policy_csv:
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    wrote_header = False
    for path in (baseline_csv, policy_csv):
        if not path or not os.path.exists(path):
            continue
        for chunk in pd.read_csv(path, chunksize=chunksize):
            # Ensure required identifiers exist
            if "scenario" not in chunk.columns:
                chunk["scenario"] = "baseline" if "baseline" in os.path.basename(path).lower() else "policy"
            if "state_abbr" not in chunk.columns:
                chunk["state_abbr"] = os.path.basename(os.path.dirname(path)).upper()
            chunk.to_csv(out_csv, mode="w" if not wrote_header else "a", index=False, header=not wrote_header)
            wrote_header = True


# =============================================================================
# Savings & Aggregations
# =============================================================================

@dataclass
class SavingsConfig:
    """
    Configuration for bill savings aggregation.

    lifetime_years:
        Years of savings credited to each cohort when computing cohort lifetime totals.
    cap_to_horizon:
        If True, limit credited years for each cohort to the visible modeling horizon
        (i.e., min(lifetime_years, last_year - cohort_year + 1)).
    """
    lifetime_years: int = 25
    cap_to_horizon: bool = False


def compute_portfolio_and_cumulative_savings(
    df: pd.DataFrame,
    cfg: SavingsConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute portfolio-level bill savings by carrying cohorts forward.

    Definitions:
        - Cohort annual savings in the year of adoption:
              cohort_annual_savings = first_year_elec_bill_savings * new_adopters
          (assumes first-year savings is representative for all subsequent years)
        - Portfolio annual savings in year t:
              sum of cohort_annual_savings for all cohorts with cohort_year <= t
        - Cumulative bill savings through year t:
              cumulative sum of portfolio_annual_savings from start to t
        - Lifetime savings total (per state/scenario):
              sum over cohorts of (cohort_annual_savings * credited_years),
              where credited_years = cfg.lifetime_years or, if cap_to_horizon,
              min(cfg.lifetime_years, last_year - cohort_year + 1)

    Returns:
        (annual_portfolio_savings_df, cumulative_savings_df)
    """
    if df.empty:
        empty_annual = pd.DataFrame(columns=["state_abbr", "scenario", "year", "portfolio_annual_savings", "lifetime_savings_total"])
        empty_cum = pd.DataFrame(columns=["state_abbr", "scenario", "year", "cumulative_bill_savings", "lifetime_savings_total"])
        return empty_annual, empty_cum

    x = df.copy()
    x["new_adopters"] = pd.to_numeric(x.get("new_adopters", 0.0), errors="coerce").fillna(0.0)
    x["first_year_elec_bill_savings"] = pd.to_numeric(x.get("first_year_elec_bill_savings", 0.0), errors="coerce").fillna(0.0)

    # Cohort-level annual savings
    x["cohort_annual_savings"] = x["first_year_elec_bill_savings"] * x["new_adopters"]

    # Collapse to one row per (state, scenario, cohort_year)
    cohorts = (
        x.groupby(["state_abbr", "scenario", "year"], as_index=False)["cohort_annual_savings"]
         .sum()
         .rename(columns={"year": "cohort_year"})
         .sort_values(["state_abbr", "scenario", "cohort_year"])
    )

    # Determine year horizon; handle missing carefully
    year_vals = pd.to_numeric(df["year"], errors="coerce").dropna()
    if year_vals.empty:
        empty_annual = pd.DataFrame(columns=["state_abbr", "scenario", "year", "portfolio_annual_savings", "lifetime_savings_total"])
        empty_cum = pd.DataFrame(columns=["state_abbr", "scenario", "year", "cumulative_bill_savings", "lifetime_savings_total"])
        return empty_annual, empty_cum

    year_min = int(year_vals.min())
    year_max = int(year_vals.max())
    all_years = list(range(year_min, year_max + 1))

    annual_frames: List[pd.DataFrame] = []
    lifetime_frames: List[pd.DataFrame] = []

    for (state, scen), g in cohorts.groupby(["state_abbr", "scenario"]):
        mapping = dict(zip(g["cohort_year"], g["cohort_annual_savings"]))

        # Portfolio annual savings via prefix sum over years
        running = 0.0
        rows = []
        for y in all_years:
            running += mapping.get(y, 0.0)
            rows.append((state, scen, y, running))
        ann_df = pd.DataFrame(rows, columns=["state_abbr", "scenario", "year", "portfolio_annual_savings"])
        annual_frames.append(ann_df)

        # Lifetime totals per cohort
        lf = cfg.lifetime_years
        if cfg.cap_to_horizon:
            credited = {cy: max(0, min(lf, year_max - cy + 1)) for cy in g["cohort_year"]}
        else:
            credited = {cy: lf for cy in g["cohort_year"]}

        g_life = g.copy()
        g_life["lifetime_years_applied"] = g_life["cohort_year"].map(credited)
        g_life["lifetime_savings_for_cohort"] = g_life["cohort_annual_savings"] * g_life["lifetime_years_applied"]
        lifetime_frames.append(
            g_life.groupby(["state_abbr", "scenario"], as_index=False)["lifetime_savings_for_cohort"]
                 .sum()
                 .rename(columns={"lifetime_savings_for_cohort": "lifetime_savings_total"})
        )

    annual_portfolio = (
        pd.concat(annual_frames, ignore_index=True)
        if annual_frames else
        pd.DataFrame(columns=["state_abbr", "scenario", "year", "portfolio_annual_savings"])
    )

    lifetime_totals = (
        pd.concat(lifetime_frames, ignore_index=True)
          .groupby(["state_abbr", "scenario"], as_index=False)["lifetime_savings_total"].sum()
        if lifetime_frames else
        pd.DataFrame(columns=["state_abbr", "scenario", "lifetime_savings_total"])
    )

    # Cumulative bill savings through year
    if not annual_portfolio.empty:
        cumulative = (
            annual_portfolio.sort_values(["state_abbr", "scenario", "year"])
            .groupby(["state_abbr", "scenario"], as_index=False)
            .apply(lambda g: g.assign(cumulative_bill_savings=g["portfolio_annual_savings"].cumsum()))
            .reset_index(drop=True)
        )
    else:
        cumulative = pd.DataFrame(columns=["state_abbr", "scenario", "year", "cumulative_bill_savings"])

    # Attach lifetime totals for convenience
    annual_portfolio = annual_portfolio.merge(lifetime_totals, on=["state_abbr", "scenario"], how="left")
    cumulative = cumulative.merge(lifetime_totals, on=["state_abbr", "scenario"], how="left")

    return annual_portfolio, cumulative


def aggregate_state_metrics(df: pd.DataFrame, cfg: SavingsConfig) -> Dict[str, pd.DataFrame]:
    """
    Aggregate per-state metrics into compact frames for plotting and export.

    Returns:
        {
            "median_system_kw": DataFrame[state_abbr, year, scenario, median_system_kw],
            "totals": DataFrame[state_abbr, year, scenario, batt_kwh_cum, system_kw_cum, number_of_adopters],
            "tech_2040": DataFrame[state_abbr, scenario, number_of_adopters, customers_in_bin, percent_tech_potential],
            "portfolio_annual_savings": DataFrame[state_abbr, scenario, year, portfolio_annual_savings, lifetime_savings_total],
            "cumulative_bill_savings": DataFrame[state_abbr, scenario, year, cumulative_bill_savings, lifetime_savings_total],
            "lifetime_totals": DataFrame[state_abbr, scenario, lifetime_savings_total],
            "avg_price_2026_model": DataFrame[state_abbr, avg_elec_price_cents_per_kwh],
            "market_share_reached": DataFrame[state_abbr, year, scenario, market_potential, market_reached, market_share_reached],
        }
    """
    if df.empty:
        return {
            "median_system_kw": pd.DataFrame(),
            "totals": pd.DataFrame(),
            "tech_2040": pd.DataFrame(),
            "portfolio_annual_savings": pd.DataFrame(),
            "cumulative_bill_savings": pd.DataFrame(),
            "lifetime_totals": pd.DataFrame(),
            "avg_price_2026_model": pd.DataFrame(),
            "market_share_reached": pd.DataFrame(),
        }

    x = df.copy()

    # Enforce numeric types and defaults
    for col in (
        "new_adopters", "number_of_adopters", "first_year_elec_bill_savings",
        "system_kw", "system_kw_cum", "batt_kwh_cum",
        "customers_in_bin", "max_market_share", "avg_elec_price_cents_per_kwh"
    ):
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")

    x["new_adopters"] = x.get("new_adopters", 0.0).fillna(0.0)
    x["number_of_adopters"] = x.get("number_of_adopters", 0.0).fillna(0.0)
    x["first_year_elec_bill_savings"] = x.get("first_year_elec_bill_savings", 0.0).fillna(0.0)
    x["customers_in_bin"] = x.get("customers_in_bin", 0.0).fillna(0.0)
    x["max_market_share"] = x.get("max_market_share", 0.0).fillna(0.0)

    # Median PV system size by state/year/scenario
    median_kw = (
        x.groupby(["state_abbr", "year", "scenario"], as_index=False)["system_kw"]
         .median()
         .rename(columns={"system_kw": "median_system_kw"})
    )

    # Yearly totals for cumulative series and adopters
    totals = (
        x.groupby(["state_abbr", "year", "scenario"], as_index=False)
         .agg(
             batt_kwh_cum=("batt_kwh_cum", "sum"),
             system_kw_cum=("system_kw_cum", "sum"),
             number_of_adopters=("number_of_adopters", "sum"),
         )
    )

    # Technical potential reached in 2040
    tech_2040_src = x.loc[x["year"] == 2040, ["state_abbr", "scenario", "number_of_adopters", "customers_in_bin"]]
    tech_2040 = tech_2040_src.groupby(["state_abbr", "scenario"], as_index=False).sum()
    if not tech_2040.empty:
        tech_2040["percent_tech_potential"] = np.where(
            tech_2040["customers_in_bin"] > 0,
            100.0 * tech_2040["number_of_adopters"] / tech_2040["customers_in_bin"],
            np.nan,
        )

    # Portfolio-annual and cumulative bill savings (uses new_adopters)
    portfolio_annual, cumulative_savings = compute_portfolio_and_cumulative_savings(x, cfg)
    lifetime_totals = (
        portfolio_annual[["state_abbr", "scenario", "lifetime_savings_total"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Baseline model average price in 2026 (if present)
    if "avg_elec_price_cents_per_kwh" in x.columns and x["avg_elec_price_cents_per_kwh"].notna().any():
        price_2026 = x[(x["year"] == 2026) & (x["scenario"] == "baseline")]
        avg_price_2026_model = (
            price_2026.groupby("state_abbr", as_index=False)["avg_elec_price_cents_per_kwh"].mean()
        )
    else:
        avg_price_2026_model = pd.DataFrame(columns=["state_abbr", "avg_elec_price_cents_per_kwh"])

    # Market share reached (using cumulative number_of_adopters for that year)
    x["market_potential"] = x["customers_in_bin"] * x["max_market_share"]
    market_share = (
        x.groupby(["state_abbr", "year", "scenario"], as_index=False)
         .agg(
             market_potential=("market_potential", "sum"),
             market_reached=("number_of_adopters", "sum"),
         )
    )
    market_share["market_share_reached"] = np.where(
        market_share["market_potential"] > 0,
        market_share["market_reached"] / market_share["market_potential"],
        np.nan,
    )

    return {
        "median_system_kw": median_kw,
        "totals": totals,
        "tech_2040": tech_2040,
        "portfolio_annual_savings": portfolio_annual,
        "cumulative_bill_savings": cumulative_savings,
        "lifetime_totals": lifetime_totals,
        "avg_price_2026_model": avg_price_2026_model,
        "market_share_reached": market_share,
    }


# =============================================================================
# Parallel processing across states
# =============================================================================

def _process_one_state(args) -> Dict[str, pd.DataFrame]:
    """Private worker wrapper for multiprocessing."""
    state_dir, run_id, cfg = args
    df = load_state_df(state_dir, run_id)
    return aggregate_state_metrics(df, cfg)


def process_all_states(
    root_dir: str,
    run_id: Optional[str] = None,
    cfg: SavingsConfig = SavingsConfig(),
    n_jobs: int = 1
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate small, plot-ready DataFrames across all states.

    Args:
        root_dir:
            Parent directory containing per-state subfolders.
        run_id:
            If provided, prefer baseline_{run_id}.csv / policy_{run_id}.csv in each state folder.
        cfg:
            SavingsConfig controlling lifetime savings behavior.
        n_jobs:
            Number of parallel worker processes across states. For I/O-bound workloads on SSDs,
            4–8 is often a good starting point. On HDDs/USB drives, reduce to avoid thrashing.

    Returns:
        Dictionary of concatenated DataFrames keyed by metric name.
    """
    state_dirs = discover_state_dirs(root_dir)
    if not state_dirs:
        return {
            "median_system_kw": pd.DataFrame(),
            "totals": pd.DataFrame(),
            "tech_2040": pd.DataFrame(),
            "portfolio_annual_savings": pd.DataFrame(),
            "cumulative_bill_savings": pd.DataFrame(),
            "lifetime_totals": pd.DataFrame(),
            "avg_price_2026_model": pd.DataFrame(),
            "market_share_reached": pd.DataFrame(),
        }

    tasks = [(sd, run_id, cfg) for sd in state_dirs]
    outputs: List[Dict[str, pd.DataFrame]] = []

    n_jobs = max(1, min(n_jobs, max(1, (cpu_count() or 2) - 1)))
    if n_jobs == 1:
        outputs = [_process_one_state(t) for t in tasks]
    else:
        # On macOS, the start method is "spawn" by default; ensure top-level functions only.
        with Pool(processes=n_jobs) as pool:
            for result in pool.imap_unordered(_process_one_state, tasks):
                outputs.append(result)

    # Concatenate by metric key
    merged: Dict[str, List[pd.DataFrame]] = {}
    for out in outputs:
        for key, df in out.items():
            merged.setdefault(key, []).append(df)

    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in merged.items()}


# =============================================================================
# Plot helpers (faceting)
# =============================================================================

def facet_lines_by_state(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    xticks: Iterable[int] = (2026, 2030, 2035, 2040),
    height: float = 3.5,
    col_wrap: int = 4,
    sharey: bool = False
) -> None:
    """
    Faceted line plot by state comparing Baseline vs Policy.

    Args:
        df: Tidy DataFrame with columns ['state_abbr','year','scenario', y_col].
        y_col: Value column to plot.
        ylabel: Y-axis label.
        title: Figure title.
        xticks: X-axis ticks (years).
        height: Facet height (inches).
        col_wrap: Number of facets per row.
        sharey: Share Y axis across facets (False by default).
    """
    if df.empty:
        return
    sns.set_context("talk", rc={"lines.linewidth": 2})
    g = sns.FacetGrid(df, col="state_abbr", col_wrap=col_wrap, height=height, sharey=sharey)
    g.map_dataframe(sns.lineplot, x="year", y=y_col, hue="scenario", marker="o")
    g.set_titles("{col_name}")
    g.set_axis_labels("Year", ylabel)
    g.set(xticks=list(xticks))
    g.add_legend()
    g.fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


def bar_tech_potential_2040(tech_agg: pd.DataFrame, title: str = "Solar Technical Potential Reached in 2040") -> None:
    """
    Grouped bar plot of percent technical potential, sorted by policy scenario.
    """
    if tech_agg.empty:
        return
    order = (
        tech_agg[tech_agg["scenario"] == "policy"]
        .sort_values("percent_tech_potential", ascending=False)["state_abbr"]
        .tolist()
    )
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(
        data=tech_agg, x="state_abbr", y="percent_tech_potential",
        hue="scenario", order=order, errorbar=None
    )
    for container in ax.containers:
        for bar in container:
            h = bar.get_height()
            if h > 0 and not math.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h - 1.5,
                    f"{h:.1f}%",
                    ha="center",
                    va="top",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
    plt.title(f"{title} (Sorted by Policy %)")
    plt.ylabel("Percent of Technical Potential (%)")
    plt.xlabel("State")
    plt.xticks(rotation=45)
    plt.ylim(0, tech_agg["percent_tech_potential"].max() * 1.1)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    # I/O
    "discover_state_dirs",
    "find_state_files",
    "load_state_df",
    "stream_combine_two_csvs",

    # Config & processing
    "SavingsConfig",
    "aggregate_state_metrics",
    "compute_portfolio_and_cumulative_savings",
    "process_all_states",

    # Plotting
    "facet_lines_by_state",
    "bar_tech_potential_2040",
]
