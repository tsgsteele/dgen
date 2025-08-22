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


def find_state_files(state_dir: str, run_id: Optional[str] = None, strict_run_id: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """
    Locate per-state CSVs for baseline and policy.

    If run_id is provided and strict_run_id=True (default), return only files that match
    baseline_{run_id}.csv and policy_{run_id}.csv; otherwise return (None, None) for missing.
    If strict_run_id=False, falls back to any baseline*/policy* when both exact matches are absent.
    """
    if run_id:
        b = glob.glob(os.path.join(state_dir, f"baseline_{run_id}.csv"))
        p = glob.glob(os.path.join(state_dir, f"policy_{run_id}.csv"))
        if strict_run_id:
            return (b[0] if b else None), (p[0] if p else None)
        if b and p:
            return b[0], p[0]

    # Fallback (only used when strict_run_id=False or no run_id provided)
    b = glob.glob(os.path.join(state_dir, "baseline*.csv"))
    p = glob.glob(os.path.join(state_dir, "policy*.csv"))
    return (b[0] if b else None), (p[0] if p else None)


def _read_with_selected_cols(path: str) -> pd.DataFrame:
    """
    Read a CSV keeping only columns present from NEEDED_COLS; missing columns are added.
    If `scenario` or `state_abbr` is missing, they are inferred from file and directory names.
    Robust to empty/zero-byte CSVs.
    """
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=NEEDED_COLS)

    try:
        header = pd.read_csv(path, nrows=0)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=NEEDED_COLS)

    present = [c for c in NEEDED_COLS if c in header.columns]

    try:
        df = pd.read_csv(path, usecols=present) if present else pd.DataFrame(columns=NEEDED_COLS)
    except (pd.errors.EmptyDataError, ValueError):
        # ValueError can occur if usecols don’t match due to malformed header
        return pd.DataFrame(columns=NEEDED_COLS)

    # Add missing columns
    for c in NEEDED_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Infer scenario/state if absent
    filename = os.path.basename(path).lower()
    if df["scenario"].isna().all():
        df["scenario"] = "baseline" if "baseline" in filename else ("policy" if "policy" in filename else np.nan)

    if df["state_abbr"].isna().any():
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
    """
    lifetime_years: int = 25
    cap_to_horizon: bool = False


def compute_portfolio_and_cumulative_savings(
    df: pd.DataFrame,
    cfg: SavingsConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute portfolio-level bill savings by carrying cohorts forward.
    """
    if df.empty:
        empty_annual = pd.DataFrame(columns=["state_abbr", "scenario", "year", "portfolio_annual_savings", "lifetime_savings_total"])
        empty_cum = pd.DataFrame(columns=["state_abbr", "scenario", "year", "cumulative_bill_savings", "lifetime_savings_total"])
        return empty_annual, empty_cum

    x = df.copy()
    x["new_adopters"] = pd.to_numeric(x.get("new_adopters", 0.0), errors="coerce").fillna(0.0)
    x["first_year_elec_bill_savings"] = pd.to_numeric(x.get("first_year_elec_bill_savings", 0.0), errors="coerce").fillna(0.0)

    x["cohort_annual_savings"] = x["first_year_elec_bill_savings"] * x["new_adopters"]

    cohorts = (
        x.groupby(["state_abbr", "scenario", "year"], as_index=False)["cohort_annual_savings"]
         .sum()
         .rename(columns={"year": "cohort_year"})
         .sort_values(["state_abbr", "scenario", "cohort_year"])
    )

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

        running = 0.0
        rows = []
        for y in all_years:
            running += mapping.get(y, 0.0)
            rows.append((state, scen, y, running))
        ann_df = pd.DataFrame(rows, columns=["state_abbr", "scenario", "year", "portfolio_annual_savings"])
        annual_frames.append(ann_df)

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

    if not annual_portfolio.empty:
        annual_portfolio = annual_portfolio.sort_values(["state_abbr", "scenario", "year"])
        cumulative = annual_portfolio.copy()
        cumulative["cumulative_bill_savings"] = (
            cumulative.groupby(["state_abbr", "scenario"], observed=True)["portfolio_annual_savings"].cumsum()
        )
    else:
        cumulative = pd.DataFrame(columns=["state_abbr", "scenario", "year", "cumulative_bill_savings"])

    annual_portfolio = annual_portfolio.merge(lifetime_totals, on=["state_abbr", "scenario"], how="left")
    cumulative = cumulative.merge(lifetime_totals, on=["state_abbr", "scenario"], how="left")

    return annual_portfolio, cumulative


def aggregate_state_metrics(df: pd.DataFrame, cfg: SavingsConfig) -> Dict[str, pd.DataFrame]:
    """
    Aggregate per-state metrics into compact frames for plotting and export.
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

    x = df

    x["new_adopters"] = x.get("new_adopters", 0.0).fillna(0.0)
    x["number_of_adopters"] = x.get("number_of_adopters", 0.0).fillna(0.0)
    x["first_year_elec_bill_savings"] = x.get("first_year_elec_bill_savings", 0.0).fillna(0.0)
    x["customers_in_bin"] = x.get("customers_in_bin", 0.0).fillna(0.0)
    x["max_market_share"] = x.get("max_market_share", 0.0).fillna(0.0)

    median_kw = (
        x.groupby(["state_abbr", "year", "scenario"], observed=True)["system_kw"]
        .quantile(0.5, interpolation="linear")
        .reset_index(name="median_system_kw")
    )

    totals = (
        x.groupby(["state_abbr", "year", "scenario"], as_index=False)
         .agg(
             batt_kwh_cum=("batt_kwh_cum", "sum"),
             system_kw_cum=("system_kw_cum", "sum"),
             number_of_adopters=("number_of_adopters", "sum"),
         )
    )

    tech_2040_src = x.loc[x["year"] == 2040, ["state_abbr", "scenario", "number_of_adopters", "customers_in_bin"]]
    tech_2040 = tech_2040_src.groupby(["state_abbr", "scenario"], as_index=False).sum()
    if not tech_2040.empty:
        tech_2040["percent_tech_potential"] = np.where(
            tech_2040["customers_in_bin"] > 0,
            100.0 * tech_2040["number_of_adopters"] / tech_2040["customers_in_bin"],
            np.nan,
        )

    portfolio_annual, cumulative_savings = compute_portfolio_and_cumulative_savings(x, cfg)
    lifetime_totals = (
        portfolio_annual[["state_abbr", "scenario", "lifetime_savings_total"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if (
        "avg_elec_price_cents_per_kwh" in x.columns
        and "customers_in_bin" in x.columns
        and x["avg_elec_price_cents_per_kwh"].notna().any()
    ):
        price_2026 = x[(x["year"] == 2026) & (x["scenario"] == "baseline")].copy()
        price_2026["customers_in_bin"] = price_2026["customers_in_bin"].fillna(0.0).clip(lower=0.0)

        def _weighted_avg(g: pd.DataFrame) -> float:
            w = g["customers_in_bin"].to_numpy()
            v = g["avg_elec_price_cents_per_kwh"].to_numpy()
            ws = w.sum()
            return float(np.average(v, weights=w)) if ws > 0 else np.nan

        avg_price_2026_model = (
            price_2026.groupby("state_abbr", as_index=False)
            .apply(_weighted_avg)
            .rename(columns={None: "avg_elec_price_cents_per_kwh"})
        )
    else:
        avg_price_2026_model = pd.DataFrame(columns=["state_abbr", "avg_elec_price_cents_per_kwh"])

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
    state_dir, run_id, cfg = args
    df = load_state_df(state_dir, run_id)
    return aggregate_state_metrics(df, cfg)


def process_all_states(
    root_dir: str,
    run_id: Optional[str] = None,
    cfg: SavingsConfig = SavingsConfig(),
    n_jobs: int = 1,
    states: Optional[Iterable[str]] = None,   # <— NEW: optional state filter
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate small, plot-ready DataFrames across all states.

    Args:
        root_dir: Root folder containing per-state subdirectories (e.g., 'CA', 'ny', ...).
        run_id: If provided, only pick baseline_{run_id}.csv / policy_{run_id}.csv.
        cfg: SavingsConfig for savings computations.
        n_jobs: Parallel workers (processes).
        states: Optional iterable of 2-letter state codes to include (case-insensitive).
    """
    state_dirs = discover_state_dirs(root_dir)

    # Optional filter by explicit state codes (case-insensitive)
    if states:
        wanted = {s.strip().upper() for s in states if s and s.strip()}
        state_dirs = [
            sd for sd in state_dirs
            if os.path.basename(sd).upper() in wanted
        ]

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
        with Pool(processes=n_jobs) as pool:
            for result in pool.imap_unordered(_process_one_state, tasks):
                outputs.append(result)

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
    Faceted line plot by state comparing Baseline vs Policy, with end-of-horizon annotations
    at 2040 (or the last available year in that state).
    """
    if df.empty:
        return

    # Formatting helper for labels
    def _fmt(v: float) -> str:
        m = y_col
        if m == "number_of_adopters":
            return f"{v/1e6:.1f}M"
        if m == "system_kw_cum":
            return f"{v/1e6:.1f} GW"     # kW -> GW
        if m == "batt_kwh_cum":
            return f"{v/1e6:.1f} GWh"    # kWh -> GWh
        if m == "cumulative_bill_savings":
            return f"${v/1e9:.1f}B"
        if m == "portfolio_annual_savings":
            return f"${v/1e9:.1f}B/yr"
        if m in ("median_system_kw", "system_kw"):
            return f"{v:.1f} kW"
        if m == "market_share_reached":
            return f"{v*100:.1f}%"
        return f"{v:.2g}"

    sns.set_context("talk", rc={"lines.linewidth": 2})
    g = sns.FacetGrid(df, col="state_abbr", col_wrap=col_wrap, height=height, sharey=sharey)

    # draw lines
    g.map_dataframe(sns.lineplot, x="year", y=y_col, hue="scenario", marker="o")
    g.set_titles("{col_name}")
    g.set_axis_labels("Year", ylabel)
    g.set(xticks=list(xticks))
    g.add_legend()
    g.fig.suptitle(title, y=1.02)

    # Pick the preferred annotation year globally (fall back to max available per state)
    global_years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
    preferred_year = 2040 if (len(global_years) and 2040 in set(global_years)) else (int(global_years.max()) if len(global_years) else None)

    # annotate per facet (per state)
    if preferred_year is not None:
        for state, ax in g.axes_dict.items():
            sdf = df[(df["state_abbr"] == state)].copy()
            if sdf.empty or y_col not in sdf.columns:
                continue
            sdf = sdf.dropna(subset=[y_col, "year"])

            # pick year for this state
            years = sdf["year"].astype(int).unique()
            end_year = preferred_year if preferred_year in years else int(sdf["year"].max())

            # collect end points for each scenario present
            end_points = []
            for scen, sg in sdf.groupby("scenario"):
                sg = sg.sort_values("year")
                # prefer the exact year; else the latest <= end_year
                g_end = sg[sg["year"] == end_year]
                if g_end.empty:
                    g_end = sg[sg["year"] <= end_year].tail(1)
                if not g_end.empty:
                    x_end = float(g_end["year"].iloc[-1])
                    y_end = float(g_end[y_col].iloc[-1])
                    end_points.append((scen, x_end, y_end))

            if not end_points:
                continue

            # tiny vertical jitter to avoid overlap if values are very close
            sv = sdf[y_col].to_numpy()
            vmin = float(np.nanmin(sv)) if sv.size else 0.0
            vmax = float(np.nanmax(sv)) if sv.size else 1.0
            yrange = max(1.0, vmax - vmin)
            sep_needed = yrange * 0.01  # 1% of the local range

            used_y = []
            for scen, x_end, y_end in sorted(end_points, key=lambda t: t[2]):
                offset_pts = 0
                for uy in used_y:
                    if abs(y_end - uy) < sep_needed:
                        offset_pts += 6  # nudge down a bit
                used_y.append(y_end)

                ax.annotate(
                    _fmt(y_end),
                    xy=(x_end, y_end),
                    xytext=(6, offset_pts),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

            # # pad x-limits slightly to ensure labels at the right edge aren't clipped
            # xmin, xmax = ax.get_xlim()
            # ax.set_xlim(xmin, xmax + 0.4)

    # Avoid tight_layout warnings by relying on default layout
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
    plt.figure(figsize=(14, 6), constrained_layout=True)
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

    # US deltas & totals
    "build_national_deltas",
    "facet_lines_all_states_delta",
    "build_national_totals",
    "facet_lines_national_totals",
]

# =============================================================================
# National deltas: Policy − Baseline (summed across states)
# =============================================================================

def build_national_deltas(outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Return tidy national deltas: ['year','metric','value'], where
    value = (policy − baseline).
    """
    years_seen = []
    for k in ("totals", "portfolio_annual_savings", "cumulative_bill_savings", "market_share_reached"):
        dfk = outputs.get(k, pd.DataFrame())
        if not dfk.empty and "year" in dfk.columns:
            years_seen.append(dfk["year"])
    if not years_seen:
        return pd.DataFrame(columns=["year", "metric", "value"])

    year_min = int(pd.concat(years_seen).min())
    year_max = int(pd.concat(years_seen).max())
    all_years = pd.Index(range(year_min, year_max + 1), name="year")

    pieces: List[pd.DataFrame] = []

    def _delta_from_series(df: pd.DataFrame, value_col: str, cumulative: bool) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["year", "value"])
        def _fill(g: pd.DataFrame) -> pd.DataFrame:
            s = g.set_index("year")[[value_col]].reindex(all_years)
            s = (s.ffill() if cumulative else s.fillna(0.0)).fillna(0.0)
            s = s.reset_index()
            s["state_abbr"] = g["state_abbr"].iloc[0]
            s["scenario"] = g["scenario"].iloc[0]
            return s
        filled = (
            df.groupby(["state_abbr", "scenario"], observed=True, as_index=False)
              .apply(_fill)
              .reset_index(drop=True)
        )
        nat = filled.groupby(["year", "scenario"], as_index=False)[value_col].sum()
        piv = nat.pivot(index="year", columns="scenario", values=value_col)
        if "policy" not in piv.columns or "baseline" not in piv.columns:
            return pd.DataFrame(columns=["year", "value"])
        out = piv["policy"].sub(piv["baseline"]).rename("value").reset_index()
        return out

    totals = outputs.get("totals", pd.DataFrame())
    for col, metric in [
        ("number_of_adopters", "number_of_adopters"),
        ("system_kw_cum", "system_kw_cum"),
        ("batt_kwh_cum", "batt_kwh_cum"),
    ]:
        if not totals.empty and col in totals.columns:
            d = _delta_from_series(totals[["state_abbr","year","scenario",col]].copy(), col, cumulative=True)
            if not d.empty:
                d["metric"] = metric
                pieces.append(d)

    pas = outputs.get("portfolio_annual_savings", pd.DataFrame())
    if not pas.empty and "portfolio_annual_savings" in pas.columns:
        d = _delta_from_series(pas[["state_abbr","year","scenario","portfolio_annual_savings"]].copy(),
                               "portfolio_annual_savings", cumulative=False)
        if not d.empty:
            d["metric"] = "portfolio_annual_savings"
            pieces.append(d)

    cbs = outputs.get("cumulative_bill_savings", pd.DataFrame())
    if not cbs.empty and "cumulative_bill_savings" in cbs.columns:
        d = _delta_from_series(cbs[["state_abbr","year","scenario","cumulative_bill_savings"]].copy(),
                               "cumulative_bill_savings", cumulative=True)
        if not d.empty:
            d["metric"] = "cumulative_bill_savings"
            pieces.append(d)

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["year","metric","value"])


def facet_lines_all_states_delta(
    outputs: Dict[str, pd.DataFrame],
    metrics: Optional[Iterable[str]] = None,
    xticks: Iterable[int] = (2026, 2030, 2035, 2040),
    title: str = "Policy − Baseline (U.S. Totals)",
    ncols: int = 3,
) -> None:
    """
    Simple subplots for national deltas (policy − baseline). No annotations.
    """
    df = build_national_deltas(outputs)
    if df.empty:
        return

    if metrics:
        df = df[df["metric"].isin(set(metrics))]
        if df.empty:
            return

    nice_titles = {
        "number_of_adopters": "Δ Cumulative Adopters",
        "system_kw_cum": "Δ Cumulative PV (kW)",
        "batt_kwh_cum": "Δ Cumulative Storage (kWh)",
        "portfolio_annual_savings": "Δ Portfolio Bill Savings ($/yr)",
        "cumulative_bill_savings": "Δ Cumulative Bill Savings ($)",
    }

    metric_list = list(df["metric"].unique())
    n = len(metric_list)
    ncols = max(1, min(ncols, n))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.3*nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for i, m in enumerate(metric_list):
        ax = axes[i]
        d = df[df["metric"] == m].sort_values("year")
        ax.plot(d["year"], d["value"], marker="o")
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xticks(list(xticks))
        ax.set_title(nice_titles.get(m, m))
        ax.set_xlabel("Year")
        ax.set_ylabel("Policy − Baseline")

    # Hide any extra axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, y=1.05)
    plt.show()


# =============================================================================
# National totals: Baseline vs Policy (summed across states)
# =============================================================================

def build_national_totals(outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build tidy national totals across states, per scenario:
        ['year', 'scenario', 'metric', 'value']
    """
    years_seen = []
    for k in ("totals", "portfolio_annual_savings", "cumulative_bill_savings"):
        dfk = outputs.get(k, pd.DataFrame())
        if not dfk.empty and "year" in dfk.columns:
            years_seen.append(dfk["year"])
    if not years_seen:
        return pd.DataFrame(columns=["year", "scenario", "metric", "value"])

    year_min = int(pd.concat(years_seen).min())
    year_max = int(pd.concat(years_seen).max())
    all_years = pd.Index(range(year_min, year_max + 1), name="year")

    pieces: List[pd.DataFrame] = []

    def _sum_series(df: pd.DataFrame, value_col: str, cumulative: bool) -> pd.DataFrame:
        if df.empty or value_col not in df.columns:
            return pd.DataFrame(columns=["year", "scenario", "value"])
        def _fill(g: pd.DataFrame) -> pd.DataFrame:
            s = g.set_index("year")[[value_col]].reindex(all_years)
            s = (s.ffill() if cumulative else s.fillna(0.0)).fillna(0.0)
            s = s.reset_index()
            s["state_abbr"] = g["state_abbr"].iloc[0]
            s["scenario"] = g["scenario"].iloc[0]
            return s
        filled = (
            df.groupby(["state_abbr", "scenario"], observed=True, as_index=False)
              .apply(_fill)
              .reset_index(drop=True)
        )
        nat = filled.groupby(["year", "scenario"], as_index=False)[value_col].sum()
        nat = nat.rename(columns={value_col: "value"})
        return nat

    totals = outputs.get("totals", pd.DataFrame())
    if not totals.empty:
        for col in ("number_of_adopters", "system_kw_cum", "batt_kwh_cum"):
            s = _sum_series(totals[["state_abbr", "year", "scenario", col]].copy(), col, cumulative=True)
            if not s.empty:
                s["metric"] = col
                pieces.append(s)

    cbs = outputs.get("cumulative_bill_savings", pd.DataFrame())
    if not cbs.empty:
        s = _sum_series(cbs[["state_abbr", "year", "scenario", "cumulative_bill_savings"]].copy(),
                        "cumulative_bill_savings", cumulative=True)
        if not s.empty:
            s["metric"] = "cumulative_bill_savings"
            pieces.append(s)

    pas = outputs.get("portfolio_annual_savings", pd.DataFrame())
    if not pas.empty:
        s = _sum_series(pas[["state_abbr", "year", "scenario", "portfolio_annual_savings"]].copy(),
                        "portfolio_annual_savings", cumulative=False)
        if not s.empty:
            s["metric"] = "portfolio_annual_savings"
            pieces.append(s)

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["year", "scenario", "metric", "value"])


def facet_lines_national_totals(
    outputs: Dict[str, pd.DataFrame],
    metrics: Optional[Iterable[str]] = ("number_of_adopters", "system_kw_cum", "batt_kwh_cum", "cumulative_bill_savings"),
    xticks: Iterable[int] = (2026, 2030, 2035, 2040),
    title: str = "U.S. Totals: Baseline vs Policy",
    ncols: int = 3,
) -> None:
    """
    Subplots for national totals (Baseline vs Policy) with annotations at 2040.
    If 2040 is not available, annotate the last available year.
    """
    nat = build_national_totals(outputs)
    if nat.empty:
        return

    if metrics:
        nat = nat[nat["metric"].isin(set(metrics))]
        if nat.empty:
            return

    nice_titles = {
        "number_of_adopters": "Cumulative Adopters",
        "system_kw_cum": "Cumulative PV (kW)",
        "batt_kwh_cum": "Cumulative Storage (kWh)",
        "cumulative_bill_savings": "Cumulative Bill Savings ($)",
        "portfolio_annual_savings": "Portfolio Bill Savings ($/yr)",
    }

    # Formatting helper for end-of-horizon labels
    def _fmt(metric: str, v: float) -> str:
        if metric == "number_of_adopters":
            return f"{v/1e6:.1f}M"
        if metric == "system_kw_cum":
            # kW -> GW
            return f"{v/1e6:.1f} GW"
        if metric == "batt_kwh_cum":
            # kWh -> GWh
            return f"{v/1e6:.1f} GWh"
        if metric == "cumulative_bill_savings":
            return f"${v/1e9:.1f}B"
        if metric == "portfolio_annual_savings":
            return f"${v/1e9:.1f}B/yr"
        # fallback with 2 sig figs
        return f"{v:.2g}"

    metric_list = list(nat["metric"].unique())
    n = len(metric_list)
    ncols = max(1, min(ncols, n))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.4*nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for i, m in enumerate(metric_list):
        ax = axes[i]
        d = nat[nat["metric"] == m].sort_values("year")

        # Choose the annotation year: prefer 2040, else max available
        years = d["year"].unique()
        end_year = 2040 if 2040 in years else int(d["year"].max())

        # Plot each scenario and store end-year values for annotation
        end_points = []
        for scen, g in d.groupby("scenario"):
            g_sorted = g.sort_values("year")
            ax.plot(g_sorted["year"], g_sorted["value"], marker="o", label=scen.capitalize())
            # value at end_year (or most recent <= end_year)
            g_end = g_sorted[g_sorted["year"] == end_year]
            if g_end.empty:
                g_end = g_sorted[g_sorted["year"] <= end_year].tail(1)
            if not g_end.empty:
                end_points.append((scen, float(g_end["year"].iloc[-1]), float(g_end["value"].iloc[-1])))

        ax.set_xticks(list(xticks))
        ax.set_title(nice_titles.get(m, m))
        ax.set_xlabel("Year")
        ax.set_ylabel("U.S. Total")
        ax.legend(frameon=False, loc="best")

        # Annotations: offset to avoid overlap if values are very close
        if end_points:
            # Compute a tiny vertical jitter if two labels are nearly identical
            ys = np.array([p[2] for p in end_points])
            yrange = max(1.0, float(d["value"].max() - d["value"].min()) or 1.0)
            sep_needed = yrange * 0.01  # 1% of range
            used_offsets = {}
            for scen, x_end, y_end in sorted(end_points, key=lambda t: t[2]):
                # If another label is within sep_needed, nudge this one
                offset_pts = 0
                for other_y in used_offsets.keys():
                    if abs(y_end - other_y) < sep_needed:
                        offset_pts += 6
                used_offsets[y_end] = True

                ax.annotate(
                    _fmt(m, y_end),
                    xy=(x_end, y_end),
                    xytext=(6, offset_pts),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

    # Hide any extra axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, y=1.05)
    plt.show()

