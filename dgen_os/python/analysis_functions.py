# analysis_functions.py (drop-in)
from __future__ import annotations

import os
import re
import glob
import math
import json
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
# Columns expected in per-state CSVs. Missing columns are added as NaN/0 where appropriate.
NEEDED_COLS = [
    "state_abbr",
    "scenario",
    "year",

    # Cohort/cumulative adopters:
    "new_adopters",           # new adopters in that year (cohort size)
    "number_of_adopters",     # cumulative adopters in that year
    "customers_in_bin",

    # Savings and pricing:
    "first_year_elec_bill_savings",
    "price_per_kwh",

    # Tech potential / market share:
    "customers_in_bin",
    "max_market_share",

    # PV / storage:
    "system_kw",              # per-adopter PV size (kW) — used for medians
    "new_system_kw",          # **cohort total PV capacity (kW)** — used for attachment math
    "system_kw_cum",
    "batt_kwh",               # per-agent storage size for median calc (if absent, we add NaN)
    "batt_kwh_cum",
]

# Optional columns we will read if present (for initial stock)
OPTIONAL_COLS = ["initial_batt_kwh", "initial_number_of_adopters"]

SCHEMA_RE = re.compile(r"^diffusion_results_(baseline|policy)_([a-z]{2})_", re.IGNORECASE)


# =============================================================================
# Hourly array helpers
# =============================================================================

def _parse_array_text_to_floats(s: str) -> List[float]:
    """
    Accept Postgres array text like '{1,2,3}' or JSON-like '[1,2,3]' and return [1.0, 2.0, 3.0].
    Returns [] on any parse issue.
    """
    if not isinstance(s, str) or not s:
        return []
    t = s.strip()
    try_json = t.replace("{", "[").replace("}", "]")
    try:
        vals = json.loads(try_json)
        return [float(v) for v in vals] if isinstance(vals, list) else []
    except Exception:
        # Fallback: naive split on commas inside braces
        if t.startswith("{") and t.endswith("}"):
            inner = t[1:-1]
            if not inner:
                return []
            parts = inner.split(",")
            out = []
            for p in parts:
                p = p.strip()
                try:
                    out.append(float(p))
                except Exception:
                    return []
            return out
    return []


# =============================================================================
# Standalone hourly → daily/monthly plotter (Baseline vs Policy), with GW annotations
# =============================================================================

def _load_hourly_one(path: str) -> pd.DataFrame:
    """
    Load a single hourly CSV and ensure scenario/state/year and net_load array exist.
    Expects 'net_sum_text' with 8760-ish hourly values (MW).
    """
    df = pd.read_csv(path)
    # infer scenario from filename if missing
    if "scenario" not in df.columns:
        fname = os.path.basename(path).lower()
        df["scenario"] = "policy" if "policy" in fname else ("baseline" if "baseline" in fname else "unknown")
    # infer state from folder if missing
    if "state_abbr" not in df.columns:
        df["state_abbr"] = os.path.basename(os.path.dirname(path)).upper()
    if "net_sum_text" not in df.columns:
        raise ValueError(f"'net_sum_text' column not found in {path}")
    df["net_load"] = df["net_sum_text"].apply(_parse_array_text_to_floats)
    return df


def _pick_year_row(df: pd.DataFrame, scenario: str, year: int) -> Optional[pd.Series]:
    """Return one row for the given scenario/year with a valid net_load array."""
    sdf = df[(df["scenario"].str.lower() == scenario.lower()) & (df["year"] == year)].copy()
    if sdf.empty:
        return None
    exact = sdf[sdf["net_load"].apply(lambda a: isinstance(a, list) and len(a) > 0)]
    return (exact.iloc[0] if not exact.empty else None)


def _to_time_series(arr: List[float], year: int) -> pd.Series:
    """
    Build an hourly pandas Series from an array and a year.
    Works for 8760, 8784 (leap), or partial lengths.
    """
    n = len(arr)
    idx = pd.date_range(start=f"{year}-01-01 00:00:00", periods=n, freq="h")
    return pd.Series(arr, index=idx)


def _aggregate_series(s: pd.Series, aggregation: str = "hourly", agg_func: str = "mean") -> pd.Series:
    """
    aggregation: 'hourly' | 'daily' | 'weekly' | 'monthly'
    agg_func:    'mean' | 'sum' | 'max' | 'min' (applies when not hourly)
    """
    aggregation = aggregation.lower()
    agg_func = agg_func.lower()

    if aggregation == "hourly":
        return s

    rule = {"daily": "D", "weekly": "W", "monthly": "M"}.get(aggregation)
    if rule is None:
        raise ValueError("aggregation must be one of: 'hourly', 'daily', 'weekly', 'monthly'")

    if agg_func not in {"mean", "sum", "max", "min"}:
        raise ValueError("agg_func must be one of: 'mean', 'sum', 'max', 'min'")

    return getattr(s.resample(rule), agg_func)()


def plot_state_netload(
    state_dir: Optional[str] = None,
    baseline_csv: Optional[str] = None,
    policy_csv: Optional[str] = None,
    year: int = 2040,
    aggregation: str = "daily",     # 'hourly' | 'daily' | 'weekly' | 'monthly'
    agg_func: str = "max",          # used for non-hourly: 'mean'|'sum'|'max'|'min'
    title: Optional[str] = None,
) -> None:
    """
    Plot Baseline & Policy net load for a specific year at chosen aggregation.
    Provide either:
      - state_dir  (auto-discovers hourly_baseline* and hourly_policy*), OR
      - both baseline_csv and policy_csv explicitly.
    Adds on-plot annotations (GW) for the maxima of the aggregated series.
    """
    # Load data
    if state_dir:
        b_matches = sorted(glob.glob(os.path.join(state_dir, "hourly_baseline*.csv")))
        p_matches = sorted(glob.glob(os.path.join(state_dir, "hourly_policy*.csv")))
        frames: List[pd.DataFrame] = []
        if b_matches: frames.append(_load_hourly_one(b_matches[0]))
        if p_matches: frames.append(_load_hourly_one(p_matches[0]))
        if not frames:
            # fallback to any hourly*.csv with both scenarios inside
            any_matches = sorted(glob.glob(os.path.join(state_dir, "hourly*.csv")))
            if not any_matches:
                raise FileNotFoundError("No hourly_* CSVs found in state_dir.")
            frames = [_load_hourly_one(p) for p in any_matches]
        df = pd.concat(frames, ignore_index=True)
    else:
        if not (baseline_csv and policy_csv):
            raise ValueError("Provide either state_dir OR both baseline_csv and policy_csv.")
        df = pd.concat([_load_hourly_one(baseline_csv), _load_hourly_one(policy_csv)], ignore_index=True)

    # Ensure numeric year
    if "year" not in df.columns:
        raise ValueError("'year' column is required in hourly CSVs.")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Pick rows for each scenario at the specified year
    brow = _pick_year_row(df, "baseline", year)
    prow = _pick_year_row(df, "policy", year)
    if brow is None:
        raise ValueError(f"No baseline record found for year {year}.")
    if prow is None:
        raise ValueError(f"No policy record found for year {year}.")

    # Build time series
    s_base = _to_time_series(brow["net_load"], int(brow["year"]))
    s_poli = _to_time_series(prow["net_load"], int(prow["year"]))

    # Aggregate
    s_base_agg = _aggregate_series(s_base, aggregation=aggregation, agg_func=agg_func)
    s_poli_agg = _aggregate_series(s_poli, aggregation=aggregation, agg_func=agg_func)

    # Labels
    if title is None:
        nice_agg = aggregation.capitalize()
        nice_func = agg_func.upper() if aggregation != "hourly" else ""
        title = f"{nice_agg} Net Load — Baseline vs Policy ({year})" + (f" [{nice_func}]" if nice_func else "")

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(s_base_agg.index, s_base_agg.values, label="Baseline", alpha=0.9)
    plt.plot(s_poli_agg.index, s_poli_agg.values, label="Policy", alpha=0.9)

    plt.xlabel("Time")
    plt.ylabel("Net Load (MW)")
    plt.title(title)
    plt.legend()

    # --- Annotations (maxima, shown in GW) ---
    base_max = float(s_base_agg.max())
    poli_max = float(s_poli_agg.max())
    base_date = s_base_agg.idxmax()
    poli_date = s_poli_agg.idxmax()

    plt.annotate(f"{base_max/1000:.1f} GW (baseline)",
                 xy=(base_date, base_max),
                 xytext=(0, 8),
                 textcoords="offset points",
                 ha="center", color="black", fontweight="bold")

    plt.annotate(f"{poli_max/1000:.1f} GW (policy)",
                 xy=(poli_date, poli_max),
                 xytext=(0, -12),
                 textcoords="offset points",
                 ha="center", color="black", fontweight="bold")

    plt.tight_layout()
    plt.show()


# =============================================================================
# Hourly → state/year peaks (MW) from CSVs (used by multiple plots)
# =============================================================================

def find_state_hourly_files(state_dir: str, run_id: Optional[str] = None) -> List[str]:
    paths: List[str] = []
    if run_id:
        paths += glob.glob(os.path.join(state_dir, f"hourly_baseline_{run_id}.csv"))
        paths += glob.glob(os.path.join(state_dir, f"hourly_policy_{run_id}.csv"))
        if not paths:
            # legacy single-file fallback
            paths += glob.glob(os.path.join(state_dir, f"hourly_{run_id}.csv"))
    else:
        paths += glob.glob(os.path.join(state_dir, "hourly_baseline*.csv"))
        paths += glob.glob(os.path.join(state_dir, "hourly_policy*.csv"))
        if not paths:
            paths += glob.glob(os.path.join(state_dir, "hourly*.csv"))
    # de-dup while preserving order
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def load_state_peaks_df(state_dir: str, run_id: Optional[str] = None) -> pd.DataFrame:
    """
    Read one state's hourly CSV(s) and compute peak demand (MW) per year, by scenario.
    Returns: ['state_abbr','scenario','year','peak_mw'].
    """
    paths = find_state_hourly_files(state_dir, run_id)
    if not paths:
        return pd.DataFrame(columns=["state_abbr", "scenario", "year", "peak_mw"])

    frames: List[pd.DataFrame] = []
    for path in paths:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue

        # Ensure required cols, infer if missing
        if "scenario" not in df.columns:
            fname = os.path.basename(path).lower()
            df["scenario"] = "policy" if "policy" in fname else ("baseline" if "baseline" in fname else np.nan)
        if "state_abbr" not in df.columns:
            df["state_abbr"] = os.path.basename(os.path.dirname(path)).upper()

        needed = {"state_abbr", "scenario", "year", "net_sum_text"}
        if not needed.issubset(df.columns):
            continue

        def _peak(row) -> float:
            arr = _parse_array_text_to_floats(str(row.get("net_sum_text", "")))
            return float(np.max(arr)) if arr else np.nan

        d = df.copy()
        d["peak_mw"] = d.apply(_peak, axis=1)
        d = d[["state_abbr", "scenario", "year", "peak_mw"]].dropna(subset=["peak_mw"])
        d["year"] = pd.to_numeric(d["year"], errors="coerce")
        d = d.dropna(subset=["year"]).reset_index(drop=True)
        frames.append(d)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["state_abbr", "scenario", "year", "peak_mw"])


def process_all_states_peaks(
    root_dir: str,
    run_id: Optional[str] = None,
    n_jobs: int = 1,
    states: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate peak demand (MW) per year for all requested states.
    Returns tidy DF: ['state_abbr','scenario','year','peak_mw'].
    """
    state_dirs = discover_state_dirs(root_dir)
    if states:
        wanted = {s.strip().upper() for s in states if s and s.strip()}
        state_dirs = [sd for sd in state_dirs if os.path.basename(sd).upper() in wanted]
    if not state_dirs:
        return pd.DataFrame(columns=["state_abbr","scenario","year","peak_mw"])

    tasks = [(sd, run_id) for sd in state_dirs]
    results: List[pd.DataFrame] = []
    n_jobs = max(1, min(n_jobs, max(1, (cpu_count() or 2) - 1)))
    if n_jobs == 1:
        results = [load_state_peaks_df(sd, run_id) for sd, _ in tasks]
    else:
        with Pool(processes=n_jobs) as pool:
            for df in pool.starmap(load_state_peaks_df, tasks):
                results.append(df)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["state_abbr","scenario","year","peak_mw"])


# =============================================================================
# Facet: state-level peak (by year), with optional US delta tile
# =============================================================================

def _build_us_peak_delta(peaks_df: pd.DataFrame) -> pd.DataFrame:
    """
    From per-state peak_mw by scenario/year, build a single 'US Δ' facet that shows
    Policy − Baseline (MW). Note: sums state peaks by year; hours may differ across states.
    """
    if peaks_df.empty:
        return pd.DataFrame(columns=["state_abbr","scenario","year","peak_mw"])

    nat = (peaks_df.groupby(["year","scenario"], as_index=False)["peak_mw"].sum())
    piv = nat.pivot(index="year", columns="scenario", values="peak_mw")
    if "policy" not in piv.columns or "baseline" not in piv.columns:
        return pd.DataFrame(columns=["state_abbr","scenario","year","peak_mw"])

    d = (piv["policy"] - piv["baseline"]).rename("peak_mw").reset_index()
    d["state_abbr"] = "US Δ"
    d["scenario"] = "delta"
    return d[["state_abbr","scenario","year","peak_mw"]]


def facet_peaks_by_state(
    df: pd.DataFrame,
    ylabel: str = "Peak Demand (MW)",
    title: str = "Peak Demand by Year — Baseline vs Policy",
    xticks: Iterable[int] = (2026, 2030, 2035, 2040),
    height: float = 3.5,
    col_wrap: int = 4,
    sharey: bool = False,
    include_us_delta: bool = False,
) -> None:
    """
    Faceted line plot by state comparing Baseline vs Policy peaks (MW), annotated at 2040.
    If include_us_delta=True, appends a 'US Δ' facet showing Policy−Baseline (MW) as a single line.
    """
    if df.empty:
        return

    if include_us_delta:
        us_delta = _build_us_peak_delta(df)
        if not us_delta.empty:
            df = pd.concat([df, us_delta], ignore_index=True)

    _df = df.rename(columns={"peak_mw":"value"}).copy()
    sns.set_context("talk", rc={"lines.linewidth": 2})
    g = sns.FacetGrid(_df, col="state_abbr", col_wrap=col_wrap, height=height, sharey=sharey)
    g.map_dataframe(sns.lineplot, x="year", y="value", hue="scenario", marker="o")
    g.set_titles("{col_name}")
    g.set_axis_labels("Year", ylabel)
    g.set(xticks=list(xticks))
    g.add_legend()
    g.fig.suptitle(title, y=1.02)

    years = pd.to_numeric(_df["year"], errors="coerce").dropna().astype(int)
    preferred_year = 2040 if (len(years) and 2040 in set(years)) else (int(years.max()) if len(years) else None)
    if preferred_year is not None:
        for state, ax in g.axes_dict.items():
            sdf = _df[_df["state_abbr"] == state].dropna(subset=["value","year"])
            if sdf.empty:
                continue
            years_s = sdf["year"].astype(int).unique()
            end_year = preferred_year if preferred_year in years_s else int(sdf["year"].max())
            end_points = []
            for scen, sg in sdf.groupby("scenario"):
                sg = sg.sort_values("year")
                g_end = sg[sg["year"] == end_year]
                if g_end.empty:
                    g_end = sg[sg["year"] <= end_year].tail(1)
                if not g_end.empty:
                    end_points.append((scen, float(g_end["year"].iloc[-1]), float(g_end["value"].iloc[-1])))
            if not end_points:
                continue
            vals = np.array([p[2] for p in end_points])
            yrange = max(1.0, float(vals.max() - vals.min()) or 1.0)
            sep = yrange * 0.01
            used = []
            for scen, x_end, y_end in sorted(end_points, key=lambda t: t[2]):
                offset = 0
                for u in used:
                    if abs(y_end - u) < sep:
                        offset += 6
                used.append(y_end)
                label = f"{y_end:.1f} MW" if state != "US Δ" else f"{y_end:.1f} MW (Δ)"
                ax.annotate(label, xy=(x_end, y_end),
                            xytext=(6, offset), textcoords="offset points",
                            ha="left", va="center", fontsize=9, fontweight="bold")
    plt.show()


# =============================================================================
# Facet — state-level daily/weekly peak time series (one year)
# =============================================================================

def facet_state_peak_timeseries_from_hourly(
    root_dir: str,
    run_id: Optional[str] = None,
    year: int = 2040,
    height: float = 2.8,
    col_wrap: int = 4,         # 4 per row as requested
    sharey: bool = False,       # improves comparability across states
    title: Optional[str] = None,
    states: Optional[Iterable[str]] = None,
) -> None:
    """
    Faceted per-state WEEKLY peak net load for a single year (Baseline vs Policy).

    Changes from prior version:
      - 4 states per row (col_wrap=4).
      - Only one y-axis label per row (far-left facet).
      - Baseline plotted ON TOP of policy (higher z-order).
      - Peak values for baseline & policy shown in a fixed position
        (top-right of each facet), not at the peak location. Baseline listed first.

    Visual details:
      - x ticks: weeks 1, 25, 52
      - small annotation text with light white box
      - compact legend (baseline first)
    """
    # ---- discover/load states ----
    state_dirs = discover_state_dirs(root_dir)
    if states:
        wanted = {s.strip().upper() for s in states if s and s.strip()}
        state_dirs = [sd for sd in state_dirs if os.path.basename(sd).upper() in wanted]
    if not state_dirs:
        return

    def _load_state_hourly_two(sd: str) -> Optional[pd.DataFrame]:
        paths = find_state_hourly_files(sd, run_id)
        if not paths:
            return None
        frames = []
        for pth in paths:
            try:
                df = pd.read_csv(pth)
            except Exception:
                continue
            if "scenario" not in df.columns:
                fn = os.path.basename(pth).lower()
                df["scenario"] = "policy" if "policy" in fn else ("baseline" if "baseline" in fn else np.nan)
            if "state_abbr" not in df.columns:
                df["state_abbr"] = os.path.basename(sd).upper()
            frames.append(df)
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)

        needed = {"state_abbr", "scenario", "year", "net_sum_text"}
        if not needed.issubset(df.columns):
            return None

        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df[df["year"] == year]

        def _arr_ok(x):
            arr = _parse_array_text_to_floats(str(x))
            return (len(arr) > 0, arr)

        if df.empty:
            return None
        df["_ok"], df["_arr"] = zip(*df["net_sum_text"].map(_arr_ok))
        df = df[df["_ok"]]
        if df.empty:
            return None

        out_rows = []
        for scen in ("baseline", "policy"):
            g = df[df["scenario"].str.lower() == scen]
            if not g.empty:
                r = g.iloc[0]
                out_rows.append({
                    "state_abbr": r["state_abbr"],
                    "scenario": scen,
                    "year": int(year),
                    "net_load": r["_arr"],
                })
        return pd.DataFrame(out_rows) if out_rows else None

    rows = []
    for sd in state_dirs:
        d = _load_state_hourly_two(sd)
        if d is not None:
            rows.append(d)
    if not rows:
        return
    df_pairs = pd.concat(rows, ignore_index=True)

    # ---- weekly peaks (MW) + ISO week numbers ----
    def _weekly_max(arr: List[float]) -> pd.Series:
        s = pd.Series(arr, index=pd.date_range(f"{year}-01-01", periods=len(arr), freq="h"))
        wk = s.resample("W").max()
        weeks = wk.index.isocalendar().week.astype(int)
        wk.index = weeks
        wk.index.name = "week"
        return wk

    pieces = []
    for (state, scen), g in df_pairs.groupby(["state_abbr", "scenario"]):
        ts = _weekly_max(g["net_load"].iloc[0])
        pieces.append(pd.DataFrame({
            "state_abbr": state,
            "scenario": scen,
            "week": ts.index.astype(int),
            "peak_mw": ts.values,
        }))
    d = pd.concat(pieces, ignore_index=True)

    # ---- facets ----
    sns.set_context("talk", rc={"lines.linewidth": 1.6})
    states_order = sorted(d["state_abbr"].unique())
    n = len(states_order)
    ncols = col_wrap
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6*ncols, height*nrows), sharex=True, sharey=sharey)
    axes = np.atleast_1d(axes).ravel()

    # Colors and draw order (baseline on top)
    pal = sns.color_palette()
    color_baseline = pal[0]  # blue
    color_policy   = pal[1]  # orange
    draw_order = ("policy", "baseline")  # draw policy first, then baseline on top
    zorders = {"policy": 2, "baseline": 3}

    for idx, (ax, state) in enumerate(zip(axes, states_order)):
        sub = d[d["state_abbr"] == state]
        pol = sub[sub["scenario"] == "policy"].sort_values("week")
        bas = sub[sub["scenario"] == "baseline"].sort_values("week")

        # Plot lines (policy first, baseline second so baseline sits on top)
        if not pol.empty:
            ax.plot(pol["week"], pol["peak_mw"], color=color_policy, linewidth=2.0, label="policy", zorder=zorders["policy"])
        if not bas.empty:
            ax.plot(bas["week"], bas["peak_mw"], color=color_baseline, linewidth=2.0, label="baseline", zorder=zorders["baseline"])

        # Title and axes
        ax.set_title(state, fontsize=10, pad=8)
        ax.set_xlim(1, 53)
        ax.set_xticks([1, 25, 52])
        if idx % ncols == 0:
            ax.set_ylabel("Weekly Peak (MW)")  # only far-left facets get y label
        else:
            ax.set_ylabel(None)

        # Bottom row shows x label
        if idx >= 44:
            ax.set_xlabel("Week of Year")

        ax.grid(True, axis="y", alpha=0.25)

        # --- Fixed-position annotation of max values (top-right), baseline first ---
        # (values shown in GW with 1 decimal)
        base_peak = bas["peak_mw"].max() if not bas.empty else np.nan
        pol_peak  = pol["peak_mw"].max() if not pol.empty else np.nan
        lines = []
        if np.isfinite(base_peak):
            lines.append(f"baseline: {base_peak/1000:.1f} GW")
        if np.isfinite(pol_peak):
            lines.append(f"policy:   {pol_peak/1000:.1f} GW")
        if lines:
            ax.text(
                0.98, 0.98, "\n".join(lines),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="black",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.0),
                zorder=10,
            )

    # Hide any unused axes
    for ax in axes[len(states_order):]:
        ax.set_visible(False)

    # Legend (baseline first) in a consistent spot
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=color_baseline, linewidth=2, label="baseline"),
        Line2D([0], [0], color=color_policy,   linewidth=2, label="policy"),
    ]
    fig.legend(handles=handles, loc="lower right", ncol=2, frameon=False)

    fig.suptitle(title or f"Weekly Peak Net Load — Baseline vs Policy ({year})", y=0.995, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()


# =============================================================================
# Savings & Aggregations
# =============================================================================

@dataclass
@dataclass
class SavingsConfig:
    """Configuration for bill savings aggregation.

    bill_savings_annual_escalation_pct:
        Expected constant annual percentage change applied to each cohort's
        first-year bill savings in subsequent years.
        Example: 3% per year => 0.03. Default 0.0 (no escalation).
    """
    lifetime_years: int = 25
    cap_to_horizon: bool = False
    bill_savings_annual_escalation_pct: float = 0.0  # e.g., 0.03 for +3%/yr


def compute_portfolio_and_cumulative_savings(
    df: pd.DataFrame,
    cfg: SavingsConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute portfolio-level bill savings by carrying cohorts forward with optional escalation.

    Escalation model:
      - Each cohort contributes its first-year bill savings * escalated by a constant
        annual rate g = cfg.bill_savings_annual_escalation_pct in years after adoption.
      - Annual portfolio in year y sums all active cohorts' escalated annual savings:
            sum_{cohort_year <= y} base_cohort_annual * (1+g)^(y - cohort_year)
      - Cumulative bill savings is the cumsum of that annual portfolio series.
      - Lifetime savings for each cohort uses a geometric sum over L credited years:
            base_cohort_annual * [((1+g)^L - 1) / g]   (and = base*L when g=0)

    Notes:
      - Annual portfolio is naturally capped by the input horizon [min(year), max(year)].
      - Lifetime crediting uses cfg.lifetime_years, optionally capped to horizon if
        cfg.cap_to_horizon=True.
    """
    # Empty guard
    if df.empty:
        empty_annual = pd.DataFrame(columns=["state_abbr", "scenario", "year",
                                             "portfolio_annual_savings", "lifetime_savings_total"])
        empty_cum = pd.DataFrame(columns=["state_abbr", "scenario", "year",
                                          "cumulative_bill_savings", "lifetime_savings_total"])
        return empty_annual, empty_cum

    # Basic typing & base cohort annual savings
    x = df.copy()
    x["new_adopters"] = pd.to_numeric(x.get("new_adopters", 0.0), errors="coerce").fillna(0.0)
    x["first_year_elec_bill_savings"] = pd.to_numeric(
        x.get("first_year_elec_bill_savings", 0.0), errors="coerce"
    ).fillna(0.0)
    x["year"] = pd.to_numeric(x.get("year", np.nan), errors="coerce")

    # base (year-of-adoption) cohort annual savings = per-HH first-year savings * cohort size
    x["cohort_annual_savings"] = x["first_year_elec_bill_savings"] * x["new_adopters"]

    # Collapse to one row per cohort (state, scenario, cohort_year)
    cohorts = (
        x.groupby(["state_abbr", "scenario", "year"], as_index=False)["cohort_annual_savings"]
         .sum()
         .rename(columns={"year": "cohort_year"})
         .sort_values(["state_abbr", "scenario", "cohort_year"])
    )

    # Horizon years present in the input
    year_vals = pd.to_numeric(df["year"], errors="coerce").dropna()
    if year_vals.empty:
        empty_annual = pd.DataFrame(columns=["state_abbr", "scenario", "year",
                                             "portfolio_annual_savings", "lifetime_savings_total"])
        empty_cum = pd.DataFrame(columns=["state_abbr", "scenario", "year",
                                          "cumulative_bill_savings", "lifetime_savings_total"])
        return empty_annual, empty_cum

    year_min = int(year_vals.min())
    year_max = int(year_vals.max())
    all_years = list(range(year_min, year_max + 1))

    g = float(getattr(cfg, "bill_savings_annual_escalation_pct", 0.0) or 0.0)
    # guard against pathological values (e.g., g <= -1 causes division by zero later)
    if g <= -0.999999999:
        g = -0.999999999

    def _geom_factor(growth: float, n_years: int) -> float:
        """Geometric sum factor for 1 + growth over n_years (>=0)."""
        if n_years <= 0:
            return 0.0
        if abs(growth) < 1e-12:
            return float(n_years)
        return float(((1.0 + growth) ** n_years - 1.0) / growth)

    # ---- Annual portfolio with escalation (per state/scenario/year) ----
    annual_frames: List[pd.DataFrame] = []

    for (state, scen), gcoh in cohorts.groupby(["state_abbr", "scenario"]):
        mapping = dict(zip(gcoh["cohort_year"].astype(int), gcoh["cohort_annual_savings"].astype(float)))

        rows = []
        for y in all_years:
            total_y = 0.0
            # sum escalated contributions from all cohorts active by year y
            for cy, base in mapping.items():
                if cy <= y:
                    age = y - cy  # years since adoption (0 in cohort year)
                    total_y += base * ((1.0 + g) ** age if abs(g) >= 1e-12 else 1.0)
            rows.append((state, scen, y, total_y))

        ann_df = pd.DataFrame(rows, columns=["state_abbr", "scenario", "year", "portfolio_annual_savings"])
        annual_frames.append(ann_df)

    annual_portfolio = (
        pd.concat(annual_frames, ignore_index=True)
        if annual_frames else
        pd.DataFrame(columns=["state_abbr", "scenario", "year", "portfolio_annual_savings"])
    )

    # ---- Lifetime totals with escalation (closed-form geometric sum) ----
    lifetime_frames: List[pd.DataFrame] = []

    for (state, scen), gcoh in cohorts.groupby(["state_abbr", "scenario"]):
        lf = int(getattr(cfg, "lifetime_years", 25))

        # Optionally cap credited years to horizon length for each cohort
        if cfg.cap_to_horizon:
            credited = {int(cy): max(0, min(lf, year_max - int(cy) + 1)) for cy in gcoh["cohort_year"]}
        else:
            credited = {int(cy): lf for cy in gcoh["cohort_year"]}

        # geometric (or linear when g==0) present-value-like accumulator
        g_life = gcoh.copy()
        g_life["cohort_year"] = g_life["cohort_year"].astype(int)
        g_life["lifetime_years_applied"] = g_life["cohort_year"].map(credited)
        g_life["lifetime_savings_for_cohort"] = g_life.apply(
            lambda r: float(r["cohort_annual_savings"]) * _geom_factor(g, int(r["lifetime_years_applied"])),
            axis=1
        )

        lifetime_frames.append(
            g_life.groupby(["state_abbr", "scenario"], as_index=False)["lifetime_savings_for_cohort"]
                 .sum()
                 .rename(columns={"lifetime_savings_for_cohort": "lifetime_savings_total"})
        )

    lifetime_totals = (
        pd.concat(lifetime_frames, ignore_index=True)
          .groupby(["state_abbr", "scenario"], as_index=False)["lifetime_savings_total"].sum()
        if lifetime_frames else
        pd.DataFrame(columns=["state_abbr", "scenario", "lifetime_savings_total"])
    )

    # ---- Cumulative (time path) ----
    if not annual_portfolio.empty:
        annual_portfolio = annual_portfolio.sort_values(["state_abbr", "scenario", "year"])
        cumulative = annual_portfolio.copy()
        cumulative["cumulative_bill_savings"] = (
            cumulative.groupby(["state_abbr", "scenario"], observed=True)["portfolio_annual_savings"].cumsum()
        )
    else:
        cumulative = pd.DataFrame(columns=["state_abbr", "scenario", "year", "cumulative_bill_savings"])

    # Attach lifetime totals to both frames (like before)
    annual_portfolio = annual_portfolio.merge(lifetime_totals, on=["state_abbr", "scenario"], how="left")
    cumulative = cumulative.merge(lifetime_totals, on=["state_abbr", "scenario"], how="left")

    return annual_portfolio, cumulative


def aggregate_state_metrics(df: pd.DataFrame, cfg: SavingsConfig) -> Dict[str, pd.DataFrame]:
    """
    Aggregate per-state metrics for plotting and exports.

    NOTE: Attachment-rate post-processing has been REMOVED.
    Storage totals now use the model-provided fields directly (e.g., batt_kwh_cum).
    """
    if df.empty:
        return {
            "median_system_kw": pd.DataFrame(),
            "median_storage_kwh": pd.DataFrame(),
            "totals": pd.DataFrame(),
            "tech_2040": pd.DataFrame(),
            "portfolio_annual_savings": pd.DataFrame(),
            "cumulative_bill_savings": pd.DataFrame(),
            "lifetime_totals": pd.DataFrame(),
            "avg_price_2026_model": pd.DataFrame(),
            "market_share_reached": pd.DataFrame(),
        }

    x = df.copy()
    x["state_abbr"] = x.get("state_abbr", "").astype(str).str.strip().str.upper()

    # --- numeric coercions / fills ---
    for c in ("year","new_adopters","number_of_adopters","first_year_elec_bill_savings",
              "customers_in_bin","max_market_share","system_kw","new_system_kw",
              "system_kw_cum","batt_kwh","batt_kwh_cum","price_per_kwh",
              "initial_batt_kwh"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    defaults = {
        "new_adopters": 0.0, "number_of_adopters": 0.0, "first_year_elec_bill_savings": 0.0,
        "customers_in_bin": 0.0, "max_market_share": 0.0, "system_kw": 0.0,
        "new_system_kw": 0.0, "system_kw_cum": 0.0, "batt_kwh_cum": 0.0,
        "initial_batt_kwh": 0.0,
    }
    for c, v in defaults.items():
        if c in x.columns:
            x[c] = x[c].fillna(v)

    # --------------------------
    # MEDIANS (weighted by adopters; adopters-only)
    # --------------------------
    def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
        v = pd.to_numeric(values, errors="coerce")
        w = pd.to_numeric(weights, errors="coerce").fillna(0).clip(lower=0)
        mask = v.notna() & (w > 0)
        if not mask.any():
            return np.nan
        v = v[mask].to_numpy()
        w = w[mask].to_numpy()
        order = np.argsort(v)
        v = v[order]
        w = w[order]
        cw = np.cumsum(w)
        cutoff = 0.5 * w.sum()
        idx = int(np.searchsorted(cw, cutoff, side="left"))
        return float(v[min(idx, len(v) - 1)])

    adopt = x.copy()
    adopt["new_adopters"] = pd.to_numeric(adopt.get("new_adopters", 0.0), errors="coerce").fillna(0.0)
    adopt = adopt[adopt["new_adopters"] > 0]

    # Weighted median per-adopter PV size by state/year/scenario
    if "system_kw" in adopt.columns and not adopt.empty:
        median_kw = (
            adopt.groupby(["state_abbr", "year", "scenario"], observed=True)
                 .apply(lambda g: g["system_kw_cum"].sum()/g["number_of_adopters"].sum())
                 .reset_index(name="median_system_kw")
        )
    else:
        median_kw = pd.DataFrame(columns=["state_abbr", "year", "scenario", "median_system_kw"])

    # Weighted median storage size among storage adopters (optional)
    if "batt_kwh" in adopt.columns:
        has_storage = adopt.copy()
        has_storage["batt_kwh"] = pd.to_numeric(has_storage["batt_kwh"], errors="coerce")
        has_storage = has_storage[has_storage["batt_kwh"] > 0]
        if not has_storage.empty:
            median_storage = (
                has_storage.groupby(["state_abbr", "year", "scenario"], observed=True)
                           .apply(lambda g: _weighted_median(g["batt_kwh"], g["new_adopters"]))
                           .reset_index(name="median_batt_kwh")
            )
        else:
            median_storage = pd.DataFrame(columns=["state_abbr", "year", "scenario", "median_batt_kwh"])
    else:
        median_storage = pd.DataFrame(columns=["state_abbr", "year", "scenario", "median_batt_kwh"])

    # --- totals (DIRECTLY from model outputs; now includes new_adopters + new_system_kw) ---
    totals = (
        x.groupby(["state_abbr","year","scenario"], as_index=False)
         .agg(
             new_adopters=("new_adopters","sum"),                 # <-- added
             new_system_kw=("new_system_kw","sum"),               # <-- optional convenience
             number_of_adopters=("number_of_adopters","sum"),
             system_kw_cum=("system_kw_cum","sum"),
             batt_kwh_cum=("batt_kwh_cum","sum"),
         )
    )

    # --- tech potential (unchanged) ---
    tech_2040_src = x.loc[x["year"] == 2040, ["state_abbr","scenario","number_of_adopters","customers_in_bin"]]
    tech_2040 = tech_2040_src.groupby(["state_abbr","scenario"], as_index=False).sum()
    if not tech_2040.empty:
        tech_2040["percent_tech_potential"] = np.where(
            tech_2040["customers_in_bin"] > 0,
            100.0 * tech_2040["number_of_adopters"] / tech_2040["customers_in_bin"],
            np.nan,
        )

    # --- savings (unchanged) ---
    portfolio_annual, cumulative_savings = compute_portfolio_and_cumulative_savings(x, cfg)
    lifetime_totals = (
        portfolio_annual[["state_abbr","scenario","lifetime_savings_total"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # --- avg price 2026 (unchanged) ---
    if (
        "price_per_kwh" in x.columns
        and "customers_in_bin" in x.columns
        and x["price_per_kwh"].notna().any()
    ):
        price_2026 = x[(x["year"] == 2026) & (x["scenario"] == "baseline")].copy()
        price_2026["customers_in_bin"] = price_2026["customers_in_bin"].fillna(0.0).clip(lower=0.0)

        def _weighted_avg(g: pd.DataFrame) -> float:
            w = g["customers_in_bin"].to_numpy()
            v = g["price_per_kwh"].to_numpy()
            ws = w.sum()
            return float(np.average(v, weights=w)) if ws > 0 else np.nan

        avg_price_2026_model = (
            price_2026.groupby("state_abbr", as_index=False)
                      .apply(_weighted_avg)
                      .rename(columns={None: "price_per_kwh"})
        )
    else:
        avg_price_2026_model = pd.DataFrame(columns=["state_abbr","price_per_kwh"])

    # --- market share (unchanged) ---
    x["market_potential"] = x["customers_in_bin"] * x["max_market_share"]
    market_share = (
        x.groupby(["state_abbr","year","scenario"], as_index=False)
         .agg(
             market_potential=("market_potential","sum"),
             market_reached=("number_of_adopters","sum"),
         )
    )
    market_share["market_share_reached"] = np.where(
        market_share["market_potential"] > 0,
        market_share["market_reached"] / market_share["market_potential"],
        np.nan,
    )

    return {
        "median_system_kw": median_kw,
        "median_storage_kwh": median_storage,
        "totals": totals,  # now also has new_adopters + new_system_kw
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

def discover_state_dirs(root_dir: str) -> List[str]:
    """Return absolute paths of immediate subdirectories under `root_dir` (state folders)."""
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
    Read a CSV keeping only columns present from NEEDED_COLS (+ OPTIONAL_COLS when present).
    Missing columns are added. If `scenario` or `state_abbr` is missing, they are inferred
    from file and directory names. Robust to empty/zero-byte CSVs.
    """
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=NEEDED_COLS)

    try:
        header = pd.read_csv(path, nrows=0)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=NEEDED_COLS)

    core = [c for c in NEEDED_COLS if c in header.columns]
    opt  = [c for c in OPTIONAL_COLS if c in header.columns]
    usecols = core + opt

    try:
        df = pd.read_csv(path, usecols=usecols) if usecols else pd.DataFrame(columns=NEEDED_COLS)
    except (pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame(columns=NEEDED_COLS)

    # Add missing core columns
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


def _process_one_state(args) -> Dict[str, pd.DataFrame]:
    state_dir, run_id, cfg = args
    df = pd.DataFrame(columns=NEEDED_COLS)
    b_csv, p_csv = find_state_files(state_dir, run_id)
    parts: List[pd.DataFrame] = []
    for p in (b_csv, p_csv):
        if p:
            parts.append(_read_with_selected_cols(p))
    if parts:
        df = pd.concat(parts, ignore_index=True)
        # Basic typing
        for col in ("year", "new_adopters", "number_of_adopters",
                    "first_year_elec_bill_savings", "system_kw",
                    "system_kw_cum", "batt_kwh", "batt_kwh_cum",
                    "customers_in_bin", "max_market_share",
                    "avg_elec_price_cents_per_kwh",
                    "initial_batt_kwh", "initial_number_of_adopters"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Fill non-negative quantities
        for col in ("new_adopters", "number_of_adopters", "first_year_elec_bill_savings",
                    "customers_in_bin", "max_market_share",
                    "system_kw_cum", "batt_kwh_cum"):
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
    return aggregate_state_metrics(df, cfg)


def process_all_states(
    root_dir: str,
    run_id: Optional[str] = None,
    cfg: SavingsConfig = SavingsConfig(),
    n_jobs: int = 1,
    states: Optional[Iterable[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate small, plot-ready DataFrames across all states.
    """
    state_dirs = discover_state_dirs(root_dir)

    # Optional filter by explicit state codes (case-insensitive)
    if states:
        wanted = {s.strip().upper() for s in states if s and s.strip()}
        state_dirs = [sd for sd in state_dirs if os.path.basename(sd).upper() in wanted]

    if not state_dirs:
        return {
            "median_system_kw": pd.DataFrame(),
            "median_storage_kwh": pd.DataFrame(),
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
# Generic state-level faceter (unchanged)
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
    Faceted line plot by state comparing Baseline vs Policy, with end-of-horizon annotations.
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
        if m == "median_batt_kwh":
            return f"{v:.1f} kWh"
        return f"{v:.2g}"

    sns.set_context("talk", rc={"lines.linewidth": 2})
    g = sns.FacetGrid(df, col="state_abbr", col_wrap=col_wrap, height=height, sharey=sharey)

    g.map_dataframe(sns.lineplot, x="year", y=y_col, hue="scenario", marker="o")
    g.set_titles("{col_name}")
    g.set_axis_labels("Year", ylabel)
    g.set(xticks=list(xticks))
    g.add_legend()
    g.fig.suptitle(title, y=1.02)

    # Preferred annotation year globally (fallback to max available per state)
    global_years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
    preferred_year = 2040 if (len(global_years) and 2040 in set(global_years)) else (int(global_years.max()) if len(global_years) else None)

    # annotate per facet
    if preferred_year is not None:
        for state, ax in g.axes_dict.items():
            sdf = df[(df["state_abbr"] == state)].copy()
            if sdf.empty or y_col not in sdf.columns:
                continue
            sdf = sdf.dropna(subset=[y_col, "year"])

            years = sdf["year"].astype(int).unique()
            end_year = preferred_year if preferred_year in years else int(sdf["year"].max())

            end_points = []
            for scen, sg in sdf.groupby("scenario"):
                sg = sg.sort_values("year")
                g_end = sg[sg["year"] == end_year]
                if g_end.empty:
                    g_end = sg[sg["year"] <= end_year].tail(1)
                if not g_end.empty:
                    x_end = float(g_end["year"].iloc[-1])
                    y_end = float(g_end[y_col].iloc[-1])
                    end_points.append((scen, x_end, y_end))

            if not end_points:
                continue

            sv = sdf[y_col].to_numpy()
            vmin = float(np.nanmin(sv)) if sv.size else 0.0
            vmax = float(np.nanmax(sv)) if sv.size else 1.0
            yrange = max(1.0, vmax - vmin)
            sep_needed = yrange * 0.01  # 1%

            used_y = []
            for scen, x_end, y_end in sorted(end_points, key=lambda t: t[2]):
                offset_pts = 0
                for uy in used_y:
                    if abs(y_end - uy) < sep_needed:
                        offset_pts += 6
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

    plt.show()


# =============================================================================
# National deltas & totals (EXTENDED TO INCLUDE PEAKS)
# =============================================================================

def build_national_deltas(outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Return tidy national deltas: ['year','metric','value'], value = (policy − baseline).
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


def build_national_totals(
    outputs: Dict[str, pd.DataFrame],
    peaks_df: Optional[pd.DataFrame] = None,   # NEW: optionally include peak_mw series
) -> pd.DataFrame:
    """
    Build tidy national totals across states, per scenario:
        ['year', 'scenario', 'metric', 'value']

    If peaks_df is provided (columns: state_abbr, scenario, year, peak_mw),
    we add a metric 'peak_mw' to the national totals.
    """
    years_seen = []
    for k in ("totals", "portfolio_annual_savings", "cumulative_bill_savings"):
        dfk = outputs.get(k, pd.DataFrame())
        if not dfk.empty and "year" in dfk.columns:
            years_seen.append(dfk["year"])
    if peaks_df is not None and not peaks_df.empty:
        years_seen.append(peaks_df["year"])

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

    # NEW: include national peak_mw totals (sum of state peaks by year/scenario)
    if peaks_df is not None and not peaks_df.empty:
        if set(["state_abbr","scenario","year","peak_mw"]).issubset(peaks_df.columns):
            s = _sum_series(peaks_df[["state_abbr","year","scenario","peak_mw"]].copy(),
                            "peak_mw", cumulative=False)
            if not s.empty:
                s["metric"] = "peak_mw"
                pieces.append(s)

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["year", "scenario", "metric", "value"])


def facet_lines_national_totals(
    outputs: Dict[str, pd.DataFrame],
    peaks_df: Optional[pd.DataFrame] = None,   # NEW: pass peaks here to include 'peak_mw'
    metrics: Optional[Iterable[str]] = ("number_of_adopters", "system_kw_cum", "batt_kwh_cum", "cumulative_bill_savings", "peak_mw"),
    xticks: Iterable[int] = (2026, 2030, 2035, 2040),
    title: str = "U.S. Totals: Baseline vs Policy",
    ncols: int = 3,
) -> None:
    """
    Subplots for national totals (Baseline vs Policy) with annotations.
    If peaks_df is provided, includes 'peak_mw' in the facet set so U.S. peaks appear
    alongside the other U.S. totals as requested.
    """
    nat = build_national_totals(outputs, peaks_df=peaks_df)
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
        "peak_mw": "U.S. Peak Demand (MW)",   # NEW
    }

    # Formatting helper for end-of-horizon labels
    def _fmt(metric: str, v: float) -> str:
        if metric == "number_of_adopters":
            return f"{v/1e6:.1f}M"
        if metric == "system_kw_cum":
            return f"{v/1e6:.1f} GW"         # kW -> GW
        if metric == "batt_kwh_cum":
            return f"{v/1e6:.1f} GWh"        # kWh -> GWh
        if metric == "cumulative_bill_savings":
            return f"${v/1e9:.1f}B"
        if metric == "peak_mw":
            return f"{v/1e3:.1f} GW"         # MW -> GW
        # portfolio_annual_savings shown raw:
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

        # Annotations: small vertical jitter to avoid overlap
        if end_points:
            ys = np.array([p[2] for p in end_points])
            yrange = max(1.0, float(d["value"].max() - d["value"].min()) or 1.0)
            sep_needed = yrange * 0.01
            used_offsets = {}
            for scen, x_end, y_end in sorted(end_points, key=lambda t: t[2]):
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


# =============================================================================
# Other plots
# =============================================================================

def facet_median_storage_by_state(
    outputs: Dict[str, pd.DataFrame],
    xticks: Iterable[int] = (2026, 2030, 2035, 2040),
    height: float = 3.5,
    col_wrap: int = 4,
    sharey: bool = False,
    title: str = "Median Storage Size (kWh) — Baseline vs Policy",
) -> None:
    """
    Faceted line plot of per-state median storage size (kWh) by scenario.
    Requires outputs['median_storage_kwh'] produced by aggregate_state_metrics.
    """
    df = outputs.get("median_storage_kwh", pd.DataFrame())
    if df is None or df.empty:
        return

    d = df.rename(columns={"median_batt_kwh": "value"}).copy()
    d = d.dropna(subset=["value"])

    sns.set_context("talk", rc={"lines.linewidth": 2})
    g = sns.FacetGrid(d, col="state_abbr", col_wrap=col_wrap, height=height, sharey=sharey)
    g.map_dataframe(sns.lineplot, x="year", y="value", hue="scenario", marker="o")
    g.set_titles("{col_name}")
    g.set_axis_labels("Year", "Median Storage Size (kWh)")
    g.set(xticks=list(xticks))
    g.add_legend()
    g.fig.suptitle(title, y=1.02)

    # annotate like the others
    global_years = pd.to_numeric(d["year"], errors="coerce").dropna().astype(int)
    preferred_year = 2040 if (len(global_years) and 2040 in set(global_years)) else (int(global_years.max()) if len(global_years) else None)
    if preferred_year is not None:
        for state, ax in g.axes_dict.items():
            sdf = d[(d["state_abbr"] == state)].dropna(subset=["value","year"])
            if sdf.empty:
                continue
            years = sdf["year"].astype(int).unique()
            end_year = preferred_year if preferred_year in years else int(sdf["year"].max())

            end_points = []
            for scen, sg in sdf.groupby("scenario"):
                sg = sg.sort_values("year")
                g_end = sg[sg["year"] == end_year]
                if g_end.empty:
                    g_end = sg[sg["year"] <= end_year].tail(1)
                if not g_end.empty:
                    end_points.append((scen, float(g_end["year"].iloc[-1]), float(g_end["value"].iloc[-1])))

            if not end_points:
                continue

            vals = np.array([p[2] for p in end_points])
            yrange = max(1.0, float(vals.max() - vals.min()) or 1.0)
            sep = yrange * 0.01
            used = []
            for scen, x_end, y_end in sorted(end_points, key=lambda t: t[2]):
                offset = 0
                for u in used:
                    if abs(y_end - u) < sep:
                        offset += 6
                used.append(y_end)
                ax.annotate(f"{y_end:.1f} kWh", xy=(x_end, y_end),
                            xytext=(6, offset), textcoords="offset points",
                            ha="left", va="center", fontsize=9, fontweight="bold")
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
    # I/O & discovery
    "discover_state_dirs",
    "find_state_files",
    "load_state_peaks_df",
    "process_all_states_peaks",

    # Hourly plotting helpers
    "plot_state_netload",
    "facet_state_peak_timeseries_from_hourly",

    # Config & processing
    "SavingsConfig",
    "aggregate_state_metrics",
    "compute_portfolio_and_cumulative_savings",
    "process_all_states",

    # State plots
    "facet_lines_by_state",
    "facet_peaks_by_state",
    "facet_median_storage_by_state",

    # National
    "build_national_deltas",
    "build_national_totals",
    "facet_lines_national_totals",

    # Other
    "bar_tech_potential_2040",
]
