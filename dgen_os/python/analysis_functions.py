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
import geopandas as gpd


# =============================================================================
# Discovery & I/O
# =============================================================================

# Columns expected in per-state CSVs. Missing columns are added as NaN/0 where appropriate.
# Updated to new savings arrays + battery cohort split; kept original metrics otherwise.
NEEDED_COLS = [
    "state_abbr",
    "scenario",
    "year",

    # Cohort/cumulative adopters:
    "new_adopters",                 # new adopters in that year (cohort size)
    "number_of_adopters",           # cumulative adopters in that year
    "customers_in_bin",
    "batt_adopters_this_year",      # storage adopters in that cohort (may be 0/missing)

    # Savings arrays (25-year, from agent_finance_series):
    "cf_energy_value_pv_only",
    "cf_energy_value_pv_batt",

    # Utility bill arrays (25-year, from agent_finance_series):
    "utility_bill_w_sys_pv_only",
    "utility_bill_w_sys_pv_batt",
    "utility_bill_wo_sys_pv_only",
    "utility_bill_wo_sys_pv_batt",

    # Pricing:
    "price_per_kwh",
    "load_kwh_per_customer_in_bin_initial",

    # Tech potential / market share:
    "customers_in_bin",
    "max_market_share",

    # PV / storage:
    "system_kw",              # per-adopter PV size (kW) — used for medians
    "new_system_kw",          # cohort total PV capacity (kW)
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

def _parse_array_text_to_floats(s) -> List[float]:
    """
    Accept Postgres array text like '{1,2,3}' or JSON-like '[1,2,3]' or a Python list/tuple/ndarray
    and return [1.0, 2.0, 3.0]. Returns [] on any parse issue.
    """
    if s is None:
        return []
    if isinstance(s, (list, tuple, np.ndarray)):
        try:
            return [float(v) for v in s]
        except Exception:
            return []
    if not isinstance(s, str) or not s:
        return []
    t = s.strip()
    try_json = t.replace("{", "[").replace("}", "]")
    try:
        vals = json.loads(try_json)
        return [float(v) for v in vals] if isinstance(vals, list) else []
    except Exception:
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
# (unchanged)
# =============================================================================

def _load_hourly_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "scenario" not in df.columns:
        fname = os.path.basename(path).lower()
        df["scenario"] = "policy" if "policy" in fname else ("baseline" if "baseline" in fname else "unknown")
    if "state_abbr" not in df.columns:
        df["state_abbr"] = os.path.basename(os.path.dirname(path)).upper()
    if "net_sum_text" not in df.columns:
        raise ValueError(f"'net_sum_text' column not found in {path}")
    df["net_load"] = df["net_sum_text"].apply(_parse_array_text_to_floats)
    return df


def _pick_year_row(df: pd.DataFrame, scenario: str, year: int) -> Optional[pd.Series]:
    sdf = df[(df["scenario"].str.lower() == scenario.lower()) & (df["year"] == year)].copy()
    if sdf.empty:
        return None
    exact = sdf[sdf["net_load"].apply(lambda a: isinstance(a, list) and len(a) > 0)]
    return (exact.iloc[0] if not exact.empty else None)


def _to_time_series(arr: List[float], year: int) -> pd.Series:
    n = len(arr)
    idx = pd.date_range(start=f"{year}-01-01 00:00:00", periods=n, freq="h")
    return pd.Series(arr, index=idx)


def _aggregate_series(s: pd.Series, aggregation: str = "hourly", agg_func: str = "mean") -> pd.Series:
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
    aggregation: str = "daily",
    agg_func: str = "max",
    title: Optional[str] = None,
) -> None:
    # unchanged logic
    if state_dir:
        b_matches = sorted(glob.glob(os.path.join(state_dir, "hourly_baseline*.csv")))
        p_matches = sorted(glob.glob(os.path.join(state_dir, "hourly_policy*.csv")))
        frames: List[pd.DataFrame] = []
        if b_matches: frames.append(_load_hourly_one(b_matches[0]))
        if p_matches: frames.append(_load_hourly_one(p_matches[0]))
        if not frames:
            any_matches = sorted(glob.glob(os.path.join(state_dir, "hourly*.csv")))
            if not any_matches:
                raise FileNotFoundError("No hourly_* CSVs found in state_dir.")
            frames = [_load_hourly_one(p) for p in any_matches]
        df = pd.concat(frames, ignore_index=True)
    else:
        if not (baseline_csv and policy_csv):
            raise ValueError("Provide either state_dir OR both baseline_csv and policy_csv.")
        df = pd.concat([_load_hourly_one(baseline_csv), _load_hourly_one(policy_csv)], ignore_index=True)

    if "year" not in df.columns:
        raise ValueError("'year' column is required in hourly CSVs.")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    brow = _pick_year_row(df, "baseline", year)
    prow = _pick_year_row(df, "policy", year)
    if brow is None:
        raise ValueError(f"No baseline record found for year {year}.")
    if prow is None:
        raise ValueError(f"No policy record found for year {year}.")

    s_base = _to_time_series(brow["net_load"], int(brow["year"]))
    s_poli = _to_time_series(prow["net_load"], int(prow["year"]))

    s_base_agg = _aggregate_series(s_base, aggregation=aggregation, agg_func=agg_func)
    s_poli_agg = _aggregate_series(s_poli, aggregation=aggregation, agg_func=agg_func)

    if title is None:
        nice_agg = aggregation.capitalize()
        nice_func = agg_func.upper() if aggregation != "hourly" else ""
        title = f"{nice_agg} Net Load — Baseline vs Policy ({year})" + (f" [{nice_func}]" if nice_func else "")

    plt.figure(figsize=(14, 6))
    plt.plot(s_base_agg.index, s_base_agg.values, label="Baseline", alpha=0.9)
    plt.plot(s_poli_agg.index, s_poli_agg.values, label="Policy", alpha=0.9)

    plt.xlabel("Time")
    plt.ylabel("Net Load (MW)")
    plt.title(title)
    plt.legend()

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
    # NEW: prefer <STATE>/<RUN_ID>/baseline_hourly.csv & policy_hourly.csv
    if run_id:
        sub = os.path.join(state_dir, run_id)
        if os.path.isdir(sub):
            c1 = os.path.join(sub, "baseline_hourly.csv")
            c2 = os.path.join(sub, "policy_hourly.csv")
            if os.path.exists(c1): paths.append(c1)
            if os.path.exists(c2): paths.append(c2)
            if not paths:
                paths += glob.glob(os.path.join(sub, "hourly_baseline*.csv"))
                paths += glob.glob(os.path.join(sub, "hourly_policy*.csv"))
                if not paths:
                    paths += glob.glob(os.path.join(sub, "hourly*.csv"))
        # legacy file names at state root with run_id token
        if not paths:
            paths += glob.glob(os.path.join(state_dir, f"hourly_baseline_{run_id}.csv"))
            paths += glob.glob(os.path.join(state_dir, f"hourly_policy_{run_id}.csv"))
            if not paths:
                paths += glob.glob(os.path.join(state_dir, f"hourly_{run_id}.csv"))
    # legacy fallback
    if not paths:
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
        if "scenario" not in df.columns:
            fname = os.path.basename(path).lower()
            df["scenario"] = "policy" if "policy" in fname else ("baseline" if "baseline" in fname else np.nan)
        if "state_abbr" not in df.columns:
            df["state_abbr"] = os.path.basename(os.path.dirname(os.path.dirname(path) if os.path.basename(os.path.dirname(path)).lower() in {"policy", "baseline"} else os.path.dirname(path))).upper()
            # if using <STATE>/<RUN_ID>/..., dirname(path) is the run_id folder; its parent is STATE

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
# Facet helpers (unchanged visuals)
# =============================================================================

def _build_us_peak_delta(peaks_df: pd.DataFrame) -> pd.DataFrame:
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
# (unchanged interface/plot; only file discovery already updated above)
# =============================================================================

def facet_state_peak_timeseries_from_hourly(
    root_dir: str,
    run_id: Optional[str] = None,
    year: int = 2040,
    height: float = 2.8,
    col_wrap: int = 4,
    sharey: bool = False,
    title: Optional[str] = None,
    states: Optional[Iterable[str]] = None,
) -> None:
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
                # if using <STATE>/<RUN_ID>/..., dirname(dirname(path)) is the state folder
                state_guess = os.path.basename(os.path.dirname(os.path.dirname(pth)))
                df["state_abbr"] = state_guess.upper()
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

    sns.set_context("talk", rc={"lines.linewidth": 1.6})
    states_order = sorted(d["state_abbr"].unique())
    n = len(states_order)
    ncols = col_wrap
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6*ncols, height*nrows), sharex=True, sharey=sharey)
    axes = np.atleast_1d(axes).ravel()

    pal = sns.color_palette()
    color_baseline = pal[0]
    color_policy   = pal[1]
    zorders = {"policy": 2, "baseline": 3}

    for idx, (ax, state) in enumerate(zip(axes, states_order)):
        sub = d[d["state_abbr"] == state]
        pol = sub[sub["scenario"] == "policy"].sort_values("week")
        bas = sub[sub["scenario"] == "baseline"].sort_values("week")

        if not pol.empty:
            ax.plot(pol["week"], pol["peak_mw"], color=color_policy, linewidth=2.0, label="policy", zorder=zorders["policy"])
        if not bas.empty:
            ax.plot(bas["week"], bas["peak_mw"], color=color_baseline, linewidth=2.0, label="baseline", zorder=zorders["baseline"])

        ax.set_title(state, fontsize=10, pad=8)
        ax.set_xlim(1, 53)
        ax.set_xticks([1, 25, 52])
        if idx % ncols == 0:
            ax.set_ylabel("Weekly Peak (MW)")
        else:
            ax.set_ylabel(None)

        if idx >= 44:
            ax.set_xlabel("Week of Year")

        ax.grid(True, axis="y", alpha=0.25)

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

    for ax in axes[len(states_order):]:
        ax.set_visible(False)

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
# Savings & Aggregations (UPDATED to use 25-yr arrays + battery split)
# =============================================================================

@dataclass
@dataclass
class SavingsConfig:
    lifetime_years: int = 25
    cap_to_horizon: bool = False
    bill_savings_annual_escalation_pct: float = 0.0  # kept for compatibility; not used with arrays


def _arr25(x) -> List[float]:
    """Parse and clip/pad to length 25."""
    a = _parse_array_text_to_floats(x)
    a = [float(v) for v in a][:25]
    if len(a) < 25:
        a = a + [0.0] * (25 - len(a))
    return a


def compute_portfolio_and_cumulative_savings(
    df: pd.DataFrame,
    cfg: SavingsConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (1) annual portfolio bill savings by rolling each cohort's 25-yr array forward,
    and (2) cumulative bill savings (cumsum of annual), plus (3) lifetime totals per state/scenario.

    Uses:
      - new_adopters and batt_adopters_this_year to split cohorts into pv_only vs pv_batt.
      - cf_energy_value_pv_only, cf_energy_value_pv_batt arrays (per-adopter savings).
    """
    if df.empty:
        empty_annual = pd.DataFrame(columns=["state_abbr","scenario","year","portfolio_annual_savings","lifetime_savings_total"])
        empty_cum    = pd.DataFrame(columns=["state_abbr","scenario","year","cumulative_bill_savings","lifetime_savings_total"])
        return empty_annual, empty_cum

    x = df.copy()

    # cohort sizes
    x["new_adopters"] = pd.to_numeric(x.get("new_adopters", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    x["batt_adopters_this_year"] = pd.to_numeric(x.get("batt_adopters_this_year", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    x["pv_batt_n"] = np.minimum(x["batt_adopters_this_year"], x["new_adopters"])
    x["pv_only_n"] = (x["new_adopters"] - x["pv_batt_n"]).clip(lower=0.0)

    # arrays
    x["cf_pv_only"] = x.get("cf_energy_value_pv_only", np.nan).apply(_arr25) if "cf_energy_value_pv_only" in x.columns else [[]]*len(x)
    x["cf_pv_batt"] = x.get("cf_energy_value_pv_batt", np.nan).apply(_arr25) if "cf_energy_value_pv_batt" in x.columns else [[]]*len(x)

    # horizon
    x["year"] = pd.to_numeric(x.get("year", np.nan), errors="coerce")
    years = x["year"].dropna()
    if years.empty:
        empty_annual = pd.DataFrame(columns=["state_abbr","scenario","year","portfolio_annual_savings","lifetime_savings_total"])
        empty_cum    = pd.DataFrame(columns=["state_abbr","scenario","year","cumulative_bill_savings","lifetime_savings_total"])
        return empty_annual, empty_cum
    y_min, y_max = int(years.min()), int(years.max())
    L = int(getattr(cfg, "lifetime_years", 25) or 25)

    # annual portfolio by rolling arrays forward
    contrib = []
    for r in x.itertuples(index=False):
        state = r.state_abbr
        scen  = r.scenario
        y0    = int(r.year) if not pd.isna(r.year) else None
        if y0 is None or ((r.pv_only_n <= 0) and (r.pv_batt_n <= 0)):
            continue
        a_only, a_batt = list(r.cf_pv_only or []), list(r.cf_pv_batt or [])
        for k in range(25):
            y = y0 + k
            if cfg.cap_to_horizon and y > y_max:
                break
            if y < y_min or y > y_max:
                continue
            v = 0.0
            if k < len(a_only) and r.pv_only_n > 0: v += a_only[k] * r.pv_only_n
            if k < len(a_batt) and r.pv_batt_n > 0: v += a_batt[k] * r.pv_batt_n
            if v != 0.0:
                contrib.append((state, scen, y, v))

    annual = (
        pd.DataFrame(contrib, columns=["state_abbr","scenario","year","portfolio_annual_savings"])
          .groupby(["state_abbr","scenario","year"], as_index=False)["portfolio_annual_savings"].sum()
    ) if contrib else pd.DataFrame(columns=["state_abbr","scenario","year","portfolio_annual_savings"])

    # lifetime totals (sum credited years of each cohort × cohort size)
    life_rows = []
    for r in x.itertuples(index=False):
        y0 = int(r.year) if not pd.isna(r.year) else None
        if y0 is None:
            continue
        credited = L if not cfg.cap_to_horizon else max(0, min(L, y_max - y0 + 1))
        if credited <= 0:
            continue
        tot = 0.0
        if r.pv_only_n > 0 and r.cf_pv_only: tot += sum(r.cf_pv_only[:credited]) * r.pv_only_n
        if r.pv_batt_n > 0 and r.cf_pv_batt: tot += sum(r.cf_pv_batt[:credited]) * r.pv_batt_n
        if tot != 0.0:
            life_rows.append((r.state_abbr, r.scenario, tot))

    lifetime = (
        pd.DataFrame(life_rows, columns=["state_abbr","scenario","lifetime_savings_for_cohort"])
          .groupby(["state_abbr","scenario"], as_index=False)["lifetime_savings_for_cohort"].sum()
          .rename(columns={"lifetime_savings_for_cohort": "lifetime_savings_total"})
    ) if life_rows else pd.DataFrame(columns=["state_abbr","scenario","lifetime_savings_total"])

    # cumulative
    if not annual.empty:
        annual = annual.sort_values(["state_abbr","scenario","year"])
        cumulative = annual.copy()
        cumulative["cumulative_bill_savings"] = cumulative.groupby(["state_abbr","scenario"], observed=True)[
            "portfolio_annual_savings"
        ].cumsum()
    else:
        cumulative = pd.DataFrame(columns=["state_abbr","scenario","year","cumulative_bill_savings"])

    # attach lifetime totals
    annual     = annual.merge(lifetime, on=["state_abbr","scenario"], how="left")
    cumulative = cumulative.merge(lifetime, on=["state_abbr","scenario"], how="left")
    return annual, cumulative


def aggregate_state_metrics(df: pd.DataFrame, cfg: SavingsConfig) -> Dict[str, pd.DataFrame]:
    """
    Aggregate per-state metrics for plotting and exports.
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
    for c in ("year","new_adopters","number_of_adopters",
              "customers_in_bin","max_market_share","system_kw","new_system_kw",
              "system_kw_cum","batt_kwh","batt_kwh_cum","price_per_kwh",
              "initial_batt_kwh","batt_adopters_this_year","load_kwh_per_customer_in_bin_initial"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    defaults = {
        "new_adopters": 0.0, "number_of_adopters": 0.0,
        "customers_in_bin": 0.0, "max_market_share": 0.0, "system_kw": 0.0,
        "new_system_kw": 0.0, "system_kw_cum": 0.0, "batt_kwh_cum": 0.0,
        "initial_batt_kwh": 0.0, "batt_adopters_this_year": 0.0,
        "load_kwh_per_customer_in_bin_initial": 0.0,
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

    if "system_kw" in adopt.columns and not adopt.empty:
        median_kw = (
            adopt.groupby(["state_abbr", "year", "scenario"], observed=True)
                 .apply(lambda g: g["system_kw_cum"].sum()/g["number_of_adopters"].sum())
                 .reset_index(name="median_system_kw")
        )
    else:
        median_kw = pd.DataFrame(columns=["state_abbr", "year", "scenario", "median_system_kw"])

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

    # --- totals ---
    totals = (
        x.groupby(["state_abbr","year","scenario"], as_index=False)
         .agg(
             new_adopters=("new_adopters","sum"),
             new_system_kw=("new_system_kw","sum"),
             number_of_adopters=("number_of_adopters","sum"),
             system_kw_cum=("system_kw_cum","sum"),
             batt_kwh_cum=("batt_kwh_cum","sum"),
         )
    )

    # --- tech potential ---
    tech_2040_src = x.loc[x["year"] == 2040, ["state_abbr","scenario","number_of_adopters","customers_in_bin"]]
    tech_2040 = tech_2040_src.groupby(["state_abbr","scenario"], as_index=False).sum()
    if not tech_2040.empty:
        tech_2040["percent_tech_potential"] = np.where(
            tech_2040["customers_in_bin"] > 0,
            100.0 * tech_2040["number_of_adopters"] / tech_2040["customers_in_bin"],
            np.nan,
        )

    # --- savings (new arrays) ---
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
        "totals": totals,
        "tech_2040": tech_2040,
        "portfolio_annual_savings": portfolio_annual,
        "cumulative_bill_savings": cumulative_savings,
        "lifetime_totals": lifetime_totals,
        "avg_price_2026_model": avg_price_2026_model,
        "market_share_reached": market_share,
    }


# =============================================================================
# Parallel processing across states (UPDATED discovery to support subfolders)
# =============================================================================

def discover_state_dirs(root_dir: str) -> List[str]:
    return sorted(
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )


def find_state_files(state_dir: str, run_id: Optional[str] = None, strict_run_id: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """
    Locate per-state CSVs for baseline and policy.

    New preferred layout:
        <STATE>/<RUN_ID>/baseline.csv
        <STATE>/<RUN_ID>/policy.csv

    Legacy fallbacks preserved.
    """
    if run_id:
        sub = os.path.join(state_dir, run_id)
        if os.path.isdir(sub):
            b = os.path.join(sub, "baseline.csv")
            p = os.path.join(sub, "policy.csv")
            return (b if os.path.exists(b) else None), (p if os.path.exists(p) else None)
        if strict_run_id:
            b = glob.glob(os.path.join(state_dir, f"baseline_{run_id}.csv"))
            p = glob.glob(os.path.join(state_dir, f"policy_{run_id}.csv"))
            return (b[0] if b else None), (p[0] if p else None)
        else:
            b = glob.glob(os.path.join(state_dir, f"baseline_{run_id}.csv"))
            p = glob.glob(os.path.join(state_dir, f"policy_{run_id}.csv"))
            if b and p:
                return b[0], p[0]

    # Fallbacks (legacy)
    b = glob.glob(os.path.join(state_dir, "baseline*.csv"))
    p = glob.glob(os.path.join(state_dir, "policy*.csv"))
    return (b[0] if b else None), (p[0] if p else None)


def _read_with_selected_cols(path: str) -> pd.DataFrame:
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

    for c in NEEDED_COLS:
        if c not in df.columns:
            df[c] = np.nan

    filename = os.path.basename(path).lower()
    if df["scenario"].isna().all():
        df["scenario"] = "baseline" if "baseline" in filename else ("policy" if "policy" in filename else np.nan)

    if df["state_abbr"].isna().any():
        # if path is <STATE>/<RUN_ID>/<file>.csv, parent of parent is the state folder
        state = os.path.basename(os.path.dirname(os.path.dirname(path))) if os.path.basename(os.path.dirname(path)) in {"policy", "baseline"} else os.path.basename(os.path.dirname(os.path.dirname(path)))
        # safer: fallback to immediate parent if above logic misfires
        if not state or len(state) != 2:
            state = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if not state or len(state) != 2:
            state = os.path.basename(os.path.dirname(path))
        df.loc[df["state_abbr"].isna(), "state_abbr"] = state.upper()

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
        for col in ("year", "new_adopters", "number_of_adopters",
                    "system_kw", "system_kw_cum", "batt_kwh", "batt_kwh_cum",
                    "customers_in_bin", "max_market_share",
                    "avg_elec_price_cents_per_kwh",
                    "initial_batt_kwh", "initial_number_of_adopters",
                    "batt_adopters_this_year", "load_kwh_per_customer_in_bin_initial"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("new_adopters", "number_of_adopters",
                    "customers_in_bin", "max_market_share",
                    "system_kw_cum", "batt_kwh_cum",
                    "batt_adopters_this_year", "load_kwh_per_customer_in_bin_initial"):
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
    state_dirs = discover_state_dirs(root_dir)
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
    if df.empty:
        return

    def _fmt(v: float) -> str:
        m = y_col
        if m == "number_of_adopters":
            return f"{v/1e6:.1f}M"
        if m == "system_kw_cum":
            return f"{v/1e6:.1f} GW"
        if m == "batt_kwh_cum":
            return f"{v/1e6:.1f} GWh"
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

    global_years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
    preferred_year = 2040 if (len(global_years) and 2040 in set(global_years)) else (int(global_years.max()) if len(global_years) else None)

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
# National deltas & totals + peaks (unchanged)
# =============================================================================

def build_national_deltas(outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
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
    peaks_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
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
    peaks_df: Optional[pd.DataFrame] = None,
    metrics: Optional[Iterable[str]] = ("number_of_adopters", "system_kw_cum", "batt_kwh_cum", "cumulative_bill_savings", "peak_mw"),
    xticks: Iterable[int] = (2026, 2030, 2035, 2040),
    title: str = "U.S. Totals: Baseline vs Policy",
    ncols: int = 3,
) -> None:
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
        "peak_mw": "U.S. Peak Demand (MW)",
    }

    def _fmt(metric: str, v: float) -> str:
        if metric == "number_of_adopters":
            return f"{v/1e6:.1f}M"
        if metric == "system_kw_cum":
            return f"{v/1e6:.1f} GW"
        if metric == "batt_kwh_cum":
            return f"{v/1e6:.1f} GWh"
        if metric == "cumulative_bill_savings":
            return f"${v/1e9:.1f}B"
        if metric == "peak_mw":
            return f"{v/1e3:.1f} GW"
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

        years = d["year"].unique()
        end_year = 2040 if 2040 in years else int(d["year"].max())

        end_points = []
        for scen, g in d.groupby("scenario"):
            g_sorted = g.sort_values("year")
            ax.plot(g_sorted["year"], g_sorted["value"], marker="o", label=scen.capitalize())
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

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, y=1.05)
    plt.show()


# =============================================================================
# Other plots (unchanged)
# =============================================================================

def facet_median_storage_by_state(
    outputs: Dict[str, pd.DataFrame],
    xticks: Iterable[int] = (2026, 2030, 2035, 2040),
    height: float = 3.5,
    col_wrap: int = 4,
    sharey: bool = False,
    title: str = "Median Storage Size (kWh) — Baseline vs Policy",
) -> None:
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
# Convenience exports (unchanged)
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

def export_compiled_results_to_excel(
    outputs: Dict[str, pd.DataFrame],
    run_id: str,
    base_dir: str = "/Volumes/Seagate Portabl/permit_power/dgen_runs",
    peaks_df: Optional[pd.DataFrame] = None,
    include_national: bool = True,
) -> str:
    import datetime
    import pandas as pd
    import os
    import re

    compiled_dir = os.path.join(base_dir, "compiled_results")
    os.makedirs(compiled_dir, exist_ok=True)
    out_path = os.path.join(compiled_dir, f"{run_id}.xlsx")

    def _sheet_name(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9 _\-]", "_", str(name))
        return cleaned[:31] if cleaned else "Sheet"

    nat_totals = None
    nat_deltas = None
    if include_national:
        try:
            nat_totals = build_national_totals(outputs, peaks_df=peaks_df)
        except Exception:
            nat_totals = None
        try:
            nat_deltas = build_national_deltas(outputs)
        except Exception:
            nat_deltas = None

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as xw:
        meta = pd.DataFrame({
            "run_id": [run_id],
            "generated_at": [datetime.datetime.now().isoformat(timespec="seconds")],
            "tables_included": [", ".join(sorted([k for k, v in outputs.items() if isinstance(v, pd.DataFrame) and not v.empty]))]
        })
        meta.to_excel(xw, index=False, sheet_name=_sheet_name("README"))

        for key, df in outputs.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(xw, index=False, sheet_name=_sheet_name(key))

        if peaks_df is not None and isinstance(peaks_df, pd.DataFrame) and not peaks_df.empty:
            peaks_df.to_excel(xw, index=False, sheet_name=_sheet_name("peaks"))

        if include_national and isinstance(nat_totals, pd.DataFrame) and not nat_totals.empty:
            nat_totals.to_excel(xw, index=False, sheet_name=_sheet_name("national_totals"))

        if include_national and isinstance(nat_deltas, pd.DataFrame) and not nat_deltas.empty:
            nat_deltas.to_excel(xw, index=False, sheet_name=_sheet_name("national_deltas"))

    return out_path

def choropleth_pv_delta_gw_policy_vs_baseline(
    outputs: Dict[str, pd.DataFrame],
    shapefile_path: str = "../../../data/states.shp",
    year: int = 2040,
    k_bins: int = 10,  # Jenks classes
) -> pd.DataFrame:
    """
    Map absolute PV delta in GW:  ΔGW = (Policy_2040 − Baseline_2040) / 1e6.

    Uses outputs["totals"] from process_all_states(...) which already contains
    state/year/scenario aggregates including system_kw_cum. No CSV reads here.

    Returns a tidy DataFrame with:
      ['state_abbr','baseline_kw','policy_kw','delta_kw','delta_gw']
    """

    # ---- grab processed totals (already built upstream) ----
    totals = outputs.get("totals", pd.DataFrame())
    if totals.empty:
        raise ValueError("outputs['totals'] is empty; run process_all_states(...) first.")

    need = {"state_abbr", "year", "scenario", "system_kw_cum"}
    if not need.issubset(totals.columns):
        missing = need - set(totals.columns)
        raise ValueError(f"outputs['totals'] missing columns: {sorted(missing)}")

    s = totals.loc[totals["year"] == year, ["state_abbr", "scenario", "system_kw_cum"]].copy()
    # pivot to [state] x {baseline, policy}
    piv = s.pivot_table(index="state_abbr", columns="scenario", values="system_kw_cum", aggfunc="sum")
    # Make sure expected columns exist
    for col in ("baseline", "policy"):
        if col not in piv.columns:
            piv[col] = np.nan

    df = piv.reset_index().rename_axis(None, axis=1)
    df["delta_kw"] = df["policy"] - df["baseline"]
    df["delta_gw"] = df["delta_kw"] / 1_000_000.0
    df = df.rename(columns={"baseline": "baseline_kw", "policy": "policy_kw"})

    # ---- map join & plot (contiguous U.S.; exclude AK & HI) ----
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:5070")
    if "STUSPS" not in gdf.columns:
        for c in ("stusps", "STATE_ABBR", "STATE", "STATEFP"):
            if c in gdf.columns:
                gdf["STUSPS"] = gdf[c].astype(str).str.upper()
                break
    if "STUSPS" not in gdf.columns:
        raise ValueError("Shapefile must include a 'STUSPS' two-letter state code.")

    gdf["STUSPS"] = gdf["STUSPS"].astype(str).str.upper()
    gdf = gdf[~gdf["STUSPS"].isin({"AK", "HI", "PR", "GU", "VI", "AS", "MP", "DC"})].copy()

    plot_df = gdf.merge(df.rename(columns={"state_abbr": "STUSPS"}), on="STUSPS", how="left")

    # Jenks (Fisher-Jenks) when available; fallback to quantiles
    try:
        import mapclassify  # noqa: F401
        scheme = "fisher_jenks"
        plot_kwargs_extra = dict(k=int(k_bins))
    except Exception:
        scheme = "quantiles"
        plot_kwargs_extra = dict(k=int(k_bins))

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    plot_df.plot(
        column="delta_gw",
        cmap="Blues",
        linewidth=0.6,
        edgecolor="grey",
        legend=True,
        scheme=scheme,
        legend_kwds={"title": f"Policy − Baseline PV in {year} (GW)", 
                     "ncols":2, "fmt": "{:.1f}", "loc":"lower left"},
        ax=ax,
        missing_kwds={"color": "lightgray"},
        **plot_kwargs_extra,
    )
    ax.set_title(f"PV Capacity Δ in {year}: Policy − Baseline (GW) — Contiguous U.S.", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    # Tidy table out
    out = (plot_df[["STUSPS", "baseline_kw", "policy_kw", "delta_kw", "delta_gw"]]
           .rename(columns={"STUSPS": "state_abbr"})
           .sort_values("state_abbr")
           .reset_index(drop=True))
    return out

def policy_only_bill_price_diffs_after_adoption(
    root_dir: str,
    year: int = 2040,
    level: str = "US",   # "US" or "state"
    run_id: str | None = None,
    strict_run_id: bool = True,
) -> "pd.DataFrame":
    """
    Policy-only comparison of bills and prices among adopters in a given year.

    Weighted by adopters:
      price_with_kwh = sum(bill_with * adopters) / sum(load_kwh * adopters)
      price_wo_kwh   = sum(bill_wo   * adopters) / sum(load_kwh * adopters)

    Also returns average bill per adopter and differences.

    Returns columns:
      ['geo','year',
       'price_with_c_per_kwh','price_wo_c_per_kwh',
       'diff_price_c_per_kwh','pct_change_price',
       'avg_bill_with_per_adopter','avg_bill_wo_per_adopter',
       'diff_avg_bill_per_adopter']
    """
    import os
    import pandas as pd
    import numpy as np

    # Required columns (assumed present)
    USECOLS = [
        "state_abbr", "year", "new_adopters",
        "first_year_elec_bill_with_system",
        "first_year_elec_bill_wo_system",
        "load_kwh_per_customer_in_bin_initial",
    ]

    # Collect POLICY files only
    frames = []
    for state_dir in discover_state_dirs(root_dir):
        _b_csv, p_csv = find_state_files(state_dir, run_id=run_id, strict_run_id=strict_run_id)
        if not p_csv:
            continue
        df = pd.read_csv(p_csv, usecols=USECOLS)
        frames.append(df)

    x = pd.concat(frames, ignore_index=True)

    # Focus year + adopters only
    x = x[x["year"] == year].copy()
    x = x[x["new_adopters"] > 0].copy()

    # Adoption-weighted totals
    x["tot_bill_with"] = x["first_year_elec_bill_with_system"] * x["new_adopters"]
    x["tot_bill_wo"]   = x["first_year_elec_bill_wo_system"]   * x["new_adopters"]
    x["tot_load"]      = x["load_kwh_per_customer_in_bin_initial"] * x["new_adopters"]

    if level.lower() == "state":
        g = (x.groupby("state_abbr", as_index=False)
              [["tot_bill_with","tot_bill_wo","tot_load","new_adopters"]].sum())
        geo = g["state_abbr"]
    else:
        s = x[["tot_bill_with","tot_bill_wo","tot_load","new_adopters"]].sum()
        g = pd.DataFrame([s])
        geo = pd.Series(["US"])

    # Prices (USD/kWh) and bills ($/adopter)
    price_with = g["tot_bill_with"] / g["tot_load"]
    price_wo   = g["tot_bill_wo"]   / g["tot_load"]
    avg_with   = g["tot_bill_with"] / g["new_adopters"]
    avg_wo     = g["tot_bill_wo"]   / g["new_adopters"]

    out = pd.DataFrame({
        "geo": geo.values,
        "year": year,
        "price_with_c_per_kwh": price_with * 100.0,
        "price_wo_c_per_kwh":   price_wo   * 100.0,
    })
    out["diff_price_c_per_kwh"] = out["price_with_c_per_kwh"] - out["price_wo_c_per_kwh"]
    out["pct_change_price"] = 100.0 * (out["diff_price_c_per_kwh"] / out["price_wo_c_per_kwh"])

    out["avg_bill_with_per_adopter"] = avg_with
    out["avg_bill_wo_per_adopter"]   = avg_wo
    out["diff_avg_bill_per_adopter"] = out["avg_bill_with_per_adopter"] - out["avg_bill_wo_per_adopter"]

    return out.sort_values("geo").reset_index(drop=True)


def plot_us_cum_adopters_grouped(outputs: Dict[str, pd.DataFrame],
                                 xticks: Iterable[int] = (2026, 2030, 2035, 2040),
                                 title: str = "U.S. Cumulative Adopters — Baseline vs Policy (Grouped Bars)") -> pd.DataFrame:
    """
    Grouped bar plot of *national cumulative adopters* by year,
    with Baseline vs Policy as the bar groups.

    Reuses existing national aggregation logic (via build_national_totals in this file).

    Returns the tidy table used for plotting:
        ['year','scenario','value'] where value = U.S. total cumulative adopters
    """

    nat = build_national_totals(outputs)
    if nat.empty:
        raise ValueError("No national totals found. Run process_all_states(...) first.")

    d = nat[nat["metric"] == "number_of_adopters"].copy()
    if d.empty:
        raise ValueError("National totals lack 'number_of_adopters' metric.")

    # Plot
    plt.figure(figsize=(12, 5), constrained_layout=True)
    ax = sns.barplot(data=d, x="year", y="value", hue="scenario", errorbar=None, palette=["#a2e0fc", "#1bb3ef"])
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative adopters (millions)")
    # ax.set_xticks(list(xticks))

    # annotate bars in millions
    for c in ax.containers:
        ax.bar_label(c, labels=[f"{v/1e6:.1f}M" if np.isfinite(v) else "" for v in c.datavalues],
                     padding=2, fontsize=9)

    plt.legend(title=None, frameon=False)
    plt.show()
    return d.sort_values(["year", "scenario"]).reset_index(drop=True)






