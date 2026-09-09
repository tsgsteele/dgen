"""
Derive a different flat battery-attachment scenario from an ALREADY-EXPORTED run,
without re-running the model.

Why this is exact rather than an approximation
----------------------------------------------
`financial_functions.calc_system_size_and_performance` computes BOTH the PV-only and
the PV+battery economics for every agent on every run -- it never looks at
`storage_attachment_rate`. The attachment rate is consumed only afterwards, by
`attachment_rate_functions._allocate_battery_adopters_integer`, which decides how many
of that year's PV adopters are *counted* as battery adopters. So every exported run
already carries the full PV+battery cash flows (`cf_energy_value_pv_batt`,
`cf_debt_payment_total_pv_batt`) for all agents.

That also means PV adoption itself is independent of the attachment rate, which was
confirmed empirically: across 24 states that finished both the 5% and 75% runs,
`new_adopters` and `new_system_kw` were identical and battery adopters differed by
exactly 15.00x (= 75/5).

Re-deriving a scenario therefore means re-running the allocation at a new rate and
recomputing the battery columns that depend on it. Everything else is untouched.

Usage
-----
    from synthesize_attachment_scenario import synthesize_rate
    agents_100 = synthesize_rate(agents_5pct_df, rate=1.0)

Then feed `agents_100` to analysis_functions exactly like a real run's agents.

Always back-test before trusting a synthesized scenario for a deliverable:
`backtest_rate()` re-derives a rate you actually ran and diffs it against the real
output.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from attachment_rate_functions import _allocate_battery_adopters_integer

# Columns the allocation rewrites; everything else passes through unchanged.
BATTERY_COLS: List[str] = [
    "storage_attachment_rate",
    "batt_adopters_added_this_year",
    "new_batt_kw",
    "new_batt_kwh",
    "batt_kw_cum",
    "batt_kwh_cum",
    "batt_kw_cum_last_year",
    "batt_kwh_cum_last_year",
]


def synthesize_rate(agents: pd.DataFrame, rate: float) -> pd.DataFrame:
    """
    Re-derive the battery-attachment columns of an exported agent frame at a new
    flat rate, replicating the model's own per-year, per-(state,sector) largest-
    remainder allocation.

    Parameters
    ----------
    agents : pd.DataFrame
        Exported agent-year frame from a completed run (all years stacked). Must
        carry agent_id, state_abbr, sector_abbr, year, new_adopters, batt_kw, batt_kwh.
    rate : float
        Flat attachment rate in [0, 1] applied to every state, matching what
        FLAT_STORAGE_ATTACHMENT_RATE would do in the model.

    Returns
    -------
    pd.DataFrame
        Copy of `agents` with the battery columns recomputed. Adoption columns
        (new_adopters, new_system_kw, system_kw, ...) and all cash-flow arrays are
        left exactly as they were.
    """
    if not (0.0 <= float(rate) <= 1.0):
        raise ValueError(f"rate must be in [0,1], got {rate}")

    # The pre-existing storage stock is REQUIRED to rebuild cumulatives (see below).
    # analysis_functions.AGENT_USECOLS does NOT include initial_batt_kw/kwh, so a frame
    # loaded via DataWarehouse.from_disk is missing them -- seeding from zero then
    # silently understates every cumulative. Fail loudly instead: read the exported
    # CSVs directly with these columns in usecols.
    missing = [c for c in ("initial_batt_kw", "initial_batt_kwh") if c not in agents.columns]
    if missing:
        raise ValueError(
            f"agents frame is missing {missing}, which is needed to seed the battery "
            "cumulatives from each agent's pre-existing stock. Load the exported "
            "baseline.csv/policy.csv directly (these columns are not in "
            "analysis_functions.AGENT_USECOLS, so a DataWarehouse frame will not have them)."
        )

    df = agents.copy()
    df["storage_attachment_rate"] = float(rate)

    # The model allocates one year at a time, and its cumulative columns are carried
    # forward per agent. The exported *_cum_last_year values belong to the SOURCE
    # run's rate, so they must not be reused -- zero them here and rebuild the
    # cumulatives after all years are allocated.
    df["batt_kw_cum_last_year"] = 0.0
    df["batt_kwh_cum_last_year"] = 0.0

    out_years = []
    for yr, g in df.groupby("year", sort=True):
        out_years.append(_allocate_battery_adopters_integer(g, int(yr)))
    df = pd.concat(out_years, ignore_index=True)

    # Rebuild per-agent cumulatives across years (the model does this via the
    # *_cum_last_year hand-off between solve years, seeded in the first solve year
    # from the agent's PRE-EXISTING storage stock -- see agent_mutation.elec, where
    # initial_batt_kwh is set from batt_kwh_cum_last_year). Seeding from zero instead
    # understates every cumulative by that starting stock, which a back-test against
    # the real 75% run caught as a ~2.8% shortfall.
    #
    # IMPORTANT: agent_id is NOT globally unique -- it restarts at 0 in every state
    # (the model runs one state per Batch task, so it never notices). Grouping the
    # cumsum by agent_id alone silently merges different states' agents; a multi-state
    # back-test caught that as a large cumulative overstatement. Key on the full agent
    # identity instead.
    key = [c for c in ("state_abbr", "sector_abbr", "agent_id") if c in df.columns]
    df = df.sort_values(key + ["year"], kind="mergesort")
    for new_col, cum_col, init_col in (
        ("new_batt_kw", "batt_kw_cum", "initial_batt_kw"),
        ("new_batt_kwh", "batt_kwh_cum", "initial_batt_kwh"),
    ):
        seed = (pd.to_numeric(df[init_col], errors="coerce").fillna(0.0)
                if init_col in df.columns else 0.0)
        df[cum_col] = seed + df.groupby(key, sort=False)[new_col].cumsum()
        df[cum_col + "_last_year"] = df[cum_col] - df[new_col]

    return df.reset_index(drop=True)


def compare_frames(
    got: pd.DataFrame,
    expected: pd.DataFrame,
    keys: Optional[List[str]] = None,
    cols: Optional[List[str]] = None,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> pd.DataFrame:
    """
    Diff two agent frames on the battery columns. Returns one row per compared
    column with max absolute/relative deviation and a pass flag.
    """
    keys = keys or [c for c in ("state_abbr", "sector_abbr", "agent_id", "year")
                    if c in got.columns and c in expected.columns]
    cols = cols or [c for c in BATTERY_COLS if c in got.columns and c in expected.columns]

    a = got.set_index(keys).sort_index()
    b = expected.set_index(keys).sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    rows = []
    for c in cols:
        x = pd.to_numeric(a[c], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(b[c], errors="coerce").to_numpy(dtype=float)
        d = np.abs(x - y)
        denom = np.where(np.abs(y) > 0, np.abs(y), np.nan)
        rows.append({
            "column": c,
            "n": len(x),
            "max_abs_diff": float(np.nanmax(d)) if len(d) else 0.0,
            "max_rel_diff": float(np.nanmax(d / denom)) if len(d) else 0.0,
            "sum_got": float(np.nansum(x)),
            "sum_expected": float(np.nansum(y)),
            "match": bool(np.allclose(np.nan_to_num(x), np.nan_to_num(y), rtol=rtol, atol=atol)),
        })
    return pd.DataFrame(rows)


def backtest_rate(
    source_agents: pd.DataFrame,
    actual_agents: pd.DataFrame,
    rate: float,
) -> Dict[str, object]:
    """
    Synthesize `rate` from `source_agents` and diff it against `actual_agents`
    (a real model run at that same rate).

    Returns {"diff": DataFrame, "all_match": bool, "adoption_identical": bool}.
    A True/True result means synthesizing other rates from this source is sound.
    """
    synth = synthesize_rate(source_agents, rate)
    diff = compare_frames(synth, actual_agents)

    # Adoption must be identical between the two runs for the method to hold at all.
    adopt_cols = [c for c in ("new_adopters", "new_system_kw", "system_kw")
                  if c in source_agents.columns and c in actual_agents.columns]
    a = source_agents.set_index(["agent_id", "year"]).sort_index()
    b = actual_agents.set_index(["agent_id", "year"]).sort_index()
    common = a.index.intersection(b.index)
    adoption_identical = all(
        np.allclose(pd.to_numeric(a.loc[common, c], errors="coerce").fillna(0).to_numpy(float),
                    pd.to_numeric(b.loc[common, c], errors="coerce").fillna(0).to_numpy(float),
                    rtol=1e-9, atol=1e-9)
        for c in adopt_cols
    ) if len(common) else False

    return {
        "diff": diff,
        "all_match": bool(diff["match"].all()),
        "adoption_identical": adoption_identical,
        "n_compared": int(len(common)),
    }
