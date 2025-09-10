from __future__ import annotations

import os
import re
import sys
import argparse
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Dict, Iterable, List, Optional, Tuple, Sequence

import pandas as pd
import psycopg2 as pg
from psycopg2 import sql

# -----------------------------
# Schema discovery & parsing
# -----------------------------

SCHEMA_PREFIX = "diffusion_results_"
SCHEMA_RE = re.compile(r"^diffusion_results_(baseline|policy)_([a-z]{2})_", re.IGNORECASE)


def parse_schema_name(schema_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse scenario ('baseline'|'policy') and 2-letter state from a schema name.
    Example: diffusion_results_baseline_ca_2040_YYYYMMDD_xxx -> ('baseline', 'ca')
    """
    m = SCHEMA_RE.match(schema_name)
    if not m:
        return None, None
    return m.group(1).lower(), m.group(2).lower()


# -----------------------------
# Connection utilities
# -----------------------------

@dataclass(frozen=True)
class ConnParams:
    dbname: str = "dgendb"
    user: str = "postgres"
    password: str = "postgres"
    host: str = "127.0.0.1"
    port: int = 5432

    @classmethod
    def from_env(cls) -> "ConnParams":
        return cls(
            dbname=os.getenv("PGDATABASE", "dgendb"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"),
            host=os.getenv("PGHOST", "127.0.0.1"),
            port=int(os.getenv("PGPORT", "5432")),
        )


def _connect(cp: ConnParams):
    return pg.connect(
        dbname=cp.dbname, user=cp.user, password=cp.password, host=cp.host, port=cp.port
    )


def list_diffusion_schemas(cp: ConnParams, like_prefix: str = SCHEMA_PREFIX) -> List[str]:
    """Return schema names that start with the given prefix."""
    q = """
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name LIKE %s
    """
    with _connect(cp) as conn:
        df = pd.read_sql(q, conn, params=(like_prefix + "%",))
    return sorted(df["schema_name"].tolist())


def filter_existing_schemas(cp: ConnParams, requested: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Intersect requested schemas with those that actually exist; return (existing, missing)."""
    req = sorted({s.strip() for s in requested if s and s.strip()})
    if not req:
        return [], []
    existing_all = set(list_diffusion_schemas(cp, like_prefix=""))
    exist = [s for s in req if s in existing_all]
    missing = [s for s in req if s not in existing_all]
    return exist, missing


# -----------------------------
# Filesystem helpers
# -----------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run_subdir(out_dir: str, state_abbr: str, run_id: str) -> str:
    """
    Create/return <out_dir>/<STATE>/<RUN_ID>.
    """
    d = os.path.join(out_dir, state_abbr.upper(), str(run_id))
    os.makedirs(d, exist_ok=True)
    return d


# -----------------------------
# SQL builders
# -----------------------------

from psycopg2 import sql as _sql

def _select_agent_outputs_with_finance_arrays_stmt(
    schema_name: str,
    selected_cols: Sequence[str],
    scenario_literal: str,   # from schema name
    state_literal: str,      # from schema name
) -> _sql.SQL:
    """
    Select from {schema}.agent_outputs AS a and LEFT JOIN pivoted
    {schema}.agent_finance_series to add:
      - cf_energy_value_pv_only / cf_energy_value_pv_batt
      - utility_bill_w_sys_pv_only / utility_bill_w_sys_pv_batt
      - utility_bill_wo_sys_pv_only / utility_bill_wo_sys_pv_batt
    Adds literal columns: scenario, schema, state_abbr.
    """
    col_list_sql = _sql.SQL(", ").join(
        [_sql.SQL("a.{}").format(_sql.Identifier(c)) for c in selected_cols]
    )

    return _sql.SQL("""
        WITH afs AS (
          SELECT
            agent_id,
            year,
            MAX(CASE WHEN scenario_case = 'pv_only' THEN cf_energy_value     END) AS cf_energy_value_pv_only,
            MAX(CASE WHEN scenario_case = 'pv_batt' THEN cf_energy_value     END) AS cf_energy_value_pv_batt,
            MAX(CASE WHEN scenario_case = 'pv_only' THEN utility_bill_w_sys  END) AS utility_bill_w_sys_pv_only,
            MAX(CASE WHEN scenario_case = 'pv_batt' THEN utility_bill_w_sys  END) AS utility_bill_w_sys_pv_batt,
            MAX(CASE WHEN scenario_case = 'pv_only' THEN utility_bill_wo_sys END) AS utility_bill_wo_sys_pv_only,
            MAX(CASE WHEN scenario_case = 'pv_batt' THEN utility_bill_wo_sys END) AS utility_bill_wo_sys_pv_batt
          FROM {schema}.agent_finance_series
          GROUP BY agent_id, year
        )
        SELECT
          {col_list},
          afs.cf_energy_value_pv_only,
          afs.cf_energy_value_pv_batt,
          afs.utility_bill_w_sys_pv_only,
          afs.utility_bill_w_sys_pv_batt,
          afs.utility_bill_wo_sys_pv_only,
          afs.utility_bill_wo_sys_pv_batt,
          {scen_lit} AS "scenario",
          {sch_lit}  AS "schema",
          {st_lit}   AS "state_abbr"
        FROM {schema}.agent_outputs AS a
        LEFT JOIN afs
          ON afs.agent_id = a.agent_id
         AND afs.year     = a.year
        ORDER BY a.year, a.agent_id
    """).format(
        schema=_sql.Identifier(schema_name),
        col_list=col_list_sql,
        scen_lit=_sql.Literal(scenario_literal),
        sch_lit=_sql.Literal(schema_name),
        st_lit=_sql.Literal(state_literal.upper()),
    )

# -----------------------------
# Per-schema export
# -----------------------------

def export_one_schema(
    cp: ConnParams,
    schema_name: str,
    out_dir: str,
    run_id: str,
    chunksize: int = 200_000,  # unused by COPY
    overwrite: bool = True,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Export per-state files for this schema into:
        <OUT_DIR>/<STATE>/<RUN_ID>/
            baseline.csv | policy.csv
            baseline_hourly.csv | policy_hourly.csv

    The flat CSV excludes heavy/hourly columns from agent_outputs,
    and includes 25-year arrays from agent_finance_series (pv_only & pv_batt).

    We derive 'scenario' and 'state_abbr' from the schema name and add them as literal columns.
    We do NOT filter on a.scenario/state_abbr (those cols may not exist in agent_outputs).
    """
    from psycopg2 import sql as _sql

    scenario, state = parse_schema_name(schema_name)
    if not scenario or not state:
        return (schema_name, None, f"Unrecognized schema pattern: {schema_name}")

    per_state_dir = _run_subdir(out_dir, state, run_id)  # nested layout

    # --- Discover columns and build SELECT ---
    EXCLUDE_COLS = {
        "wholesale_prices",
        "consumption_hourly_list",
        "generation_hourly_list",
        "batt_dispatch_profile_list",
        # if any legacy large columns exist in your env, add them here:
        "baseline_net_hourly",
        "adopter_net_hourly",
        "adopter_batt_hourly",
    }
    exclude_regex = os.getenv("EXPORT_EXCLUDE_REGEX", r"(?:_hourly_|_ndarray$|_profile_list$|_list$)")
    exclude_re = re.compile(exclude_regex, re.IGNORECASE)

    try:
        with _connect(cp) as conn, conn.cursor() as cur:
            # Columns in agent_outputs
            meta_sql = """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = 'agent_outputs'
                ORDER BY ordinal_position
            """
            cols_df = pd.read_sql(meta_sql, conn, params=(schema_name,))
            all_cols = [r["column_name"] for _, r in cols_df.iterrows()]

            selected_cols = [c for c in all_cols if c not in EXCLUDE_COLS and not exclude_re.search(c)]
            out_csv = os.path.join(per_state_dir, f"{scenario}.csv")

            if not selected_cols:
                # ensure empty CSV with header still appears
                if overwrite or not (os.path.exists(out_csv) and os.path.getsize(out_csv) > 0):
                    pd.DataFrame().to_csv(out_csv, index=False)
                agent_done = True
            else:
                stmt = _select_agent_outputs_with_finance_arrays_stmt(
                    schema_name=schema_name,
                    selected_cols=selected_cols,
                    scenario_literal=scenario,
                    state_literal=state,
                )

                # --- Flat export ---
                if overwrite or not (os.path.exists(out_csv) and os.path.getsize(out_csv) > 0):
                    copy_stmt = _sql.SQL("COPY ({}) TO STDOUT WITH CSV HEADER").format(stmt)
                    with open(out_csv, "w", newline="") as f:
                        cur.copy_expert(copy_stmt.as_string(conn), f)
                agent_done = True

            # --- Hourly export, if present ---
            hourly_csv = os.path.join(per_state_dir, f"{scenario}_hourly.csv")
            if overwrite or not (os.path.exists(hourly_csv) and os.path.getsize(hourly_csv) > 0):
                # Check table existence
                cur.execute(
                    """
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = 'state_hourly_agg'
                    LIMIT 1
                    """,
                    (schema_name,),
                )
                exists = cur.fetchone() is not None
                if exists:
                    select_hourly = _sql.SQL(
                        """
                        SELECT
                          {scen} AS "scenario",
                          {sch}  AS "schema",
                          {st}   AS "state_abbr",
                          a.year,
                          a.n_hours,
                          a.net_sum::text AS net_sum_text
                        FROM {schema}.state_hourly_agg AS a
                        ORDER BY a.year
                        """
                    ).format(
                        scen=_sql.Literal(scenario),
                        sch=_sql.Literal(schema_name),
                        st=_sql.Literal(state.upper()),
                        schema=_sql.Identifier(schema_name),
                    )
                    copy_hourly = _sql.SQL("COPY ({}) TO STDOUT WITH CSV HEADER").format(select_hourly)
                    with open(hourly_csv, "w", newline="") as f2:
                        cur.copy_expert(copy_hourly.as_string(conn), f2)

        return (schema_name, out_csv if agent_done else None, None)

    except Exception as e:
        return (schema_name, None, f"{type(e).__name__}: {e}")


# -----------------------------
# Per-state combining
# -----------------------------

def combine_two_csvs(
    baseline_csv: str,
    policy_csv: str,
    combined_csv: str,
    chunksize: int = 200_000,
) -> None:
    """Stream-append baseline + policy into combined_csv. Creates directory if missing."""
    _ensure_dir(os.path.dirname(combined_csv))
    wrote_header = False

    def _append(path: str):
        nonlocal wrote_header
        if not path or not os.path.exists(path):
            return
        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunk.to_csv(
                combined_csv,
                mode=("w" if not wrote_header else "a"),
                index=False,
                header=not wrote_header,
            )
            wrote_header = True

    _append(baseline_csv)
    _append(policy_csv)


def _find_state_files_for_run(state_dir: str, run_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (baseline_csv, policy_csv) under the nested layout."""
    subdir = os.path.join(state_dir, str(run_id))
    b = os.path.join(subdir, "baseline.csv")
    p = os.path.join(subdir, "policy.csv")
    return (b if os.path.exists(b) else None, p if os.path.exists(p) else None)


# -----------------------------
# Batch export / CLI
# -----------------------------

def export_all(
    cp: ConnParams,
    out_dir: str,
    run_id: str,
    chunksize: int = 200_000,
    jobs: int = 1,
    overwrite: bool = True,
    only_scenarios: Optional[Iterable[str]] = None,
    states_filter: Optional[Iterable[str]] = None,
    combine_per_state: bool = False,
    schemas_include: Optional[Iterable[str]] = None,
) -> Dict[str, List[str]]:
    """
    Export selected diffusion_result schemas (optionally limited explicitly) in parallel,
    then optionally combine per-state files.

    Returns a summary dict with keys: exported, skipped, failed, states, missing.
    """
    missing_requested: List[str] = []
    if schemas_include:
        schemas, missing_requested = filter_existing_schemas(cp, schemas_include)
    else:
        schemas = list_diffusion_schemas(cp)

    if not schemas:
        return {"exported": [], "skipped": [], "failed": [], "states": [], "missing": missing_requested}

    scen_set = {s.lower() for s in only_scenarios} if only_scenarios else None
    state_set = {s.lower() for s in states_filter} if states_filter else None

    tasks: List[Tuple[ConnParams, str, str, str, int, bool]] = []
    for s in sorted(schemas):
        scen, st = parse_schema_name(s)
        if not scen or not st:
            continue
        if scen_set and scen not in scen_set:
            continue
        if state_set and st.lower() not in state_set:
            continue
        tasks.append((cp, s, out_dir, run_id, chunksize, overwrite))

    if not tasks:
        return {"exported": [], "skipped": [], "failed": [], "states": [], "missing": missing_requested}

    jobs = max(1, min(jobs, max(1, (cpu_count() or 2) - 1)))
    if jobs == 1:
        results = [export_one_schema(*t) for t in tasks]
    else:
        results = []
        with Pool(processes=jobs) as pool:
            for res in pool.starmap(export_one_schema, tasks):
                results.append(res)

    exported, skipped, failed = [], [], []
    states_out = set()

    for schema_name, out_csv, err in results:
        scen, st = parse_schema_name(schema_name)
        if err:
            failed.append(f"{schema_name} -> {err}")
            continue
        if out_csv is None:
            skipped.append(schema_name)
        else:
            exported.append(schema_name)
            if st:
                states_out.add(st.upper())

    # Optional per-state combine (baseline + policy for the same run_id) — write inside run subfolder
    if combine_per_state:
        for st in sorted(states_out):
            sdir = os.path.join(out_dir, st)
            b_csv, p_csv = _find_state_files_for_run(sdir, run_id)
            subdir = os.path.join(sdir, str(run_id))
            if b_csv or p_csv:
                _ensure_dir(subdir)
                combined = os.path.join(subdir, "both_scenarios.csv")
                combine_two_csvs(b_csv, p_csv, combined, chunksize=chunksize)

    return {
        "exported": exported,
        "skipped": skipped,
        "failed": failed,
        "states": sorted(states_out),
        "missing": missing_requested,
    }


def _read_schemas_file(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream-export diffusion_results_* schemas to per-state CSVs (parallel)."
    )
    p.add_argument("--dbname", default=os.getenv("PGDATABASE", "dgendb"))
    p.add_argument("--user", default=os.getenv("PGUSER", "postgres"))
    p.add_argument("--password", default=os.getenv("PGPASSWORD", "postgres"))
    p.add_argument("--host", default=os.getenv("PGHOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    p.add_argument("--out-dir", required=True, help="Output root dir (state subfolders will be created)")
    p.add_argument("--run-id", required=True, help="Run label to create subfolder names, e.g. r_20250906_0130")
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument("--jobs", type=int, default=max(1, (cpu_count() or 2) // 2))
    p.add_argument("--no-overwrite", action="store_true", help="Skip schemas whose CSV already exists")
    p.add_argument("--scenarios", nargs="*", choices=["baseline", "policy"], help="Only export these scenarios")
    p.add_argument("--states", nargs="*", help="Only export these states (2-letter codes)")
    p.add_argument("--combine-per-state", action="store_true", help="Also write both_scenarios.csv per state/run_id")
    # explicit schema selection (exact matches)
    p.add_argument("--schemas", nargs="*", help="Explicit schema names to export")
    p.add_argument("--schemas-file", help="Path to newline-delimited schema names to export")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cp = ConnParams(dbname=args.dbname, user=args.user, password=args.password, host=args.host, port=args.port)

    explicit_schemas: List[str] = []
    if args.schemas_file:
        explicit_schemas.extend(_read_schemas_file(args.schemas_file))
    if args.schemas:
        explicit_schemas.extend(args.schemas)
    # Dedup while preserving order
    seen = set()
    explicit_schemas = [s for s in explicit_schemas if not (s in seen or seen.add(s))]

    summary = export_all(
        cp=cp,
        out_dir=args.out_dir,
        run_id=args.run_id,
        chunksize=args.chunksize,
        jobs=args.jobs,
        overwrite=not args.no_overwrite,
        only_scenarios=args.scenarios,
        states_filter=args.states,
        combine_per_state=args.combine_per_state,
        schemas_include=explicit_schemas if explicit_schemas else None,
    )

    print(f"Exported: {len(summary['exported'])}")
    if summary["skipped"]:
        print(f"Skipped:  {len(summary['skipped'])}")
    if summary["failed"]:
        print("Failures:")
        for line in summary["failed"]:
            print(f"  - {line}")
    if summary.get("missing"):
        print("Requested but not found:")
        for s in summary["missing"]:
            print(f"  - {s}")
    print(f"States:   {', '.join(summary['states']) if summary['states'] else '(none)'}")
    return 0 if not summary["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
