from __future__ import annotations

import os
import re
import sys
import glob
import json
import argparse
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import psycopg2 as pg
from psycopg2 import sql

SCHEMA_PREFIX = "diffusion_results_"
SCHEMA_RE = re.compile(r"^diffusion_results_(baseline|policy)_([a-z]{2})_", re.IGNORECASE)


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
    # Pull all schemas once and intersect in Python (simpler & safe)
    existing_all = set(list_diffusion_schemas(cp, like_prefix=""))
    exist = [s for s in req if s in existing_all]
    missing = [s for s in req if s not in existing_all]
    return exist, missing


def parse_schema_name(schema_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse scenario ('baseline'|'policy') and 2-letter state from a schema name.
    Example: diffusion_results_baseline_ca_2040_YYYYMMDD_xxx
    """
    m = SCHEMA_RE.match(schema_name)
    if not m:
        return None, None
    return m.group(1).lower(), m.group(2).lower()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_one_schema(
    cp: ConnParams,
    schema_name: str,
    out_dir: str,
    run_id: str,
    chunksize: int = 200_000,  # unused by COPY
    overwrite: bool = True,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Stream-export ONLY selected columns from "{schema}.agent_outputs", explicitly excluding
    the huge text columns that contain hourly arrays/lists.
    """
    from psycopg2 import sql as _sql

    EXCLUDE_COLS = {
        "wholesale_prices",
        "consumption_hourly_list",
        "generation_hourly_list",
        "batt_dispatch_profile_list",
    }
    exclude_regex = os.getenv("EXPORT_EXCLUDE_REGEX", r"(?:_hourly_|_ndarray$|_profile_list$|_list$)")
    exclude_re = re.compile(exclude_regex, re.IGNORECASE)

    scenario, state = parse_schema_name(schema_name)
    if not scenario or not state:
        return (schema_name, None, f"Unrecognized schema pattern: {schema_name}")

    per_state_dir = os.path.join(out_dir, state)
    _ensure_dir(per_state_dir)
    out_csv = os.path.join(per_state_dir, f"{scenario}_{run_id}.csv")

    if not overwrite and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        return (schema_name, None, None)

    try:
        with _connect(cp) as conn, conn.cursor() as cur:
            meta_sql = """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
            """
            cols_df = pd.read_sql(meta_sql, conn, params=(schema_name, "agent_outputs"))
            all_cols = [r["column_name"] for _, r in cols_df.iterrows()]

            selected_cols = [c for c in all_cols if c not in EXCLUDE_COLS and not exclude_re.search(c)]
            if not selected_cols:
                return (schema_name, None, f"No columns left after exclusion for {schema_name}.agent_outputs")

            col_list_sql = _sql.SQL(", ").join([_sql.SQL("a.{}").format(_sql.Identifier(c)) for c in selected_cols])

            copy_stmt = _sql.SQL(
                """
                COPY (
                  SELECT
                    {col_list},
                    {scen} AS "scenario",
                    {sch}  AS "schema",
                    {st}   AS "state_abbr"
                  FROM {schema}.{table} AS a
                ) TO STDOUT WITH CSV HEADER
                """
            ).format(
                col_list=col_list_sql,
                scen=_sql.Literal(scenario),
                sch=_sql.Literal(schema_name),
                st=_sql.Literal(state.upper()),
                schema=_sql.Identifier(schema_name),
                table=_sql.Identifier("agent_outputs"),
            )

            with open(out_csv, "w", newline="") as f:
                cur.copy_expert(copy_stmt.as_string(conn), f)

        return (schema_name, out_csv, None)

    except Exception as e:
        return (schema_name, None, f"{type(e).__name__}: {e}")


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
    """Return (baseline_csv, policy_csv) for a given run_id in a state folder, if present."""
    base = glob.glob(os.path.join(state_dir, f"baseline_{run_id}.csv"))
    pol = glob.glob(os.path.join(state_dir, f"policy_{run_id}.csv"))
    return (base[0] if base else None, pol[0] if pol else None)


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
    schemas_include: Optional[Iterable[str]] = None,   # <— NEW
) -> Dict[str, List[str]]:
    """
    Export selected diffusion_result schemas (optionally limited explicitly) in parallel,
    then optionally combine per-state files.

    Returns a summary dict with keys: exported, skipped, failed, states, missing.
    """
    missing_requested: List[str] = []
    if schemas_include:
        # Explicit list given: intersect with what's actually in the DB
        schemas, missing_requested = filter_existing_schemas(cp, schemas_include)
    else:
        schemas = list_diffusion_schemas(cp)

    if not schemas:
        return {"exported": [], "skipped": [], "failed": [], "states": [], "missing": missing_requested}

    # Build tasks with optional scenario/state filters
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

    # Parallel export
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

    # Optional per-state combine (baseline + policy for the same run_id)
    if combine_per_state:
        for st in sorted(states_out):
            sdir = os.path.join(out_dir, st)
            b_csv, p_csv = _find_state_files_for_run(sdir, run_id)
            if b_csv or p_csv:
                combined = os.path.join(sdir, f"both_scenarios_{run_id}.csv")
                combine_two_csvs(b_csv, p_csv, combined, chunksize=chunksize)

    return {
        "exported": exported,
        "skipped": skipped,
        "failed": failed,
        "states": sorted(states_out),
        "missing": missing_requested,
    }


# -----------------------------
# Command-line interface (CLI)
# -----------------------------

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
    p.add_argument("--run-id", required=True, help="Run label to suffix filenames, e.g. test_run_20250821")
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument("--jobs", type=int, default=max(1, (cpu_count() or 2) // 2))
    p.add_argument("--no-overwrite", action="store_true", help="Skip schemas whose CSV already exists")
    p.add_argument("--scenarios", nargs="*", choices=["baseline", "policy"], help="Only export these scenarios")
    p.add_argument("--states", nargs="*", help="Only export these states (2-letter codes)")
    p.add_argument("--combine-per-state", action="store_true", help="Also write both_scenarios_<run_id>.csv per state")
    # NEW: explicit schema selection
    p.add_argument("--schemas", nargs="*", help="Explicit schema names to export (exact matches)")
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
