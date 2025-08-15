from __future__ import annotations

import os
import re
import sys
import glob
import math
import json
import time
import argparse
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import psycopg2 as pg


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
    sql = """
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name LIKE %s
    """
    with _connect(cp) as conn:
        df = pd.read_sql(sql, conn, params=(like_prefix + "%",))
    # stable order
    return sorted(df["schema_name"].tolist())


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
    chunksize: int = 200_000,
    overwrite: bool = True,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Stream-export "{schema}.agent_outputs" to a per-state CSV:
      <out_dir>/<state>/<scenario>_{run_id}.csv

    Returns:
        (schema_name, out_csv_path_or_None_if_skipped, error_message_or_None)
    """
    scenario, state = parse_schema_name(schema_name)
    if not scenario or not state:
        return (schema_name, None, f"Unrecognized schema pattern: {schema_name}")

    per_state_dir = os.path.join(out_dir, state)
    _ensure_dir(per_state_dir)

    out_csv = os.path.join(per_state_dir, f"{scenario}_{run_id}.csv")
    if not overwrite and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        return (schema_name, None, None)  # skip

    q = f'SELECT * FROM "{schema_name}".agent_outputs'
    first = True
    rowcount = 0

    try:
        with _connect(cp) as conn:
            for chunk in pd.read_sql(q, conn, chunksize=chunksize):
                # Enrich minimal metadata (handy later)
                if "scenario" not in chunk.columns:
                    chunk["scenario"] = scenario
                if "schema" not in chunk.columns:
                    chunk["schema"] = schema_name
                if "state_abbr" not in chunk.columns:
                    chunk["state_abbr"] = state.upper()

                chunk.to_csv(out_csv, mode=("w" if first else "a"),
                             index=False, header=first)
                first = False
                rowcount += len(chunk)

        # If query returned zero rows, write an empty CSV with at least headers
        if first:
            pd.DataFrame(columns=["scenario", "schema", "state_abbr"]).to_csv(out_csv, index=False)
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
    combine_per_state: bool = False,   # <— new: default off
) -> Dict[str, List[str]]:
    """
    Export all diffusion_result schemas in parallel, then combine per-state files.

    Returns a summary dict with:
      {
        "exported": [list of schema names exported],
        "skipped":  [schemas skipped],
        "failed":   [schemas that errored],
        "states":   [state codes produced]
      }
    """
    schemas = list_diffusion_schemas(cp)
    if not schemas:
        return {"exported": [], "skipped": [], "failed": [], "states": []}

    # Filter schemas by scenario/state if requested
    tasks: List[Tuple[ConnParams, str, str, str, int, bool]] = []
    for s in schemas:
        scen, st = parse_schema_name(s)
        if not scen or not st:
            continue
        if only_scenarios and scen not in set(map(str.lower, only_scenarios)):
            continue
        if states_filter and st.lower() not in set(map(str.lower, states_filter)):
            continue
        tasks.append((cp, s, out_dir, run_id, chunksize, overwrite))

    if not tasks:
        return {"exported": [], "skipped": [], "failed": [], "states": []}

    # Parallel export
    jobs = max(1, min(jobs, max(1, (cpu_count() or 2) - 1)))
    results: List[Tuple[str, Optional[str], Optional[str]]] = []
    if jobs == 1:
        results = [export_one_schema(*t) for t in tasks]
    else:
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
        if out_csv is None:  # skipped due to overwrite=False and file exists
            skipped.append(schema_name)
        else:
            exported.append(schema_name)
            if st:
                states_out.add(st.upper())

    # Optionally combine per state
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
    }

# -----------------------------
# Command-line interface (CLI)
# -----------------------------

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
    p.add_argument("--run-id", required=True, help="Run label to suffix filenames, e.g. test_run_20250814")
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument("--jobs", type=int, default=max(1, (cpu_count() or 2) // 2))
    p.add_argument("--no-overwrite", action="store_true", help="Skip schemas whose CSV already exists")
    p.add_argument("--scenarios", nargs="*", choices=["baseline", "policy"], help="Only export these scenarios")
    p.add_argument("--states", nargs="*", help="Only export these states (2-letter codes)")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cp = ConnParams(dbname=args.dbname, user=args.user, password=args.password, host=args.host, port=args.port)

    summary = export_all(
        cp=cp,
        out_dir=args.out_dir,
        run_id=args.run_id,
        chunksize=args.chunksize,
        jobs=args.jobs,
        overwrite=not args.no_overwrite,
        only_scenarios=args.scenarios,
        states_filter=args.states,
    )

    print(f"Exported: {len(summary['exported'])}")
    if summary["skipped"]:
        print(f"Skipped:  {len(summary['skipped'])}")
    if summary["failed"]:
        print("Failures:")
        for line in summary["failed"]:
            print(f"  - {line}")
    print(f"States:   {', '.join(summary['states'])}")
    return 0 if not summary["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
