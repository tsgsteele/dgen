from typing import Iterable, Any, List
import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine
import input_data_functions as iFuncs  # same helper used by export_state_hourly_with_storage_mix

_TABLE = "agent_finance_series"

def _norm25(x: Iterable[Any]) -> List[float]:
    """Return a 25-length list[float] (pad/truncate; replace non-finite with 0)."""
    try:
        a = np.asarray(list(x), dtype=float).ravel()
    except Exception:
        return [0.0] * 25
    if a.size < 25:
        a = np.pad(a, (0, 25 - a.size))
    elif a.size > 25:
        a = a[:25]
    a = np.where(np.isfinite(a), a, 0.0)
    return [float(v) for v in a.tolist()]

def export_agent_finance_series(
    engine: Engine,
    schema: str,
    owner: str,
    year: int,
    df_agents: pd.DataFrame
) -> None:
    """
    Mirror export_state_hourly_with_storage_mix:
      - Build records with plain Python lists (25-length)
      - Single DataFrame
      - Single df_to_psql append
    """
    need_any = [
        "cf_energy_value_pv_only", "utility_bill_w_sys_pv_only", "utility_bill_wo_sys_pv_only",
        "cf_energy_value_pv_batt", "utility_bill_w_sys_pv_batt", "utility_bill_wo_sys_pv_batt",
    ]
    if not any(c in df_agents.columns for c in need_any):
        return

    df = df_agents
    if df.index.name == "agent_id" and "agent_id" not in df.columns:
        df = df.reset_index()

    recs = []
    for _, r in df.iterrows():
        aid = int(r.get("agent_id", -1))

        # pv_only row (only if lists are present)
        if any(isinstance(r.get(c), (list, tuple)) for c in
               ["cf_energy_value_pv_only","utility_bill_w_sys_pv_only","utility_bill_wo_sys_pv_only"]):
            recs.append({
                "agent_id": aid,
                "year": int(year),
                "scenario_case": "pv_only",
                "cf_energy_value":     _norm25(r.get("cf_energy_value_pv_only", [])),
                "utility_bill_w_sys":  _norm25(r.get("utility_bill_w_sys_pv_only", [])),
                "utility_bill_wo_sys": _norm25(r.get("utility_bill_wo_sys_pv_only", [])),
            })

        # pv_batt row
        if any(isinstance(r.get(c), (list, tuple)) for c in
               ["cf_energy_value_pv_batt","utility_bill_w_sys_pv_batt","utility_bill_wo_sys_pv_batt"]):
            recs.append({
                "agent_id": aid,
                "year": int(year),
                "scenario_case": "pv_batt",
                "cf_energy_value":     _norm25(r.get("cf_energy_value_pv_batt", [])),
                "utility_bill_w_sys":  _norm25(r.get("utility_bill_w_sys_pv_batt", [])),
                "utility_bill_wo_sys": _norm25(r.get("utility_bill_wo_sys_pv_batt", [])),
            })

    if not recs:
        return

    out = pd.DataFrame.from_records(recs)
    iFuncs.df_to_psql(
        out, engine, schema, owner, _TABLE,
        if_exists="append", append_transformations=False
    )
