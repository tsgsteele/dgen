"""
urdb_to_pysam.py
----------------
Convert a URDB JSON record (from usurdb.json) to the PySAM tariff_dict format
expected by financial_functions.normalize_tariff / process_tariff.

URDB format reference:
  https://openei.org/services/doc/rest/util_rates

PySAM UtilityRate5 field reference:
  ur_ec_tou_mat   rows: [period(1-based), tier(1-based), max_kwh, units_code, $/kWh, sell_$/kWh]
  ur_dc_flat_mat  rows: [period(1-based), tier(1-based), max_kw, $/kW]
  ur_dc_tou_mat   rows: [period(1-based), tier(1-based), max_kw, $/kW]
  ur_ec_sched_*   12x24 int matrix, 1-based period indices
  ur_dc_sched_*   12x24 int matrix, 1-based period indices
"""

from __future__ import annotations
from typing import Any

_BIG = 1e38  # PySAM's "no upper bound" sentinel

# URDB energy unit string → PySAM units code
_UNIT_MAP = {
    "kWh":          0,
    "kWh/kW":       1,
    "kWh daily":    2,
    "kWh/kW daily": 3,
}


def _plus1_sched(mat: list | None) -> list[list[int]]:
    """Convert 0-based 12x24 URDB period schedule to 1-based PySAM schedule."""
    if not mat:
        return [[1] * 24 for _ in range(12)]
    out = []
    for r in range(12):
        row = mat[r] if r < len(mat) else []
        fixed = []
        for c in range(24):
            v = row[c] if c < len(row) else 0
            try:
                fixed.append(int(v) + 1)
            except Exception:
                fixed.append(1)
        out.append(fixed)
    return out


def _build_ec_tou_mat(
    energy_rate_strux: list[dict],
    unit_code: int = 0,
    sell_rate: float = 0.0,
) -> list[list[float]]:
    """
    Build ur_ec_tou_mat from URDB energyRateStrux.

    energyRateStrux is a list of periods (0-indexed). Each period has
    energyRateTiers: list of {max?, unit?, rate, adj?}.

    Output row: [period(1-based), tier(1-based), max_kwh, units_code, rate+adj, sell_rate]
    """
    rows: list[list[float]] = []
    for p_idx, period in enumerate(energy_rate_strux):
        tiers = period.get("energyRateTiers") or []
        for t_idx, tier in enumerate(tiers):
            rate = float(tier.get("rate", 0.0))
            adj  = float(tier.get("adj",  0.0))
            max_kwh = float(tier["max"]) if "max" in tier else _BIG
            rows.append([
                float(p_idx + 1),
                float(t_idx + 1),
                max_kwh,
                float(unit_code),
                rate + adj,
                sell_rate,
            ])
    return rows


def _build_dc_flat_mat(flat_demand_strux: list[dict]) -> list[list[float]]:
    """
    Build ur_dc_flat_mat from URDB flatDemandStructure.

    flatDemandStructure is a list of months (0-indexed). Each month has
    flatDemandTiers: list of {max?, rate, adj?}.

    Output row: [month(1-based), tier(1-based), max_kw, $/kW]
    """
    rows: list[list[float]] = []
    for m_idx, month in enumerate(flat_demand_strux):
        tiers = month.get("flatDemandTiers") or []
        for t_idx, tier in enumerate(tiers):
            rate   = float(tier.get("rate", 0.0))
            adj    = float(tier.get("adj",  0.0))
            max_kw = float(tier["max"]) if "max" in tier else _BIG
            rows.append([
                float(m_idx + 1),
                float(t_idx + 1),
                max_kw,
                rate + adj,
            ])
    return rows


def _build_dc_tou_mat(demand_rate_strux: list[dict]) -> list[list[float]]:
    """
    Build ur_dc_tou_mat from URDB demandRateStructure.

    demandRateStructure is a list of periods (0-indexed). Each period has
    demandRateTiers: list of {max?, rate, adj?}.

    Output row: [period(1-based), tier(1-based), max_kw, $/kW]
    """
    rows: list[list[float]] = []
    for p_idx, period in enumerate(demand_rate_strux):
        tiers = period.get("demandRateTiers") or []
        for t_idx, tier in enumerate(tiers):
            rate   = float(tier.get("rate", 0.0))
            adj    = float(tier.get("adj",  0.0))
            max_kw = float(tier["max"]) if "max" in tier else _BIG
            rows.append([
                float(p_idx + 1),
                float(t_idx + 1),
                max_kw,
                rate + adj,
            ])
    return rows


def urdb_to_pysam(record: dict[str, Any], sell_rate: float = 0.0) -> dict[str, Any]:
    """
    Convert a single URDB JSON record to a PySAM-compatible tariff_dict.

    Parameters
    ----------
    record    : dict from usurdb.json (one element of the top-level list)
    sell_rate : net sell rate $/kWh applied to all energy export tiers (default 0)

    Returns
    -------
    dict compatible with financial_functions.normalize_tariff / process_tariff
    """
    out: dict[str, Any] = {}

    # --- Always enable electricity rates ---
    out["en_electricity_rates"] = 1
    out["ur_metering_option"]   = 0  # Net metering (normalize_tariff may override)

    # --- Fixed monthly charge ---
    out["ur_monthly_fixed_charge"] = float(record.get("fixedChargeFirstMeter") or 0.0)

    # --- Energy rate unit code ---
    unit_str  = record.get("energyRateUnits") or "kWh"
    unit_code = _UNIT_MAP.get(unit_str, 0)

    # --- Energy charge schedules (0-based in URDB → 1-based in PySAM) ---
    out["ur_ec_sched_weekday"] = _plus1_sched(record.get("energyWeekdaySched"))
    out["ur_ec_sched_weekend"] = _plus1_sched(record.get("energyWeekendSched"))

    # --- Energy TOU matrix ---
    energy_strux = record.get("energyRateStrux") or []
    out["ur_ec_tou_mat"] = _build_ec_tou_mat(energy_strux, unit_code, sell_rate)

    # --- Flat demand charges ---
    flat_strux = record.get("flatDemandStructure") or []
    dc_flat    = _build_dc_flat_mat(flat_strux)
    out["ur_dc_flat_mat"] = dc_flat

    # --- TOU demand charges ---
    tou_d_strux = record.get("demandRateStructure") or []
    dc_tou      = _build_dc_tou_mat(tou_d_strux)
    out["ur_dc_tou_mat"] = dc_tou

    # --- Demand schedules (0-based → 1-based) ---
    out["ur_dc_sched_weekday"] = _plus1_sched(record.get("demandWeekdaySched"))
    out["ur_dc_sched_weekend"] = _plus1_sched(record.get("demandWeekendSched"))

    # --- Enable demand charges if any structure present ---
    out["ur_dc_enable"]            = 1 if (dc_flat or dc_tou) else 0
    out["ur_enable_billing_demand"] = False

    return out
