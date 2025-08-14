# batt_dispatch_helpers.py
import numpy as np
import os

# ---------- Safe value helpers ----------

def _scalarize(x, how="first"):
    """Coerce PySAM outputs (which can be scalars, tuples, or lists) to float."""
    try:
        return float(x)
    except (TypeError, ValueError):
        pass
    if x is None:
        return 0.0
    try:
        if len(x) == 0:
            return 0.0
        if how == "sum":
            return float(np.sum(np.array(x, dtype=float)))
        # default: first element
        return float(np.array(x, dtype=float).flat[0])
    except Exception:
        return 0.0


def get_power_limits_kw(batt):
    """Return (discharge_kw_max, charge_kw_max) in kW (AC if available)."""
    def _get(attr_list, default=0.0):
        for grp, attr in attr_list:
            obj = getattr(batt, grp, None)
            if obj is not None and hasattr(obj, attr):
                try:
                    v = float(getattr(obj, attr))
                    if v > 0:
                        return v
                except Exception:
                    pass
        return default

    discharge_kw_max = _get([
        ("BatterySystem", "batt_power_discharge_max_kwac"),
        ("Outputs",       "batt_power_discharge_max_kwac"),
        ("BatterySystem", "batt_power_discharge_max_kwdc"),
        ("Outputs",       "batt_power_discharge_max_kwdc"),
    ], default=0.0)

    charge_kw_max = _get([
        ("BatterySystem", "batt_power_charge_max_kwac"),
        ("Outputs",       "batt_power_charge_max_kwac"),
        ("BatterySystem", "batt_power_charge_max_kwdc"),
        ("Outputs",       "batt_power_charge_max_kwdc"),
    ], default=0.0)

    return discharge_kw_max, charge_kw_max


# ---------- Retail-rate dispatch configuration (choice = 4) ----------

def configure_retail_rate_dispatch(
    batt,
    *,
    lookahead_hours=6,
    allow_export=True,
    allow_grid_charge=False,
    charge_only_when_surplus=True,
    load_forecast_choice=0,   # 0 = provide hourly load forecast
    wf_forecast_choice=0      # 0 = provide hourly PV forecast
):
    """
    Configure PySAM Battery for Retail Rate Dispatch (choice = 4).

    - Behind-the-meter, net-billing compatible.
    - No grid charging (PV-only charge) if allow_grid_charge=False.
    - Allow export if allow_export=True.
    - Prefer self-consumption when buy > sell; export when sell > buy (handled by retail-rate dispatch logic).
    - Uses provided PV/load forecasts with a specified look-ahead horizon.
    """
    bd = batt.BatteryDispatch

    # Core mode
    bd.batt_dispatch_choice = 4  # Retail Rate Dispatch (aka price/retail-rate driven)

    # Forecast controls (SAM uses these to plan charge/discharge)
    bd.batt_dispatch_load_forecast_choice = load_forecast_choice
    bd.batt_dispatch_wf_forecast_choice   = wf_forecast_choice
    bd.batt_look_ahead_hours              = int(lookahead_hours)
    bd.batt_dispatch_update_frequency_hours = 1  # hourly

    # Charging/discharging permissions
    bd.batt_dispatch_auto_can_gridcharge = 1 if allow_grid_charge else 0
    bd.batt_dispatch_charge_only_system_exceeds_load = 1 if charge_only_when_surplus else 0
    bd.batt_dispatch_discharge_only_load_exceeds_system = 0  # allow discharge even if load > PV
    # Behind-the-meter export permission
    try:
        bd.batt_dispatch_auto_btm_can_discharge_to_grid = 1 if allow_export else 0
    except Exception:
        # Some older bindings may not expose this; ignore if unavailable
        pass


# ---------- Diagnostics ----------

def dispatch_export_diags(
    agent,
    batt,
    gen,
    load_hourly,
    costs,
    optimization_result,
    utilityrate=None,
    *,
    sell_ts_kwh=None,        # prefer passing the exact $/kWh sell series you used
    night_eps=1e-6,          # gen < night_eps ⇒ “night”
    midday_hours=(11, 15),   # inclusive range for “midday”
):
    """
    Minimal, robust diagnostics for:
      • Midday PV surplus capture (PV→Batt vs PV→Grid),
      • Battery night export (kWh & $),
      • Basic bottlenecks (charge power cap vs SOC full).

    Revenue is computed against the sell price time series:
      - Prefer `sell_ts_kwh` if provided (recommended).
      - Else, fall back to `utilityrate.ElectricityRates.ur_ts_sell_rate`.

    All inputs are hourly.
    """
    agent_id = agent.loc['agent_id']

    # ---- Align horizons ----
    gen  = np.asarray(gen, dtype=float)
    load = np.asarray(load_hourly, dtype=float)
    H0   = min(gen.size, load.size) if (gen.size and load.size) else 0
    if H0 == 0:
        print(f"=== Dispatch/export diagnostics for {agent_id}: no gen/load data ===")
        return {}

    # ---- Sell series (robust) ----
    src = "override"
    if sell_ts_kwh is not None:
        sell = np.asarray(sell_ts_kwh, dtype=float).ravel()
    else:
        src = "utilityrate"
        if utilityrate is not None:
            sell = np.asarray(
                getattr(utilityrate.ElectricityRates, 'ur_ts_sell_rate', []),
                dtype=float
            ).ravel()
        else:
            sell = np.array([], dtype=float)

    # Normalize length: broadcast singletons; truncate/clip to H0
    if sell.size == 0:
        sell = np.zeros(H0, dtype=float)
    elif sell.size == 1 and H0 > 1:
        sell = np.full(H0, float(sell[0]), dtype=float)
    else:
        sell = sell[:H0]
    H = min(H0, sell.size)

    gen, load, sell = gen[:H], load[:H], sell[:H]

    # ---- Masks: midday & night ----
    hod = (np.arange(H) % 24)
    mid_lo, mid_hi = midday_hours
    midday = (hod >= mid_lo) & (hod <= mid_hi)
    night  = (gen < night_eps)  # dark or nearly dark

    # ---- PV surplus (post-load) ----
    surplus = np.maximum(gen - load, 0.0)
    surplus_total   = float(surplus.sum())
    surplus_mid_kwh = float(surplus[midday].sum())

    # ---- Safe pulls from Battery outputs ----
    def _a(name, default=0.0):
        try:
            return np.asarray(getattr(batt.Outputs, name), dtype=float)[:H]
        except Exception:
            return np.full(H, default, dtype=float)

    soc = _a('batt_SOC')
    s2b = _a('system_to_batt')   # PV → batt
    s2g = _a('system_to_grid')   # PV → grid
    b2l = _a('batt_to_load')
    b2g = _a('batt_to_grid')
    g2b = _a('grid_to_batt')

    # ---- Totals & capture ----
    s2b_total = float(s2b.sum())
    s2g_total = float(s2g.sum())
    s2b_mid   = float(s2b[midday].sum())
    s2g_mid   = float(s2g[midday].sum())
    cap_mid   = (s2b_mid / surplus_mid_kwh) if surplus_mid_kwh > 1e-9 else 0.0

    b2l_total = float(b2l.sum())
    b2g_total = float(b2g.sum())
    b2g_night = float(b2g[night].sum())

    # Approximate PV direct-to-load (PV self-consumption not via battery)
    s2l = np.maximum(gen - s2b - s2g, 0.0)
    s2l_total = float(s2l.sum())

    # ---- Sell stats (day vs night) ----
    day_mask   = ~night
    sell_day   = float(np.mean(sell[day_mask])) if np.any(day_mask) else 0.0
    sell_night = float(np.mean(sell[night]))    if np.any(night)    else 0.0
    sell_min, sell_max, sell_mean = float(sell.min()), float(sell.max()), float(sell.mean())

    # ---- Bottlenecks at midday ----
    try:
        dis_max_kw, chg_max_kw = get_power_limits_kw(batt)
    except NameError:
        from batt_dispatch_helpers import get_power_limits_kw
        dis_max_kw, chg_max_kw = get_power_limits_kw(batt)

    power_bound_mid = int(np.sum((surplus > (float(chg_max_kw) + 1e-6)) & (soc < 95.0) & midday))
    soc_bound_mid   = int(np.sum((surplus > 1e-6) & (soc >= 95.0) & midday))

    # ---- Bottlenecks at all times ----
    power_bound_all = int(np.sum((surplus > (float(chg_max_kw) + 1e-6)) & (soc < 95.0)))
    soc_bound_all   = int(np.sum((surplus > 1e-6) & (soc >= 95.0)))

    # ---- Revenues ($) using sell series ----
    batt_rev_all   = float((b2g * sell).sum())
    batt_rev_night = float((b2g[night] * sell[night]).sum())
    pv_rev_all     = float((s2g * sell).sum())
    pv_rev_mid     = float((s2g[midday] * sell[midday]).sum())

    # ---- BUY rate series (or scalar approx) for avoided-cost math ----
    def _approx_avg_buy_rate_from_utilityrate(ur):
        ER = ur.ElectricityRates
        mat = np.asarray(getattr(ER, "ur_ec_tou_mat", []), dtype=float)
        if mat.size == 0:
            return 0.0
        first_tier = mat[mat[:, 1] == 1]
        if first_tier.size == 0:
            return 0.0
        periods = (first_tier[:, 0].astype(int) - 1)
        prices  = first_tier[:, 4]
        period_price = {p: float(np.mean(prices[periods == p])) for p in np.unique(periods)}
        wkday = np.asarray(getattr(ER, "ur_ec_sched_weekday", []), dtype=int) - 1
        wkend = np.asarray(getattr(ER, "ur_ec_sched_weekend", []), dtype=int) - 1
        if wkday.size == 0 or wkend.size == 0:
            return float(np.mean(list(period_price.values())))
        counts = {p: 0 for p in period_price}
        for row in wkday:
            for p in row:
                counts[p] = counts.get(p, 0) + 5
        for row in wkend:
            for p in row:
                counts[p] = counts.get(p, 0) + 2
        total = sum(counts.values())
        if total == 0:
            return float(np.mean(list(period_price.values())))
        return sum(period_price[p] * counts[p] for p in period_price) / total

    if (utilityrate is not None) and int(getattr(utilityrate.ElectricityRates, "ur_en_ts_buy_rate", 0)) == 1:
        buy = np.asarray(getattr(utilityrate.ElectricityRates, "ur_ts_buy_rate", []), dtype=float).ravel()[:H]
        buy_src = "ts"
        if buy.size < H:
            buy = np.pad(buy, (0, H - buy.size), constant_values=buy[-1] if buy.size else 0.0)
    else:
        avg_buy = _approx_avg_buy_rate_from_utilityrate(utilityrate) if (utilityrate is not None) else 0.0
        buy = np.full(H, float(avg_buy), dtype=float)
        buy_src = "approx"

    # ---- Avoided retail spend ($) ----
    avoided_pv_self   = float((s2l * buy).sum())  # PV direct-to-load
    avoided_batt_self = float((b2l * buy).sum())  # Battery-to-load

    # ---- Approx system sizes ----
    def _fmt(v):
        return f"{v:.3f}" if np.isfinite(v) else "n/a"

    # PV kW: get the result from the optimization
    pv_kw_est = optimization_result.x

    # Battery kWh: prefer agent['batt_kwh']; else pull from Outputs if present
    batt_kwh_est = np.nan
    if 'batt_kwh' in agent.index:
        try: batt_kwh_est = float(agent.loc['batt_kwh'])
        except Exception: batt_kwh_est = np.nan
    if not np.isfinite(batt_kwh_est) and hasattr(batt, 'Outputs') and hasattr(batt.Outputs, 'batt_bank_installed_capacity'):
        try: batt_kwh_est = float(batt.Outputs.batt_bank_installed_capacity)
        except Exception: batt_kwh_est = np.nan

    # Battery costs
    u_kwh = float(costs.get('batt_capex_per_kwh_combined', costs.get('batt_capex_per_kwh', 0.0)))
    cap_mult = float(costs.get('cap_cost_multiplier', 1.0))
    batt_capex_core   = batt_kwh_est * u_kwh
    batt_capex_gross  = batt_capex_core * cap_mult

    if not os.environ.get('PG_CONN_STRING'):
        # ---- Prints (concise) ----
        print(f"=== Dispatch/export diagnostics for {agent_id} ===")
        print(f"Approx sizes: PV≈{_fmt(pv_kw_est)} kW | Battery≈{_fmt(batt_kwh_est)} kWh")
        print(f"sell source={src}, len={sell.size}, min/mean/max={sell_min:.4f}/{sell_mean:.4f}/{sell_max:.4f}, "
            f"mean_day={sell_day:.4f}, mean_night={sell_night:.4f}")
        print(f"PV surplus kWh: total={surplus_total:.3f} | midday={surplus_mid_kwh:.3f}")
        print(f"PV→Batt kWh: total={s2b_total:.3f} | midday={s2b_mid:.3f} | capture_mid={cap_mid:.3f}")
        print(f"PV→Grid kWh: total={s2g_total:.3f} | midday={s2g_mid:.3f}")
        print(f"Batt use kWh: to_load={b2l_total:.3f} | to_grid={b2g_total:.3f} | to_grid_at_night={b2g_night:.3f}")
        print(f"Revenue $: batt_all={batt_rev_all:.2f} | batt_night={batt_rev_night:.2f} | pv_all={pv_rev_all:.2f} | pv_midday={pv_rev_mid:.2f}")
        print(f"Avoided spend $ (buy={buy_src}): pv_self={avoided_pv_self:.2f} | batt_self={avoided_batt_self:.2f} | buy_mean={float(np.mean(buy)):.4f}")
        print(f"Battery system total installed cost $: ${round(batt_capex_gross,2)}")
        print(f"Limits: charge_cap_kW={float(chg_max_kw):.3f} | discharge_cap_kW={float(dis_max_kw):.3f} | "
            f"midday_power_bound_hrs={power_bound_mid} | midday_SOC_bound_hrs={soc_bound_mid}")
        print(f"Bottlenecks (all hours with PV surplus): power_bound_hrs={power_bound_all} | SOC_bound_hrs={soc_bound_all}")


    # ---- Return summary ----
    return {
        "surplus_total_kwh": surplus_total,
        "surplus_mid_kwh": surplus_mid_kwh,
        "pv_to_batt_total_kwh": s2b_total,
        "pv_to_batt_mid_kwh": s2b_mid,
        "pv_to_grid_total_kwh": s2g_total,
        "pv_to_grid_mid_kwh": s2g_mid,
        "pv_direct_to_load_total_kwh": s2l_total,
        "batt_to_load_kwh": b2l_total,
        "batt_to_grid_total_kwh": b2g_total,
        "batt_to_grid_night_kwh": b2g_night,
        "capture_mid_frac": cap_mid,
        "sell_mean_day": sell_day,
        "sell_mean_night": sell_night,
        "batt_rev_all_usd": batt_rev_all,
        "batt_rev_night_usd": batt_rev_night,
        "pv_rev_all_usd": pv_rev_all,
        "pv_rev_midday_usd": pv_rev_mid,
        "avoided_pv_self_usd": avoided_pv_self,
        "avoided_batt_self_usd": avoided_batt_self,
        "avg_buy_rate_used": float(np.mean(buy)),
        "pv_kw_est": pv_kw_est,
        "batt_kwh_est": batt_kwh_est,
        "batt_total_installed_cost": batt_capex_gross
    }


