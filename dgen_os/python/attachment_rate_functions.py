### Helper functions for exogenous application of storage attachment rates
import numpy as np
import pandas as pd
import input_data_functions as iFuncs


def _load_state_attachment_rates(csv_path: str = "../input_data/ohm_attachment_rates.csv") -> pd.DataFrame:
    """
    Load quarterly attachment data and compute a **state-level weighted average**
    attachment rate using **install_volume** as weights.

    CSV schema
    ----------
    Columns: state_abbr, metric (one of {'attachment_rate','install_volume'}),
             q2_24, q3_24, q4_24, q1_25  (quarterly values)
    Each state has two rows: one for 'attachment_rate', one for 'install_volume'.

    Returns
    -------
    pd.DataFrame with columns:
      - state_abbr
      - storage_attachment_rate  (weighted average in [0,1])
    """


    qcols = ["q2_24", "q3_24", "q4_24", "q1_25"]

    df = pd.read_csv(csv_path)
    # Split into rates and weights
    rates = df[df["metric"] == "attachment_rate"][["state_abbr"] + qcols].set_index("state_abbr")
    vols  = df[df["metric"] == "install_volume"][["state_abbr"] + qcols].set_index("state_abbr")

    # Align indexes and coerce numeric
    rates = rates.apply(pd.to_numeric, errors="coerce")
    vols  = vols.apply(pd.to_numeric, errors="coerce")
    rates, vols = rates.align(vols, join="outer")

    # Weighted average per state across available quarters
    weights = vols.fillna(0.0).to_numpy(dtype=float)
    values  = rates.to_numpy(dtype=float)

    # If all weights are zero or NaN, fall back to simple mean over available quarters
    wsum = np.nansum(weights, axis=1)
    num  = np.nansum(values * weights, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        wavg = num / wsum

    simple_mean = np.nanmean(values, axis=1)
    use_simple  = ~np.isfinite(wavg) | (wsum <= 0)
    out = np.where(use_simple, simple_mean, wavg)
    out = np.clip(out, 0.0, 1.0)  # keep in [0,1]

    res = pd.DataFrame({"state_abbr": rates.index, "storage_attachment_rate": out}).reset_index(drop=True)
    res["storage_attachment_rate"] = res["storage_attachment_rate"].fillna(0.0)
    return res


def _allocate_battery_adopters_integer(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Allocate integer battery adopters by state×sector for THIS YEAR
    using the largest remainders method.

    Notes
    -----
    • Diffusion (in diffusion_functions_elec.calc_diffusion_solar) already
      produces the correct per-year PV cohort via `new_adopters`.
    • This function must NOT overwrite `new_adopters`.
      (Earlier versions incorrectly recomputed it as
       number_of_adopters - initial_number_of_adopters, which
       double-counted all prior years.)
    • We only allocate which of those *new* PV adopters take storage,
      according to the state-level `storage_attachment_rate`.

    Parameters
    ----------
    df : pandas.DataFrame
        Agent-year frame (after diffusion). Must include:
          - 'state_abbr','sector_abbr','agent_id'
          - 'new_adopters','number_of_adopters'
          - 'batt_kw','batt_kwh',
            'batt_kw_cum_last_year','batt_kwh_cum_last_year'
          - 'storage_attachment_rate' (per state, in [0,1])
    year : int
        Current solve year (not used here, kept for compatibility).

    Returns
    -------
    pandas.DataFrame
        Copy of `df` with added columns:
          - 'batt_adopters_added_this_year' (int)
          - 'new_batt_kw','new_batt_kwh'
          - 'batt_kw_cum','batt_kwh_cum'
    """
    df = df.copy()

    # Ensure required columns exist (fill missing with 0)
    need = [
        'state_abbr','sector_abbr','agent_id','new_adopters','number_of_adopters',
        'batt_kw','batt_kwh','batt_kw_cum_last_year','batt_kwh_cum_last_year',
        'storage_attachment_rate'
    ]
    for c in need:
        if c not in df.columns:
            df[c] = df.index.astype(str) if c == 'agent_id' else 0.0

    # Integer allocation per state×sector using largest remainders
    alloc = pd.Series(0, index=df.index, dtype=int)

    for (s, sec), g in df.groupby(['state_abbr', 'sector_abbr'], sort=False):
        idx = g.index
        r = float(g['storage_attachment_rate'].iloc[0]) if len(g) else 0.0
        r = max(0.0, min(1.0, r))  # clamp to [0,1]

        n = g['new_adopters'].to_numpy(dtype=float)
        if n.sum() <= 0 or r <= 0:
            continue

        # Target number of battery adopters in this group
        target = int(round(r * n.sum()))

        # Initial floor allocation
        f = r * n
        base = np.floor(f).astype(int)
        rem = target - base.sum()

        if rem > 0:
            frac = f - base
            order_idx = (
                g.assign(_frac=frac, _aid=g['agent_id'].astype(str))
                 .sort_values(['_frac','_aid'], ascending=[False,True])
                 .index.to_numpy()
            )
            winners = order_idx[:rem]
            winners_mask = np.isin(idx.to_numpy(), winners)
            base = base.copy()
            base[winners_mask] += 1

        alloc.loc[idx] = base

    df['batt_adopters_added_this_year'] = alloc.reindex(df.index).astype(int).to_numpy()

    # Capacity additions and cumulatives
    df['new_batt_kw']  = df['batt_adopters_added_this_year'] * df['batt_kw']
    df['new_batt_kwh'] = df['batt_adopters_added_this_year'] * df['batt_kwh']
    df['batt_kw_cum']  = df['batt_kw_cum_last_year']  + df['new_batt_kw']
    df['batt_kwh_cum'] = df['batt_kwh_cum_last_year'] + df['new_batt_kwh']

    return df


def export_state_hourly_with_storage_mix(engine, schema, owner, year: int, solar_agents_df: pd.DataFrame) -> None:

    req = {"baseline_net_hourly","adopter_net_hourly_pvonly","adopter_net_hourly_with_batt"}
    if not req.issubset(solar_agents_df.columns):
        return

    def _len_safe(x):
        try: return len(x)
        except Exception: return 0

    records = []
    eps = 1e-9

    for state, g in solar_agents_df.groupby("state_abbr", sort=False):
        n_hours = int(min(
            g["baseline_net_hourly"].map(_len_safe).replace(0, np.nan).min(),
            g["adopter_net_hourly_pvonly"].map(_len_safe).replace(0, np.nan).min(),
            g["adopter_net_hourly_with_batt"].map(_len_safe).replace(0, np.nan).min(),
        ))
        if not (np.isfinite(n_hours) and n_hours > 0):
            continue

        def _arr(a):
            a = np.asarray(a, dtype=float)
            return a[:n_hours] if a.size >= n_hours else np.pad(a, (0, n_hours - a.size))

        net_sum_kw = np.zeros(n_hours, dtype=float)

        for _, r in g.iterrows():
            base = _arr(r["baseline_net_hourly"])
            pvo  = _arr(r["adopter_net_hourly_pvonly"])
            wbt  = _arr(r["adopter_net_hourly_with_batt"])

            n_cust  = float(r.get("customers_in_bin", 0.0))
            n_adopt = float(r.get("number_of_adopters", 0.0))
            n_non   = max(n_cust - n_adopt, 0.0)

            prev_batt_cum = float(r.get("batt_kw_cum_last_year", 0.0)) / max(float(r.get("batt_kw", 0.0)) or eps, eps)
            prev_batt_cum = int(round(max(prev_batt_cum, 0.0)))
            batt_add_this_year = int(r.get("batt_adopters_added_this_year", 0))
            batt_cum = max(prev_batt_cum + batt_add_this_year, 0)
            pvo_cum  = max(int(round(n_adopt)) - batt_cum, 0)

            net_sum_kw += (pvo * pvo_cum) + (wbt * batt_cum) + (base * n_non)

        records.append({
            "state_abbr": state,
            "year": int(year),
            "n_hours": int(n_hours),
            "net_sum": (net_sum_kw / 1000.0).tolist(),  # MW
        })

    if records:
        rec = pd.DataFrame.from_records(records)
        iFuncs.df_to_psql(rec, engine, schema, owner, "state_hourly_agg",
                          if_exists="append", append_transformations=False)
        
def export_rto_hourly_with_storage_mix(engine, schema, owner, year: int, solar_agents_df: pd.DataFrame) -> None:
    """
    Simple export of aggregated hourly net load at the RTO level after diffusion.
    Groups by 'rto' and writes to rto_hourly_agg table.
    """

    records = []
    eps = 1e-9

    for rto, g in solar_agents_df.groupby("rto"):
        n_hours = min(
            g["baseline_net_hourly"].map(len).min(),
            g["adopter_net_hourly_pvonly"].map(len).min(),
            g["adopter_net_hourly_with_batt"].map(len).min(),
        )
        net_sum_kw = np.zeros(n_hours, dtype=float)

        for _, r in g.iterrows():
            base = np.asarray(r["baseline_net_hourly"], dtype=float)[:n_hours]
            pvo  = np.asarray(r["adopter_net_hourly_pvonly"], dtype=float)[:n_hours]
            wbt  = np.asarray(r["adopter_net_hourly_with_batt"], dtype=float)[:n_hours]

            n_cust  = float(r["customers_in_bin"])
            n_adopt = float(r["number_of_adopters"])
            n_non   = max(n_cust - n_adopt, 0.0)

            prev_batt_cum = float(r["batt_kw_cum_last_year"]) / max(float(r["batt_kw"]) or eps, eps)
            prev_batt_cum = int(round(max(prev_batt_cum, 0.0)))
            batt_add_this_year = int(r["batt_adopters_added_this_year"])
            batt_cum = max(prev_batt_cum + batt_add_this_year, 0)
            pvo_cum  = max(int(round(n_adopt)) - batt_cum, 0)

            net_sum_kw += (pvo * pvo_cum) + (wbt * batt_cum) + (base * n_non)

        records.append({
            "rto": str(rto),
            "year": int(year),
            "n_hours": int(n_hours),
            "net_sum": (net_sum_kw / 1000.0).tolist(),  # MW
        })

    if records:
        rec = pd.DataFrame.from_records(records)
        iFuncs.df_to_psql(rec, engine, schema, owner, "rto_hourly_agg",
                          if_exists="append", append_transformations=False)
