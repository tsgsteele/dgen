"""
Distributed Generation Market Demand Model (dGen) - Open Source Release
National Renewable Energy Lab

This is the main module of the dGen Model. 
Running this module requires a properly installed environment with applicable scenario files. 
"""

import time
import os
import pandas as pd
import psycopg2.extras as pgx
import psycopg2.extensions
import pg8000.native
import numpy as np
import data_functions as datfunc
import utility_functions as utilfunc
import settings
import agent_mutation
import diffusion_functions_elec
import financial_functions
from functools import partial
import input_data_functions as iFuncs
import PySAM
import multiprocessing
from financial_functions import size_chunk, _init_worker
import logging
from sqlalchemy import event
from sqlalchemy.engine import Engine

# raise numpy and pandas warnings as exceptions
pd.set_option('mode.chained_assignment', None)
# Suppress pandas warnings
import warnings
warnings.simplefilter("ignore")

### Helper functions for exogenous application of storage attachment rates

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
    import numpy as np
    import pandas as pd

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
    import numpy as np, pandas as pd

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

def main(mode=None, resume_year=None, endyear=None, ReEDS_inputs=None):
    model_settings = settings.init_model_settings()
    os.makedirs(model_settings.out_dir, exist_ok=True)
    logger = utilfunc.get_logger(os.path.join(model_settings.out_dir, 'dg_model.log'))
    print(f"Detected CPUs = {os.cpu_count()}, multiprocessing.cpu_count() = {multiprocessing.cpu_count()}", flush=True)
    print(f"model_settings.local_cores = {model_settings.local_cores}")

    con, cur = utilfunc.make_con(model_settings.pg_conn_string, model_settings.role)
    engine = utilfunc.make_engine(model_settings.pg_engine_string)

    if isinstance(con, psycopg2.extensions.connection):
        pgx.register_hstore(con)
        logger.info(f"Connected to Postgres with: {model_settings.pg_params_log}")
    owner = model_settings.role

    scenario_names = []
    dup_n = 1
    out_subfolders = {'wind': [], 'solar': []}

    for i, scenario_file in enumerate(model_settings.input_scenarios, start=1):
        scenario_start_time = round(time.time())
        logger.info('============================================')
        logger.info(f"Running Scenario {i} of {len(model_settings.input_scenarios)}")

        scenario_settings = settings.init_scenario_settings(scenario_file, model_settings, con, cur, i-1)
        scenario_settings.input_data_dir = model_settings.input_data_dir
        datfunc.summarize_scenario(scenario_settings, model_settings)

        input_scenario = scenario_settings.input_scenario
        scen_name = scenario_settings.scen_name
        out_scen_path, scenario_names, dup_n = datfunc.create_scenario_results_folder(
            input_scenario, scen_name, scenario_names, model_settings.out_dir, dup_n
        )
        scenario_settings.dir_to_write_input_data = os.path.join(out_scen_path, 'input_data')
        scenario_settings.scen_output_dir = out_scen_path
        os.makedirs(scenario_settings.dir_to_write_input_data, exist_ok=True)

        schema = scenario_settings.schema
        max_market_share = datfunc.get_max_market_share(con, schema)
        inflation_rate = datfunc.get_annual_inflation(con, schema)
        bass_params = datfunc.get_bass_params(con, schema)
        agent_file_status = scenario_settings.agent_file_status

        logger.info("--------------Creating Agents---------------")
        if scenario_settings.techs in [['wind'], ['solar']]:
            solar_agents = iFuncs.import_agent_file(
                scenario_settings, con, cur, engine, model_settings,
                agent_file_status, input_name='agent_file'
            )
            #Subset to only single family and no renters
            solar_agents.df = (
                solar_agents.df[
                    (solar_agents.df['owner_occupancy_status'] == 1) &
                    (solar_agents.df['crb_model'] != "Multi-Family with 5+ Units")

                ]
            )
            cols_base = list(solar_agents.df.columns)

        if scenario_settings.techs == ['solar']:
            # load all static inputs
            state_incentives = datfunc.get_state_incentives(con)
            itc_options = datfunc.get_itc_incentives(con, schema)
            nem_state_capacity_limits = datfunc.get_nem_state(con, schema)
            nem_state_and_sector_attributes = datfunc.get_nem_state_by_sector(con, schema)
            nem_utility_and_sector_attributes = datfunc.get_nem_utility_by_sector(con, schema)
            nem_selected_scenario = datfunc.get_selected_scenario(con, schema)
            rate_switch_table = agent_mutation.elec.get_rate_switch_table(con)

            if os.environ.get('PG_CONN_STRING'):
                deprec_sch = pd.read_sql_table(
                    "deprec_sch_FY19",
                    con=engine,
                    schema="diffusion_shared"
                )

                carbon_intensities = pd.read_sql_table(
                    "carbon_intensities_FY19",
                    con=engine,
                    schema="diffusion_shared"
                )

                wholesale_elec_prices = pd.read_sql_table(
                    "ATB23_Mid_Case_wholesale",
                    con=engine,
                    schema="diffusion_shared"
                )

                pv_tech_traj = pd.read_sql_table(
                    "pv_tech_performance_defaultFY19",
                    con=engine,
                    schema="diffusion_shared"
                )

                elec_price_change_traj = pd.read_sql_table(
                    "ATB23_Mid_Case_retail",
                    con=engine,
                    schema="diffusion_shared"
                )

                load_growth = pd.read_sql_table(
                    "load_growth_to_model_adjusted",
                    con=engine,
                    schema="diffusion_shared"
                )
                # ADJUST PRICES BASED ON SCENARIO
                if "baseline" in scenario_settings.schema:
                    pv_price_traj = pd.read_sql_table(
                        "pv_price_baseline",
                        con=engine,
                        schema="diffusion_shared"
                    )

                    batt_price_traj = pd.read_sql_table(
                        "batt_prices_baseline",
                        con=engine,
                        schema="diffusion_shared"
                    )

                    pv_plus_batt_price_traj = pd.read_sql_table(
                        "pv_plus_batt_baseline",
                        con=engine,
                        schema="diffusion_shared"
                    )
                else:    
                    pv_price_traj = pd.read_sql_table(
                        "pv_price_dollar_per_watt",
                        con=engine,
                        schema="diffusion_shared"
                    )

                    batt_price_traj = pd.read_sql_table(
                        "batt_prices_dollar_per_watt",
                        con=engine,
                        schema="diffusion_shared"
                    )

                    pv_plus_batt_price_traj = pd.read_sql_table(
                        "pv_plus_batt_dollar_per_watt",
                        con=engine,
                        schema="diffusion_shared"
                    )

                financing_terms = pd.read_sql_table(
                    "financing_atb_FY23",
                    con=engine,
                    schema="diffusion_shared"
                )

                batt_tech_traj = pd.read_sql_table(
                    "batt_tech_performance_FY19",
                    con=engine,
                    schema="diffusion_shared"
                )

                value_of_resiliency = pd.read_sql_table(
                    "vor_FY20_mid",
                    con=engine,
                    schema="diffusion_shared"
                )

            else:
                # ingest static tables once
                deprec_sch = iFuncs.import_table(scenario_settings, con, engine, owner,
                                                input_name='depreciation_schedules', csv_import_function=iFuncs.deprec_schedule)
                carbon_intensities = iFuncs.import_table(scenario_settings, con, engine, owner,
                                                        input_name='carbon_intensities', csv_import_function=iFuncs.melt_year('grid_carbon_intensity_tco2_per_kwh'))
                wholesale_elec_prices = iFuncs.import_table(scenario_settings, con, engine, owner,
                                                        input_name='wholesale_electricity_prices', csv_import_function=iFuncs.process_wholesale_elec_prices)
                pv_tech_traj = iFuncs.import_table(scenario_settings, con, engine, owner,
                                                input_name='pv_tech_performance', csv_import_function=iFuncs.stacked_sectors)
                elec_price_change_traj = iFuncs.import_table(scenario_settings, con, engine, owner,
                                                            input_name='elec_prices', csv_import_function=iFuncs.process_elec_price_trajectories)
                load_growth = iFuncs.import_table(scenario_settings, con, engine, owner,
                                                input_name='load_growth', csv_import_function=iFuncs.stacked_sectors)
                pv_price_traj = iFuncs.import_table(scenario_settings, con, engine, owner,
                                                input_name='pv_prices', csv_import_function=iFuncs.stacked_sectors)
                batt_price_traj = iFuncs.import_table(scenario_settings, con, engine, owner,
                                                    input_name='batt_prices', csv_import_function=iFuncs.stacked_sectors)
                pv_plus_batt_price_traj = iFuncs.import_table(scenario_settings, con, engine, owner,
                                                            input_name='pv_plus_batt_prices', csv_import_function=iFuncs.stacked_sectors)
                financing_terms = iFuncs.import_table(scenario_settings, con, engine, owner,
                                                    input_name='financing_terms', csv_import_function=iFuncs.stacked_sectors)
                batt_tech_traj = iFuncs.import_table(scenario_settings, con, engine, owner,
                                                    input_name='batt_tech_performance', csv_import_function=iFuncs.stacked_sectors)
                value_of_resiliency = iFuncs.import_table(scenario_settings, con, engine, owner,
                                                        input_name='value_of_resiliency', csv_import_function=None)
            
            # Load the quarterly attachment rates and compute a state-level weighted average
            _state_rates = _load_state_attachment_rates("../input_data/ohm_attachment_rates.csv")

            # per-year loop
            for year in scenario_settings.model_years:
                logger.info(f'\tWorking on {year}')
                # reset new-year columns
                cols = list(solar_agents.df.columns)
                drop_cols = [c for c in cols if c not in cols_base]
                solar_agents.df.drop(drop_cols, axis=1, inplace=True)
                solar_agents.df['year'] = year
                is_first_year = (year == model_settings.start_year)

                # apply growth, rates, profiles, incentives…
                solar_agents.on_frame(agent_mutation.elec.apply_load_growth, [load_growth])
                cf_during_peak_demand = pd.read_csv('cf_during_peak_demand.csv')
                peak_demand_mw = pd.read_csv('peak_demand_mw.csv')
                if is_first_year:
                    last_year_installed_capacity = agent_mutation.elec.get_state_starting_capacities(con, schema)

                state_capacity_by_year = agent_mutation.elec.calc_state_capacity_by_year(
                    con, schema, load_growth, peak_demand_mw,
                    is_first_year, year, solar_agents, last_year_installed_capacity
                )
                net_metering_state_df, net_metering_utility_df = agent_mutation.elec.get_nem_settings(
                    nem_state_capacity_limits, nem_state_and_sector_attributes,
                    nem_utility_and_sector_attributes, nem_selected_scenario,
                    year, state_capacity_by_year, cf_during_peak_demand
                )
                solar_agents.on_frame(agent_mutation.elec.apply_export_tariff_params,
                                       [net_metering_state_df, net_metering_utility_df])
                solar_agents.on_frame(agent_mutation.elec.apply_elec_price_multiplier_and_escalator,
                                       [year, elec_price_change_traj])
                solar_agents.on_frame(agent_mutation.elec.apply_batt_tech_performance, 
                                       [batt_tech_traj])
                solar_agents.on_frame(agent_mutation.elec.apply_pv_tech_performance,
                                       [pv_tech_traj])
                solar_agents.on_frame(agent_mutation.elec.apply_pv_prices,
                                       [pv_price_traj])
                solar_agents.on_frame(agent_mutation.elec.apply_batt_prices,
                                       [batt_price_traj, batt_tech_traj, year])
                solar_agents.on_frame(agent_mutation.elec.apply_pv_plus_batt_prices,
                                       [pv_plus_batt_price_traj, batt_tech_traj, year])
                solar_agents.on_frame(agent_mutation.elec.apply_value_of_resiliency,
                                       [value_of_resiliency])
                solar_agents.on_frame(agent_mutation.elec.apply_depreciation_schedule,
                                       [deprec_sch])
                solar_agents.on_frame(agent_mutation.elec.apply_carbon_intensities,
                                       [carbon_intensities])
                solar_agents.on_frame(agent_mutation.elec.apply_wholesale_elec_prices,
                                       [wholesale_elec_prices])
                solar_agents.on_frame(agent_mutation.elec.apply_financial_params,
                                       [financing_terms, itc_options, inflation_rate])
                solar_agents.on_frame(agent_mutation.elec.apply_state_incentives,
                                       [state_incentives, year, model_settings.start_year, state_capacity_by_year])

                # ── parallel system‐sizing ──
                if os.name == 'posix':
                    cores = model_settings.local_cores
                else:
                    cores = None
                print(f"Using {cores} cores for parallel processing", flush=True)

                if cores is None:
                    solar_agents.chunk_on_row(
                        financial_functions.calc_system_size_and_performance,
                        sectors=scenario_settings.sectors,
                        cores=None,
                        rate_switch_table=rate_switch_table
                    )
                else:
                    from multiprocessing import get_context, Manager

                    # build a spawn‐based Pool with a DB connection in each worker
                    ctx = get_context('spawn')
                    pool = ctx.Pool(
                        processes=cores,
                        initializer=_init_worker,
                        initargs=(model_settings.pg_conn_string, model_settings.role)
                    )

                    worker_pids = [p.pid for p in pool._pool]
                    logger.info(f"Spawned {len(worker_pids)} workers, PIDs={worker_pids}")

                    # drop any large or per‐hour columns before splitting
                    drop_cols = [c for c in solar_agents.df.columns if c.endswith('_hourly')]
                    static_df = solar_agents.df.drop(columns=drop_cols).copy()

                    # split by agent ID
                    all_ids      = static_df.index.tolist()
                    chunks       = np.array_split(all_ids, cores)
                    total_agents = len(all_ids)

                    tasks = [
                        (
                            static_df.loc[chunk_ids],
                            scenario_settings.sectors,
                            rate_switch_table,
                            "simple"
                        )
                        for idx, chunk_ids in enumerate(chunks)
                    ]

                    logger.info(f"Sizing {total_agents} agents in {len(tasks)} chunks with {cores} workers")

                    # set up shared counters for progress tracking
                    manager          = Manager()
                    completed_chunks = manager.Value('i', 0)
                    processed_agents = manager.Value('i', 0)
                    lock             = manager.Lock()

                    def on_done(result, idx):
                        """
                        result: (df_chunk, agg)
                        """
                        df_chunk, agg = result
                        elapsed = time.time() - chunk_start[idx]
                        with lock:
                            completed_chunks.value += 1
                            processed_agents.value += len(df_chunk)
                            pct = processed_agents.value / total_agents

                        print(
                            f"[Chunk {idx+1}/{len(tasks)}] "
                            f"[{df_chunk['state_abbr'].iloc[0]}] "
                            f"sized {len(df_chunk)} agents in {elapsed:.2f}s → "
                            f"{processed_agents.value}/{total_agents} ({pct:.0%})",
                            flush=True
                        )
                        # return the result intact so parent can collect both df and agg
                        return (df_chunk, agg)

                    # dispatch
                    results = []
                    chunk_start = {}
                    for idx, args in enumerate(tasks):
                        chunk_start[idx] = time.time()
                        res = pool.apply_async(size_chunk, args=args, callback=partial(on_done, idx=idx))
                        results.append(res)

                    pool.close()
                    pool.join()

                    # collect & combine
                    got = [r.get() for r in results]  # list of (df_chunk, agg)
                    sized_chunks = [g[0] for g in got]
                    solar_agents.df = pd.concat(sized_chunks, axis=0)


                # downstream: max market share, developable load, market last year, diffusion…
                solar_agents.on_frame(financial_functions.calc_max_market_share, [max_market_share])
                solar_agents.on_frame(agent_mutation.elec.calculate_developable_customers_and_load)
                if is_first_year:
                    state_starting_capacities_df = agent_mutation.elec.get_state_starting_capacities(con, schema)
                    solar_agents.on_frame(agent_mutation.elec.estimate_initial_market_shares,
                                           [state_starting_capacities_df])
                    market_last_year_df = None
                else:
                    solar_agents.on_frame(agent_mutation.elec.apply_market_last_year,
                                           [market_last_year_df])

                solar_agents.df, market_last_year_df = diffusion_functions_elec.calc_diffusion_solar(
                    solar_agents.df, is_first_year, bass_params, year
                )

                # ensure agent_id is a real column before merge (only if the index is already agent_id)
                if solar_agents.df.index.name == 'agent_id' and 'agent_id' not in solar_agents.df.columns:
                    solar_agents.df = solar_agents.df.reset_index()  # creates 'agent_id' column from the index

                # 1) Merge onto the agent frame by state; fill missing with 0
                solar_agents.df = solar_agents.df.merge(_state_rates, on="state_abbr", how="left")
                solar_agents.df["storage_attachment_rate"] = solar_agents.df["storage_attachment_rate"].fillna(0.0)

                # restore agent_id as index without dropping the column (idempotent)
                if 'agent_id' in solar_agents.df.columns and solar_agents.df.index.name != 'agent_id':
                    solar_agents.df = solar_agents.df.set_index('agent_id', drop=False)

                # 2) Allocate **integer** battery adopters for THIS year and compute new/cum batt capacity
                solar_agents.df = _allocate_battery_adopters_integer(solar_agents.df, year)

                # 2a) Swap in cumulative battery capacity from the battery adoption allocation
                # --- simplest per-agent update of battery cumulatives for next year's handoff ---
                ml  = market_last_year_df.set_index("agent_id")
                cur = solar_agents.df.set_index("agent_id")[["batt_kw_cum", "batt_kwh_cum"]]

                # write the updated cumulatives into the "last_year" fields (by agent_id)
                ml.loc[cur.index, "batt_kw_cum_last_year"]  = cur["batt_kw_cum"].to_numpy()
                ml.loc[cur.index, "batt_kwh_cum_last_year"] = cur["batt_kwh_cum"].to_numpy()

                market_last_year_df = ml.reset_index()

                # 3) Export state-level hourly net using the actual cumulative mix (PV-only vs PV+Batt)
                export_state_hourly_with_storage_mix(engine, schema, owner, year, solar_agents.df)

                # 4) Update cumulatives for next year's state capacity table
                last_year_installed_capacity = solar_agents.df[['state_abbr','system_kw_cum','batt_kw_cum','batt_kwh_cum','year']].copy()
                last_year_installed_capacity = last_year_installed_capacity.loc[last_year_installed_capacity['year'] == year]
                last_year_installed_capacity = last_year_installed_capacity.groupby('state_abbr')[['system_kw_cum','batt_kw_cum','batt_kwh_cum']].sum().reset_index()

                # write outputs… (same as original)
                drop_list = [f for f in [
                    'index','reeds_reg','customers_in_bin_initial',
                    'load_kwh_per_customer_in _bin_initial','load_kwh_in_bin_initial',
                    'sector','roof_adjustment','load_kwh_in_bin','naep',
                    'first_year_elec_bill_savings_frac','metric',
                    'developable_load_kwh_in_bin',
                    'initial_pv_kw','initial_market_value',
                    'market_value_last_year','teq_yr1','mms_fix_zeros','ratio',
                    'teq2','f','new_adopt_fraction','bass_market_share',
                    'diffusion_market_share','new_market_value','market_value',
                    'total_gen_twh','tariff_dict','deprec_sch','cash_flow',
                    'cbi','ibi','pbi','cash_incentives','state_incentives',
                    'export_tariff_results'
                ] if f in solar_agents.df.columns]
                df_write = solar_agents.df.drop(drop_list, axis=1)
                df_write.to_pickle(os.path.join(out_scen_path, f'agent_df_{year}.pkl'))
                mode = 'replace' if year == scenario_settings.model_years[0] else 'append'
                iFuncs.df_to_psql(df_write, engine, schema, owner,
                                    'agent_outputs', if_exists=mode, append_transformations=True)
                del df_write

            # teardown and finish
            logger.info("---------Saving Model Results---------")
            out_subfolders = datfunc.create_tech_subfolders(out_scen_path, scenario_settings.techs, out_subfolders)
            pool.close(); pool.join()

        if i < len(model_settings.input_scenarios):
            pass
        else:
            engine.dispose()
            con.close()
        datfunc.drop_output_schema(model_settings.pg_conn_string, schema, model_settings.delete_output_schema)
        scenario_endtime = round(time.time())
        logger.info(f"-------------Model Run Complete in {round(scenario_start_time-scenario_endtime,1)}s-------------")

if __name__ == '__main__':
    main()

