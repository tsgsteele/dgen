# dGen Solar Modeling (ResStock Solar Modeling) — Documentation & Runbook

This repository contains a customized implementation of NREL’s **dGen** distributed-generation adoption model, tailored for:
- national counterfactuals (e.g., **“$ / W”** PV cost trajectories), and
- **state-level policy** sensitivity runs (e.g., instant permitting in a specific state).

This README is organized as:

1. Methodological overview of this dGen implementation  
2. End-to-end runbook for running dGen on Google Cloud and producing outputs  
3. How to switch between a national **$/W** run and a **single-state** policy run  
4. Input data updates, modifications, and known “out-of-date” areas  

---

## Repo orientation

### Core model scripts
- `dgen_model.py` — main model runner/orchestrator (scenario loop, year loop, IO, parallel sizing, diffusion, battery post-processing)
- `financial_functions.py` — PV (and PV+battery) sizing + economics + hourly tariff simulation (PySAM)
- `diffusion_functions_elec.py` — Bass diffusion implementation (PV diffusion; battery is applied later)
- `attachment_rate_functions.py` — battery attachment rate ingestion + integer allocation + hourly mix exports
- `input_data_functions.py` — imports (CSV → DB) and helpers for array/dict serialization to Postgres
- `utility_functions.py` — logging + DB connection utilities (Cloud SQL connector / private IP / local)

### Results & analysis scripts
- `schema_exporter.py` — exports model output schemas from Postgres (Cloud SQL) to local CSV files for analysis
- `analysis_functions.py` — loads exported CSVs and computes portfolio metrics (adoption, capacity, net load, savings, etc.)

### Cloud execution assets
- `docker/dgen/Dockerfile` — Docker image definition (referenced by `build_and_submit.sh`)
- `build_and_submit.sh` — builds/pushes the image to Artifact Registry and submits a batch job
- `dgen-batch-job-large-states.yaml` — example Google Cloud Batch job spec (multi-state)

### Data-prep notebooks
- `adjust_pv_batt_price_trajectories.ipynb`
- `adjust_state_level_prices.ipynb`
- `output_state_input_csvs.ipynb`
- `adjust_agent_file.ipynb`
- `process_load_growth.ipynb`

---

# 1. Methodological overview

This section documents the modeling mechanism (what the model is doing), in the order it is executed.

## 1.1 High-level pipeline (conceptual)

At a high level, each scenario-year loop is:

1. **Read inputs**
   - Inputs are primarily stored in **Cloud SQL** (e.g., solar potential/performance, hourly profiles, retail + wholesale prices, starting capacities, etc.).
2. **Initialize a run output schema**
   - Each scenario writes results into a run-specific Postgres schema (see §1.8).
3. **Apply year-specific inputs to each agent**
   - PV/battery prices, electricity price trajectories, load growth, incentives/financing terms, etc.
4. **Agent-level techno-economic optimization (PySAM)**
   - Size PV (PV-only) to maximize NPV (or equivalently minimize negative NPV), then evaluate PV+battery at that PV size.
5. **Economic metric calculation**
   - Payback, NPV, bill savings, etc.
6. **Market ceiling + adoption dynamics**
   - Translate economics (primarily payback) into a **maximum market share** ceiling.
   - Feed payback / market ceiling into **Bass diffusion** (innovation + imitation) to update market share and “new adopters”.
7. **Battery attachment (post-diffusion)**
   - Apply an **exogenous attachment rate** and allocate **integer** battery adopters; compute new and cumulative battery capacity.
8. **Write outputs**
   - Agent outputs, finance-series arrays, and state/RTO hourly aggregates.

The run typically proceeds through years up to the configured end year (e.g., **2040**), completes the **baseline** scenario, and then repeats the full year loop for the **policy** scenario.

---

## 1.2 Agent-based representation

- The model uses ~**25,000 agents** representing a statistically weighted subset of the U.S. housing stock (ResStock-derived).  
- Agents are tied to geography (county/utility territory or similar) and contain **weights** so that each agent represents many households/customers.
- Practically: agents in dense areas often have higher weights (representing more households), while rural agents may represent fewer.

---

## 1.3 Key input datasets (what lives in Cloud SQL)

- Solar generation potential / technology performance (PV performance tables)
- Hourly load profiles
- Existing solar installations / starting capacities
- Retail electricity price trajectories and escalation
- Wholesale electricity prices (for avoided-cost/value components, depending on tariff logic)
- Other scenario settings (Bass diffusion parameters, financing terms, etc.)

**Why Cloud SQL:** the SQL DB acts as a fast, centralized datastore (the data is too large / unwieldy for “just CSVs”, and often faster to work with than object storage when the model is running).

### 1.3.1 Tables read directly from `diffusion_shared` during a run

When running on Cloud SQL, `dgen_model.py` reads many “static” inputs from the `diffusion_shared` schema (via `pandas.read_sql_table`). The most operationally important ones to keep in sync are:

- **PV technology performance**
  - `pv_tech_performance_defaultFY19`
- **Retail electricity price change trajectory**
  - `ATB23_Mid_Case_retail`
- **Load growth**
  - `load_growth_to_model_adjusted`
- **Financing terms**
  - `financing_atb_FY23`
- **Baseline PV prices**
  - `pv_plus_batt_baseline`
- **Policy PV prices**
  - `pv_plus_batt_dollar_per_watt`

#### PV + battery cost table selection (baseline vs policy)

The model selects PV/battery cost trajectories based on the scenario schema name:

- If `"baseline"` is in `scenario_settings.schema`, it reads:
  - `pv_plus_batt_baseline`
- Otherwise (policy), it reads:
  - `pv_plus_batt_dollar_per_watt`

**Implication:** for a standard baseline/policy pair, you usually update the *policy* tables once (national run), or create/update state-specific tables/rows for a targeted policy run (state run).

**The baseline tables are state-specific.** `pv_price_baseline` and `pv_plus_batt_baseline` carry a
`state_abbr` column (49 states x 25 years, `res` only), anchored to LBNL *Tracking the Sun* 2025
published state medians: 13 states with a sample of n>=100 use their own median, the remaining 36 use
the national median ($3.6159/W). Battery cost stays national ($1,199.3/kWh, LBNL TTS 2024 — LBNL has
not published a clean 2025 storage median). The anchor is a level-shift onto model year 2026; the
forward decline keeps the NREL ATB/FY23 shape, and no inflation adjustment is applied. Built by
`Notebooks/build_baseline_upfront_cost.ipynb` -> `data/state_upfront_cost_lbnl_2025.csv` ->
`Notebooks/adjust_pv_batt_price_trajectories.ipynb`.

`agent_mutation.elec.apply_pv_prices` / `apply_pv_plus_batt_prices` merge on
`['state_abbr','sector_abbr','year']` when the price table has a `state_abbr` column and fall back to
`['sector_abbr','year']` when it does not (the policy dollar-per-watt tables are national by design).
Both log a warning when they take the national path, and when a merge leaves any agent without a
price — a silent collapse to one national price is otherwise invisible.

**Caveat:** the non-cloud read path (`USE_PRIVATE_IP_DIRECT` unset) fetches prices via an explicit
`SELECT` in `data_functions.get_technology_costs_solar` that does **not** include `state_abbr`, so a
local run silently collapses to a single national price. Every Batch yaml sets
`USE_PRIVATE_IP_DIRECT=1`, so cloud runs are unaffected.

---

## 1.4 Optimal system sizing and economics (PV-only, then PV+battery)

### Key design choice
This implementation sizes PV **once** (PV-only), then evaluates a battery at a fixed ratio (or configured sizing rule) and records *both* cases.

Mechanically (as implemented in `financial_functions.calc_system_size_and_performance()`):
1. **Optimize PV-only capacity** (`kW*`) via repeated PySAM evaluations and a 1-D optimizer.
2. **Run PV+battery once** at PV fixed to `kW*` (no re-optimization of PV with battery).
3. Store both:
   - PV-only hourly arrays + scalars (these drive diffusion economics)
   - PV+battery hourly arrays + scalars (used for storage accounting + hourly mix exports)

### Key hourly time-series fields produced by PySAM

For each agent-year, the sizing/economics step can write hourly arrays (lists) such as:

- `baseline_net_hourly` — hourly net load for non-adopters (no PV)
- `adopter_net_hourly_pvonly` — adopter hourly net load with PV only
- `adopter_net_hourly_with_batt` — adopter hourly net load with PV + battery

These hourly arrays are large, so downstream export/analysis often uses **aggregated** hourly outputs written to `state_hourly_agg` / `rto_hourly_agg` rather than pulling every agent’s hourly series.

### “What is optimized?”
In the walkthrough, the sizing step was described as selecting PV to **maximize NPV** and improve/pay down payback. In code, the objective is implemented as minimizing `-NPV` (equivalent).

---

## 1.5 Incentives and ITC handling

**There is no federal ITC in the current model.** `financial_functions.calc_system_size_and_performance`
pins `loan.TaxCreditIncentives.itc_fed_percent = [0]` on every call, and batteries carry their full
installed cost (an earlier `* 0.7` battery multiplier, which stood in for the ITC, has been removed).
Both changes are unconditional — they apply to every year, state and scenario.

Two things to know:

- The `input_main_itc_options` table is still read (`data_functions.get_itc_incentives`) and
  `itc_fraction_of_capex` is still merged onto agents, but nothing reads it any more. The plumbing is
  inert; reinstating an ITC is a one-line change at the `itc_fed_percent` assignment.
- That assignment is **load-bearing, not redundant**. `_init_pv_batt_stack` tries three named PySAM
  configs in order, and two of them (`PVWattsBatteryResidential`, `PVBatteryResidential`) ship with a
  30% `itc_fed_percent` default. Only `CustomGenerationBatteryResidential` defaults to 0. If the
  explicit zero were removed and the first config ever failed to construct, a 30% ITC would silently
  reappear.

State production-based incentives are separate and unaffected — see `config.PRODUCTION_INCENTIVES`
and the `pbi_sta_*` block in `financial_functions` (used for the NJ SREC study).

---

## 1.6 Market size, market share, and Bass diffusion

### Maximum market share (ceiling)
Economics (exclusively **payback period**) determine a ceiling on adoption (maximum market share). Shorter payback → higher maximum market share. This is based on empirical research describing the relationships between payback period in years and maximum market share. For a given year, maximum market share is identified using a look-up table, ` stored in the Cloud SQL database with maximum market share proportions for a given paypack period. 

### Bass diffusion formulation (used in dGen)

dGen models PV adoption using a **Bass diffusion process**, parameterized by an innovation term `p` and an imitation term `q`, and scaled by an economically determined maximum market share.

The cumulative fraction of the market that has adopted by time `t` is:

$$
F(t) = \frac{1 - e^{-(p+q)t}}{1 + \frac{q}{p} e^{-(p+q)t}}
$$

where:
- `p` is the **coefficient of innovation** (external influence),
- `q` is the **coefficient of imitation** (peer / word-of-mouth effects),
- `t` is time since diffusion began (in years).

#### Equivalent time formulation

Rather than using calendar time directly, dGen evaluates the diffusion curve using an **equivalent time** variable that reflects the current market position given historical adoption and economics:

- `t_eq`: equivalent time on the diffusion curve  
- `t_eq_yr1`: initial time offset applied in the first model year  
- `t_eq_2`: effective time used for the current solve year  

The time update is:

$$
t_{eq,2} =
\begin{cases}
t_{eq} + t_{eq,yr1}, & \text{first model year} \\
t_{eq} + 1, & \text{subsequent years}
\end{cases}
$$

#### Market share and new adoption

The Bass-implied market share is:

$$
M_{\text{bass}} = M_{\max} \cdot F(t_{eq,2})
$$

where `M_max` is the **maximum market share**, determined endogenously from system economics (e.g., payback).

Final market share is constrained to be non-decreasing:

$$
M = \max(M_{\text{bass}}, M_{t-1})
$$

New adoption in the current year is then:

$$
\Delta M = M - M_{t-1}
$$

$$
\text{New Adopters} = \Delta M \cdot W
$$

where `W` is the **developable agent weight** (eligible customers).

> **Important:** In this implementation, diffusion is performed **for PV adoption only**. Battery adoption is applied *after diffusion* using exogenous state-level attachment rates and integer allocation.

---

## 1.7 Battery attachment methodology (post-diffusion)

Battery adoption is applied after PV diffusion as an **exogenous attachment rate** process.

### Attachment rate source
`attachment_rate_functions._load_state_attachment_rates()` loads and aggregates attachment-rate data by:
- state
- sector
- year (weighted over quarters by install volume where possible)

### Integer allocation (largest remainder method)
New PV adopters per agent are often fractional (because agents are weighted). Battery adopters are allocated as integers:

`attachment_rate_functions._allocate_battery_adopters_integer()`:
- computes expected battery adopters = `attachment_rate × new PV adopters`
- allocates integer adopters across agents with **largest remainder**
- updates `new_batt_kw`, `new_batt_kwh`, and cumulatives

### Hourly exports use PV-only vs PV+battery mix
State/RTO hourly net load aggregates use the **cumulative adoption mix** to blend:
- baseline net load (non-adopters)
- PV-only adopter net load
- PV+battery adopter net load

### Flat attachment-rate scenarios (sensitivity runs)
Setting the env var `FLAT_STORAGE_ATTACHMENT_RATE` (a float in `[0,1]`, read in `config.py`) overrides
the per-state Ohm rates with one flat rate for every state, for both the baseline and policy scenario.
`data_functions.create_output_schema` then tags the schema `_a<pct>` (e.g. `_a75`) so runs are
self-describing in Cloud SQL. `submit_all_attach.sh <rate>` injects the variable into temp copies of
the Batch yamls and submits the full 49-state run; the checked-in yamls are untouched.

Keep that tag short — Postgres truncates identifiers at 63 bytes and the untagged schema name is
already ~56 characters. The original `_attach<pct>` tag pushed `baseline` names to 64 and every
schema creation failed with `IdentifierError`, while the two-characters-shorter `policy` names
survived. `create_output_schema` now trims the microsecond suffix and raises if a name still will not
fit.

### Attachment rate does not affect PV adoption
`calc_system_size_and_performance` computes **both** the PV-only and PV+battery economics for every
agent and never reads `storage_attachment_rate`; the rate is consumed only by the post-diffusion
allocation. Adoption is driven by PV-only payback. Verified empirically: the 5% and 75% runs produced
byte-identical `new_adopters`, `new_system_kw`, `system_capex_per_kw_combined`, `payback_period`, and
all per-agent cash-flow arrays, with battery counts differing by exactly 15.00x (= 75/5).

That makes any flat rate **derivable from an existing exported run** —
`synthesize_attachment_scenario.synthesize_rate(agents, rate)` re-runs the allocation at a new rate
and rebuilds the battery columns, no model run required. Proven exact: re-deriving 75% from the
exported 5% run reproduced the real 75% run's `totals` and `portfolio_annual_savings` with a maximum
difference of 0.0000 on every column. Two traps it handles, both caught by that back-test:
cumulatives must be seeded from each agent's pre-existing stock (`initial_batt_kwh`, which is *not* in
`analysis_functions.AGENT_USECOLS`), and `agent_id` is not globally unique — it restarts at 0 in each
state, so any per-agent cumsum must key on `(state_abbr, sector_abbr, agent_id)`. What it *cannot*
derive is `peaks` / `coincident`, because those come from the hourly aggregates, which weight the
PV-only and PV+battery load shapes by battery counts and whose per-agent hourly profiles are excluded
from the CSV export.

### Battery sizing is a fixed ratio, 2-hour duration
Battery size is pinned to PV size in `financial_functions`, not chosen independently:
`battery kWh = PV kW / 0.8` and `battery kW = battery kWh / 2`. Every battery is therefore a **2-hour**
system (verified: `batt_kwh / batt_kw` = 2.000 across all agent-years, zero variance). Reported
`batt_kwh` / `new_batt_kwh` are **installed nameplate energy capacity, not dispatched energy**.
Typical run values: 8.76 kW PV -> 11.47 kWh / 5.74 kW battery.

---

## 1.8 Outputs written per run schema

Each run writes to a dedicated output schema, typically named like:
- `diffusion_results_baseline_<state>_<endyear>_<timestamp>`
- `diffusion_results_policy_<state>_<endyear>_<timestamp>`

Key tables include:
- `agent_outputs`
- `agent_finance_series` (arrays for PV-only and PV+batt)
- `state_hourly_agg`
- `rto_hourly_agg`

Schema names are additionally tagged `_a<pct>` when `FLAT_STORAGE_ATTACHMENT_RATE` is set.

### Analysis-layer columns (computed on export, not stored per run)
`analysis_functions` adds these on top of the raw tables:

- `totals`: `new_batt_adopters`, `new_batt_kwh` alongside the solar `new_adopters` / `new_system_kw`.
- `portfolio_annual_savings`: `portfolio_annual_upfront_cost` with `_pv` / `_batt` split (**gross**
  installed capex in the install year — no tax credits, no netting of financing),
  `portfolio_annual_down_payment_cost` (the 30% cash share; `financing_atb_FY23.down_payment_fraction`
  = 0.7 becomes SAM's `debt_fraction = 70`), `portfolio_annual_financed_cost` (annual debt service,
  rolled forward by calendar year), and `portfolio_annual_out_of_pocket_cost` = down payment + loan
  payments. All six roll up into `national_totals`.

Note `portfolio_annual_financed_cost` is 0 in the first model year: SAM's debt arrays are zero at
index 0, so a 2026 install makes its first annual payment in 2027. The financed component then
accumulates as each year adds a new cohort of 20-year loans, and never declines within a 2026-2040
horizon because no loan matures before 2046.

---

# 2. Runbook: running dGen on Google Cloud and producing results

This runbook covers two workflows:

- **National run (49 states: lower 48 + DC)**: submit *multiple* Google Cloud Batch jobs (small/mid/mid-large/large state groups) to parallelize the national run and right-size compute per state.
- **State run (single state)**: submit *one* Batch job sized appropriately for the selected state (often used for policy-specific runs like instant permitting).

---

## 2.1 Authentication (local terminal → GCP)

### 2.1.1 Login + project selection

```bash
# 1) Login (interactive)
gcloud auth login

# 2) Select the project
gcloud config set project <GCP_PROJECT_ID>

# 3) (Often needed) set the default region
gcloud config set compute/region us-east1
```

### 2.1.2 Application Default Credentials (ADC)

```bash
# Needed for many Python libraries / Cloud SQL tools
gcloud auth application-default login
```

### 2.1.3 Docker / Artifact Registry auth

```bash
# Allow docker push to Artifact Registry (adjust region if needed)
gcloud auth configure-docker us-east1-docker.pkg.dev
```

---

## 2.2 National run (49 states: lower 48 + DC): build once, submit 11 Batch jobs

> **Washington DC** was added to the run set in 2026-09 (`states.csv`, `small_states_r2b.csv`,
> `DGEN_STATES` in the price-trajectory notebook, and the cost CSV where it takes the national
> fallback). DC has only 13 agents, which exposed a deadlock: `dgen_model` split agents into
> `LOCAL_CORES` chunks via `np.array_split`, and when a state has fewer agents than cores the empty
> chunks made the progress callback raise inside the multiprocessing pool, killing the result handler
> so `pool.join()` blocked forever — the task sat in RUNNING, silent, with no traceback. Chunks are
> now capped at `min(cores, n_agents)` with empties dropped, and the callback is wrapped defensively
> (any exception in a Pool callback deadlocks the run). All other states have >=32 agents.


### 2.2.1 Why the national run is split across multiple Batch jobs

A “national run” is executed as **multiple independent Batch jobs**, each processing a *subset* of states.
States are grouped by an approximate **agent-count / population proxy** so that:

- large-agent states run on larger machines (more cores/RAM),
- small-agent states run on smaller machines, and
- we increase parallelism by running multiple jobs at once.

### 2.2.2 Batch job “size classes” at a glance

| State-size class | Batch YAML | State list file (downloaded by tasks) | Typical machine type | `LOCAL_CORES` |
|---|---|---|---|---|
| Small | `dgen-batch-job-small-states.yaml` | `small_states_test.csv` | `c2d-highcpu-8` | 8 |
| Mid | `dgen-batch-job-mid-states.yaml` | `mid_states_test.csv` | `c2d-highcpu-16` | 16 |
| Mid-large | `dgen-batch-job-mid-large-states.yaml` | `mid_large_states_test.csv` | `c2d-highcpu-32` | 32 |
| Large | `dgen-batch-job-large-states.yaml` | `large_states_test.csv` | `c2d-highcpu-32` | 32 |

> “Typical machine type” reflects what `submit_all.sh` currently uses. You can change it, but keep the YAML `computeResource` requests consistent with the machine.  

### 2.2.3 How state selection works inside each Batch job

Each Batch job has `taskCount = number_of_states_in_that_group`. Each *task*:

1. downloads the baseline/policy templates and the group’s state-list CSV from GCS,
2. selects *one* state using `BATCH_TASK_INDEX`,
3. exports `BATCH_STATE="<STATE_ABBR>"` for `dgen_model.py`, and
4. generates exactly two scenario spreadsheets (baseline + policy) for that state before running the model.

**Implication:** your state-list CSVs must be correct, and the YAML `taskCount` must match the number of lines in the CSV.

### 2.2.4 Step-by-step: national run

#### Step 0 — Start the Cloud SQL instance (if stopped)

- Start the Cloud SQL instance in the GCP Console (Cloud SQL → Instances → Start)
- Confirm you know the:
  - **instance connection name** (e.g., `PROJECT:REGION:INSTANCE`)
  - whether you’re connecting via **private IP** (Batch jobs) and/or **Cloud SQL Proxy** (local tools)

#### Step 1 — Generate the state input CSVs (must cover all 49 states, incl. DC)

Run the notebook:

- `output_state_input_csvs.ipynb`

This should generate the four state-list CSVs used by the Batch yamls (one CSV per state-size class), and upload these CSVs to Google Cloud Storage. They are then downloaded from Google Cloud Storage during the model run.
Each CSV should have:

- **no header**
- one state per line: `ABBR,FULL_STATE_NAME`

#### Step 2 - Alter the PV price trajectories in accordance with the policy you want to model

- Run the `adjust_pv_batt_price_trajectories.ipynb`
  - For the baseline scenario, the current version of the notebook pulls LBNL *Tracking the Sun* data to calculate median solar prices, which then decline a certain percentage each year for the duration of the model run. This can be changed in the notebook. 
  - For the policy scenario, the starting solar price is hard-coded in the notebook and prices decline a certain percentage each year for the duration of the model run. This can also be changed in the notebook.

#### Step 3 — Set `taskCount` and `parallelism` in each Batch yaml

For each yaml in `batch_job_yamls/`:

- `taskCount` should equal the number of states (lines) in that group’s CSV.
- `parallelism` controls how many tasks run concurrently (≤ `taskCount`).

Example pattern:

```yaml
taskGroups:
  - taskCount: "<NUM_STATES_IN_GROUP>"
    parallelism: "<CONCURRENCY_FOR_GROUP>"
```

A common approach is:
- **large states**: `parallelism = taskCount` (run all large states at once)
- **small states**: `parallelism < taskCount` (avoid too many simultaneous workers)

#### Step 4 — Build + push the Docker image

Run:

```bash
bash build_and_submit.sh
```

For national runs, `build_and_submit.sh` should call `submit_all.sh` (not `submit_one.sh`).

This script submits 4 Batch jobs (one per state-size class).

---

## 2.3 State run (single state): submit 1 Batch job sized for that state

A “state run” uses the same machinery as the national run, but you:

- run **one state only** (often to test/implement a specific policy),
- set the selected group’s state-list CSV to contain **only that state**, and
- set the relevant Batch yaml `taskCount=1` and `parallelism=1`.

### 2.3.1 Step-by-step: state run

#### Step 0 — Identify the state-size class (large / mid-large / mid / small)

Use the same classification logic as the national run (agent-count / population proxy).
Pick the corresponding YAML + machine type.

> Practical note: the easiest approach is usually to edit the existing group CSV (e.g., `mid_states_test.csv`) so it contains only your state.

#### Step 1 — Update the state list CSV to contain just the target state

Create/update the relevant file so it contains exactly **one** line:

```csv
MD,Maryland
```

Upload it to GCS under the exact filename the YAML fetches (e.g., `mid_states_test.csv`).

#### Step 2 - Alter the PV price trajectories in accordance with the policy you want to model

- In the `input_data` folder in the repository, and then in the `pv_plus_batt_prices` folder, alter two csvs to change the PV price trajectories that dGen will use. 
  - For the baseline scenario, open the `pv_plus_batt_prices_FY23_mid.csv` and change the values in the `system_capex_per_kw_res` column to your desired baseline policy price. These should reflect national prices. Change the years 2026-2040, and ensure that prices are in $/kW. Save the file as `pv_plus_batt_prices_FY23_{STATE_ABBR}_baseline.csv`
  - For the policy scenario, follow the same steps as above, but alter the values in the  `pv_plus_batt_prices_FY23_dollar_watt.csv`, and save teh file as `pv_plus_batt_prices_FY23_{STATE_ABBR}_policy.csv`
- Run the `adjust_state_level_prices.ipynb` notebook - changing the state abbreviation input as needed.

#### Step 3 — Set `taskCount=1` and `parallelism=1` in the chosen Batch yaml

Example:

```yaml
taskGroups:
  - taskCount: "1"
    parallelism: "1"
```

#### Step 4 — Build + push Docker image (optional but common)

If you changed code, rebuild/push:

```bash
docker buildx build --platform linux/amd64   -f docker/dgen/Dockerfile   -t us-east1-docker.pkg.dev/<PROJECT>/<REPO>/dgen:latest   --push .
```

#### Step 5 — Submit one job

For state runs, `build_and_submit.sh` should call `submit_one.sh` (not `submit_all.sh`)

Update `submit_one.sh` to:
- name the job for your target state, and
- point at the correct yaml for that state’s size class.

#### Step 6 — Monitor and sanity-check

Sanity checks that catch “silently wrong” configs:

- logs should show the correct `BATCH_STATE` abbreviation
- the number of processed agents should be plausible for the chosen state (too low/high often indicates the wrong state list)

---

## 2.4 Export results and run analysis

### 2.4.1 Connect locally to Cloud SQL (Cloud SQL Proxy)

This pattern is commonly used for *local* export + analysis steps:

```bash
# Terminal A: run the proxy (leave running)
cloud-sql-proxy dgen-466702:us-east1:dgen-db
```

### 2.4.2 Export run schemas to local flat files

In the `Notebooks` folder, runt he `loading_model_results.ipynb` notebook to pull the schemas from the Cloud SQL instance, lightly process the results, and save them locally.

Be sure to change the `RUN_ID` to help identify which run you are exporting.

### 2.4.3 Run analysis notebooks / scripts

- Point your analysis notebook `analysis_of_model_results.ipynb` at the exported output root directory.
- Keep your exported “flat files” as the reproducible artifact for downstream aggregation/plots.

---

## 2.5 Cloud SQL cleanup and cost management

### 2.5.1 Optional: remove old result schemas before a run

If old `diffusion_results_*` schemas exist and you want a clean export, drop them before starting a new run.

To do this, you will have to `Activate Cloud Shell` using a button in the top right of the Google Cloud Project web interface. Once the cloud shell opens, in the command line, enter the following code:

`gcloud sql connect dgen-db --user=postgres --database=dgendb`

This code connects to the dGen Cloud SQL instance. It will prompt you for a password. The password is `postgres`.

Once you are connected to the SQL instance, run the following SQL code to drop schemas from previous model runs:

```
DO $$
DECLARE r record;
BEGIN
  PERFORM set_config('lock_timeout','5s', true);
  FOR r IN SELECT nspname FROM pg_namespace WHERE nspname LIKE 'diffusion_results%' LOOP
    BEGIN
      EXECUTE format('DROP SCHEMA IF EXISTS %I CASCADE;', r.nspname);
    EXCEPTION
      WHEN lock_not_available THEN
        RAISE NOTICE 'Skipped % due to lock timeout', r.nspname;
      WHEN insufficient_privilege THEN
        RAISE NOTICE 'Skipped % due to insufficient privilege', r.nspname;
      WHEN others THEN
        RAISE NOTICE 'Skipped % due to %', r.nspname, SQLERRM;
    END;
  END LOOP;
END$$;
```

### 2.5.2 Post-run: stop the SQL instance

1. Export results from Cloud SQL to local files.
2. **Stop** the Cloud SQL instance when done to control costs.
3. **Do not delete** the Cloud SQL instance.

---

# 3. Input data updates and known out-of-date areas

This section documents the custom updates we've made to the underlying dGen data, which notebooks can update these data sources going forward, and how to alter the PV price trajectories to model different policies.

## 3.1 Loads and load growth
`process_load_growth.ipynb` builds a revised load-growth table (and uploads it).

Refresh risk:
- Load forecasts (AEO vintages, IRPs, etc.) update regularly.

Checklist:
- update source dataset version
- confirm mappings and manual fixes
- re-upload to Cloud SQL
- validate state totals

---

## 3.2 Agent weighting and population totals
`adjust_agent_file.ipynb` updates the agent file (weights, load totals, tariffs, etc.).

Refresh risk:
- Housing counts and retail tariffs change often; calibration vintages drift.

Checklist:
- preserve required columns/flags
- validate state totals (customers and load)
- validate tariff structures remain compatible with PySAM tariff logic

---

## 3.3 Retail rates and tariff assumptions
Refresh risk:
- net-billing / NEM regimes and TOU schedules can change quickly.

Checklist:
- confirm special-case handling for states with complex tariffs (e.g., CA)
- confirm interpretation in PySAM remains consistent

---

## 3.4 PV + storage prices
Notebooks:
- `adjust_pv_batt_price_trajectories.ipynb` (national trajectories)
- `adjust_state_level_prices.ipynb` (state-specific adjustments)

Note:
- default baseline PV prices (NREL tech baselines) can be “too high” depending on use case; you may choose a lower annual decrease (example mentioned: ~0.5%/yr).

Checklist:
- confirm data vintage and filtering
- confirm inflation/base-year alignment
- confirm units (\$/W vs $/kW) and column naming

---

# Appendix A — Troubleshooting

## A.1 “dGen can’t connect to Cloud SQL”
- Confirm you are using the intended connection mode:
  - Batch job: private IP direct (VPC routing required)
  - Local export/update: Cloud SQL Auth Proxy
- Confirm IAM/network permissions and that the instance is running.

## A.2 “Docker build/push fails due to dependency issues”
- Identify which package became unavailable or incompatible.
- Pin versions in the Docker/conda environment.
- Rebuild and confirm the image exists in Artifact Registry.
- If urgent and model code hasn’t changed, you can temporarily reuse the last known-good image.

