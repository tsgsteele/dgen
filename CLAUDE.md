# CLAUDE.md — dGen Solar Modeling (ResStock Solar Modeling fork)

This file gives Claude Code the context it needs to assist effectively with this repository.

---

## Project overview

This is a **fork of NREL's dGen** (Distributed Generation) model, customized for ResStock-based solar adoption research. It models residential PV (and PV+battery) adoption across U.S. states using Bass diffusion, agent-based economics, and PySAM for system sizing.

Primary use cases:
- National counterfactual runs (e.g., varying $/W PV cost trajectories across all 48 states)
- Single-state policy sensitivity runs (e.g., instant-permitting policy in one state)

The model runs on **Google Cloud** (Cloud Batch + Cloud SQL + Artifact Registry). Local development is mainly for data prep, analysis, and debugging.

---

## Repo layout

```
dgen/
├── dgen_os/
│   ├── python/              # All runnable model code + notebooks
│   │   ├── dgen_model.py            # Main model runner (entry point)
│   │   ├── financial_functions.py   # PySAM sizing + economics
│   │   ├── diffusion_functions_elec.py  # Bass diffusion
│   │   ├── attachment_rate_functions.py # Battery attachment
│   │   ├── input_data_functions.py  # CSV → DB imports
│   │   ├── utility_functions.py     # Logging + DB connection (Cloud SQL / local)
│   │   ├── analysis_functions.py    # Post-run portfolio metrics
│   │   ├── schema_exporter.py       # Cloud SQL → local CSV export
│   │   ├── config.py                # Model config (start_year, cores, flags)
│   │   ├── dg3n.yml                 # Conda environment (used in Docker)
│   │   ├── pg_params_connect.json   # DB credentials — GITIGNORED, never commit
│   │   └── Notebooks/               # Jupyter notebooks for data prep + analysis
│   ├── input_data/          # Static input CSVs (prices, battery, financing, etc.)
│   ├── input_agents/        # Agent pickle files — GITIGNORED (too large)
│   └── input_scenarios/     # Scenario spreadsheets — GITIGNORED
├── docker/
│   ├── dgen/Dockerfile      # dGen app image (continuumio/miniconda3 base)
│   ├── postgis/Dockerfile   # PostGIS image (postgis/postgis:11-3.3)
│   └── docker-compose.yml   # Local full-stack (dgen + postgis)
├── .devcontainer/
│   └── devcontainer.json    # VS Code dev container config (uses docker-compose)
├── batch_job_yamls/         # Google Cloud Batch job specs (4 state-size classes)
├── state_input_csvs/        # State list CSVs used by Batch tasks
├── build_envs/
│   └── environment.yml      # Conda env for CI (dg3n_build)
├── build_and_submit.sh      # Build Docker image + submit Batch jobs
├── submit_all.sh / submit_one.sh
└── states.csv               # State abbreviation → full name lookup
```

---

## Environment and dependencies

**Conda environment:** `dg3n` (defined in [dgen_os/python/dg3n.yml](dgen_os/python/dg3n.yml))

Key packages:
- `nrel-pysam=7.1.0` — PV/battery system simulation (NREL SAM)
- `psycopg2`, `pg8000` — PostgreSQL drivers
- `cloud-sql-python-connector[pg8000]` — Google Cloud SQL connector
- `google-cloud-storage` — GCS access
- `pandas`, `numpy`, `scipy`, `matplotlib` — data science stack
- `sqlalchemy` — ORM / SQL helpers

**Python version:** 3.11 (pinned in Docker; CI also tests 3.10, 3.12)

---

## Local development setup

### With Docker (recommended)

```bash
cd docker/
docker-compose up --build -d
docker attach dgen_1
conda activate dg3n
cd /opt/dgen_os/python/
python dgen_model.py
```

This starts the full stack (dGen container + PostGIS). Data downloads to `~/dgen_data/` on the host.

### VS Code / GitHub Codespaces

Open the repo and VS Code will prompt to reopen in the dev container (`.devcontainer/devcontainer.json`). This uses `docker-compose.yml` and sets up the `dg3n` Python interpreter automatically.

### Conda (bare metal)

```bash
conda env create -f dgen_os/python/dg3n.yml
conda activate dg3n
cd dgen_os/python/
python dgen_model.py
```

You will also need a local PostgreSQL instance (or Cloud SQL proxy) and a populated `pg_params_connect.json`.

---

## Database connection

The model connects to PostgreSQL via `pg_params_connect.json` (local/Docker) or the Google Cloud SQL connector (Cloud Batch).

`pg_params_connect.json` is **gitignored**. Template:
```json
{
  "dbname": "dgen_db",
  "host": "localhost",
  "port": "5432",
  "user": "postgres",
  "password": "postgres"
}
```

For Cloud SQL connections, set environment variable `USE_CLOUD_SQL=true` and ensure ADC credentials are available (`gcloud auth application-default login`).

---

## Running the model

### Entry point

```bash
cd dgen_os/python/
python dgen_model.py
```

The model reads scenario inputs from the `input_scenarios/` directory (gitignored) and writes results to a new Postgres schema named `diffusion_results_<scenario>_<state>_<endyear>_<timestamp>`.

### Config knobs ([dgen_os/python/config.py](dgen_os/python/config.py))

| Variable | Default | Notes |
|---|---|---|
| `start_year` | 2026 | First model year |
| `pg_procs` | 12 (or CPU count on Cloud) | Postgres parallel processes |
| `local_cores` | `cpu_count // 2` | Local multiprocessing |
| `dynamic_system_sizing` | True | Enable PySAM optimizer |
| `delete_output_schema` | False | Cleanup after run |
| `VERBOSE` | False | Extra logging |

---

## Cloud execution (Google Cloud Batch)

### One-time auth

```bash
gcloud auth login
gcloud config set project <GCP_PROJECT_ID>
gcloud auth application-default login
gcloud auth configure-docker us-east1-docker.pkg.dev
```

### Build and submit

```bash
# National run (4 Batch jobs, one per state-size class)
bash build_and_submit.sh   # calls submit_all.sh internally

# Single-state run
bash build_and_submit.sh   # configure to call submit_one.sh
```

States are grouped into four size classes (`small / mid / mid-large / large`) with matching YAML files in `batch_job_yamls/` and CSV files in `state_input_csvs/`.

### Exporting results

1. Start Cloud SQL Proxy: `cloud-sql-proxy dgen-466702:us-east1:dgen-db`
2. Run `dgen_os/python/Notebooks/loading_model_results.ipynb` to pull schemas to local CSVs.
3. Run `dgen_os/python/Notebooks/analysis_of_model_results.ipynb` for portfolio metrics.

---

## Key notebooks ([dgen_os/python/Notebooks/](dgen_os/python/Notebooks/))

| Notebook | Purpose |
|---|---|
| `adjust_pv_batt_price_trajectories.ipynb` | Set national PV/battery price paths (baseline vs policy) |
| `adjust_state_level_prices.ipynb` | State-specific price overrides |
| `output_state_input_csvs.ipynb` | Generate + upload state-list CSVs for Batch jobs |
| `adjust_agent_file.ipynb` | Update agent weights, loads, tariffs |
| `process_load_growth.ipynb` | Rebuild load-growth table |
| `loading_model_results.ipynb` | Export run schemas from Cloud SQL to local CSV |
| `analysis_of_model_results.ipynb` | Portfolio metrics, charts, comparisons |
| `upload_non_price_inputs_to_cloud_sql.ipynb` | Upload static inputs (financing, battery, etc.) |

---

## Fork-specific customizations vs upstream NREL dGen

- **ResStock agents**: ~25,000 statistically-weighted residential agents derived from ResStock (not the default NREL agent files).
- **Cloud-native**: upstream dGen targets local Postgres; this fork runs on Cloud SQL + Cloud Batch.
- **Dual-scenario loop**: each run produces paired `baseline` + `policy` schemas in one execution.
- **Battery attachment**: exogenous state-level attachment rates + largest-remainder integer allocation (not in upstream).
- **No federal ITC**: the residential ITC is fully removed — `itc_fed_percent` is pinned to 0 in `financial_functions.py` and batteries carry full (undiscounted) capex. The `input_main_itc_options` DB plumbing still loads but is inert.
- **PV sizing**: PV optimized once (PV-only), then PV+battery evaluated at fixed PV size.
- **Hourly aggregates**: `state_hourly_agg` and `rto_hourly_agg` tables written per run schema.
- **Analysis layer**: `analysis_functions.py` and `schema_exporter.py` are fork additions.

---

## Secrets and credentials — never commit these

| File / pattern | What it contains |
|---|---|
| `**/pg_params*.json` | PostgreSQL host/user/password |
| `*-service-account*.json` | GCP service account keys |
| `application_default_credentials.json` | ADC token cache |
| `.env`, `.env.*` | Any env-var secret files |

All of the above are covered by `.gitignore`.

---

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on push/PR to `master` and on a daily cron. It tests Python 3.10, 3.11, and 3.12 on Ubuntu, macOS, and Windows using the `build_envs/environment.yml` conda spec.
