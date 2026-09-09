import os
import json
import multiprocessing



#==============================================================================
#   get postgres connection parameters
#==============================================================================
# get the path of the current file
model_path = os.path.dirname(os.path.abspath(__file__))

# set the name of the pg_params_file
pg_params_file = 'pg_params_connect.json'

#==============================================================================
#   model start year
#==============================================================================
start_year = 2026

#==============================================================================
#   set number of parallel processes to run postgres queries
#==============================================================================
if os.environ.get('LOCAL_CORES'):
    pg_procs = multiprocessing.cpu_count()
else:
    pg_procs = 12

#==============================================================================
#   set role for database (default is postgres)
#==============================================================================
role = 'postgres'

#==============================================================================
#   local cores
#==============================================================================
local_cores = multiprocessing.cpu_count()//2
#==============================================================================
#  Should the output schema be deleted after the model run
#==============================================================================
delete_output_schema = False

#==============================================================================
#  Set switch for dynamic sizing
#==============================================================================
dynamic_system_sizing = True

#==============================================================================
#  Runtime Tests
#==============================================================================
NULL_COLUMN_EXCEPTIONS = ['state_incentives', 'pct_state_incentives', 'batt_dispatch_profile', 'export_tariff_results']

CHANGED_DTYPES_EXCEPTIONS = ['state_abbr', 'sector_abbr', 'county_id']
MISSING_COLUMN_EXCEPTIONS = []

#==============================================================================
#  Detailed Output
#==============================================================================
VERBOSE = False

#==============================================================================
#  Define Directories
#==============================================================================

cwd = os.getcwd() #should be /python
pdir = os.path.abspath('..') #should be /dgen or whatever it is called

OBSERVED_DEPLOYMENT_BY_STATE = os.path.join(pdir, 'input_data','observed_deployment_by_state_sector_2020.csv')

#==============================================================================
#  State production-based incentives (SREC / PBI)
#==============================================================================
# Flat $/MWh paid on GROSS PV production for `term_yrs` years, added (untaxed,
# like value_of_resiliency) to the system's annual energy value in
# financial_functions.calc_system_performance. Gated by state_abbr, so only the
# listed states are affected. Empty dict = OFF (no state gets an incentive).
#
# Driven by the PRODUCTION_INCENTIVES env var (JSON) so one Docker image can run
# both the no-SREC control and the with-SREC run without a rebuild:
#   No-SREC run:   leave the env var unset  ->  {}
#   With-SREC run: PRODUCTION_INCENTIVES='{"NJ": {"usd_per_mwh": 76.5, "term_yrs": 15}}'
# NJ SREC/TREC study (SuSI): $76.50/MWh for 15 years.
_pbi_env = os.environ.get("PRODUCTION_INCENTIVES", "").strip()
PRODUCTION_INCENTIVES = json.loads(_pbi_env) if _pbi_env else {}

#==============================================================================
#   flat battery storage attachment rate (sensitivity scenarios)
#==============================================================================
# Overrides the per-state Ohm attachment rates with a single flat rate applied to
# EVERY state (baseline and policy alike). Driven by the FLAT_STORAGE_ATTACHMENT_RATE
# env var so one Docker image can run the 5% / 75% / 100% attachment sensitivities
# (for the Synapse deliverable) without a rebuild:
#   Normal run (Ohm per-state rates): leave the env var unset -> None
#   Flat sensitivity run:             FLAT_STORAGE_ATTACHMENT_RATE=0.75  (fraction in [0,1])
# When set, dgen_model overrides the merged `storage_attachment_rate` column, and the
# output schema name is tagged `_attach{pct}` (e.g. _attach75) so runs are self-describing.
_flat_attach = os.environ.get("FLAT_STORAGE_ATTACHMENT_RATE", "").strip()
FLAT_STORAGE_ATTACHMENT_RATE = float(_flat_attach) if _flat_attach else None
if FLAT_STORAGE_ATTACHMENT_RATE is not None and not (0.0 <= FLAT_STORAGE_ATTACHMENT_RATE <= 1.0):
    raise ValueError(f"FLAT_STORAGE_ATTACHMENT_RATE must be in [0,1], got {FLAT_STORAGE_ATTACHMENT_RATE}")