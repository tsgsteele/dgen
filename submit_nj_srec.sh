#!/usr/bin/env bash
set -euo pipefail

# NJ SREC study: two single-state NJ runs on identical (immediate $1/W) prices,
# differing ONLY by the PRODUCTION_INCENTIVES env var baked into each YAML.
#   control -> no SREC        (PRODUCTION_INCENTIVES="{}")
#   srec    -> $76.50/MWh 15y (PRODUCTION_INCENTIVES='{"NJ":{"usd_per_mwh":76.5,"term_yrs":15}}')
#
# PREREQS (must be done first):
#   1. DB holds the IMMEDIATE $1/W prices (price notebook POLICY_SCENARIO="immediate_1w", re-uploaded).
#   2. Docker image rebuilt with the new financial_functions.py + config.py.
#   3. nj_state.csv uploaded to the dgen-assets bucket.

LOCATION="us-east1"
JOB_TS=$(date -u +"%Y%m%d-%H%M%S")
PROVISIONING="STANDARD"

# Control (no SREC)
gcloud batch jobs submit "dgen-nj-control-${JOB_TS}" \
  --location="${LOCATION}" \
  --config="batch_job_yamls/dgen-batch-job-nj-control.yaml" \
  --machine-type="c2d-highcpu-32" \
  --provisioning-model="${PROVISIONING}"

# Treatment (SREC)
gcloud batch jobs submit "dgen-nj-srec-${JOB_TS}" \
  --location="${LOCATION}" \
  --config="batch_job_yamls/dgen-batch-job-nj-srec.yaml" \
  --machine-type="c2d-highcpu-32" \
  --provisioning-model="${PROVISIONING}"
