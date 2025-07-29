#!/usr/bin/env bash
set -euo pipefail

# Set location and timestamp for job naming
LOCATION="us-east1"
JOB_TS=$(date -u +"%Y%m%d-%H%M%S")

# Define job name with timestamp
JOB_NAME="dgen-large-states-${JOB_TS}"

# submit the third job
gcloud batch jobs submit "${JOB_NAME}" \
  --location="${LOCATION}" \
  --config="batch_job_yamls/dgen-batch-job-large-states.yaml" \
  --machine-type="c2d-highcpu-56"  \
  --provisioning-model="SPOT"