#!/usr/bin/env bash
set -euo pipefail

LOCATION="us-east1"
JOB_TS=$(date -u +"%Y%m%d-%H%M%S")


# Define job name with timestamp
JOB_NAME="dgen-small-states-${JOB_TS}"

# submit the last job
gcloud batch jobs submit "${JOB_NAME}" \
  --location="${LOCATION}" \
  --config="batch_job_yamls/dgen-batch-job-small-states.yaml" \
  --machine-type="c2d-highcpu-8"  \
  --provisioning-model="SPOT"