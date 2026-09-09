#!/usr/bin/env bash
set -euo pipefail

LOCATION="us-east1"
JOB_TS=$(date -u +"%Y%m%d-%H%M%S")
PROVISIONING="STANDARD"   # STANDARD = on-demand (starts fast, subject to quota); SPOT = cheap but waits for capacity

# CA retry after tier-cap fix
gcloud batch jobs submit "dgen-ca-r2-${JOB_TS}" \
  --location="${LOCATION}" \
  --config="batch_job_yamls/dgen-batch-job-ca.yaml" \
  --machine-type="c2d-highcpu-32" \
  --provisioning-model="${PROVISIONING}"