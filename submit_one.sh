#!/usr/bin/env bash
set -euo pipefail

gcloud batch jobs submit dgen-all \
  --location="us-east1" \
  --config="batch_job_yamls/dgen-batch-job-mid-states.yaml" \
  --machine-type="c2d-highcpu-32" \
  --provisioning-model="SPOT"