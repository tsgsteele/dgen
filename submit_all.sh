#!/usr/bin/env bash
set -euo pipefail

LOCATION="us-east1"
JOB_TS=$(date -u +"%Y%m%d-%H%M%S")
PROVISIONING="STANDARD"   # STANDARD = on-demand (starts fast, subject to quota); SPOT = cheap but waits for capacity

# ── Round 1 states: MA, MD, NJ (mid); IL, PA, VA (mid-large); TX, NY (large) ──
gcloud batch jobs submit "dgen-mid-r1-${JOB_TS}" \
  --location="${LOCATION}" \
  --config="batch_job_yamls/dgen-batch-job-mid-states.yaml" \
  --machine-type="c2d-highcpu-16" \
  --provisioning-model="${PROVISIONING}"

gcloud batch jobs submit "dgen-mid-large-r1-${JOB_TS}" \
  --location="${LOCATION}" \
  --config="batch_job_yamls/dgen-batch-job-mid-large-states.yaml" \
  --machine-type="c2d-highcpu-32" \
  --provisioning-model="${PROVISIONING}"

gcloud batch jobs submit "dgen-large-r1-${JOB_TS}" \
  --location="${LOCATION}" \
  --config="batch_job_yamls/dgen-batch-job-large-states.yaml" \
  --machine-type="c2d-highcpu-32" \
  --provisioning-model="${PROVISIONING}"

# ── Round 2: remaining 40 states ───────────────────────────────────────────

# CA (1 task, 32 vCPU)
gcloud batch jobs submit "dgen-ca-r2-${JOB_TS}" \
  --location="${LOCATION}" \
  --config="batch_job_yamls/dgen-batch-job-ca.yaml" \
  --machine-type="c2d-highcpu-32" \
  --provisioning-model="${PROVISIONING}"

# FL, NC (2 tasks, 32 vCPU)
gcloud batch jobs submit "dgen-large-r2-${JOB_TS}" \
  --location="${LOCATION}" \
  --config="batch_job_yamls/dgen-batch-job-large-states-r2.yaml" \
  --machine-type="c2d-highcpu-32" \
  --provisioning-model="${PROVISIONING}"

# OH, GA, MI (3 tasks, 32 vCPU)
gcloud batch jobs submit "dgen-mid-large-r2a-${JOB_TS}" \
  --location="${LOCATION}" \
  --config="batch_job_yamls/dgen-batch-job-mid-large-states-r2a.yaml" \
  --machine-type="c2d-highcpu-32" \
  --provisioning-model="${PROVISIONING}"

# WA, AZ, CO, SC (4 tasks, 32 vCPU)
gcloud batch jobs submit "dgen-mid-large-r2b-${JOB_TS}" \
  --location="${LOCATION}" \
  --config="batch_job_yamls/dgen-batch-job-mid-large-states-r2b.yaml" \
  --machine-type="c2d-highcpu-32" \
  --provisioning-model="${PROVISIONING}"

# LA, MO, MN, WI, IN, TN (6 tasks, 16 vCPU)
gcloud batch jobs submit "dgen-mid-r2a-${JOB_TS}" \
  --location="${LOCATION}" \
  --config="batch_job_yamls/dgen-batch-job-mid-states-r2a.yaml" \
  --machine-type="c2d-highcpu-16" \
  --provisioning-model="${PROVISIONING}"

# KY, OR, UT, NV, NM, CT, AR (7 tasks, 16 vCPU)
gcloud batch jobs submit "dgen-mid-r2b-${JOB_TS}" \
  --location="${LOCATION}" \
  --config="batch_job_yamls/dgen-batch-job-mid-states-r2b.yaml" \
  --machine-type="c2d-highcpu-16" \
  --provisioning-model="${PROVISIONING}"

# OK, IA, ID, MT, KS, WV, NE, MS, AL (9 tasks, 16 vCPU)
gcloud batch jobs submit "dgen-small-r2a-${JOB_TS}" \
  --location="${LOCATION}" \
  --config="batch_job_yamls/dgen-batch-job-small-states-r2a.yaml" \
  --machine-type="c2d-highcpu-16" \
  --provisioning-model="${PROVISIONING}"

# NH, ME, WY, SD, ND, VT, RI, DE (8 tasks, 16 vCPU)
gcloud batch jobs submit "dgen-small-r2b-${JOB_TS}" \
  --location="${LOCATION}" \
  --config="batch_job_yamls/dgen-batch-job-small-states-r2b.yaml" \
  --machine-type="c2d-highcpu-16" \
  --provisioning-model="${PROVISIONING}"
