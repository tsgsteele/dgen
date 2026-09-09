#!/usr/bin/env bash
set -euo pipefail

# Submit the full 48-state national run with a FLAT battery storage attachment rate applied to
# EVERY state (Synapse attachment-rate sensitivity). Runs both baseline and policy scenarios
# (whatever input_scenarios the image is built with).
#
# Usage:
#   bash submit_all_attach.sh 0.05      # 5%  attachment, all states
#   bash submit_all_attach.sh 0.75      # 75%
#   bash submit_all_attach.sh 1.0       # 100%
#
# One Docker image serves all three (build once via build_and_submit.sh / docker push, then run
# this three times). FLAT_STORAGE_ATTACHMENT_RATE is injected into each job's env block at submit
# time into a TEMP copy of the yaml -- the checked-in yamls are never modified. The model tags each
# output schema `_a<pct>` (e.g. diffusion_results_baseline_..._a75_<ts>) so runs are
# self-describing in Cloud SQL. Leave this script unused for a normal (Ohm per-state) run.

RATE="${1:?usage: bash submit_all_attach.sh <rate in [0,1]>   e.g. 0.05 | 0.75 | 1.0}"
PCT=$(python3 -c "r=float('$RATE'); assert 0<=r<=1, 'rate must be in [0,1]'; print(int(round(r*100)))")

LOCATION="us-east1"
JOB_TS=$(date -u +"%Y%m%d-%H%M%S")
PROVISIONING="STANDARD"   # STANDARD = on-demand; SPOT = cheaper but waits for capacity
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# label | yaml | machine-type   (mirrors submit_all.sh: 11 jobs = all 48 states)
JOBS=(
  "mid-r1|dgen-batch-job-mid-states.yaml|c2d-highcpu-16"
  "mid-large-r1|dgen-batch-job-mid-large-states.yaml|c2d-highcpu-32"
  "large-r1|dgen-batch-job-large-states.yaml|c2d-highcpu-32"
  "ca-r2|dgen-batch-job-ca.yaml|c2d-highcpu-32"
  "large-r2|dgen-batch-job-large-states-r2.yaml|c2d-highcpu-32"
  "mid-large-r2a|dgen-batch-job-mid-large-states-r2a.yaml|c2d-highcpu-32"
  "mid-large-r2b|dgen-batch-job-mid-large-states-r2b.yaml|c2d-highcpu-32"
  "mid-r2a|dgen-batch-job-mid-states-r2a.yaml|c2d-highcpu-16"
  "mid-r2b|dgen-batch-job-mid-states-r2b.yaml|c2d-highcpu-16"
  "small-r2a|dgen-batch-job-small-states-r2a.yaml|c2d-highcpu-16"
  "small-r2b|dgen-batch-job-small-states-r2b.yaml|c2d-highcpu-16"
)

echo "Submitting 48-state national run  |  FLAT_STORAGE_ATTACHMENT_RATE=${RATE}  (attach${PCT})  |  ts=${JOB_TS}"

for spec in "${JOBS[@]}"; do
  IFS='|' read -r label yaml mtype <<< "$spec"
  src="batch_job_yamls/${yaml}"
  tmp="${TMPDIR}/${yaml}"

  # inject FLAT_STORAGE_ATTACHMENT_RATE into the env block, right after LOCAL_CORES,
  # matching that line's indentation (portable; no BSD/GNU sed differences).
  FLAT_RATE="$RATE" python3 - "$src" "$tmp" <<'PYEOF'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
rate = os.environ["FLAT_RATE"]
out, injected = [], False
for ln in open(src):
    out.append(ln)
    if (not injected) and "LOCAL_CORES:" in ln:
        indent = ln[: len(ln) - len(ln.lstrip())]
        out.append(f'{indent}FLAT_STORAGE_ATTACHMENT_RATE: "{rate}"\n')
        injected = True
assert injected, f"LOCAL_CORES anchor not found in {src}"
open(dst, "w").writelines(out)
PYEOF

  gcloud batch jobs submit "dgen-attach${PCT}-${label}-${JOB_TS}" \
    --location="${LOCATION}" \
    --config="${tmp}" \
    --machine-type="${mtype}" \
    --provisioning-model="${PROVISIONING}"
done

echo "Done: submitted attach${PCT} run (${#JOBS[@]} jobs). Schemas will be tagged _a${PCT}."
