#!/usr/bin/env bash
set -euo pipefail

DETECTOR_SESSION="${DETECTOR_SESSION:-rfdetr_skyball_large_v1}"
VLLM_SESSION="${VLLM_SESSION:-vllm_qwen}"
ROOT="/home/atao/vsdevel/skyball/jersey"
OUT_ROOT="/mnt/t/output/jersey_sgd/soccernet_visibility_qwen_v0"
LOG_DIR="${OUT_ROOT}/logs"
MANIFEST_DIR="${OUT_ROOT}/manifests"
REPORT_DIR="${OUT_ROOT}/reports"
WATCHER_LOG="${LOG_DIR}/soccernet_after_large_watcher.log"
MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
ENDPOINT="http://127.0.0.1:8000/v1"
STARTED_VLLM=0

mkdir -p "${LOG_DIR}" "${MANIFEST_DIR}" "${REPORT_DIR}"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "${WATCHER_LOG}"
}

tmux_has_session() {
  tmux has-session -t "$1" 2>/dev/null
}

cleanup() {
  if [[ "${STARTED_VLLM}" == "1" ]] && tmux_has_session "${VLLM_SESSION}"; then
    log "Stopping ${VLLM_SESSION}."
    tmux kill-session -t "${VLLM_SESSION}" || true
  fi
}
trap cleanup EXIT

wait_for_detector() {
  log "Waiting for detector tmux session ${DETECTOR_SESSION} to exit."
  while tmux_has_session "${DETECTOR_SESSION}"; do
    sleep 300
  done
  log "Detector session ${DETECTOR_SESSION} is no longer running."
}

start_vllm_if_needed() {
  if curl -fsS "${ENDPOINT}/models" >/dev/null 2>&1; then
    log "Existing vLLM endpoint is ready at ${ENDPOINT}."
    return
  fi

  if tmux_has_session "${VLLM_SESSION}"; then
    log "Found existing ${VLLM_SESSION}; waiting for endpoint readiness."
  else
    log "Starting ${VLLM_SESSION} on GPUs 0,1,2,3."
    tmux new-session -d -s "${VLLM_SESSION}" "
cd ${ROOT} &&
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_ALLREDUCE_USE_FLASHINFER=0 \
conda run --no-capture-output -n vllm5090 \
  vllm serve ${MODEL} \
    --served-model-name ${MODEL} \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --gpu-memory-utilization 0.72 \
    --max-model-len 32768 \
  > ${LOG_DIR}/vllm_qwen.log 2>&1
"
    STARTED_VLLM=1
  fi

  until curl -fsS "${ENDPOINT}/models" >/dev/null 2>&1; do
    log "Waiting for vLLM readiness."
    sleep 30
  done
  log "vLLM endpoint is ready."
}

run_label_manifest() {
  local split="$1"
  local expected_rows="$2"
  local source_manifest="/mnt/t/output/trn/jersey_recognition/soccernet_jersey_2023/manifests/soccernet_${split}_visibility_labels.jsonl"
  local crop_root="/mnt/t/data/soccernet/jersey-2023/${split}/images"
  local output_jsonl="${MANIFEST_DIR}/soccernet_${split}_visibility_labels.jsonl"
  local summary_json="${REPORT_DIR}/soccernet_${split}_visibility_summary.json"
  local label_log="${LOG_DIR}/label_soccernet_${split}_vllm.log"

  log "Starting/resuming SoccerNet ${split} visibility labeling."
  cd "${ROOT}"
  conda run --no-capture-output -n vllm5090 python -m jersey_sdg.cli label-manifest \
    --source-manifest "${source_manifest}" \
    --crop-root "${crop_root}" \
    --output-jsonl "${output_jsonl}" \
    --summary-json "${summary_json}" \
    --endpoint "${ENDPOINT}" \
    --model "${MODEL}" \
    --workers 96 \
    --timeout 240 \
    --chunk-size 5000 \
    >> "${label_log}" 2>&1

  local rows
  rows="$(wc -l < "${output_jsonl}")"
  log "SoccerNet ${split} labels now have ${rows}/${expected_rows} rows."
  if [[ "${rows}" != "${expected_rows}" ]]; then
    log "Expected ${expected_rows} rows for ${split}; stopping before next split."
    return 1
  fi
}

main() {
  wait_for_detector
  start_vllm_if_needed
  run_label_manifest train 733001
  run_label_manifest test 564547
  log "SoccerNet train+test visibility labeling completed."
}

main "$@"
