#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/atao/vsdevel/skyball/jersey"
OUT_ROOT="/mnt/t/output/trn/jersey_recognition"
QWEN_ROOT="/mnt/t/output/jersey_sgd/soccernet_visibility_qwen_v0"
COMBINED_ROOT="/mnt/t/output/jersey_sgd/combined_soccernet_v1_4x_20260608"
LOG_DIR="${QWEN_ROOT}/logs"
PIPELINE_LOG="${OUT_ROOT}/soccernet_jersey_pipeline_20260608.log"
MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
ENDPOINT="http://127.0.0.1:8000/v1"
VLLM_PID=""

mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${QWEN_ROOT}/manifests" "${QWEN_ROOT}/reports" "${COMBINED_ROOT}/manifests"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "${PIPELINE_LOG}"
}

train_jersey() {
  local name="$1"
  local train_labels="$2"
  local val_labels="$3"
  local train_root="$4"
  local val_root="$5"
  local output="${OUT_ROOT}/${name}"
  local log_path="${output}.launch.log"

  if [[ -s "${output}/summary.json" && -s "${output}/best.pt" ]]; then
    log "Skipping completed jersey run ${output}."
    return
  fi

  mkdir -p "${output}"
  log "Starting jersey run ${name}."
  cd "${ROOT}"
  CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n pt5090new \
    python -m jersey_sdg.jersey_number_train \
      --gallery-root /mnt/t/data/soccernet/jersey-2023 \
      --train-crops-root "${train_root}" \
      --val-crops-root "${val_root}" \
      --train-labels "${train_labels}" \
      --val-labels "${val_labels}" \
      --output "${output}" \
      --backbone resnet50 \
      --head independent \
      --epochs 15 \
      --batch-size 128 \
      --workers 5 \
      --lr 2e-4 \
      --weight-decay 1e-4 \
      --amp \
      --data-parallel \
    >> "${log_path}" 2>&1
  log "Completed jersey run ${name}."
}

start_vllm() {
  if curl -fsS "${ENDPOINT}/models" >/dev/null 2>&1; then
    log "Using existing vLLM endpoint at ${ENDPOINT}."
    return
  fi

  log "Starting vLLM Qwen on GPUs 2,3."
  cd "${ROOT}"
  CUDA_VISIBLE_DEVICES=2,3 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  conda run --no-capture-output -n vllm5090 \
    vllm serve "${MODEL}" \
      --served-model-name "${MODEL}" \
      --host 127.0.0.1 \
      --port 8000 \
      --tensor-parallel-size 2 \
      --trust-remote-code \
      --gpu-memory-utilization 0.82 \
      --max-model-len 16384 \
    > "${LOG_DIR}/vllm_qwen_gpus23.log" 2>&1 &
  VLLM_PID=$!

  for _ in {1..240}; do
    if curl -fsS "${ENDPOINT}/models" >/dev/null 2>&1; then
      log "vLLM endpoint is ready."
      return
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      log "vLLM exited before readiness."
      exit 1
    fi
    sleep 15
  done
  log "Timed out waiting for vLLM readiness."
  exit 1
}

stop_vllm() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    log "Stopping vLLM pid ${VLLM_PID}."
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap stop_vllm EXIT

label_split() {
  local split="$1"
  local workers="$2"
  local expected="$3"
  local source="/mnt/t/output/trn/jersey_recognition/soccernet_jersey_2023/manifests/soccernet_${split}_visibility_labels.jsonl"
  local crop_root="/mnt/t/data/soccernet/jersey-2023/${split}/images"
  local output="${QWEN_ROOT}/manifests/soccernet_${split}_visibility_labels.jsonl"
  local summary="${QWEN_ROOT}/reports/soccernet_${split}_visibility_summary.json"
  local label_log="${LOG_DIR}/label_soccernet_${split}_vllm_gpus23.log"

  log "Labeling/resuming SoccerNet ${split}."
  cd "${ROOT}"
  conda run --no-capture-output -n vllm5090 python -m jersey_sdg.cli label-manifest \
    --source-manifest "${source}" \
    --crop-root "${crop_root}" \
    --output-jsonl "${output}" \
    --summary-json "${summary}" \
    --endpoint "${ENDPOINT}" \
    --model "${MODEL}" \
    --workers "${workers}" \
    --timeout 240 \
    --chunk-size 5000 \
    >> "${label_log}" 2>&1

  local rows
  rows="$(wc -l < "${output}")"
  log "SoccerNet ${split} labels: ${rows}/${expected} rows."
  if [[ "${rows}" != "${expected}" ]]; then
    log "Expected ${expected} rows for ${split}; aborting downstream SoccerNet all-data runs."
    exit 1
  fi
}

make_combined_manifests() {
  log "Building combined SoccerNet + v1-4x manifests."
  cd "${ROOT}"
  conda run --no-capture-output -n pt5090new python - <<'PY'
import json
from pathlib import Path

out = Path("/mnt/t/output/jersey_sgd/combined_soccernet_v1_4x_20260608/manifests")
out.mkdir(parents=True, exist_ok=True)

jobs = [
    (
        Path("/mnt/t/output/jersey_sgd/soccernet_visibility_qwen_v0/manifests/soccernet_train_visibility_labels.jsonl"),
        Path("/mnt/t/data/soccernet/jersey-2023/train/images"),
        out / "train_visibility_labels.jsonl",
    ),
    (
        Path("/mnt/t/output/jersey_sgd/gallery_visibility_4x_v1/manifests/gallery_train_visibility_labels.jsonl"),
        Path("/mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/v1_4x/train/crops"),
        out / "train_visibility_labels.jsonl",
    ),
    (
        Path("/mnt/t/output/jersey_sgd/soccernet_visibility_qwen_v0/manifests/soccernet_test_visibility_labels.jsonl"),
        Path("/mnt/t/data/soccernet/jersey-2023/test/images"),
        out / "val_visibility_labels.jsonl",
    ),
    (
        Path("/mnt/t/output/jersey_sgd/gallery_visibility_4x_v1/manifests/gallery_val_visibility_labels.jsonl"),
        Path("/mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/v1_4x/val/crops"),
        out / "val_visibility_labels.jsonl",
    ),
]

for target in {job[2] for job in jobs}:
    target.write_text("")

counts = {}
for src, root, target in jobs:
    count = 0
    with src.open() as fin, target.open("a") as fout:
        for line in fin:
            row = json.loads(line)
            crop_path = Path(row["crop_path"])
            if not crop_path.is_absolute():
                row["crop_path"] = str(root / crop_path)
            fout.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    counts[str(src)] = count

(out / "summary.json").write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
PY
  log "Combined manifests are ready."
}

main() {
  train_jersey \
    "soccernet_qwen_partial_resnet50_bs128_lr2e4_e15_gpus23_20260608" \
    "${QWEN_ROOT}/manifests/soccernet_train_visibility_labels.jsonl" \
    "/mnt/t/output/trn/jersey_recognition/soccernet_jersey_2023/manifests/soccernet_test_visibility_labels.jsonl" \
    "/mnt/t/data/soccernet/jersey-2023/train/images" \
    "/mnt/t/data/soccernet/jersey-2023/test/images"

  start_vllm
  label_split train 48 733001
  label_split test 48 564547
  stop_vllm
  VLLM_PID=""

  train_jersey \
    "soccernet_qwen_all_resnet50_bs128_lr2e4_e15_gpus23_20260608" \
    "${QWEN_ROOT}/manifests/soccernet_train_visibility_labels.jsonl" \
    "${QWEN_ROOT}/manifests/soccernet_test_visibility_labels.jsonl" \
    "/mnt/t/data/soccernet/jersey-2023/train/images" \
    "/mnt/t/data/soccernet/jersey-2023/test/images"

  make_combined_manifests
  train_jersey \
    "soccernet_qwen_all_plus_v1_4x_resnet50_bs128_lr2e4_e15_gpus23_20260608" \
    "${COMBINED_ROOT}/manifests/train_visibility_labels.jsonl" \
    "${COMBINED_ROOT}/manifests/val_visibility_labels.jsonl" \
    "/" \
    "/"

  log "SoccerNet jersey pipeline completed."
}

main "$@"
