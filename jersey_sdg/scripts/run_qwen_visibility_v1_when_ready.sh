#!/usr/bin/env bash
set -euo pipefail

gallery_root="/mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/v1_4x"
output_root="/mnt/t/output/jersey_sgd/gallery_visibility_4x_v1"
repo_root="/home/atao/vsdevel/skyball/jersey"
log_dir="$output_root/logs"
wait_log="$log_dir/qwen_visibility_v1_wait.log"
vllm_log="$log_dir/vllm_qwen.log"

mkdir -p "$log_dir"

log() {
  echo "[$(date)] $*" >> "$wait_log"
}

wait_for_4x_gallery() {
  log "waiting for 4x gallery v1"
  while [[ ! -s "$gallery_root/summary.json" ||
           ! -s "$gallery_root/train/manifests/gallery_samples.jsonl" ||
           ! -s "$gallery_root/val/manifests/gallery_samples.jsonl" ]]; do
    if ! tmux has-session -t jersey_gallery_rec_v1_4x_wait 2>/dev/null &&
       [[ ! -s "$gallery_root/summary.json" ]]; then
      log "4x gallery session ended before summary"
      exit 1
    fi
    sleep 60
  done
  log "4x gallery ready"
}

wait_for_labeling_gpus() {
  log "waiting for CLIP-ReID GPU jobs"
  while tmux has-session -t clip_reid_v1_wait 2>/dev/null ||
        tmux has-session -t clip_reid_v1_e10_wait 2>/dev/null ||
        tmux has-session -t clip_reid_v1_eval_v0_wait 2>/dev/null; do
    sleep 120
  done
  log "CLIP-ReID sessions are complete"
}

wait_for_vllm() {
  local vllm_pid="$1"
  log "waiting for vLLM readiness, pid=$vllm_pid"
  for _ in {1..240}; do
    if curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
      log "vLLM ready"
      return 0
    fi
    if ! kill -0 "$vllm_pid" 2>/dev/null; then
      log "vLLM exited before readiness"
      exit 1
    fi
    sleep 15
  done
  log "vLLM readiness timeout"
  exit 1
}

wait_for_4x_gallery
wait_for_labeling_gpus

cd "$repo_root"
log "starting vLLM Qwen on GPUs 2,3"
CUDA_VISIBLE_DEVICES=2,3 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_ALLREDUCE_USE_FLASHINFER=0 \
conda run --no-capture-output -n vllm5090 \
  vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --served-model-name Qwen/Qwen3.6-35B-A3B-FP8 \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --gpu-memory-utilization 0.82 \
    --max-model-len 16384 \
  > "$vllm_log" 2>&1 &
vllm_pid=$!
trap 'kill "$vllm_pid" 2>/dev/null || true' EXIT

wait_for_vllm "$vllm_pid"

log "labeling train split"
conda run --no-capture-output -n vllm5090 python -m jersey_sdg.cli label-gallery-split \
  --gallery-root "$gallery_root" \
  --split train \
  --output "$output_root" \
  --endpoint http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --workers 48 \
  --timeout 240 \
  --chunk-size 5000 >> "$wait_log" 2>&1

log "labeling val split"
conda run --no-capture-output -n vllm5090 python -m jersey_sdg.cli label-gallery-split \
  --gallery-root "$gallery_root" \
  --split val \
  --output "$output_root" \
  --endpoint http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --workers 48 \
  --timeout 240 \
  --chunk-size 5000 >> "$wait_log" 2>&1

log "visibility labeling complete"
