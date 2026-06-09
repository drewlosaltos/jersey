#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/atao/vsdevel/skyball/jersey/clip_reid"
gallery_root="/mnt/t/data/vball/skyball/jersey/gallery/v0"
out_root="/mnt/t/output/trn/clip_reid/eval_v1_on_v0_val"
log_path="$out_root/eval.log"

run_e20="/mnt/t/output/trn/clip_reid/skyball_roster_v1_vitl14_lr1e-5_e20_seed1"
run_e10="/mnt/t/output/trn/clip_reid/skyball_roster_v1_vitl14_lr1e-5_e10_seed1"

mkdir -p "$out_root"

log() {
  echo "[$(date)] $*" >> "$log_path"
}

wait_for_session() {
  local session="$1"
  log "waiting for $session"
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 60
  done
}

evaluate_checkpoint() {
  local name="$1"
  local checkpoint="$2"
  local output_json="$out_root/${name}.json"

  if [[ ! -s "$checkpoint" ]]; then
    log "missing checkpoint for $name: $checkpoint"
    return 1
  fi
  if [[ -s "$output_json" ]]; then
    log "eval already exists for $name: $output_json"
    return 0
  fi

  log "evaluating $name on v0 val"
  cd "$repo_root"
  CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n pt5090new python evaluate_skyball.py \
    --gallery-root "$gallery_root" \
    --split val \
    --checkpoint "$checkpoint" \
    --output-json "$output_json" >> "$log_path" 2>&1
}

wait_for_session clip_reid_v1_wait
wait_for_session clip_reid_v1_e10_wait

evaluate_checkpoint "v1_e20_weights_e10_on_v0_val" "$run_e20/weights_e10.pth"
evaluate_checkpoint "v1_e20_weights_e20_on_v0_val" "$run_e20/weights_e20.pth"
evaluate_checkpoint "v1_e10_weights_e10_on_v0_val" "$run_e10/weights_e10.pth"

log "v1-on-v0 val evaluation complete"
