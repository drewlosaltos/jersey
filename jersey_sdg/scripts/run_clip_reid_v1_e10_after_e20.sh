#!/usr/bin/env bash
set -euo pipefail

gallery_root="/mnt/t/data/vball/skyball/jersey/gallery/v1"
repo_root="/home/atao/vsdevel/skyball/jersey/clip_reid"
output_root="/mnt/t/output/trn/clip_reid/skyball_roster_v1_vitl14_lr1e-5_e10_seed1"
log_path="$output_root.launch.log"

mkdir -p "$output_root"

log() {
  echo "[$(date)] $*" >> "$log_path"
}

log "waiting for 20-epoch v1 CLIP-ReID run"
while tmux has-session -t clip_reid_v1_wait 2>/dev/null; do
  sleep 60
done

if [[ -s "$output_root/weights_e10.pth" ]]; then
  log "10-epoch run already complete: $output_root/weights_e10.pth"
  exit 0
fi

log "starting 10-epoch v1 CLIP-ReID run"
cd "$repo_root"
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n pt5090new python train_skyball.py \
  --gallery-root "$gallery_root" \
  --epochs 10 \
  --lr 1e-5 \
  --lr-end 5e-7 \
  --warmup-epochs 0.5 \
  --checkpoint-start ./model/ViT-L-14_openai/fold-1_seed_1/weights_e4.pth \
  --output-dir "$output_root" >> "$log_path" 2>&1

log "10-epoch v1 CLIP-ReID run complete"
