#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/atao/vsdevel/skyball/jersey"
output_root="/mnt/t/output/trn/jersey_recognition/uncertainty_vit_b_aug_4x_v1"
log_path="$output_root.launch.log"

mkdir -p "$output_root"

log() {
  echo "[$(date)] $*" >> "$log_path"
}

log "waiting for v0 4x ResNet50 jersey-recognition run"
while tmux has-session -t jersey_recognition_v0_4x_resnet50_wait 2>/dev/null; do
  sleep 60
done

if [[ -s "$output_root/summary.json" && -s "$output_root/best.pt" ]]; then
  log "v1 4x ViT-B run already complete: $output_root"
  exit 0
fi

log "starting v1 4x ViT-B jersey-recognition run"
cd "$repo_root"
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n pt5090new python -m jersey_sdg.jersey_number_train \
  --gallery-root /mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/v1_4x \
  --train-crops-root /mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/v1_4x/train/crops \
  --val-crops-root /mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/v1_4x/val/crops \
  --train-labels /mnt/t/output/jersey_sgd/gallery_visibility_4x_v1/manifests/gallery_train_visibility_labels.jsonl \
  --val-labels /mnt/t/output/jersey_sgd/gallery_visibility_4x_v1/manifests/gallery_val_visibility_labels.jsonl \
  --output "$output_root" \
  --backbone vit_b_16 \
  --head independent \
  --epochs 30 \
  --batch-size 128 \
  --workers 16 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --amp \
  --data-parallel >> "$log_path" 2>&1

log "v1 4x ViT-B jersey-recognition run complete"
