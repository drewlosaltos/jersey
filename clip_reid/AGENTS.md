# Repository Guidelines

## Scope

`clip_reid` trains and evaluates CLIP-ReID models on SkyBall jersey galleries. It consumes galleries built by `../jersey_gallery` at paths like `/mnt/t/data/vball/skyball/jersey/gallery/v1`.

## Commands

Train the baseline from this directory:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_skyball.py \
  --gallery-root /mnt/t/data/vball/skyball/jersey/gallery/v1 \
  --epochs 20 \
  --lr 1e-5 \
  --lr-end 5e-7 \
  --warmup-epochs 0.5 \
  --checkpoint-start ./model/ViT-L-14_openai/fold-1_seed_1/weights_e4.pth \
  --output-dir /mnt/t/output/trn/clip_reid/skyball_roster_v1_vitl14_lr1e-5_e20_seed1
```

Evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_skyball.py \
  --gallery-root /mnt/t/data/vball/skyball/jersey/gallery/v1 \
  --split val \
  --checkpoint /mnt/t/output/trn/clip_reid/<run>/best_by_rerank.pth
```

## Rules

Do not commit `model/`, downloaded datasets, checkpoints, or debug images. Record completed metrics in `../workflows/results.jsonl`.

