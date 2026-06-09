# Repository Guidelines

## Scope

`jersey_gallery` builds deterministic SkyBall crop galleries from unified dataset split files. Its outputs feed both `../clip_reid` and `../jersey_sdg`.

## Commands

Build the standard 15-crop ReID gallery:

```bash
python jersey_gallery.py \
  --group-scope match \
  --data-root /mnt/t/data/vball/skyball/unified_dataset \
  --split-file /mnt/t/data/vball/skyball/unified_dataset/splits/v1.json \
  --split-names train val \
  --output /mnt/t/data/vball/skyball/jersey/gallery/v1 \
  --target-count 15
```

Build the 4x jersey-recognition gallery:

```bash
python jersey_gallery.py \
  --group-scope match \
  --data-root /mnt/t/data/vball/skyball/unified_dataset \
  --split-file /mnt/t/data/vball/skyball/unified_dataset/splits/v1.json \
  --split-names train val \
  --output /mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/v1_4x \
  --target-count 60 \
  --num-workers 8
```

Run tests with `PYTHONPATH=. conda run -n pt5090new python -m pytest tests -q`.

## Rules

Do not overwrite existing versioned galleries unless explicitly requested. Keep split versions in output names (`v1`, `v2`, `v2_4x`).
