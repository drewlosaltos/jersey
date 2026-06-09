# SkyBall Jersey Dataset Workflow

Use this workflow whenever a new unified split file arrives, for example:

```text
/mnt/t/data/vball/skyball/unified_dataset/splits/v2.json
```

Set `VERSION=v2` and keep all output names versioned.

## 1. Build ReID Gallery

```bash
cd /home/atao/vsdevel/skyball/jersey/jersey_gallery
python jersey_gallery.py \
  --group-scope match \
  --data-root /mnt/t/data/vball/skyball/unified_dataset \
  --split-file /mnt/t/data/vball/skyball/unified_dataset/splits/${VERSION}.json \
  --split-names train val \
  --output /mnt/t/data/vball/skyball/jersey/gallery/${VERSION} \
  --target-count 15
```

Expected manifests:

```text
/mnt/t/data/vball/skyball/jersey/gallery/${VERSION}/{train,val}/manifests/gallery_samples.jsonl
```

## 2. Train and Evaluate CLIP-ReID

```bash
cd /home/atao/vsdevel/skyball/jersey/clip_reid
CUDA_VISIBLE_DEVICES=0,1 python train_skyball.py \
  --gallery-root /mnt/t/data/vball/skyball/jersey/gallery/${VERSION} \
  --epochs 20 \
  --lr 1e-5 \
  --lr-end 5e-7 \
  --warmup-epochs 0.5 \
  --checkpoint-start ./model/ViT-L-14_openai/fold-1_seed_1/weights_e4.pth \
  --output-dir /mnt/t/output/trn/clip_reid/skyball_roster_${VERSION}_vitl14_lr1e-5_e20_seed1
```

Evaluate on the new validation gallery and, when comparing against older datasets, evaluate the new checkpoint on the previous best validation gallery too. Treat result recording as part of validation: append the completed run to `workflows/results.jsonl`, then compare its primary metric against the previous CLIP-ReID best in `workflows/results_summary.md`.

## 3. Build 4x Jersey-Recognition Gallery

```bash
cd /home/atao/vsdevel/skyball/jersey/jersey_gallery
python jersey_gallery.py \
  --group-scope match \
  --data-root /mnt/t/data/vball/skyball/unified_dataset \
  --split-file /mnt/t/data/vball/skyball/unified_dataset/splits/${VERSION}.json \
  --split-names train val \
  --output /mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/${VERSION}_4x \
  --target-count 60 \
  --num-workers 8
```

## 4. Label Visibility with Qwen

Use `jersey_sdg/scripts/run_qwen_visibility_v1_when_ready.sh` as the template. For a new version, write a versioned script or command that labels:

```text
/mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/${VERSION}_4x
```

to:

```text
/mnt/t/output/jersey_sgd/gallery_visibility_4x_${VERSION}
```

The labeler should mark `synthetic_visible=true` only when Qwen reads the visible number and it matches the ground-truth jersey number.

## 5. Train Jersey Recognition

Use the strongest current baseline first:

```bash
cd /home/atao/vsdevel/skyball/jersey
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n pt5090new \
  python -m jersey_sdg.jersey_number_train \
    --gallery-root /mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/${VERSION}_4x \
    --train-crops-root /mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/${VERSION}_4x/train/crops \
    --val-crops-root /mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/${VERSION}_4x/val/crops \
    --train-labels /mnt/t/output/jersey_sgd/gallery_visibility_4x_${VERSION}/manifests/gallery_train_visibility_labels.jsonl \
    --val-labels /mnt/t/output/jersey_sgd/gallery_visibility_4x_${VERSION}/manifests/gallery_val_visibility_labels.jsonl \
    --output /mnt/t/output/trn/jersey_recognition/uncertainty_convnext_base_in22k_aug_4x_${VERSION}_bs128_lr2e4_tda_mb_e15_gpus23_augfix \
    --backbone timm:convnext_base.fb_in22k_ft_in1k \
    --head tda_mb \
    --epochs 15 \
    --batch-size 128 \
    --workers 5 \
    --lr 2e-4 \
    --weight-decay 1e-4 \
    --digit-weight 0.3 \
    --seed 20260607 \
    --tb-image-every 1 \
    --tb-examples 16 \
    --amp \
    --data-parallel
```

## 6. Validate and Record Results

Every retrained network must be validated against the historical best before the workflow is considered complete.

Append one JSON object per completed run to `workflows/results.jsonl`. Include at least:

```json
{"task":"jersey_recognition","dataset_version":"v2","dataset":"v2_4x","run_name":"...","primary_metric":"best_visible_acc","primary_value":0.0,"output":"..."}
```

Then update `workflows/results_summary.md` with:

- previous best run and metric value
- new run and metric value
- `pass` if the new primary metric is greater than or equal to the previous best, otherwise `regression`
- notes about validation-set changes or cross-evaluation caveats

For jersey recognition, the primary metric is currently `best_visible_acc`. For CLIP-ReID, use reranked mAP unless a workflow explicitly chooses a different primary metric.
