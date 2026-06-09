# Results Summary

## Current Bests

| Task | Dataset | Run | Primary Metric | Value |
| --- | --- | --- | --- | ---: |
| CLIP-ReID | v1 trained, v0 val cross-eval | `skyball_roster_v1_vitl14_lr1e-5_e20_seed1/weights_e20` | rerank mAP | 99.07 |
| Jersey recognition | v1_4x | `uncertainty_convnext_base_in22k_aug_4x_v1_bs128_lr2e4_tda_mb_e15_gpus23_augfix` | best visible acc | 0.8057 |

## Jersey Recognition Comparisons

| Dataset | Backbone | Head | Run | Best Visible Acc | Notes |
| --- | --- | --- | --- | ---: | --- |
| v1_4x | ResNet50 | independent | `...w5_e15_gpus23` | 0.6080 | Pre augmentation fix |
| v0_4x | ResNet50 | independent | `...v0...augfix` | 0.6214 | v0 comparison |
| v1_4x | ResNet50 | independent | `...v1...augfix` | 0.7007 | Safer augmentation |
| v1_4x | ResNet50 | tda_mb | `...tda_mb...augfix` | 0.7588 | Digit-compositional improvement |
| v1_4x | ViT-B | tda_mb | `...vit_b...tda_mb...augfix` | 0.3758 | Underperformed |
| v1_4x | ConvNeXt-Base IN22K | tda_mb | `...convnext_base...tda_mb...augfix` | 0.8057 | Current best |

Use `workflows/results.jsonl` as the source of truth. Update this table after appending new completed runs.

