# SoccerNet Visibility Labeling

This file records the current SoccerNet dense visibility-labeling checkpoint and
how to resume it.

## Current Checkpoint

Labeling was paused on `2026-06-07 18:07 PDT`.

- SoccerNet train manifest: `733,001` images
- SoccerNet test manifest: `564,547` images
- Train labels completed: `456,000 / 733,001`
- Train completion: `62.2%`
- Train images remaining: `277,001`
- Test labels completed: `0 / 564,547`
- Total train+test completed: `456,000 / 1,297,548`
- Total train+test completion: `35.1%`

Completed train labels are in:

```text
/mnt/t/output/jersey_sgd/soccernet_visibility_qwen_v0/manifests/soccernet_train_visibility_labels.jsonl
```

The train labeler, after-train test watcher, and vLLM server were stopped. GPUs
were free after stopping.

## Throughput and ETA

The warmed vLLM path was benchmarked at roughly `47.7 labels/s` with 96 client
workers on 4x RTX 5090.

Estimated remaining time at the same speed:

- Train only: about `1.5 hours`
- Train + test: about `4.7 hours`

Actual time can vary with image sizes, server warmup, and transient request
errors.

## Resume Server

Run from the repository root:

```bash
cd /home/atao/vsdevel/skyball/jersey
```

Start Qwen through vLLM:

```bash
mkdir -p /mnt/t/output/jersey_sgd/soccernet_visibility_qwen_v0/logs

tmux new-session -d -s vllm_qwen '
cd /home/atao/vsdevel/skyball/jersey &&
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_ALLREDUCE_USE_FLASHINFER=0 \
conda run --no-capture-output -n vllm5090 \
  vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --served-model-name Qwen/Qwen3.6-35B-A3B-FP8 \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --gpu-memory-utilization 0.72 \
    --max-model-len 32768 \
  > /mnt/t/output/jersey_sgd/soccernet_visibility_qwen_v0/logs/vllm_qwen.log 2>&1
'
```

Wait for readiness:

```bash
curl http://127.0.0.1:8000/v1/models
tail -f /mnt/t/output/jersey_sgd/soccernet_visibility_qwen_v0/logs/vllm_qwen.log
```

The first server startup can take several minutes because vLLM loads the model,
profiles memory, and captures CUDA graphs.

## Resume Train Labeling

The labeler is resumable by `crop_path`. If the output JSONL exists, completed
rows are skipped and only pending rows are appended. Do not pass `--overwrite`
unless intentionally discarding the checkpoint.

```bash
tmux new-session -d -s label_soccernet_train_vllm '
cd /home/atao/vsdevel/skyball/jersey &&
conda run --no-capture-output -n vllm5090 python -m jersey_sdg.cli label-manifest \
  --source-manifest /mnt/t/output/trn/jersey_recognition/soccernet_jersey_2023/manifests/soccernet_train_visibility_labels.jsonl \
  --crop-root /mnt/t/data/soccernet/jersey-2023/train/images \
  --output-jsonl /mnt/t/output/jersey_sgd/soccernet_visibility_qwen_v0/manifests/soccernet_train_visibility_labels.jsonl \
  --summary-json /mnt/t/output/jersey_sgd/soccernet_visibility_qwen_v0/reports/soccernet_train_visibility_summary.json \
  --endpoint http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --workers 96 \
  --timeout 240 \
  --chunk-size 5000 \
  >> /mnt/t/output/jersey_sgd/soccernet_visibility_qwen_v0/logs/label_soccernet_train_vllm.log 2>&1
'
```

Check progress:

```bash
wc -l /mnt/t/output/jersey_sgd/soccernet_visibility_qwen_v0/manifests/soccernet_train_visibility_labels.jsonl
tail -f /mnt/t/output/jersey_sgd/soccernet_visibility_qwen_v0/logs/label_soccernet_train_vllm.log
nvidia-smi
```

## Resume Test Labeling

Only start test labeling after the train JSONL reaches exactly `733,001` rows.

```bash
tmux new-session -d -s label_soccernet_test_vllm '
cd /home/atao/vsdevel/skyball/jersey &&
conda run --no-capture-output -n vllm5090 python -m jersey_sdg.cli label-manifest \
  --source-manifest /mnt/t/output/trn/jersey_recognition/soccernet_jersey_2023/manifests/soccernet_test_visibility_labels.jsonl \
  --crop-root /mnt/t/data/soccernet/jersey-2023/test/images \
  --output-jsonl /mnt/t/output/jersey_sgd/soccernet_visibility_qwen_v0/manifests/soccernet_test_visibility_labels.jsonl \
  --summary-json /mnt/t/output/jersey_sgd/soccernet_visibility_qwen_v0/reports/soccernet_test_visibility_summary.json \
  --endpoint http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --workers 96 \
  --timeout 240 \
  --chunk-size 5000 \
  >> /mnt/t/output/jersey_sgd/soccernet_visibility_qwen_v0/logs/label_soccernet_test_vllm.log 2>&1
'
```

## Important Notes

- The model is `Qwen/Qwen3.6-35B-A3B-FP8`.
- The serving environment is `vllm5090`.
- vLLM uses the checkpoint's FP8 quantization config automatically.
- `VLLM_USE_FLASHINFER_SAMPLER=0` is set because the FlashInfer sampler path hit
  a 5090 capability-check issue. This workload is dominated by image
  encoding/prefill rather than decode sampling.
- Train labeling had only a few isolated request errors before pausing. Failed
  rows remain in the JSONL as `vlm_status="error"` and count as not visible.
