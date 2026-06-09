#!/usr/bin/env bash
set -u

RF_SESSION="${RF_SESSION:-rfdetr_skyball_large_v1}"
RF_LOG="${RF_LOG:-/mnt/t/output/rf-detr-mf/runs/skyball/skyball_large_stage3_coco_e90_v1_20260608/launch_retry_20260608.log}"
JERSEY_LOG="${JERSEY_LOG:-/mnt/t/output/trn/jersey_recognition/soccernet_qwen_partial_resnet50_bs128_lr2e4_e15_gpus23_20260608.launch.log}"
MONITOR_LOG="${MONITOR_LOG:-/mnt/t/output/trn/jersey_recognition/skyball_monitor_20260608.log}"
INTERVAL="${INTERVAL:-300}"

mkdir -p "$(dirname "$MONITOR_LOG")"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

latest_jersey_epoch() {
  if [[ ! -f "$JERSEY_LOG" ]]; then
    echo "missing"
    return
  fi
  rg -o "'epoch': [0-9]+" "$JERSEY_LOG" | tail -n 1 | awk '{print $2}'
}

latest_rf_epoch_line() {
  if [[ ! -f "$RF_LOG" ]]; then
    echo "missing"
    return
  fi
  rg "Epoch: \\[[0-9]+\\]|Training config|falling back|Fallback|out of memory|CUDA.*OOM|RuntimeError|Traceback" "$RF_LOG" -i | tail -n 5 | tr '\n' ' '
}

while true; do
  {
    echo "[$(timestamp)] monitor tick"

    if tmux has-session -t "$RF_SESSION" 2>/dev/null; then
      echo "rf_session=ok"
    else
      echo "rf_session=missing"
    fi

    echo "gpu_status:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

    rf_markers="$(latest_rf_epoch_line)"
    echo "rf_recent=${rf_markers:-none}"

    jersey_epoch="$(latest_jersey_epoch)"
    echo "jersey_epoch=${jersey_epoch}"

    qwen_processes="$(ps -eo pid,ppid,stat,comm,args | rg 'vllm|label[-_]manifest|visibility_label_manifest|qwen.*server|server.*qwen' -i | rg -v ' rg ' || true)"
    if [[ -n "$qwen_processes" ]]; then
      if [[ "$jersey_epoch" =~ ^[0-9]+$ && "$jersey_epoch" -ge 15 ]]; then
        echo "qwen_status=running_after_partial"
      else
        echo "qwen_status=WARNING_running_before_partial_complete"
      fi
      echo "$qwen_processes"
    else
      echo "qwen_status=not_running"
    fi

    echo
  } >> "$MONITOR_LOG" 2>&1

  sleep "$INTERVAL"
done
