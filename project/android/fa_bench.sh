#!/bin/bash
# FlashAttention A/B benchmark on device.
#   ./fa_bench.sh <0|1 flash> <promptLen> [decodeLen] [warmup] [repeat]
# Runs one prompt length per process so the sampled peak VmRSS belongs to that
# length only, and samples /proc/<pid>/status on the device while it runs.
set -u
SERIAL=${SERIAL:-192.168.124.101:47954}
REMOTE_ROOT=${REMOTE_ROOT:-/data/local/tmp/mnn-fa}
MODEL_DIR=${MODEL_DIR:-/data/local/tmp/mnn-dbg/m}
D="adb -s $SERIAL"

FLASH=${1:-0}
P=${2:-128}
DEC=${3:-1}
WARM=${4:-2}
REP=${5:-3}
OUT_DIR=${OUT_DIR:-$(pwd)/fa_results}
mkdir -p "$OUT_DIR"
TAG="fa${FLASH}_p${P}"

busy=$($D shell "ps -A -o NAME" 2>/dev/null | tr -d '\r' | grep -xE "spec_eval|profiler|llm_demo")
if [ -n "$busy" ]; then
  echo "device busy: $busy" >&2
  exit 1
fi

$D shell "rm -f $REMOTE_ROOT/run.log $REMOTE_ROOT/mem.log"
$D shell "cd $MODEL_DIR && export LD_LIBRARY_PATH=$REMOTE_ROOT:\$LD_LIBRARY_PATH && \
  export MNN_OPENCL_FLASH_ATTENTION=$FLASH && \
  nohup $REMOTE_ROOT/profiler config.json --prompt-lens=$P --decode-lens=$DEC --warmup=$WARM --repeat=$REP > $REMOTE_ROOT/run.log 2>&1 &
  sleep 1
  pid=\$(pidof profiler)
  echo \"pid=\$pid\"
  peak=0
  while [ -d /proc/\$pid ]; do
    r=\$(grep VmRSS /proc/\$pid/status 2>/dev/null | awk '{print \$2}')
    h=\$(grep VmHWM /proc/\$pid/status 2>/dev/null | awk '{print \$2}')
    if [ -n \"\$r\" ]; then echo \"\$r \$h\" >> $REMOTE_ROOT/mem.log; fi
    sleep 0.2
  done
  echo done"

$D pull "$REMOTE_ROOT/run.log" "$OUT_DIR/$TAG.log" >/dev/null 2>&1
$D pull "$REMOTE_ROOT/mem.log" "$OUT_DIR/$TAG.mem" >/dev/null 2>&1
PEAK=$(awk '{if($2>m)m=$2}END{printf "%.1f", m/1024}' "$OUT_DIR/$TAG.mem" 2>/dev/null)
echo "=== flash=$FLASH P=$P  peakVmHWM=${PEAK} MiB ==="
grep -E "flash attention ON|Prefill|prefill" "$OUT_DIR/$TAG.log" | head -8
echo "--- Attention op ---"
grep -E "^Attention" "$OUT_DIR/$TAG.log" | head -4
