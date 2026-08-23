#!/bin/bash
# Multi-config single-session benchmark with a clock witness.
#
# The shared device's GPU clock drifts >2x between (and sometimes within) sessions - all op
# types scale together, so absolute ms are not comparable across runs. Convolution does
# identical work in every variant, so its time is used as a clock witness: each variant's
# Attention time is normalised by its own Convolution time, and the resulting
# attn/conv ratio is comparable across runs even when the clock moves.
#
#   ./fa_multi.sh <promptLen> "<name:ENV=V,ENV=V> ..."
set -u
SERIAL=${SERIAL:-192.168.124.101:47954}
REMOTE_ROOT=${REMOTE_ROOT:-/data/local/tmp/mnn-fa}
MODEL_DIR=${MODEL_DIR:-/data/local/tmp/mnn-dbg/m}
D="adb -s $SERIAL"
P=${1:-1024}
shift || true
CFGS=${1:-"base:MNN_OPENCL_FLASH_ATTENTION=0 fp32:MNN_OPENCL_FLASH_ATTENTION=1,MNN_FA_MIXED=0 mix8:MNN_OPENCL_FLASH_ATTENTION=1,MNN_FA_MIXED=1,MNN_FA_SEG=8"}
OUT=${OUT:-$(pwd)/fa_multi_p$P}
mkdir -p "$OUT"

wait_free() {
  for i in $(seq 1 200); do
    b=$($D shell "ps -A -o NAME" 2>/dev/null | tr -d '\r' | grep -xE "spec_eval|profiler|llm_demo")
    [ -z "$b" ] && return 0
    sleep 15
  done
  return 1
}

run() { # $1 comma-separated env, $2 tag
  wait_free || { echo "device stayed busy" >&2; return 1; }
  local envs=$(echo "$1" | tr ',' ' ')
  $D shell "rm -f $REMOTE_ROOT/mem.log"
  $D shell "cd $MODEL_DIR && export LD_LIBRARY_PATH=$REMOTE_ROOT:\$LD_LIBRARY_PATH && export MNN_OPENCL_MEM_REPORT=1 $envs ; \
    nohup $REMOTE_ROOT/profiler config.json --prompt-lens=$P --decode-lens=1 --warmup=2 --repeat=3 > $REMOTE_ROOT/run.log 2>&1 &
    sleep 2; pid=\$(pidof profiler)
    while [ -d /proc/\$pid ]; do
      grep VmHWM /proc/\$pid/status 2>/dev/null >> $REMOTE_ROOT/mem.log
      sleep 0.3
    done" >/dev/null 2>&1
  $D pull "$REMOTE_ROOT/run.log" "$OUT/$2.log" >/dev/null 2>&1
  $D pull "$REMOTE_ROOT/mem.log" "$OUT/$2.mem" >/dev/null 2>&1
}

printf "%-8s %-9s %-9s %-11s %-10s %-12s %-9s %-6s\n" \
  "config" "attn_ms" "conv_ms" "attn/conv" "prefill" "transMiB" "rssMiB" "FA"
for spec in $CFGS; do
  name=${spec%%:*}; envs=${spec#*:}
  run "$envs" "$name"
  python3 - "$OUT/$name.log" "$OUT/$name.mem" "$name" <<'PY'
import re,sys
t=open(sys.argv[1],errors="replace").read()
i=t.find("=== Prefill Phase Ops Statistics ===")
j=t.find("=== Decode Phase Ops Statistics ===",i)
blk=t[i:j] if i>=0 else ""
def g(pat,src=blk,grp=1):
    m=re.search(pat,src,re.M)
    return float(m.group(grp)) if m else None
attn=g(r"^Attention\s+\S+\s+([\d.]+)")
aavg=g(r"^Attention\s+\S+\s+[\d.]+\s+([\d.]+)")
conv=g(r"^Convolution\s+\S+\s+([\d.]+)")
pre =g(r"Token-level time: ([\d.]+) ms")
trans=0.0
for m in re.finditer(r"poolPeak=([\d.]+) MiB",t):
    trans=max(trans,float(m.group(1)))
scratch=0.0
for m in re.finditer(r"attnScratch=([\d.]+) MiB",t):
    scratch=max(scratch,float(m.group(1)))
rss=0
try:
    for line in open(sys.argv[2],errors="replace"):
        n=re.search(r"(\d+)",line)
        if n: rss=max(rss,int(n.group(1)))
except Exception: pass
ratio = (attn/conv) if (attn and conv) else None
print("%-8s %-9s %-9s %-11s %-10s %-12s %-9s %-6s"%(
  sys.argv[3],
  f"{aavg:.2f}" if aavg else "-",
  f"{conv:.0f}" if conv else "-",
  f"{ratio:.4f}" if ratio else "-",
  f"{pre:.0f}" if pre else "-",
  f"{scratch:.2f}/{trans:.0f}" if trans else "-",
  f"{rss/1024:.0f}" if rss else "-",
  "y" if "flash attention ON" in t else "n"))
PY
done
