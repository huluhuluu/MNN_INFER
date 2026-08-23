#!/bin/bash
# Sweep FlashAttention tile parameters at one prompt length and report the
# Attention op time plus the kernel's local/private memory footprint.
#   ./fa_sweep.sh <promptLen> "<TQ:LSZ> <TQ:LSZ> ..."
set -u
SERIAL=${SERIAL:-192.168.124.101:47954}
REMOTE_ROOT=${REMOTE_ROOT:-/data/local/tmp/mnn-fa}
MODEL_DIR=${MODEL_DIR:-/data/local/tmp/mnn-dbg/m}
D="adb -s $SERIAL"
P=${1:-1024}
CFGS=${2:-"16:64 8:64 8:32 16:32 16:128 8:128 32:64 4:64"}
OUT=${OUT:-$(pwd)/fa_sweep_p$P}
mkdir -p "$OUT"

wait_free() {
  for i in $(seq 1 120); do
    b=$($D shell "ps -A -o NAME" 2>/dev/null | tr -d '\r' | grep -xE "spec_eval|profiler|llm_demo")
    [ -z "$b" ] && return 0
    sleep 15
  done
  return 1
}

run() { # $1 env-extra, $2 tag
  wait_free || { echo "device stayed busy" >&2; return 1; }
  $D shell "cd $MODEL_DIR && export LD_LIBRARY_PATH=$REMOTE_ROOT:\$LD_LIBRARY_PATH && $1 ; \
    $REMOTE_ROOT/profiler config.json --prompt-lens=$P --decode-lens=1 --warmup=1 --repeat=2 2>&1" > "$OUT/$2.log"
}

printf "%-10s %-10s %-10s %-9s %-10s %-9s\n" "config" "attn_ms" "attn_avg" "prefill" "localMem" "privMem"
run "export MNN_OPENCL_FLASH_ATTENTION=0" base
python3 - "$OUT/base.log" base <<'PY'
import re,sys
t=open(sys.argv[1],errors="replace").read()
i=t.find("=== Prefill Phase Ops Statistics ===")
j=t.find("=== Decode Phase Ops Statistics ===",i)
blk=t[i:j] if i>=0 else ""
a=re.search(r"^Attention\s+\S+\s+([\d.]+)\s+([\d.]+)",blk,re.M)
p=re.search(r"Token-level time: ([\d.]+) ms",blk)
print("%-10s %-10s %-10s %-9s %-10s %-9s"%(sys.argv[2],
  a.group(1) if a else "-", a.group(2) if a else "-", p.group(1) if p else "-","-","-"))
PY

for c in $CFGS; do
  TQ=${c%%:*}; LSZ=${c##*:}
  run "export MNN_OPENCL_FLASH_ATTENTION=1 MNN_FA_TQ=$TQ MNN_FA_LSZ=$LSZ" "fa_${TQ}_${LSZ}"
  python3 - "$OUT/fa_${TQ}_${LSZ}.log" "tq${TQ}/lsz${LSZ}" <<'PY'
import re,sys
t=open(sys.argv[1],errors="replace").read()
i=t.find("=== Prefill Phase Ops Statistics ===")
j=t.find("=== Decode Phase Ops Statistics ===",i)
blk=t[i:j] if i>=0 else ""
a=re.search(r"^Attention\s+\S+\s+([\d.]+)\s+([\d.]+)",blk,re.M)
p=re.search(r"Token-level time: ([\d.]+) ms",blk)
k=re.search(r"KERNEL_LOCAL_MEM=(\d+)\s+KERNEL_PRIVATE_MEM=(\d+)",t)
on="flash attention ON" in t
print("%-10s %-10s %-10s %-9s %-10s %-9s%s"%(sys.argv[2],
  a.group(1) if a else "-", a.group(2) if a else "-", p.group(1) if p else "-",
  k.group(1) if k else "-", k.group(2) if k else "-", "" if on else "  (FA NOT ACTIVE)"))
PY
done
