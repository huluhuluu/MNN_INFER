#!/bin/bash
# Paired A/B benchmark. The device's GPU clock drifts by >2x between sessions
# (all op types scale together), so absolute ms are only comparable inside one
# session. This interleaves base and flash back to back N times and reports the
# per-round ratio, which is robust to that drift.
#   ./fa_paired.sh <promptLen> [rounds] [TQ] [LSZ]
set -u
SERIAL=${SERIAL:-192.168.124.101:47954}
REMOTE_ROOT=${REMOTE_ROOT:-/data/local/tmp/mnn-fa}
MODEL_DIR=${MODEL_DIR:-/data/local/tmp/mnn-dbg/m}
D="adb -s $SERIAL"
P=${1:-1024}
ROUNDS=${2:-3}
TQ=${3:-4}
LSZ=${4:-128}
OUT=${OUT:-$(pwd)/fa_paired_p$P}
mkdir -p "$OUT"

wait_free() {
  for i in $(seq 1 160); do
    b=$($D shell "ps -A -o NAME" 2>/dev/null | tr -d '\r' | grep -xE "spec_eval|profiler|llm_demo")
    [ -z "$b" ] && return 0
    sleep 15
  done
  return 1
}

# one run, sampling peak VmHWM on device while it executes
run() { # $1 env, $2 tag
  wait_free || return 1
  $D shell "rm -f $REMOTE_ROOT/mem.log"
  $D shell "cd $MODEL_DIR && export LD_LIBRARY_PATH=$REMOTE_ROOT:\$LD_LIBRARY_PATH && $1 ; \
    nohup $REMOTE_ROOT/profiler config.json --prompt-lens=$P --decode-lens=1 --warmup=2 --repeat=3 > $REMOTE_ROOT/run.log 2>&1 &
    sleep 2; pid=\$(pidof profiler)
    while [ -d /proc/\$pid ]; do
      grep -H VmHWM /proc/\$pid/status 2>/dev/null | awk -F: '{print \$3}' >> $REMOTE_ROOT/mem.log
      sleep 0.3
    done" >/dev/null 2>&1
  $D pull "$REMOTE_ROOT/run.log" "$OUT/$2.log" >/dev/null 2>&1
  $D pull "$REMOTE_ROOT/mem.log" "$OUT/$2.mem" >/dev/null 2>&1
}

extract() { # $1 log -> "attn_ms attn_avg prefill_ms tok_s active"
  python3 - "$1" <<'PY'
import re,sys
t=open(sys.argv[1],errors="replace").read()
i=t.find("=== Prefill Phase Ops Statistics ===")
j=t.find("=== Decode Phase Ops Statistics ===",i)
blk=t[i:j] if i>=0 else ""
a=re.search(r"^Attention\s+\S+\s+([\d.]+)\s+([\d.]+)",blk,re.M)
p=re.search(r"Token-level time: ([\d.]+) ms \((\d+) prompt tokens, ([\d.]+) tokens/s\)",blk)
c=re.search(r"^Convolution\s+\S+\s+([\d.]+)",blk,re.M)
print(" ".join([
  a.group(1) if a else "nan", a.group(2) if a else "nan",
  p.group(1) if p else "nan", p.group(3) if p else "nan",
  c.group(1) if c else "nan",
  "1" if "flash attention ON" in t else "0"]))
PY
}
peak() { awk '{gsub(/[^0-9]/,"",$1); if($1>m)m=$1}END{if(m)printf "%.0f", m/1024; else printf "nan"}' "$1" 2>/dev/null; }

echo "P=$P  flash tile TQ=$TQ LSZ=$LSZ  rounds=$ROUNDS"
printf "%-6s %-9s %-9s %-9s %-9s %-9s %-8s %-8s\n" "round" "attn_b" "attn_f" "conv_b" "conv_f" "attnRatio" "rssB" "rssF"
for r in $(seq 1 $ROUNDS); do
  run "export MNN_OPENCL_FLASH_ATTENTION=0" "r${r}_base"
  run "export MNN_OPENCL_FLASH_ATTENTION=1 MNN_FA_TQ=$TQ MNN_FA_LSZ=$LSZ" "r${r}_flash"
  read ab aab pb tb cb onb <<< "$(extract "$OUT/r${r}_base.log")"
  read af aaf pf tf cf onf <<< "$(extract "$OUT/r${r}_flash.log")"
  rb=$(peak "$OUT/r${r}_base.mem"); rf=$(peak "$OUT/r${r}_flash.mem")
  ratio=$(python3 -c "
try: print(f'{float('$ab')/float('$af'):.2f}x')
except Exception: print('-')")
  printf "%-6s %-9s %-9s %-9s %-9s %-9s %-8s %-8s%s\n" "$r" "$ab" "$af" "$cb" "$cf" "$ratio" "$rb" "$rf" \
    "$([ "$onf" = 1 ] && echo "" || echo "  (FA NOT ACTIVE)")"
done
echo
echo "prefill token-level ms / tok_s per round:"
for r in $(seq 1 $ROUNDS); do
  read ab aab pb tb cb onb <<< "$(extract "$OUT/r${r}_base.log")"
  read af aaf pf tf cf onf <<< "$(extract "$OUT/r${r}_flash.log")"
  echo "  r$r base ${pb}ms ${tb}tok/s | flash ${pf}ms ${tf}tok/s"
done
