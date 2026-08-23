#!/usr/bin/env python3
"""Summarise fa_bench.sh output: prefill ms, Attention op ms, peak VmHWM."""
import os, re, sys, glob

RESDIR = sys.argv[1] if len(sys.argv) > 1 else "fa_results"

def parse_log(path):
    txt = open(path, errors="replace").read()
    # first "=== Prefill Phase Ops Statistics ===" block belongs to the target model
    prefill_ms = None
    m = re.search(r"=== Prefill Phase Ops Statistics ===\s*\nToken-level time: ([\d.]+) ms \((\d+) prompt tokens, ([\d.]+) tokens/s\)", txt)
    if m:
        prefill_ms, ptok, tps = float(m.group(1)), int(m.group(2)), float(m.group(3))
    else:
        ptok = tps = None
    # Attention row inside the first prefill block
    attn = None
    i = txt.find("=== Prefill Phase Ops Statistics ===")
    if i >= 0:
        j = txt.find("=== Decode Phase Ops Statistics ===", i)
        blk = txt[i:j if j > 0 else len(txt)]
        a = re.search(r"^Attention\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", blk, re.M)
        if a:
            attn = (float(a.group(2)), float(a.group(3)), int(a.group(4)))
    ok = "flash attention ON" in txt
    return prefill_ms, ptok, tps, attn, ok

def parse_mem(path):
    peak = 0
    if not os.path.exists(path):
        return None
    for line in open(path):
        f = line.split()
        if len(f) >= 2:
            try:
                peak = max(peak, int(f[1]))
            except ValueError:
                pass
    return peak / 1024.0 if peak else None

rows = []
for log in sorted(glob.glob(os.path.join(RESDIR, "fa*_p*.log"))):
    tag = os.path.basename(log)[:-4]
    m = re.match(r"fa(\d)_p(\d+)", tag)
    if not m:
        continue
    flash, P = int(m.group(1)), int(m.group(2))
    prefill, ptok, tps, attn, saw_fa = parse_log(log)
    peak = parse_mem(log[:-4] + ".mem")
    rows.append((P, flash, prefill, tps, attn, peak, saw_fa))

rows.sort(key=lambda r: (r[0], r[1]))
print(f"{'P':>6} {'path':>8} {'prefill ms':>11} {'tok/s':>8} {'Attn ms':>9} {'Attn avg':>9} {'peak RSS MiB':>13} {'FA?':>4}")
for P, flash, prefill, tps, attn, peak, saw in rows:
    print("{:>6} {:>8} {:>11} {:>8} {:>9} {:>9} {:>13} {:>4}".format(
        P, "flash" if flash else "base",
        f"{prefill:.0f}" if prefill else "-",
        f"{tps:.1f}" if tps else "-",
        f"{attn[0]:.2f}" if attn else "-",
        f"{attn[1]:.3f}" if attn else "-",
        f"{peak:.0f}" if peak else "-",
        "y" if saw else "n"))

# paired deltas
print()
by = {}
for P, flash, prefill, tps, attn, peak, saw in rows:
    by.setdefault(P, {})[flash] = (prefill, tps, attn, peak)
print(f"{'P':>6} {'prefill speedup':>16} {'Attn speedup':>13} {'peak RSS delta MiB':>19}")
for P in sorted(by):
    d = by[P]
    if 0 in d and 1 in d:
        sp = d[0][0] / d[1][0] if d[0][0] and d[1][0] else None
        asp = d[0][2][0] / d[1][2][0] if d[0][2] and d[1][2] and d[1][2][0] else None
        dm = (d[1][3] - d[0][3]) if d[0][3] and d[1][3] else None
        print("{:>6} {:>16} {:>13} {:>19}".format(
            P,
            f"{sp:.2f}x" if sp else "-",
            f"{asp:.2f}x" if asp else "-",
            f"{dm:+.0f}" if dm is not None else "-"))
