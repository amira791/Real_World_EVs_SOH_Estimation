"""
verify_assumptions.py — config-driven, no hardcoded column names or filenames
Usage:
    python verify_assumptions.py --dataset "<path>" --n-vins 3 --max-rows 500000
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from collections import Counter

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Folder containing vin*.csv")
    ap.add_argument("--pattern", default="vin*.csv")
    ap.add_argument("--n-vins", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=500_000)
    ap.add_argument("--pick", choices=["first", "random"], default="first",
                    help="Which VINs to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--packed-seps", default="~,;|",
                    help="Candidate separators for packed columns")
    return ap.parse_args()

# ---------------- Helpers ----------------
def infer_column(df, keywords, kind="first"):
    """Find a column whose name contains any of the keywords (case-insensitive)."""
    cols = [c for c in df.columns
            if any(k.lower() in c.lower() for k in keywords)]
    if not cols:
        return None
    return cols[0] if kind == "first" else cols

def infer_time_column(df):
    return infer_column(df, ["terminaltime", "timestamp", "time"])

def infer_packed_candidates(df, seps, min_elements=5):
    """Detect object-dtype columns whose string values split into many parts."""
    candidates = {}
    for c in df.select_dtypes(include=["object"]).columns:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        # test each separator
        best = None
        for sep in seps:
            counts = s.head(200).str.split(sep).str.len()
            if counts.median() >= min_elements and counts.nunique() <= 3:
                best = (sep, int(counts.median()))
                break
        if best:
            candidates[c] = best
    return candidates

def infer_status_column(df):
    return infer_column(df, ["chargestatus", "status", "state"])

# ---------------- Main ----------------
def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.dataset, args.pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern} in {args.dataset}")

    if args.pick == "random":
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(files), size=min(args.n_vins, len(files)),
                         replace=False)
        picked = [files[i] for i in idx]
    else:
        picked = files[:args.n_vins]

    seps = args.packed_seps.split(",")
    print(f"Files found: {len(files)}")
    print(f"VINs used  : {[os.path.basename(p) for p in picked]}\n")

    # ---------- SCHEMA DISCOVERY (from first file) ----------
    df0 = pd.read_csv(picked[0], nrows=min(args.max_rows, 10_000))
    time_col   = infer_time_column(df0)
    status_col = infer_status_column(df0)
    packed     = infer_packed_candidates(df0, seps)

    print("=" * 80)
    print("SCHEMA DISCOVERY")
    print("=" * 80)
    print(f"  columns        : {list(df0.columns)}")
    print(f"  time column    : {time_col}")
    print(f"  status column  : {status_col}")
    print(f"  packed columns : {packed}\n")

    # ---------- TEST 1: TIME UNIT ----------
    print("=" * 80)
    print("TEST 1: time unit")
    print("=" * 80)
    for f in picked:
        t = pd.read_csv(f, usecols=[time_col],
                        nrows=args.max_rows)[time_col].values
        d = np.diff(t)
        c = Counter(d)
        print(f"\n{os.path.basename(f)}  (n={len(t)})")
        print(f"  t[0:5]  = {t[:5].tolist()}")
        print(f"  t[-3:]  = {t[-3:].tolist()}")
        print(f"  Δt: min={d.min():.3f}  median={np.median(d):.3f}  max={d.max():.3f}")
        print(f"  Δt == 0 : {(d == 0).sum()}")
        print(f"  top gaps: {c.most_common(6)}")

        # Auto-infer: mode gap G; sampling interval is *either* G sec or 0.1*G sec.
        mode_gap = c.most_common(1)[0][0]
        span = t[-1] - t[0]
        print(f"  mode gap = {mode_gap}")
        print(f"  if 1 tick = 1 s   → span = {span/3600:8.2f} h "
              f"({span/86400:.2f} d)")
        print(f"  if 1 tick = 0.1 s → span = {span/36000:8.2f} h "
              f"({span/864000:.2f} d)")

    # ---------- TEST 2: PACKED COLUMNS ----------
    print("\n" + "=" * 80)
    print("TEST 2: packed column decoding")
    print("=" * 80)
    df = pd.read_csv(picked[0], nrows=args.max_rows)

    decoded = {}
    for col, (sep, median_n) in packed.items():
        s = df[col].dropna().astype(str)
        parts = s.str.split(sep)
        counts = parts.str.len()
        arr = np.array(parts.iloc[0], dtype=float)
        decoded[col] = (sep, arr)
        print(f"\n--- {col} ---")
        print(f"  separator           : '{sep}'")
        print(f"  n_missing           : {df[col].isna().sum()} "
              f"({df[col].isna().mean()*100:.2f}%)")
        print(f"  elements per sample : min={counts.min()}  "
              f"median={counts.median()}  max={counts.max()}")
        print(f"  first sample parsed : n={len(arr)}  "
              f"min={arr.min():.4f}  mean={arr.mean():.4f}  max={arr.max():.4f}")

    # cross-check any packed column against any numeric 'total*' column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col, (sep, arr) in decoded.items():
        n = len(arr)
        # try to find a numeric column whose per-element average matches
        candidates = [c for c in numeric_cols
                      if c.lower().startswith("total") or "voltage" in c.lower()]
        if not candidates:
            continue
        for cand in candidates:
            mean_cell = df[col].dropna().astype(str).str.split(sep) \
                          .apply(lambda x: np.mean([float(v) for v in x])) \
                          .reindex(df.index)
            expected = df[cand] / n
            diff = (mean_cell - expected).abs()
            print(f"  cross-check {col} vs {cand}/{n}: "
                  f"mean|diff|={diff.mean():.5f}  max={diff.max():.5f}")

    # ---------- TEST 3: STATUS CODES ----------
    print("\n" + "=" * 80)
    print("TEST 3: status code semantics")
    print("=" * 80)
    for f in picked:
        d = pd.read_csv(f, nrows=args.max_rows)
        print(f"\n--- {os.path.basename(f)} ---")
        aggs = {}
        for c in d.columns:
            if d[c].dtype.kind in "iuf":
                aggs[c] = ["mean", "min", "max"]
        g = d.groupby(status_col).agg(aggs)
        # flatten
        g.columns = ["_".join(x) for x in g.columns]
        # add count
        g["n"] = d.groupby(status_col).size()
        print(g.round(2).to_string())

        # transitions
        s = d[status_col].values
        trans = Counter(zip(s[:-1], s[1:]))
        print("  top transitions:",
              sorted(trans.items(), key=lambda x: -x[1])[:6])

    # ---------- TEST 4: DOWNSAMPLING ----------
    print("\n" + "=" * 80)
    print("TEST 4: downsampling feasibility (full row counts)")
    print("=" * 80)
    for f in picked:
        n = sum(1 for _ in open(f, encoding="utf-8", errors="ignore")) - 1
        print(f"  {os.path.basename(f):20s} rows = {n:,}")
        for step in [1, 10, 60, 100, 1000]:
            print(f"      every {step:>5} samples → {n//step:>10,} points")

if __name__ == "__main__":
    main()