#!/usr/bin/env python3
"""
Method B only: Relative SOH index from voltage window (3.90V - 4.05V).
Adjusted for higher data yield while maintaining physical validity.
"""

import os
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION - UPDATE THESE PATHS
# ==========================================
INPUT_CSV = r"\\FS19\HomeDir\129963\Desktop\DR2\4 Coding\sample_data\vin93.csv"
OUTPUT_DIR = r"\\FS19\HomeDir\129963\Desktop\DR2\4 Coding\sample_data\vin93_soh_results"
# ==========================================

RATED_AH = 155.0
CHG_I_MIN = 1.0         # A, min charging current (lowered from 5.0 to capture slow/CV charging)
CHG_I_MAX = 200.0       # A, max charging current (filter 1000A errors)
SPEED_MAX = 1.0         # km/h, charging only when parked
MERGE_GAP_S = 60        # merge charge runs separated by <= this
MAX_STEP_S = 30         # a time step longer than this is a gap
V_LO, V_HI = 3.90, 4.05 # Voltage window for Method B
MIN_QB_AH = 10.0        # Ah, minimum Q_B to consider window valid (filter partial windows)
CV_THRESHOLD = 0.15     # relaxed from 0.10 to 0.15 to allow fast-charging variability
SOH_CLIP_MIN = 0.5      # clip extreme low values
SOH_CLIP_MAX = 1.15     # clip extreme high values

USECOLS = ["terminaltime", "speed", "totalcurrent", "maxvoltagebattery"]

def load(path):
    df = pd.read_csv(path, usecols=USECOLS)
    df = (df.dropna(subset=["terminaltime", "totalcurrent"])
            .drop_duplicates("terminaltime")
            .sort_values("terminaltime")
            .reset_index(drop=True))
    
    # Convert mV to V if needed
    if df["maxvoltagebattery"].median() > 100:
        df["maxvoltagebattery"] = df["maxvoltagebattery"] / 1000.0
        
    df["i_ch"] = -df["totalcurrent"]       # charging current positive
    return df

def find_events_b(df, vid):
    # Filter charging events
    m = ((df.i_ch > CHG_I_MIN) & (df.i_ch < CHG_I_MAX) & (df.speed <= SPEED_MAX)).to_numpy()
    if not m.any():
        return []
    
    edges = np.diff(np.r_[0, m.astype(int), 0])
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    t = df.terminaltime.to_numpy()
    
    groups = []
    for s, e in zip(starts, ends):
        if groups and t[s] - t[groups[-1][1]] <= MERGE_GAP_S:
            groups[-1][1] = e
        else:
            groups.append([s, e])

    rows = []
    for k, (s, e) in enumerate(groups):
        if e <= s:
            continue
        g = df.iloc[s:e + 1]
        tt = g.terminaltime.to_numpy()
        ii = g.i_ch.to_numpy()
        v = g.maxvoltagebattery.to_numpy()
        dt = np.diff(tt)
        
        # Trapezoidal integration (Ah)
        dq = (ii[1:] + ii[:-1]) / 2 * dt / 3600.0
        
        # Method B: Voltage window 3.90V to 4.05V
        q_b = np.nan
        crate = np.nan
        n_w = 0
        window_ok = False
        
        ia = np.flatnonzero(v >= V_LO)
        if v[0] < V_LO and ia.size:
            a = ia[0]
            ib = np.flatnonzero(v[a:] >= V_HI)
            if ib.size:
                b = a + ib[0]
                iw = ii[a:b + 1]
                if b > a and (dt[a:b] <= MAX_STEP_S).all() and iw.mean() > 0:
                    cv = iw.std() / iw.mean()
                    q_b = dq[a:b].sum()
                    crate = iw.mean() / RATED_AH
                    n_w = b - a + 1
                    # Valid if: constant current (relaxed) AND enough charge transferred
                    window_ok = (cv < CV_THRESHOLD) and (q_b >= MIN_QB_AH)

        rows.append(dict(
            vehicle=vid, event=k, t_start=tt[0], t_end=tt[-1],
            Q_B_ah=q_b, crate_B=crate, n_window=n_w, window_ok=window_ok))
    return rows

def label_vehicle(ev):
    ev = ev.sort_values("t_start").reset_index(drop=True)
    ev["soh_B_rel"] = np.nan

    w = ev[ev.window_ok].copy()
    if len(w):
        # Bin by C-rate (0.05C increments)
        w["cbin"] = (w.crate_B / 0.05).round() * 0.05
        
        # Try to use per-bin reference
        bin_counts = w.groupby("cbin").size()
        w["ref"] = w.groupby("cbin").Q_B_ah.transform(
            lambda x: x.head(3).median() if len(x) >= 3 else np.nan)
        
        # Fallback: use global median for bins with <3 events
        global_median = w.Q_B_ah.median()
        w["ref"] = w["ref"].fillna(global_median)
        
        # Compute relative index
        w["soh_B_rel"] = w.Q_B_ah / w.ref
        
        # Clip extreme values
        w["soh_B_rel"] = w["soh_B_rel"].clip(SOH_CLIP_MIN, SOH_CLIP_MAX)
        
        ev.loc[w.index, "soh_B_rel"] = w.soh_B_rel
        
    return ev

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    vid = os.path.splitext(os.path.basename(INPUT_CSV))[0]

    try:
        df = load(INPUT_CSV)
    except Exception as exc:
        raise SystemExit(f"Could not read {INPUT_CSV}: {exc}")

    rows = find_events_b(df, vid)
    if not rows:
        raise SystemExit(f"No charging events found in {vid}.")

    ev = label_vehicle(pd.DataFrame(rows))
    
    out_path = os.path.join(OUTPUT_DIR, f"{vid}_soh_B.csv")
    ev.to_csv(out_path, index=False)

    print(f"Vehicle: {vid}")
    print(f"Total charging events detected: {len(ev)}")
    print(f"Valid Method B windows (window_ok): {int(ev.window_ok.sum())}")
    print(f"Method B SOH labels generated: {int(ev.soh_B_rel.notna().sum())}")
    print(f"Results saved to: {out_path}")

if __name__ == "__main__":
    main()