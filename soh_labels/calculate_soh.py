#!/usr/bin/env python3
"""
BMS-based SOH pseudo-labels for EV field data (Liu et al., Nat. Commun. 16, 1137).
"""

import argparse
import os
import numpy as np
import pandas as pd

RATED_AH = 155.0
CHG_I_MIN = 5.0         # A, min charging current (filters noise)
CHG_I_MAX = 200.0       # A, max charging current (filters 1000A sensor errors)
SPEED_MAX = 1.0         # km/h, charging only when parked
MERGE_GAP_S = 60        # merge charge runs separated by <= this
MAX_STEP_S = 30         # a time step longer than this is a gap
MAX_MISSING_FRAC = 0.02 # max 2% missing data allowed for Method A
MIN_DUR_S = 600         # min duration (10 mins)
FULL_SOC = 99           # event must end at BMS SOC >= this
FULL_V = 4.20           # and max cell voltage >= this
V_LO, V_HI = 3.90, 4.05 # Voltage window for Method B
REF_N = 5               # first N eligible events define per-vehicle reference
SMOOTH_W = 5            # rolling window for smoothing
REF_ODO_MAX = 1000.0    # km; reference counts as beginning-of-life below this
MIN_LABEL_EVENTS = 5

USECOLS = ["terminaltime", "soc", "speed", "totalodometer", "chargestatus",
           "totalvoltage", "totalcurrent", "minvoltagebattery",
           "maxvoltagebattery", "mintemperaturevalue", "maxtemperaturevalue"]

def load(path):
    df = pd.read_csv(path, usecols=USECOLS)
    df = (df.dropna(subset=["terminaltime", "totalcurrent", "soc"])
            .drop_duplicates("terminaltime")
            .sort_values("terminaltime")
            .reset_index(drop=True))
    # Unit conversion if needed (mV to V)
    for c in ("minvoltagebattery", "maxvoltagebattery"):
        if df[c].median() > 100:           
            df[c] = df[c] / 1000.0
    df["i_ch"] = -df["totalcurrent"]       # charging current positive
    return df

def find_events(df, vid):
    # FILTERING LOGIC: current > 5A, current < 200A, speed < 1km/h
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
        soc = g.soc.to_numpy()
        
        dt = np.diff(tt)
        ok = dt <= MAX_STEP_S
        dq = (ii[1:] + ii[:-1]) / 2 * dt / 3600.0      # Ah per step (trapezoid)
        dur = tt[-1] - tt[0]
        dQ = dq[ok].sum()
        missing = dt[~ok].sum() / dur if dur > 0 else 1.0
        dsoc = soc[-1] - soc[0]

        # METHOD B: Voltage window 3.90V to 4.05V
        q_b = np.nan; crate = np.nan; n_w = 0; window_ok = False
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
                    window_ok = cv < 0.10     # constant-current check

        full_end = (soc[-1] >= FULL_SOC) and (v[-1] >= FULL_V)
        soc_drop = bool((np.diff(soc) < -1).any())
        
        # METHOD A: Capacity Calculation
        c_a = dQ / (dsoc / 100.0) if dsoc > 0 else np.nan
        # valid_A also checks that dsoc is wide enough later (min_dsoc)
        valid_a = (full_end and not soc_drop and missing <= MAX_MISSING_FRAC
                   and dur >= MIN_DUR_S and np.isfinite(c_a)
                   and 0.4 * RATED_AH <= c_a <= 1.3 * RATED_AH)
        
        rows.append(dict(
            vehicle=vid, event=k, t_start=tt[0], t_end=tt[-1], dur_s=dur,
            odo_km=g.totalodometer.iloc[0], soc_s=soc[0], soc_e=soc[-1], dsoc=dsoc,
            dQ_ah=dQ, missing_frac=missing, vmax_end=v[-1], full_end=full_end,
            soc_drop=soc_drop, i_mean=ii.mean(), crate_mean=ii.mean() / RATED_AH,
            temp_mean=(g.mintemperaturevalue.mean() + g.maxtemperaturevalue.mean()) / 2,
            status=",".join(sorted(map(str, g.chargestatus.unique()))),
            C_A_ah=c_a, valid_A=valid_a,
            Q_B_ah=q_b, crate_B=crate, n_window=n_w, window_ok=window_ok))
    return rows

def label_vehicle(ev, min_dsoc):
    ev = ev.sort_values("t_start").reset_index(drop=True)
    ev["soh_A_rated"] = np.nan
    ev["soh_A_rel"] = np.nan
    ev["soh_A_smooth"] = np.nan
    ev["soh_A_outlier"] = False
    ev["soh_B_rel"] = np.nan

    # --- METHOD A LABELS ---
    ok = ev.valid_A & (ev.dsoc >= min_dsoc)
    if ok.any():
        a = ev[ok]
        # Reference: median of first N valid events
        ref = a.C_A_ah.head(REF_N).median()
        ev.loc[a.index, "soh_A_rated"] = a.C_A_ah / RATED_AH
        ev.loc[a.index, "soh_A_rel"] = a.C_A_ah / ref
        
        # Smoothing and Outlier Detection
        sm = ev.loc[a.index, "soh_A_rated"].rolling(SMOOTH_W, center=True, min_periods=1).median()
        res = ev.loc[a.index, "soh_A_rated"] - sm
        mad = 1.4826 * np.median(np.abs(res - np.median(res)))
        ev.loc[a.index, "soh_A_smooth"] = sm
        ev.loc[a.index, "soh_A_outlier"] = np.abs(res) > 3 * max(mad, 0.005)

    # --- METHOD B LABELS ---
    w = ev[ev.window_ok].copy()
    if len(w):
        # Bin by C-rate (0.05C increments)
        w["cbin"] = (w.crate_B / 0.05).round() * 0.05
        # Reference: median of first 3 events in the same C-rate bin
        w["ref"] = w.groupby("cbin").Q_B_ah.transform(
            lambda x: x.head(3).median() if len(x) >= 3 else np.nan)
        ev.loc[w.index, "soh_B_rel"] = w.Q_B_ah / w.ref
    return ev

def summarize(ev, vid):
    ok = lambda thr: int((ev.valid_A & (ev.dsoc >= thr)).sum())
    return dict(
        vehicle=vid, n_events=len(ev), n_ok30=ok(30), n_ok40=ok(40), n_ok50=ok(50),
        n_window=int(ev.window_ok.sum()),
        odo_min=ev.odo_km.min(), odo_max=ev.odo_km.max(),
        ref_is_bol=bool(ev.odo_km.min() < REF_ODO_MAX),
        rho_A_vs_odo=spearman(ev.soh_A_rated, ev.odo_km),
        rho_B_vs_odo=spearman(ev.soh_B_rel, ev.odo_km),
        rho_A_vs_B=spearman(ev.soh_A_rel, ev.soh_B_rel),
        label_ok=int(ev.soh_A_rated.notna().sum()) >= MIN_LABEL_EVENTS)

def spearman(x, y):
    d = pd.concat([x, y], axis=1).dropna()
    return d.corr(method="spearman").iloc[0, 1] if len(d) >= 5 else np.nan

def main():
    ap = argparse.ArgumentParser()
    # Update these paths for your environment
    ap.add_argument("--input", default=r"\\FS19\HomeDir\129963\Desktop\DR2\4 Coding\sample_data\vin93.csv")
    ap.add_argument("--out", default=r"\\FS19\HomeDir\129963\Desktop\DR2\4 Coding\sample_data\vin93_soh_results")
    ap.add_argument("--min-dsoc", type=float, default=30.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    f = args.input
    vid = os.path.splitext(os.path.basename(f))[0]

    try:
        df = load(f)
    except Exception as exc:
        raise SystemExit(f"Could not read {f}: {exc}")

    rows = find_events(df, vid)
    if not rows:
        raise SystemExit(f"No charging events found in {vid}.")

    ev = label_vehicle(pd.DataFrame(rows), args.min_dsoc)
    sm = pd.DataFrame([summarize(ev, vid)])

    ev.to_csv(os.path.join(args.out, "soh_events.csv"), index=False)
    sm.to_csv(os.path.join(args.out, "soh_vehicle_summary.csv"), index=False)

    print(f"Vehicle: {vid}")
    print(f"Total charging events: {len(ev)}")
    print(f"Valid Method A events (dsoc >= {args.min_dsoc}): {int((ev.valid_A & (ev.dsoc >= args.min_dsoc)).sum())}")
    print(f"Method B voltage-window events: {int(ev.window_ok.sum())}")
    print(f"Method A SOH labels generated: {int(ev.soh_A_rated.notna().sum())}")
    print(f"Method B relative labels generated: {int(ev.soh_B_rel.notna().sum())}")
    print(f"Results saved to: {args.out}")

if __name__ == "__main__":
    main()