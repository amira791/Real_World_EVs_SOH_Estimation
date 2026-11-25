#!/usr/bin/env python3
"""BMS-based SOH pseudo-labels, single vehicle (corrected version).

Usage:  python soh_label_generator_fixed.py path/to/vin93.csv [output_dir]
Labels are pseudo-labels: they inherit BMS SOC error (no OCV table available).
"""
import os
import sys

import numpy as np
import pandas as pd
import numpy as np

DEFAULT_INPUT = r"C:\Users\admin\Desktop\DR2\11 All Datasets\07 300-EV Real-World BMS Dataset Liu et al. (2025)\300EV_Sample\vin93.csv"
DEFAULT_OUT = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr05\Real_World_EVs_SOH_Estimation\soh_labels\results"

RATED_AH = 155.0
CHARGE_SIGN = -1        # in this dataset charging current is NEGATIVE (discharge positive)
MIN_I_CHG = 5.0         # A
MAX_I_CHG = 250.0       # A
MAX_SPEED = 2.0         # km/h
MIN_DUR_SEC = 600
MERGE_GAP_SEC = 60
RESUME_GAP_SEC = 600    # re-join an interrupted charge if odometer and SOC are continuous
MAX_STEP_SEC = 30       # a time step longer than this is a data gap (not integrated)
MAX_MISSING = 0.02      # max share of event duration that may be missing
SOC_DROP_MAX = 2        # points
MIN_DSOC = 50.0         # wide SOC window (also try 40 and 60)
FULL_SOC_MIN = 99.0     # event must END at full charge (BMS SOC jumps to 100)
FULL_V_MIN = 4.20       # AND max cell voltage near the 4.25 V cutoff

USECOLS = ["terminaltime", "soc", "speed", "totalodometer", "chargestatus",
           "totalcurrent", "minvoltagebattery", "maxvoltagebattery"]


def load_and_clean_data(path):
    print(f"[*] Reading dataset: {path}")
    df = pd.read_csv(path, usecols=lambda c: c.lower() in USECOLS)
    df.columns = [c.lower() for c in df.columns]

    # time in seconds (keep numeric values as they are; no datetime round trip)
    if pd.api.types.is_numeric_dtype(df["terminaltime"]):
        df["t_sec"] = df["terminaltime"].astype(float)
    else:
        dt = pd.to_datetime(df["terminaltime"])
        df["t_sec"] = (dt - pd.Timestamp("1970-01-01")).dt.total_seconds()

    for col in ("minvoltagebattery", "maxvoltagebattery"):
        if df[col].median() > 100.0:           # mV -> V
            df[col] = df[col] / 1000.0

    # Invalid frames decode to: current -1000 A, status 255, cell voltage 0, temp -40 C.
    n0 = len(df)
    bad = ((df["totalcurrent"].abs() >= 1000)
           | (df["maxvoltagebattery"] <= 0) | (df["minvoltagebattery"] <= 0))
    if "chargestatus" in df.columns:
        bad |= df["chargestatus"] == 255
    df = df[~bad]
    print(f"[*] Dropped {n0 - len(df)} invalid frames")

    df = (df.dropna(subset=["t_sec", "totalcurrent", "soc"])
            .drop_duplicates("t_sec").sort_values("t_sec").reset_index(drop=True))
    df["i_ch"] = CHARGE_SIGN * df["totalcurrent"]    # charging current positive
    return df


def extract_charging_events(df, vid):
    mask = ((df["i_ch"] >= MIN_I_CHG) & (df["i_ch"] <= MAX_I_CHG)
            & (df["speed"] <= MAX_SPEED)).to_numpy()
    if not mask.any():
        print("[-] No points matched charging criteria.")
        return []
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1

    t = df["t_sec"].to_numpy()
    odo = df["totalodometer"].to_numpy()
    soc_all = df["soc"].to_numpy()
    groups = []
    for s, e in zip(starts, ends):
        if groups:
            pe = groups[-1][1]
            gap = t[s] - t[pe]
            resumed = (gap <= RESUME_GAP_SEC and abs(odo[s] - odo[pe]) <= 0.2
                       and abs(soc_all[s] - soc_all[pe]) <= 2)
            if gap <= MERGE_GAP_SEC or resumed:
                groups[-1][1] = e
                continue
        groups.append([s, e])
    print(f"[*] Analyzing {len(groups)} detected charging session(s)...")

    events = []
    for k, (s, e) in enumerate(groups):
        if e <= s:
            continue
        seg = df.iloc[s:e + 1]
        tt, ii = seg["t_sec"].to_numpy(), seg["i_ch"].to_numpy()
        v_max, soc = seg["maxvoltagebattery"].to_numpy(), seg["soc"].to_numpy()
        dt = np.diff(tt)
        dur = tt[-1] - tt[0]
        dsoc = soc[-1] - soc[0]
        ok = dt <= MAX_STEP_SEC
        dq = (ii[1:] + ii[:-1]) / 2.0 * dt / 3600.0
        q_ah = dq[ok].sum()
        missing = dt[~ok].sum() / dur if dur > 0 else 1.0
        soc_drop = bool((np.diff(soc) < -SOC_DROP_MAX).any())
        is_full = bool((v_max[-1] >= FULL_V_MIN) and (soc[-1] >= FULL_SOC_MIN))
        c_calc = q_ah / (dsoc / 100.0) if 0 < dsoc <= 100 else np.nan
        valid = bool(is_full and dsoc >= MIN_DSOC and dur >= MIN_DUR_SEC
                     and missing <= MAX_MISSING and not soc_drop
                     and np.isfinite(c_calc)
                     and 0.4 * RATED_AH <= c_calc <= 1.3 * RATED_AH)
        events.append({
            "vehicle": vid, "event_id": k, "t_start_s": tt[0], "t_end_s": tt[-1],
            "odometer_km": seg["totalodometer"].iloc[0],
            "duration_min": round(dur / 60.0, 1),
            "soc_start": soc[0], "soc_end": soc[-1], "delta_soc": round(dsoc, 1),
            "integrated_ah": round(q_ah, 2), "missing_frac": round(missing, 3),
            "soc_drop": soc_drop, "v_max_end": round(v_max[-1], 3),
            "c_calculated_ah": round(c_calc, 2) if np.isfinite(c_calc) else np.nan,
            "is_full_charge": is_full, "is_valid_event": valid})
    return events


def compute_soh_metrics(ev):
    ev["soh_calculated"] = np.nan
    ev["soh_smoothed"] = np.nan
    m = ev["is_valid_event"]
    if m.any():
        soh = ev.loc[m, "c_calculated_ah"] / RATED_AH
        ev.loc[m, "soh_calculated"] = soh
        ev.loc[m, "soh_smoothed"] = soh.rolling(5, center=True, min_periods=1).median()
    return ev


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)
    vid = os.path.splitext(os.path.basename(path))[0]

    df = load_and_clean_data(path)
    events = extract_charging_events(df, vid)
    if not events:
        return
    ev = compute_soh_metrics(pd.DataFrame(events))

    # sanity check on the current sign: real charges must RAISE the SOC
    share_up = (ev["delta_soc"] > 0).mean()
    print(f"[*] Sanity check: {share_up:.0%} of detected events have delta SOC > 0")
    if share_up < 0.5:
        print("[!] WARNING: most 'charging' events lower the SOC. Check CHARGE_SIGN.")

    out = os.path.join(out_dir, f"{vid}_soh_labels.csv")
    ev.to_csv(out, index=False)
    print(f"Total events: {len(ev)} | full-charge: {int(ev.is_full_charge.sum())} | "
          f"valid labels: {int(ev.is_valid_event.sum())}\nSaved: {out}")


if __name__ == "__main__":
    main()