#!/usr/bin/env python3
"""SOH Label Generator using Wide-SOC BMS Capacity Integration

(Liu et al., Nature Communications 2025)
"""

import os
import numpy as np
import pandas as pd

# Hardcoded File Paths
INPUT_PATH = r"\\FS19\HomeDir\129963\Desktop\DR2\4 Coding\sample_data\vin93.csv"
OUTPUT_DIR = (
    r"\\FS19\HomeDir\129963\Desktop\DR2\4 Coding\sample_data\vin93_soh_results"
)

# Fleet & Filtering Constants (Relaxed for Real-World Field Noise)
RATED_AH = 155.0  # Ah capacity specified for the fleet[cite: 4]
CHG_I_MIN = 2.0  # A, filter standby current (relaxed from 5.0)
CHG_I_MAX = 200.0  # A, filter sensor glitches
SPEED_MAX = 1.0  # km/h, allow small speed noise/GPS drift[cite: 7]
MIN_DUR_S = 300  # min charging duration (5 mins / 300 s)
MAX_STEP_S = 30  # max allowable gap between timestamps
MAX_MISSING_FRAC = 0.05  # allow up to 5% missing data
FULL_V = 4.18  # V, cut-off voltage target (relaxed from 4.25)[cite: 7, 9]
MIN_DSOC = 30.0  # % min delta SOC to reduce BMS estimation error[cite: 7]
REF_N = 5  # first N valid events for relative baseline calculation[cite: 7]
SMOOTH_W = 5  # window size for rolling median outlier smoothing[cite: 7]

USECOLS = [
    "terminaltime",
    "soc",
    "speed",
    "totalodometer",
    "totalcurrent",
    "minvoltagebattery",
    "maxvoltagebattery",
]


def load_and_preprocess(path):
    """Loads CSV data, formats timestamps, and standardizes units."""
    df = pd.read_csv(path, usecols=lambda c: c.lower() in USECOLS)
    df.columns = [c.lower() for c in df.columns]

    # Convert timestamps
    if pd.api.types.is_numeric_dtype(df["terminaltime"]):
        unit = "ms" if df["terminaltime"].median() > 1e11 else "s"
        df["datetime"] = pd.to_datetime(df["terminaltime"], unit=unit)
    else:
        df["datetime"] = pd.to_datetime(df["terminaltime"])

    df["t_sec"] = df["datetime"].astype("int64") // 1e9

    # Clean duplicates and sort chronologically
    df = (
        df.dropna(subset=["t_sec", "totalcurrent", "soc"])
        .drop_duplicates("t_sec")
        .sort_values("t_sec")
        .reset_index(drop=True)
    )

    # Unit conversion: convert mV to V if necessary
    for col in ("minvoltagebattery", "maxvoltagebattery"):
        if col in df.columns and df[col].median() > 100.0:
            df[col] = df[col] / 1000.0

    # Auto-detect charging current sign
    if df["totalcurrent"].median() < 0:
        df["i_ch"] = -df["totalcurrent"]
    else:
        df["i_ch"] = df["totalcurrent"]

    return df


def extract_soh_segments(df, vid):
    """Identifies stationary charging segments and calculates capacity via Coulomb Counting."""
    mask = (
        (df.i_ch >= CHG_I_MIN)
        & (df.i_ch <= CHG_I_MAX)
        & (df.speed <= SPEED_MAX)
    ).to_numpy()

    if not mask.any():
        print(
            "[-] Diagnostic: No data points matched current and speed filters."
        )
        print(
            f"    Current range in file: {df.i_ch.min():.2f}A to {df.i_ch.max():.2f}A"
        )
        print(f"    Speed max in file: {df.speed.max():.2f} km/h")
        return []

    # Identify continuous segments
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1

    t = df.t_sec.to_numpy()
    rows = []
    rejected_reasons = {}

    for k, (s, e) in enumerate(zip(starts, ends)):
        if e <= s:
            continue

        g = df.iloc[s : e + 1]
        tt = g.t_sec.to_numpy()
        ii = g.i_ch.to_numpy()
        v_max = g.maxvoltagebattery.to_numpy()
        soc = g.soc.to_numpy()

        dt = np.diff(tt)
        if len(dt) == 0:
            continue

        ok_step = dt <= MAX_STEP_S
        dur = tt[-1] - tt[0]
        missing_ratio = (
            dt[~ok_step].sum() / dur if (dur > 0 and (~ok_step).any()) else 0.0
        )

        # Trapezoidal Ah Integration
        dq_steps = (ii[1:] + ii[:-1]) / 2.0 * dt / 3600.0
        dQ_total = dq_steps[ok_step].sum()

        dsoc = soc[-1] - soc[0]
        full_end = (v_max[-1] >= FULL_V) or (soc[-1] >= 98.0)[cite: 9]
        soc_drop = bool((np.diff(soc) < -1).any())

        c_calculated = dQ_total / (dsoc / 100.0) if dsoc > 0 else np.nan

        # Collect rejection reasons for troubleshooting
        reasons = []
        if not full_end:
            reasons.append(
                f"Not full charge (v_end={v_max[-1]:.2f}V, soc_end={soc[-1]:.1f}%)"
            )
        if dsoc < MIN_DSOC:
            reasons.append(f"Low dSOC ({dsoc:.1f}% < {MIN_DSOC}%)")
        if dur < MIN_DUR_S:
            reasons.append(f"Short duration ({dur}s < {MIN_DUR_S}s)")
        if missing_ratio > MAX_MISSING_FRAC:
            reasons.append(f"High missing fraction ({missing_ratio:.2f})")

        valid_event = len(reasons) == 0 and np.isfinite(c_calculated)

        if not valid_event:
            r_str = ", ".join(reasons) if reasons else "Invalid Capacity Math"
            rejected_reasons[r_str] = rejected_reasons.get(r_str, 0) + 1

        rows.append(
            dict(
                vehicle=vid,
                event=k,
                t_start=g.datetime.iloc[0],
                t_end=g.datetime.iloc[-1],
                dur_s=dur,
                odo_km=g.totalodometer.iloc[0] if "totalodometer" in g else np.nan,
                soc_s=soc[0],
                soc_e=soc[-1],
                dsoc=dsoc,
                dQ_ah=dQ_total,
                vmax_end=v_max[-1],
                c_calculated_ah=c_calculated,
                valid=valid_event,
            )
        )

    if rejected_reasons and not any(r["valid"] for r in rows):
        print("[-] Diagnostic: Segments detected but rejected due to:")
        for reason, count in rejected_reasons.items():
            print(f"    - {count} event(s) failed: {reason}")

    return rows


def generate_soh_labels(df_events):
    """Calculates final SOH labels and performs outlier filtering."""
    ev = df_events.sort_values("t_start").reset_index(drop=True)
    ev["soh_rated"] = np.nan
    ev["soh_relative"] = np.nan
    ev["soh_smooth"] = np.nan
    ev["is_outlier"] = False

    valid_mask = ev.valid
    if valid_mask.any():
        sub = ev[valid_mask]

        # Absolute Rated SOH (C_calc / 155 Ah)
        ev.loc[sub.index, "soh_rated"] = sub.c_calculated_ah / RATED_AH[cite: 1, 4]

        # Relative SOH against vehicle baseline
        baseline_cap = sub.c_calculated_ah.head(REF_N).median()
        ev.loc[sub.index, "soh_relative"] = sub.c_calculated_ah / baseline_cap

        # Outlier Detection
        smooth_val = (
            ev.loc[sub.index, "soh_rated"]
            .rolling(SMOOTH_W, center=True, min_periods=1)
            .median()
        )
        residual = ev.loc[sub.index, "soh_rated"] - smooth_val
        mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))

        ev.loc[sub.index, "soh_smooth"] = smooth_val
        ev.loc[sub.index, "is_outlier"] = np.abs(residual) > 3 * max(mad, 0.005)

    return ev


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vid = os.path.splitext(os.path.basename(INPUT_PATH))[0]

    print(f"[*] Processing dataset: {INPUT_PATH}")
    try:
        df_raw = load_and_preprocess(INPUT_PATH)
    except Exception as e:
        print(f"[-] Failed to load input file: {e}")
        return

    print("[*] Filtering events and computing capacity...")
    event_rows = extract_soh_segments(df_raw, vid)

    if not event_rows:
        print("[-] No charging events extracted.")
        return

    df_events = pd.DataFrame(event_rows)
    df_labeled = generate_soh_labels(df_events)

    out_file = os.path.join(OUTPUT_DIR, f"{vid}_soh_labels.csv")
    df_labeled.to_csv(out_file, index=False)

    valid_count = int(df_labeled.valid.sum())
    print("\n" + "=" * 50)
    print(f"       SOH CALCULATION RESULTS: {vid.upper()}")
    print("=" * 50)
    print(f"Total Charging Events Detected  : {len(df_labeled)}")
    print(f"Valid Wide-SOC Events (ΔSOC >= {MIN_DSOC}%) : {valid_count}")
    print(
        f"Generated SOH Ground Truth Labels: {int(df_labeled.soh_rated.notna().sum())}"
    )
    print("=" * 50)
    print(f"[+] Output saved to: {out_file}\n")


if __name__ == "__main__":
    main()