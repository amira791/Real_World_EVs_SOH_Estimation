#!/usr/bin/env python3
"""SOH Ground Truth Label Generator (Without OCV-SOC Table)

Method: Ampere-Hour Integration over Wide-SOC Charging Events
Reference: Liu et al., Nature Communications (2025)
"""

import os
import numpy as np
import pandas as pd

# ==============================================================================
# 1. FILE PATHS & FLEET PARAMETERS
# ==============================================================================
INPUT_PATH = r"\\FS19\HomeDir\129963\Desktop\DR2\4 Coding\sample_data\vin93.csv"
OUTPUT_DIR = (
    r"\\FS19\HomeDir\129963\Desktop\DR2\4 Coding\sample_data\vin93_soh_results"
)

RATED_AH = 155.0  # Rated pack capacity in Ah[cite: 4]
MIN_I_CHG = 1.0  # Minimum charging current in A to filter noise
MAX_I_CHG = 250.0  # Maximum plausible current in A to filter sensor errors
MAX_SPEED = 2.0  # Max vehicle speed in km/h to account for GPS/speed drift[cite: 7]
MIN_DUR_SEC = 300  # Minimum charging duration (5 minutes)
MAX_GAP_SEC = 60  # Merge charge sub-segments separated by <= 60 seconds
MIN_DSOC = 25.0  # Minimum required ΔSOC (%) to ensure low BMS error[cite: 7]
FULL_SOC_MIN = 95.0  # Minimum end SOC (%) for full-charge classification[cite: 7, 9]
FULL_V_MIN = 4.15  # Minimum cell max-voltage (V) for full charge[cite: 7, 9]

USECOLS = [
    "terminaltime",
    "soc",
    "speed",
    "totalodometer",
    "totalcurrent",
    "minvoltagebattery",
    "maxvoltagebattery",
]


# ==============================================================================
# 2. PREPROCESSING & SIGNAL CLEANING
# ==============================================================================
def load_and_clean_data(path):
    """Loads CSV, standardizes column names, converts timestamps and voltage units."""
    print(f"[*] Reading dataset: {path}")
    df = pd.read_csv(path, usecols=lambda c: c.lower() in USECOLS)
    df.columns = [c.lower() for c in df.columns]

    # Convert timestamps to numeric seconds
    if pd.api.types.is_numeric_dtype(df["terminaltime"]):
        unit = "ms" if df["terminaltime"].median() > 1e11 else "s"
        df["datetime"] = pd.to_datetime(df["terminaltime"], unit=unit)
    else:
        df["datetime"] = pd.to_datetime(df["terminaltime"])

    df["t_sec"] = df["datetime"].astype("int64") // 1e9

    # Sort and remove duplicate timestamps
    df = (
        df.dropna(subset=["t_sec", "totalcurrent", "soc"])
        .drop_duplicates("t_sec")
        .sort_values("t_sec")
        .reset_index(drop=True)
    )

    # Standardize cell voltage units (convert mV to V if necessary)
    for col in ("minvoltagebattery", "maxvoltagebattery"):
        if col in df.columns and df[col].median() > 100.0:
            df[col] = df[col] / 1000.0

    # Auto-detect current polarity (ensure charging current is positive)
    if df["totalcurrent"].median() < 0:
        df["i_ch"] = -df["totalcurrent"]
    else:
        df["i_ch"] = df["totalcurrent"]

    return df


# ==============================================================================
# 3. CHARGING EVENT SEGMENTATION & AMPERE-HOUR INTEGRATION
# ==============================================================================
def extract_charging_events(df, vid):
    """Isolates continuous stationary charging sessions and computes capacity."""
    # Filter stationary charging states
    is_charging = (
        (df["i_ch"] >= MIN_I_CHG)
        & (df["i_ch"] <= MAX_I_CHG)
        & (df["speed"] <= MAX_SPEED)
    ).to_numpy()

    if not is_charging.any():
        print("[-] No points matched charging criteria.")
        return []

    # Detect start and end bounds
    edges = np.diff(np.r_[0, is_charging.astype(int), 0])
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1

    t_array = df["t_sec"].to_numpy()

    # Merge consecutive sub-segments separated by brief gaps (<= MAX_GAP_SEC)
    merged_groups = []
    for s, e in zip(starts, ends):
        if (
            merged_groups
            and t_array[s] - t_array[merged_groups[-1][1]] <= MAX_GAP_SEC
        ):
            merged_groups[-1][1] = e
        else:
            merged_groups.append([s, e])

    events = []
    print(f"[*] Analyzing {len(merged_groups)} detected charging session(s)...")

    for k, (s, e) in enumerate(merged_groups):
        if e <= s:
            continue

        seg = df.iloc[s : e + 1]
        t = seg["t_sec"].to_numpy()
        i = seg["i_ch"].to_numpy()
        v_max = seg["maxvoltagebattery"].to_numpy()
        soc = seg["soc"].to_numpy()

        dt = np.diff(t)
        if len(dt) == 0:
            continue

        dur_s = t[-1] - t[0]
        dsoc = soc[-1] - soc[0]

        # Coulomb Counting (Trapezoidal integration in Ah)
        dq_steps = (i[1:] + i[:-1]) / 2.0 * (dt / 3600.0)
        total_q_ah = dq_steps.sum()

        # Full charge condition check
        is_full = (v_max[-1] >= FULL_V_MIN) or (soc[-1] >= FULL_SOC_MIN)[
            cite: 7, 9
        ]

        # Calculate extrapolated 0-100% capacity
        c_calculated = (
            total_q_ah / (dsoc / 100.0)
            if (dsoc > 0 and dsoc <= 100)
            else np.nan
        )

        # Validation rules
        valid = bool(
            is_full
            and (dsoc >= MIN_DSOC)
            and (dur_s >= MIN_DUR_SEC)
            and np.isfinite(c_calculated)
            and (0.4 * RATED_AH <= c_calculated <= 1.3 * RATED_AH)
        )

        events.append(
            {
                "vehicle": vid,
                "event_id": k,
                "t_start": seg["datetime"].iloc[0],
                "t_end": seg["datetime"].iloc[-1],
                "duration_min": round(dur_s / 60.0, 1),
                "soc_start": soc[0],
                "soc_end": soc[-1],
                "delta_soc": round(dsoc, 1),
                "integrated_ah": round(total_q_ah, 2),
                "v_max_end": round(v_max[-1], 3),
                "c_calculated_ah": round(c_calculated, 2)
                if np.isfinite(c_calculated)
                else np.nan,
                "is_full_charge": is_full,
                "is_valid_event": valid,
            }
        )

    return events


# ==============================================================================
# 4. SOH LABEL GENERATION & SMOOTHING
# ==============================================================================
def compute_soh_metrics(df_events):
    """Calculates SOH values and filters local outliers using rolling MAD."""
    df_events["soh_calculated"] = np.nan
    df_events["soh_smoothed"] = np.nan

    valid_mask = df_events["is_valid_event"]
    if valid_mask.any():
        # SOH = C_calculated / C_rated
        soh_vals = df_events.loc[valid_mask, "c_calculated_ah"] / RATED_AH[cite: 1, 4]
        df_events.loc[valid_mask, "soh_calculated"] = soh_vals

        # Rolling median smoothing
        smooth_soh = soh_vals.rolling(
            window=5, center=True, min_periods=1
        ).median()
        df_events.loc[valid_mask, "soh_smoothed"] = smooth_soh

    return df_events


# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vid = os.path.splitext(os.path.basename(INPUT_PATH))[0]

    try:
        df_raw = load_and_clean_data(INPUT_PATH)
    except Exception as err:
        print(f"[-] Error loading file: {err}")
        return

    events = extract_charging_events(df_raw, vid)
    if not events:
        print("[-] No charging events were extracted.")
        return

    df_events = pd.DataFrame(events)
    df_results = compute_soh_metrics(df_events)

    # Save to CSV
    output_file = os.path.join(OUTPUT_DIR, f"{vid}_soh_labels.csv")
    df_results.to_csv(output_file, index=False)

    # Output detailed report
    print("\n" + "=" * 65)
    print(f"          CHARGING SESSIONS DIAGNOSTIC & SOH SUMMARY")
    print("=" * 65)
    for idx, row in df_results.iterrows():
        status = "VALID" if row["is_valid_event"] else "INVALID"
        soh_str = (
            f"{row['soh_calculated']*100:.2f}%"
            if pd.notna(row["soh_calculated"])
            else "N/A"
        )
        print(
            f"Event {row['event_id']:02d} | Status: {status:<7} | "
            f"SOC: {row['soc_start']:>4.1f}% -> {row['soc_end']:>4.1f}% (ΔSOC: {row['delta_soc']:>4.1f}%) | "
            f"V_end: {row['v_max_end']:.2f}V | SOH: {soh_str}"
        )

    valid_count = int(df_results["is_valid_event"].sum())
    print("=" * 65)
    print(f"Total Charging Events Detected   : {len(df_results)}")
    print(f"Valid SOH Ground Truth Labels    : {valid_count}")
    print(f"Output File Path                 : {output_file}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()