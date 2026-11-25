# ============================================================
# explore_voltage_window.py
# ============================================================
import os
import numpy as np
import pandas as pd

INPUT_PATH = r"\\FS19\HomeDir\129963\Desktop\DR2\4 Coding\sample_data\vin93.csv"
V_START = 3.90
V_END = 4.05
CHARGE_STATUS = 1

# NumPy 2.x renamed np.trapz -> np.trapezoid; resolve once, safely
_TRAPZ = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
if _TRAPZ is None:
    raise RuntimeError("Neither np.trapezoid nor np.trapz is available.")


def load_raw(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ['terminaltime', 'soc', 'chargestatus', 'totalcurrent', 'maxvoltagebattery']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['terminaltime'])
    df = df.sort_values('terminaltime').drop_duplicates('terminaltime').reset_index(drop=True)

    df = df[(df.soc >= 0) & (df.soc <= 100)]
    df = df[(df.maxvoltagebattery > 2.0) & (df.maxvoltagebattery <= 4.30)]
    df['gap_s'] = df['terminaltime'].diff()
    return df


def explore_windows(df):
    print("=" * 80)
    print(f"EXPLORING FIXED VOLTAGE WINDOW: [{V_START}V, {V_END}V]")
    print("=" * 80)

    # 1. Find contiguous charge episodes
    is_chg = (df.chargestatus == CHARGE_STATUS).values
    chg_start = np.zeros(len(df), dtype=bool)
    chg_start[1:] = is_chg[1:] & ~is_chg[:-1]
    starts = np.where(chg_start)[0]

    chg_end = np.zeros(len(df), dtype=bool)
    chg_end[:-1] = is_chg[:-1] & ~is_chg[1:]
    ends = np.where(chg_end)[0]
    if is_chg[-1]:
        ends = np.append(ends, len(df) - 1)

    print(f"Total charge episodes found: {len(starts)}")
    print("-" * 80)

    valid_windows = 0
    stats = []

    for s, e in zip(starts, ends):
        seg = df.iloc[s:e + 1]
        if len(seg) < 10:
            continue

        mask_39 = seg.maxvoltagebattery >= V_START
        mask_405 = seg.maxvoltagebattery >= V_END
        if not mask_405.any():
            continue

        idx_39 = seg.index[mask_39]
        idx_405 = seg.index[mask_405]
        if len(idx_39) == 0:
            continue

        idx_start = idx_39[0]
        valid_405 = idx_405[idx_405 > idx_start]
        if len(valid_405) == 0:
            continue
        idx_end = valid_405[0]

        win = seg.loc[idx_start:idx_end]
        if len(win) < 2:
            continue

        t = win.terminaltime.values.astype(float)
        I = win.totalcurrent.values.astype(float)
        dQ_Ah = -_TRAPZ(I, t) / 3600.0

        soc_start = win.soc.iloc[0]
        soc_end = win.soc.iloc[-1]
        duration = t[-1] - t[0]
        dSOC = soc_end - soc_start

        valid_windows += 1
        cap = dQ_Ah / (dSOC / 100.0) if dSOC > 0 else np.nan
        soh = cap / 155.0 * 100.0 if not np.isnan(cap) else np.nan

        stats.append({
            'episode': f"{s}-{e}",
            'duration_s': duration,
            'v_in': win.maxvoltagebattery.iloc[0],
            'v_out': win.maxvoltagebattery.iloc[-1],
            'soc_in': soc_start,
            'soc_out': soc_end,
            'dSOC': dSOC,
            'dQ_Ah': dQ_Ah,
            'cap_Ah': cap,
            'SOH_pct': soh,
        })

        if valid_windows <= 20:
            print(f"Charge Episode {s}-{e}")
            print(f"  Window duration: {duration:.0f} s")
            print(f"  Voltage: {win.maxvoltagebattery.iloc[0]:.3f} V -> {win.maxvoltagebattery.iloc[-1]:.3f} V")
            print(f"  SOC: {soc_start}% -> {soc_end}% (Delta SOC = {dSOC}%)")
            print(f"  Delta Q: {dQ_Ah:.2f} Ah")
            if not np.isnan(soh):
                print(f"  Estimated Capacity: {cap:.1f} Ah (SOH: {soh:.1f}%)")
            print("-" * 40)

    print(f"\nTotal valid voltage windows found: {valid_windows}")

    # Summary statistics
    if stats:
        st = pd.DataFrame(stats)
        st.to_csv(os.path.join(os.path.dirname(INPUT_PATH), 'vin93_windows.csv'), index=False)
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS OF VALID WINDOWS")
        print("=" * 80)
        print(f"count            : {len(st)}")
        print(f"duration (s)     : min={st.duration_s.min():.0f}  median={st.duration_s.median():.0f}  max={st.duration_s.max():.0f}")
        print(f"dSOC (%)         : min={st.dSOC.min():.1f}  median={st.dSOC.median():.1f}  max={st.dSOC.max():.1f}")
        print(f"dQ (Ah)          : min={st.dQ_Ah.min():.1f}  median={st.dQ_Ah.median():.1f}  max={st.dQ_Ah.max():.1f}")
        print(f"capacity (Ah)    : min={st.cap_Ah.min():.1f}  median={st.cap_Ah.median():.1f}  max={st.cap_Ah.max():.1f}")
        print(f"SOH (%)          : min={st.SOH_pct.min():.1f}  median={st.SOH_pct.median():.1f}  max={st.SOH_pct.max():.1f}")
        print(f"\nSaved to vin93_windows.csv for further inspection.")


if __name__ == "__main__":
    print(f"Loading data from {INPUT_PATH}...")
    df = load_raw(INPUT_PATH)
    explore_windows(df)