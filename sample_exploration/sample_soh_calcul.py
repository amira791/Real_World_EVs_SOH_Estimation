# ============================================================
# aggregate_soh.py
# Compute final SOH label for a single VIN from its extracted windows.
# ============================================================
import os
import numpy as np
import pandas as pd

WINDOWS_CSV = r"\\FS19\HomeDir\129963\Desktop\DR2\3 Contributions\Real_World_EVs_SOH_Estimation\sample_exploration\output_sample\vin93_windows.csv"
OUT_CSV     = r"\\FS19\HomeDir\129963\Desktop\DR2\3 Contributions\Real_World_EVs_SOH_Estimation\sample_exploration\vin93_soh_label.csv"

# Quality filters (all derived from the paper + observed data)
MIN_DURATION_S = 300
MIN_DSOC_PCT   = 5.0
V_IN_RANGE     = (3.88, 3.92)
V_OUT_RANGE    = (4.04, 4.10)
CAP_RANGE_AH   = (80, 180)


def aggregate(win_csv):
    w = pd.read_csv(win_csv)

    print(f"Loaded {len(w)} windows")
    print(f"Initial SOH range: {w.SOH_pct.min():.1f} – {w.SOH_pct.max():.1f}% "
          f"(median {w.SOH_pct.median():.1f}%)")

    # Apply filters step-by-step so we can see the effect of each
    filters = [
        ("duration",  lambda d: d.duration_s >= MIN_DURATION_S),
        ("ΔSOC",      lambda d: d.dSOC >= MIN_DSOC_PCT),
        ("V_in",      lambda d: d.v_in.between(*V_IN_RANGE)),
        ("V_out",     lambda d: d.v_out.between(*V_OUT_RANGE)),
        ("capacity",  lambda d: d.cap_Ah.between(*CAP_RANGE_AH)),
    ]

    for name, f in filters:
        n_before = len(w)
        w = w[f(w)].reset_index(drop=True)
        print(f"after {name:10s}: {len(w):4d} windows left "
              f"(removed {n_before - len(w)})")

    print()
    print("Surviving windows statistics:")
    print(f"  count         : {len(w)}")
    print(f"  duration (s)  : {w.duration_s.min():.0f} / "
          f"{w.duration_s.median():.0f} / {w.duration_s.max():.0f}")
    print(f"  ΔSOC (%)      : {w.dSOC.min():.1f} / "
          f"{w.dSOC.median():.1f} / {w.dSOC.max():.1f}")
    print(f"  capacity (Ah) : {w.cap_Ah.min():.1f} / "
          f"{w.cap_Ah.median():.1f} / {w.cap_Ah.max():.1f}")
    print(f"  SOH (%)       : {w.SOH_pct.min():.1f} / "
          f"{w.SOH_pct.median():.1f} / {w.SOH_pct.max():.1f}")

    # Three aggregators for cross-checking
    final = {
        'vin':            'vin93',
        'n_windows':      len(w),
        'SOH_median':     float(w.SOH_pct.median()),
        'SOH_trimmed_mean': float(
            w.SOH_pct.sort_values().iloc[int(0.1*len(w)):
                                          int(0.9*len(w))].mean()
        ),
        'SOH_weighted_by_dSOC': float(
            (w.SOH_pct * w.dSOC).sum() / w.dSOC.sum()
        ),
    }

    print()
    print("Aggregated SOH labels:")
    for k, v in final.items():
        if k not in ('vin', 'n_windows'):
            print(f"  {k:24s}: {v:.2f}%")
    print(f"  {'n_windows':24s}: {final['n_windows']}")

    pd.DataFrame([final]).to_csv(OUT_CSV, index=False)
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    aggregate(WINDOWS_CSV)