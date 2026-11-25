# ============================================================
# plot_soh_curve.py
# Plot SOH vs. time to inspect monotonicity.
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WINDOWS_CSV = r"\\FS19\HomeDir\129963\Desktop\DR2\3 Contributions\Real_World_EVs_SOH_Estimation\sample_exploration\output_sample\vin93_windows.csv"
RAW_CSV     = r"\\FS19\HomeDir\129963\Desktop\DR2\4 Coding\sample_data\vin93.csv"
OUT_PNG     = r"\\FS19\HomeDir\129963\Desktop\DR2\3 Contributions\Real_World_EVs_SOH_Estimation\sample_exploration\output_sample\vin93_SOH_vs_time.png"

# Same filters as aggregate_soh.py
MIN_DURATION_S = 300
MIN_DSOC_PCT   = 5.0
V_IN_RANGE     = (3.88, 3.92)
V_OUT_RANGE    = (4.04, 4.10)
CAP_RANGE_AH   = (80, 180)


def load_time_lookup(raw_csv):
    """Read raw CSV and return row_idx -> terminaltime (seconds)."""
    df = pd.read_csv(raw_csv, usecols=['terminaltime'])
    df = pd.to_numeric(df['terminaltime'], errors='coerce')
    return df.values


def make_plot(win_csv, raw_csv, out_png):
    w = pd.read_csv(win_csv)
    times = load_time_lookup(raw_csv)

    # Parse "start-end" episode into start row index
    w['start_idx'] = w['episode'].str.split('-').str[0].astype(int)
    w['t_seconds'] = times[w['start_idx']]

    # Convert to days since first charge event
    t0 = w['t_seconds'].min()
    w['t_days'] = (w['t_seconds'] - t0) / 86400.0

    # Apply the same quality filters
    w = w[
        (w.duration_s >= MIN_DURATION_S) &
        (w.dSOC       >= MIN_DSOC_PCT) &
        (w.v_in.between(*V_IN_RANGE)) &
        (w.v_out.between(*V_OUT_RANGE)) &
        (w.cap_Ah.between(*CAP_RANGE_AH))
    ].reset_index(drop=True).sort_values('t_days').reset_index(drop=True)

    print(f"Plotting {len(w)} filtered windows")
    print(f"Time span: {w.t_days.min():.1f} to {w.t_days.max():.1f} days "
          f"({(w.t_days.max() - w.t_days.min()) / 365:.1f} years)")
    print(f"SOH: {w.SOH_pct.min():.1f} – {w.SOH_pct.max():.1f}%, "
          f"median {w.SOH_pct.median():.2f}%")

    # Rolling median (window = 20 windows)
    rolling = w.SOH_pct.rolling(20, center=True, min_periods=5).median()

    # --- plot ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(w.t_days, w.SOH_pct, s=14, alpha=0.55,
               color='steelblue', label='individual windows')
    ax.plot(w.t_days, rolling, linewidth=2.5, color='darkorange',
            label='rolling median (20-window)')

    ax.axhline(100, color='gray', linestyle='--', linewidth=1, alpha=0.6,
               label='nominal 100%')
    ax.axhline(w.SOH_pct.median(), color='crimson', linestyle=':',
               linewidth=1.5, label=f'global median {w.SOH_pct.median():.2f}%')

    ax.set_xlabel('Time since first qualifying charge (days)')
    ax.set_ylabel('SOH (%)')
    ax.set_title('vin93 — SOH vs. time (filtered windows)')
    ax.set_ylim(60, 130)
    ax.grid(alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"\nSaved figure → {out_png}")

    # --- monotonicity diagnostics ---
    print("\n" + "=" * 60)
    print("MONOTONICITY CHECK")
    print("=" * 60)

    # (a) Linear regression slope of SOH vs time
    x = w.t_days.values
    y = w.SOH_pct.values
    slope_per_year, intercept = np.polyfit(x / 365.0, y, 1)  # SOH %/year
    print(f"Linear slope: {slope_per_year:+.3f} %/year  (negative = degrading)")

    # (b) Spearman rank correlation (robust monotonic trend)
    from scipy.stats import spearmanr
    rho, pval = spearmanr(x, y)
    print(f"Spearman rho: {rho:+.3f}  (p = {pval:.3e})")
    print("  interpretation:")
    print("    |rho| < 0.1 : essentially no trend (flat)")
    print("    0.1-0.3     : weak trend")
    print("    0.3-0.6     : moderate trend")
    print("    > 0.6       : strong trend")

    # (c) Mann-Kendall (non-parametric monotonic trend test)
    try:
        from scipy.stats import kendalltau
        tau, kp = kendalltau(x, y)
        print(f"Kendall tau: {tau:+.3f}  (p = {kp:.3e})")
    except Exception:
        pass

    # (d) first-50 vs last-50 windows comparison
    head = w.SOH_pct.iloc[:50].median()
    tail = w.SOH_pct.iloc[-50:].median()
    print(f"First 50 windows median: {head:.2f}%")
    print(f"Last  50 windows median: {tail:.2f}%")
    print(f"Change (tail - head): {tail - head:+.2f}% "
          f"({'degrading' if tail < head else 'increasing'})")


if __name__ == "__main__":
    make_plot(WINDOWS_CSV, RAW_CSV, OUT_PNG)