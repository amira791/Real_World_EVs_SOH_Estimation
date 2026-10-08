"""
usage_profiles.py
Extract per-VIN usage features (F1-F20) and generate fleet visualizations.
Just run:  python usage_profiles.py
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

# ==================================================================
# >>>>>  EDIT THIS BLOCK — everything you'd otherwise pass on CLI
# ==================================================================
CONFIG = {
    # Path to the folder that contains vin*.csv files
    "dataset": r"Z:\129963\Desktop\DR2\1 Datasets\07 300-EV Real-World BMS Dataset Liu et al. (2025)\300 EVs dataset",

    # Where to write outputs (CSV + PNGs)
    "outdir": r"./usage_out",

    # How many VINs to process (set to None to use all found)
    "n_vins": 12,

    # Pick mode: "first" (alphabetical) or "random"
    "pick": "first",
    "seed": 0,

    # Keep every Nth row (1 = keep all). 10 = 100 s resolution
    "downsample": 10,

    # Optional row cap per file for quick tests; None = no cap
    "max_rows": None,

    # Fast-charge threshold (A) — tune once you see peak-current distribution
    "fast_charge_threshold_A": 40.0,

    # Min segment length in samples (6 = 60 s at 10 s sampling)
    "min_segment_len": 6,
}
# ==================================================================
# <<<<<  END OF EDIT BLOCK
# ==================================================================


# ------------------------------------------------------------------
# Schema discovery
# ------------------------------------------------------------------
def discover_schema(df):
    cols = list(df.columns)
    def find(key):
        for c in cols:
            if key in c.lower():
                return c
        return None
    return {
        "time":     find("terminaltime"),
        "soc":      find("soc"),
        "speed":    find("speed"),
        "odo":      find("odometer"),
        "status":   find("chargestatus"),
        "v_pack":   find("totalvoltage"),
        "i_pack":   find("totalcurrent"),
        "t_min":    find("mintemperaturevalue"),
        "t_max":    find("maxtemperaturevalue"),
        "v_cells":  find("batteryvoltage"),
        "t_probes": find("probetemperatures"),
    }


# ------------------------------------------------------------------
# Segment helper
# ------------------------------------------------------------------
def segments(mask, min_len=3):
    m = np.asarray(mask)
    if len(m) == 0:
        return []
    d = np.diff(m.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends   = list(np.where(d == -1)[0])
    if m[0]:  starts = [0] + starts
    if m[-1]: ends   = ends + [len(m) - 1]
    return [(s, e) for s, e in zip(starts, ends) if (e - s + 1) >= min_len]


# ------------------------------------------------------------------
# Feature extraction for one VIN
# ------------------------------------------------------------------
def extract_features(path, cfg):
    df = pd.read_csv(path, nrows=cfg["max_rows"])
    sch = discover_schema(df)

    df = df[df[sch["status"]] != 255].copy()
    df = df.dropna(subset=[sch["v_cells"], sch["t_probes"]])
    df = df[df[sch["v_pack"]] > 0]

    if cfg["downsample"] > 1:
        df = df.iloc[::cfg["downsample"]].reset_index(drop=True)

    t = df[sch["time"]].values.astype(float)
    span_days = (t[-1] - t[0]) / 86400 if len(t) > 1 else np.nan
    weeks = span_days / 7 if span_days and span_days > 0 else np.nan

    odo = df[sch["odo"]].values
    total_km = float(np.nanmax(odo) - np.nanmin(odo))
    daily_km = total_km / span_days if span_days and span_days > 0 else np.nan

    soc  = df[sch["soc"]].values.astype(float)
    spd  = df[sch["speed"]].values.astype(float)
    tmin = df[sch["t_min"]].values.astype(float)
    tmax = df[sch["t_max"]].values.astype(float)
    I    = df[sch["i_pack"]].values.astype(float)
    V    = df[sch["v_pack"]].values.astype(float)

    avg_speed = float(np.nanmean(spd[spd > 0])) if (spd > 0).any() else np.nan
    avg_soc   = float(np.nanmean(soc))

    status = df[sch["status"]].values
    mlen   = cfg["min_segment_len"]
    drive_segs  = segments(status == 3, min_len=mlen)
    charge_segs = segments(status == 1, min_len=mlen)
    idle_segs   = segments(status == 4, min_len=mlen)

    n = len(df)
    drive_pct  = 100 * sum(e - s + 1 for s, e in drive_segs)  / n if n else np.nan
    charge_pct = 100 * sum(e - s + 1 for s, e in charge_segs) / n if n else np.nan
    idle_pct   = 100 * sum(e - s + 1 for s, e in idle_segs)   / n if n else np.nan

    chg_soc_delta, chg_peak_I, chg_dur_min, chg_temp_rise = [], [], [], []
    fast_ct = 0
    FAST_I = cfg["fast_charge_threshold_A"]

    for s, e in charge_segs:
        soc_seg = soc[s:e + 1]
        if len(soc_seg) < 2:
            continue
        delta = soc_seg[-1] - soc_seg[0]
        if delta <= 0:
            continue
        I_seg = I[s:e + 1]
        T_seg = tmax[s:e + 1]
        peak_I = float(np.nanmax(np.abs(I_seg)))
        chg_soc_delta.append(delta)
        chg_peak_I.append(peak_I)
        chg_dur_min.append((e - s + 1) * 10 / 60.0)
        chg_temp_rise.append(float(np.nanmax(T_seg) - np.nanmin(T_seg)))
        if peak_I > FAST_I:
            fast_ct += 1

    avg_chg_delta     = float(np.mean(chg_soc_delta))  if chg_soc_delta else np.nan
    avg_chg_peak_I    = float(np.mean(chg_peak_I))     if chg_peak_I    else np.nan
    avg_chg_dur_min   = float(np.mean(chg_dur_min))    if chg_dur_min   else np.nan
    avg_chg_temp_rise = float(np.mean(chg_temp_rise))  if chg_temp_rise else np.nan
    fast_frac         = fast_ct / max(1, len(chg_soc_delta))

    dt_h = 10 / 3600.0
    power_kW = V * I / 1000.0
    energy_kWh = float(np.nansum(np.abs(power_kW)) * dt_h)

    return {
        "vin":               os.path.basename(path).replace(".csv", ""),
        "rows_kept":         n,
        "span_days":         span_days,
        "total_km":          total_km,
        "daily_km":          daily_km,
        "n_drive":           len(drive_segs),
        "trips_per_day":     len(drive_segs) / span_days if span_days else np.nan,
        "avg_speed":         avg_speed,
        "drive_pct":         drive_pct,
        "charge_pct":        charge_pct,
        "idle_pct":          idle_pct,
        "n_charge":          len(chg_soc_delta),
        "charges_per_week":  len(chg_soc_delta) / weeks if weeks else np.nan,
        "avg_chg_delta_soc": avg_chg_delta,
        "avg_chg_peak_I":    avg_chg_peak_I,
        "avg_chg_dur_min":   avg_chg_dur_min,
        "avg_chg_temp_rise": avg_chg_temp_rise,
        "avg_soc":           avg_soc,
        "fast_frac":         fast_frac,
        "energy_kWh":        energy_kWh,
    }


# ------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------
def plot_heatmap(df, outdir):
    feat_cols = [c for c in df.columns if c not in ("vin", "rows_kept")]
    M = df[feat_cols].copy()
    M_norm = (M - M.min()) / (M.max() - M.min() + 1e-12)
    M_norm.index = df["vin"].values
    M_norm["_score"] = M_norm.mean(axis=1)
    M_norm = M_norm.sort_values("_score", ascending=False).drop(columns="_score")

    fig, ax = plt.subplots(figsize=(14, max(4, 0.35 * len(M_norm))))
    sns.heatmap(M_norm, cmap="viridis",
                cbar_kws={"label": "normalized (0–1)"},
                ax=ax, yticklabels=True, xticklabels=True)
    ax.set_title("Fleet Usage Profile Heatmap (sorted by intensity)")
    ax.set_xlabel("Usage feature")
    ax.set_ylabel("VIN")
    plt.xticks(rotation=45, ha="right")
    fig.savefig(os.path.join(outdir, "fleet_usage_heatmap.png"))
    plt.close(fig)


def plot_radar(df, outdir, n_vins=6):
    feat_cols = [c for c in df.columns if c not in ("vin", "rows_kept")]
    M = df[feat_cols].copy()
    M_norm = (M - M.min()) / (M.max() - M.min() + 1e-12)
    M_norm.index = df["vin"].values

    score = M_norm.mean(axis=1).sort_values()
    idx = np.linspace(0, len(score) - 1, min(n_vins, len(score))).astype(int)
    chosen = score.index[idx]

    labels = list(M_norm.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    cmap = plt.get_cmap("tab10")
    for k, vin in enumerate(chosen):
        vals = M_norm.loc[vin].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.8, label=vin, color=cmap(k % 10))
        ax.fill(angles, vals, alpha=0.08, color=cmap(k % 10))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Representative VIN Usage Profiles (normalized)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.savefig(os.path.join(outdir, "fleet_usage_radar.png"))
    plt.close(fig)


def plot_pca(df, outdir):
    feat_cols = [c for c in df.columns if c not in ("vin", "rows_kept")]
    X = df[feat_cols].fillna(df[feat_cols].median())
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=df["daily_km"], cmap="plasma",
                    s=90, edgecolor="k", linewidth=0.5)
    for i, vin in enumerate(df["vin"]):
        ax.annotate(vin, (Z[i, 0], Z[i, 1]), fontsize=7, alpha=0.75,
                    xytext=(4, 4), textcoords="offset points")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("daily km")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Fleet Usage Space (PCA of F1–F20 features)")
    fig.savefig(os.path.join(outdir, "fleet_usage_pca.png"))
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    cfg = CONFIG
    os.makedirs(cfg["outdir"], exist_ok=True)

    files = sorted(glob.glob(os.path.join(cfg["dataset"], "vin*.csv")))
    if not files:
        raise FileNotFoundError(f"No vin*.csv in {cfg['dataset']}")
    print(f"Found {len(files)} VIN files.")

    if cfg["pick"] == "random":
        rng = np.random.default_rng(cfg["seed"])
        idx = rng.choice(len(files),
                         size=min(cfg["n_vins"] or len(files), len(files)),
                         replace=False)
        files = [files[i] for i in idx]
    else:
        if cfg["n_vins"] is not None:
            files = files[:cfg["n_vins"]]

    print(f"Processing {len(files)} VINs...\n")
    rows = []
    for i, f in enumerate(files, 1):
        try:
            feat = extract_features(f, cfg)
            rows.append(feat)
            print(f"  [{i:>3}/{len(files)}] {feat['vin']:<12} "
                  f"rows={feat['rows_kept']:>9,}  "
                  f"daily_km={feat['daily_km']:.1f}  "
                  f"chg/wk={feat['charges_per_week']:.2f}")
        except Exception as e:
            print(f"  [{i:>3}/{len(files)}] {os.path.basename(f)}  ERROR: {e}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(cfg["outdir"], "usage_profiles.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved features → {csv_path}")
    print(df.head().to_string(index=False))

    if len(df) >= 3:
        print("\nGenerating visuals...")
        plot_heatmap(df, cfg["outdir"])
        plot_radar(df, cfg["outdir"], n_vins=min(6, len(df)))
        plot_pca(df, cfg["outdir"])
        print(f"  → {cfg['outdir']}/fleet_usage_heatmap.png")
        print(f"  → {cfg['outdir']}/fleet_usage_radar.png")
        print(f"  → {cfg['outdir']}/fleet_usage_pca.png")
    else:
        print("Not enough VINs to plot (need ≥ 3).")


if __name__ == "__main__":
    main()