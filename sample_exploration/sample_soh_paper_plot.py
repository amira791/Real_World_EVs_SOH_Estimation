# ============================================================
# plot_paper_ground_truth.py
# Plot only the Ground Truth SOH labels from the paper.
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Update this path to where your file is located
INPUT_CSV = r"\\FS19\HomeDir\129963\Desktop\DR2\3 Contributions\Real_World_EVs_SOH_Estimation\sample_exploration\code_sample_data.csv"
OUT_PNG   = r"\\FS19\HomeDir\129963\Desktop\DR2\3 Contributions\Real_World_EVs_SOH_Estimation\sample_exploration\paper_ground_truth_soh.png"

def plot_ground_truth(csv_path, out_png):
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Clean column names just in case
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    
    # Extract the Ground Truth
    x = df.index
    y_true = df['SOH_labels'] * 100  # Convert 0.897 to 89.7%

    # Create a rolling median to see the trend through the noise
    rolling_true = y_true.rolling(50, center=True, min_periods=10).median()

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Scatter plot for individual labels
    ax.scatter(x, y_true, s=25, color='steelblue', alpha=0.7, label='Ground Truth SOH Label')
    
    # Line plot for the rolling median
    ax.plot(x, rolling_true, color='darkblue', linewidth=2.5, label='Rolling Median (50 samples)')
    
    # Add a reference line for our calculated vin93 SOH (89.84%)
    ax.axhline(89.84, color='crimson', linestyle='--', linewidth=1.5, 
               label='Your vin93 Calculated SOH (89.84%)')

    ax.set_ylabel('SOH (%)', fontsize=12)
    ax.set_xlabel('Sample Index (0 to 567)', fontsize=12)
    ax.set_title('Paper Ground Truth SOH Labels', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_ylim(60, 110)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"Saved plot to {out_png}")
    
    # Print some statistics
    print("\n" + "="*40)
    print("GROUND TRUTH STATISTICS")
    print("="*40)
    print(f"Total samples: {len(df)}")
    print(f"SOH Labels range: {y_true.min():.2f}% - {y_true.max():.2f}%")
    print(f"Mean:   {y_true.mean():.2f}%")
    print(f"Median: {y_true.median():.2f}%")
    
    # Where does your vin93 fit?
    vin93_soh = 89.84
    percentile = (y_true < vin93_soh).mean() * 100
    print(f"\nYour vin93 label (89.84%) is at the {percentile:.1f}th percentile of the paper's fleet.")

if __name__ == "__main__":
    plot_ground_truth(INPUT_CSV, OUT_PNG)