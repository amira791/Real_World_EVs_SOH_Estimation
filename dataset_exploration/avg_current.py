import pandas as pd
import numpy as np
import os
import glob

# --- CONFIGURATION ---
DATA_DIR = r"\\FS19\HomeDir\129963\Desktop\DR2\4 Coding\sample_data" # Update this path
USECOLS = ["terminaltime", "totalcurrent", "speed", "maxvoltagebattery"]

def analyze_current_distribution():
    # Find all CSV files
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not all_files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    print(f"Found {len(all_files)} files. Analyzing charging current...\n")
    
    all_charging_currents = []
    
    for f in all_files:
        try:
            df = pd.read_csv(f, usecols=USECOLS)
            
            # Clean data
            df = df.dropna(subset=["totalcurrent", "speed"])
            
            # Convert current sign (assume negative is charging, so invert)
            df["i_ch"] = -df["totalcurrent"]
            
            # Filter ONLY when parked and current is positive (flowing in)
            parked_charging = df[(df["speed"] <= 1.0) & (df["i_ch"] > 0)]
            
            if not parked_charging.empty:
                all_charging_currents.extend(parked_charging["i_ch"].tolist())
                
        except Exception as e:
            print(f"Error reading {os.path.basename(f)}: {e}")

    if not all_charging_currents:
        print("No parked charging data found.")
        return

    # Convert to numpy array for easy stats
    currents = np.array(all_charging_currents)
    
    # 1. PRINT STATISTICS
    print("="*40)
    print(" CHARGING CURRENT STATISTICS (Parked) ")
    print("="*40)
    print(f"Total data points (rows): {len(currents):,}")
    print(f"Min Current:   {np.min(currents):.2f} A")
    print(f"Max Current:   {np.max(currents):.2f} A")
    print(f"Mean Current:  {np.mean(currents):.2f} A")
    print(f"Median:        {np.median(currents):.2f} A")
    print(f"95th Percentile: {np.percentile(currents, 95):.2f} A")
    print(f"99th Percentile: {np.percentile(currents, 99):.2f} A")
    print("-" * 40)
    
    # 2. BINNING (How much data falls in each range?)
    bins = [0, 1, 2, 5, 10, 20, 50, 100, np.inf]
    labels = ["0-1 A", "1-2 A", "2-5 A", "5-10 A", "10-20 A", "20-50 A", "50-100 A", ">100 A"]
    
    counts, _ = np.histogram(currents, bins=bins)
    
    print(" DISTRIBUTION ACROSS CURRENT RANGES ")
    print(f"{'Range':<12} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 40)
    for i, count in enumerate(counts):
        pct = (count / len(currents)) * 100
        print(f"{labels[i]:<12} | {count:<10,} | {pct:.2f}%")
    
    print("="*40)
    print("\nINTERPRETATION:")
    print("Look at the 0-5 A ranges. If a large percentage of your")
    print("data is in the 2-5 A range, a 5 A threshold will lose a lot of data.")
    print("If almost all data is > 10 A, the 5 A threshold is perfectly safe.")

if __name__ == "__main__":
    analyze_current_distribution()