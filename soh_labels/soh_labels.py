import pandas as pd
import numpy as np

INPUT_PATH = r"\\FS19\HomeDir\129963\Desktop\DR2\4 Coding\sample_data\vin93.csv"

df = pd.read_csv(INPUT_PATH)
df.columns = [c.lower() for c in df.columns]

print("Total rows:", len(df))

# Check current distribution
print("\n--- Current (totalcurrent) ---")
print(df["totalcurrent"].describe())

# Check speed distribution
print("\n--- Speed ---")
print(df["speed"].describe())
print(f"Rows with speed <= 2.0: {(df['speed'] <= 2.0).sum()}")
print(f"Rows with speed is NaN: {df['speed'].isna().sum()}")

# Check SOC
print("\n--- SOC ---")
print(df["soc"].describe())

# Determine polarity: when SOC is increasing, what is the current sign?
df["soc_diff"] = df["soc"].diff()
charging_rows = df[df["soc_diff"] > 0]
print("\n--- Current during SOC increase (charging) ---")
print(charging_rows["totalcurrent"].describe())
print(f"Median charging current: {charging_rows['totalcurrent'].median():.2f}")

# Apply the same logic as the script
if len(charging_rows) > 0 and charging_rows["totalcurrent"].median() < 0:
    df["i_ch"] = -df["totalcurrent"]
    print("\n[INFO] Inverted current: charging current is now positive.")
else:
    df["i_ch"] = df["totalcurrent"]
    print("\n[INFO] Did NOT invert current: charging current remains as is.")

# Now count how many rows pass each filter condition
cond1 = df["i_ch"] >= 1.0
cond2 = df["i_ch"] <= 250.0
cond3 = df["speed"] <= 2.0

print(f"\n--- Filter counts ---")
print(f"Rows with i_ch >= 1.0: {cond1.sum()}")
print(f"Rows with i_ch <= 250.0: {cond2.sum()}")
print(f"Rows with speed <= 2.0: {cond3.sum()}")
print(f"Rows with ALL three conditions: {(cond1 & cond2 & cond3).sum()}")

# Show a sample of rows that pass all conditions
passed = df[cond1 & cond2 & cond3]
if len(passed) > 0:
    print(f"\nSample of first 5 rows passing all conditions:")
    print(passed[["terminaltime", "i_ch", "speed", "soc"]].head())
else:
    print("\n[WARNING] No rows passed all conditions. Here is a sample of i_ch values:")
    print(df["i_ch"].describe())