import pandas as pd

INPUT_PATH = r"\\FS19\HomeDir\129963\Desktop\DR2\4 Coding\sample_data\vin93.csv"

df = pd.read_csv(INPUT_PATH)
df.columns = [c.lower() for c in df.columns]

print("Columns:", df.columns.tolist())
print("\nTotal rows:", len(df))

# Current statistics
print("\n--- Current ---")
print(df["totalcurrent"].describe())
print(f"Median current: {df['totalcurrent'].median():.2f}")

# Speed statistics
print("\n--- Speed ---")
print(df["speed"].describe())
print(f"Rows with speed <= 2.0: {(df['speed'] <= 2.0).sum()}")
print(f"Rows with speed is NaN: {df['speed'].isna().sum()}")

# SOC statistics
print("\n--- SOC ---")
print(df["soc"].describe())

# Check polarity: when SOC is increasing, what is the current?
df["soc_diff"] = df["soc"].diff()
charging = df[df["soc_diff"] > 0]
print("\n--- Current during SOC increase (charging) ---")
print(charging["totalcurrent"].describe())
print(f"Median charging current: {charging['totalcurrent'].median():.2f}")

# Check speed during SOC increase
print("\n--- Speed during SOC increase ---")
print(charging["speed"].describe())
print(f"Median speed during charging: {charging['speed'].median():.2f}")

# Check voltage units
print("\n--- Max cell voltage ---")
print(df["maxvoltagebattery"].describe())
print(f"Median voltage: {df['maxvoltagebattery'].median():.2f}")