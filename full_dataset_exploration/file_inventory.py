
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import pandas as pd
import csv

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = Path(
    r"C:\Users\admin\Desktop\DR2\11 All Datasets\07 300-EV Real-World BMS Dataset Liu et al. (2025)\CompleteDataset_New\CSV_Data"
)

OUTPUT_DIR = DATA_DIR / "EDA_Inventory"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_VEHICLES = 300
CHUNK_SIZE = 100_000

TIMESTAMP_KEYWORDS = (
    "timestamp", "datetime", "date", "time",
    "时间", "日期", "时刻"
)

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def find_timestamp_column(columns):
    """Identify a likely timestamp column from its name."""
    for col in columns:
        name = str(col).strip().lower()
        if any(keyword in name for keyword in TIMESTAMP_KEYWORDS):
            return col
    return None


def read_header(path, encoding):
    """Read the raw CSV header to detect duplicate column names."""
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        return next(reader, [])


def scan_csv(path):
    """
    Scan one CSV in chunks.
    Returns file statistics and per-column statistics.
    """
    last_error = None

    # Try common encodings, including one used by some Chinese datasets.
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            raw_header = read_header(path, encoding)

            if not raw_header:
                return {
                    "status": "EMPTY_FILE",
                    "error": "",
                    "encoding": encoding,
                    "raw_header": [],
                }, {}

            # Pandas may rename duplicate columns automatically.
            duplicate_headers = sorted({
                col for col in raw_header if raw_header.count(col) > 1
            })

            rows = 0
            missing_counts = defaultdict(int)
            dtype_values = defaultdict(set)
            timestamp_min = None
            timestamp_max = None

            reader = pd.read_csv(
                path,
                encoding=encoding,
                chunksize=CHUNK_SIZE,
                on_bad_lines="error",
                low_memory=True,
            )

            columns = None
            timestamp_col = None

            for chunk in reader:
                if columns is None:
                    columns = list(chunk.columns)
                    timestamp_col = find_timestamp_column(columns)

                rows += len(chunk)

                for col in chunk.columns:
                    missing_counts[col] += int(chunk[col].isna().sum())
                    dtype_values[col].add(str(chunk[col].dtype))

                if timestamp_col is not None:
                    parsed = pd.to_datetime(
                        chunk[timestamp_col],
                        errors="coerce",
                        utc=True,
                    ).dropna()

                    if not parsed.empty:
                        current_min = parsed.min()
                        current_max = parsed.max()

                        if timestamp_min is None or current_min < timestamp_min:
                            timestamp_min = current_min

                        if timestamp_max is None or current_max > timestamp_max:
                            timestamp_max = current_max

            columns = columns or []

            file_stats = {
                "status": "OK",
                "error": "",
                "encoding": encoding,
                "rows": rows,
                "columns_count": len(columns),
                "columns": columns,
                "schema": " | ".join(map(str, columns)),
                "schema_signature": tuple(map(str, columns)),
                "duplicate_header_names": " | ".join(duplicate_headers),
                "timestamp_column": timestamp_col,
                "first_timestamp_utc": (
                    timestamp_min.isoformat() if timestamp_min is not None else ""
                ),
                "last_timestamp_utc": (
                    timestamp_max.isoformat() if timestamp_max is not None else ""
                ),
                "missing_counts": dict(missing_counts),
                "dtypes": {
                    col: " / ".join(sorted(dtype_values[col]))
                    for col in columns
                },
            }

            return file_stats, {}

        except (UnicodeDecodeError, pd.errors.ParserError,
                pd.errors.EmptyDataError, OSError, StopIteration) as exc:
            last_error = exc

    return {
        "status": "ERROR",
        "error": str(last_error),
        "encoding": "",
        "raw_header": [],
    }, {}


# ============================================================
# 3. FIND CSV FILES
# ============================================================

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found:\n{DATA_DIR}")

csv_files = sorted(
    DATA_DIR.rglob("*.csv"),
    key=lambda p: p.name.lower()
)

# Exclude any reports from previous runs.
csv_files = [
    p for p in csv_files
    if OUTPUT_DIR not in p.parents
]

print("=" * 70)
print("EV DATASET — FILE INVENTORY")
print("=" * 70)
print(f"Dataset folder : {DATA_DIR}")
print(f"CSV files found: {len(csv_files)}")
print(f"Expected       : {EXPECTED_VEHICLES}")
print()

# ============================================================
# 4. SCAN EACH FILE
# ============================================================

inventory = []
column_stats = defaultdict(
    lambda: {
        "files": 0,
        "rows_with_column": 0,
        "missing_values": 0,
        "dtypes": set(),
    }
)
errors = []

for i, path in enumerate(csv_files, start=1):
    print(f"[{i}/{len(csv_files)}] Scanning {path.name} ...")

    size_bytes = path.stat().st_size
    result, _ = scan_csv(path)

    if result["status"] == "ERROR":
        errors.append({
            "file": str(path),
            "error": result["error"],
        })

        inventory.append({
            "file": path.name,
            "relative_path": str(path.relative_to(DATA_DIR)),
            "size_bytes": size_bytes,
            "size_MB": round(size_bytes / 1e6, 3),
            "status": "ERROR",
            "rows": None,
            "columns_count": None,
            "schema": "",
            "duplicate_header_names": "",
            "timestamp_column": "",
            "first_timestamp_utc": "",
            "last_timestamp_utc": "",
            "error": result["error"],
        })
        continue

    columns = result.get("columns", [])
    rows = result.get("rows", 0)
    missing = result.get("missing_counts", {})
    dtypes = result.get("dtypes", {})

    # Aggregate column-level statistics across files.
    for col in columns:
        stats = column_stats[str(col)]
        stats["files"] += 1
        stats["rows_with_column"] += rows
        stats["missing_values"] += missing.get(col, 0)
        stats["dtypes"].update(dtypes.get(col, "").split(" / "))

    inventory.append({
        "file": path.name,
        "relative_path": str(path.relative_to(DATA_DIR)),
        "size_bytes": size_bytes,
        "size_MB": round(size_bytes / 1e6, 3),
        "status": result["status"],
        "rows": rows,
        "columns_count": result.get("columns_count", 0),
        "schema": result.get("schema", ""),
        "duplicate_header_names": result.get("duplicate_header_names", ""),
        "timestamp_column": result.get("timestamp_column", ""),
        "first_timestamp_utc": result.get("first_timestamp_utc", ""),
        "last_timestamp_utc": result.get("last_timestamp_utc", ""),
        "error": result.get("error", ""),
    })


# ============================================================
# 5. CHECK EXPECTED VEHICLE FILES
# ============================================================

# Match filenames such as vin1.csv, vin2.csv, ..., vin300.csv.
# The lookup is case-insensitive.
files_by_name = defaultdict(list)

for path in csv_files:
    files_by_name[path.name.lower()].append(path)

expected_names = {
    f"vin{i}.csv".lower() for i in range(1, EXPECTED_VEHICLES + 1)
}

present_names = set(files_by_name)
missing_names = sorted(
    expected_names - present_names,
    key=lambda name: int(name[3:-4]),
)

unexpected_names = sorted(present_names - expected_names)

duplicate_filenames = {
    name: paths
    for name, paths in files_by_name.items()
    if len(paths) > 1
}

# ============================================================
# 6. BUILD COLUMN SUMMARY
# ============================================================

column_rows = []

for col, stats in sorted(column_stats.items()):
    total_rows = stats["rows_with_column"]
    missing = stats["missing_values"]

    column_rows.append({
        "column": col,
        "files_containing_column": stats["files"],
        "total_rows_in_files_with_column": total_rows,
        "missing_values": missing,
        "missing_percent": (
            round(100 * missing / total_rows, 4)
            if total_rows else None
        ),
        "observed_dtypes": " | ".join(sorted(stats["dtypes"])),
    })

# ============================================================
# 7. SAVE REPORTS
# ============================================================

inventory_df = pd.DataFrame(inventory)
columns_df = pd.DataFrame(column_rows)
errors_df = pd.DataFrame(errors)

inventory_df.to_csv(
    OUTPUT_DIR / "vehicle_inventory.csv",
    index=False,
    encoding="utf-8-sig",
)

columns_df.to_csv(
    OUTPUT_DIR / "column_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

errors_df.to_csv(
    OUTPUT_DIR / "inventory_errors.csv",
    index=False,
    encoding="utf-8-sig",
)

# Schema consistency report: count files sharing each column layout.
schema_rows = []

for schema, group in inventory_df[
    inventory_df["status"] == "OK"
].groupby("schema", dropna=False):
    schema_rows.append({
        "number_of_files": len(group),
        "example_file": group.iloc[0]["file"],
        "columns_count": group.iloc[0]["columns_count"],
        "schema": schema,
    })

pd.DataFrame(schema_rows).to_csv(
    OUTPUT_DIR / "schema_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# Save missing expected vehicle filenames.
pd.DataFrame({
    "missing_expected_file": missing_names
}).to_csv(
    OUTPUT_DIR / "missing_vehicle_files.csv",
    index=False,
    encoding="utf-8-sig",
)

# ============================================================
# 8. PRINT SUMMARY
# ============================================================

successful = inventory_df[inventory_df["status"] == "OK"]

print("\n" + "=" * 70)
print("INVENTORY SUMMARY")
print("=" * 70)

print(f"CSV files found          : {len(csv_files)}")
print(f"Successfully scanned     : {len(successful)}")
print(f"Files with errors        : {len(errors)}")
print(f"Missing expected VIN CSVs: {len(missing_names)}")
print(f"Unexpected CSV filenames : {len(unexpected_names)}")
print(f"Repeated filenames       : {len(duplicate_filenames)}")

if not successful.empty:
    print(f"Total rows               : {int(successful['rows'].sum()):,}")
    print(f"Total CSV size           : "
          f"{inventory_df['size_bytes'].sum() / 1e9:.3f} GB")
    print(f"Minimum rows per file    : {int(successful['rows'].min()):,}")
    print(f"Maximum rows per file    : {int(successful['rows'].max()):,}")
    print(f"Distinct schemas         : {successful['schema'].nunique()}")

if missing_names:
    print("\nMissing expected files:")
    print(", ".join(missing_names[:30]))

if unexpected_names:
    print("\nUnexpected CSV filenames:")
    print(", ".join(unexpected_names[:30]))

print(f"\nReports saved in:\n{OUTPUT_DIR}")
print("\nInventory completed.")
