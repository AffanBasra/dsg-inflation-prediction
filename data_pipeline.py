"""
DSG Data Pipeline — Inflation Forecasting in Pakistan
=====================================================
Transforms raw annual macroeconomic data into a large-scale, normalized,
sharded dataset for Distributed Subgradient (DSG) optimization.

Built with Polars (data manipulation) + SciPy (cubic spline interpolation)
+ NumPy (jittering / random ops).
"""

import json
import os
from pathlib import Path
from datetime import date as dt_date

import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURABLE PARAMETERS — adjust these as needed
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Paths
    "input_csv": "Data for Inflation Forecasting in Pakistan.csv",
    "output_dir": "output",

    # Phase 2: Temporal upsampling
    "upsample_freq_months": 1,          # 1 = monthly, 3 = quarterly, 6 = semi-annual

    # Phase 3: Synthetic volume generation
    "target_total_rows": 50_000,        # Approximate final synthetic dataset size
    "jitter_alpha": 0.02,               # Noise intensity (fraction of column std)
    "random_seed": 42,                  # Reproducibility

    # Phase 4: Preprocessing & sharding
    "num_shards": 4,                    # K — number of worker node shards
    "num_lags": 3,                      # Lag features for inflation
    "train_ratio": 0.80,               # Temporal train/test split ratio
}

# ──────────────────────────────────────────────────────────────────────────────
# COLUMN NAME MAPPING
# ──────────────────────────────────────────────────────────────────────────────

COLUMN_RENAME_MAP = {
    "Year": "year",
    "Inflation": "inflation",
    "Exchange Rate": "exchange_rate",
    "GDPGrowth": "gdp_growth",
    "Unemployment": "unemployment",
    "BroadMoney": "broad_money",
    "Exports": "exports",
    "Imports": "imports",
    "Oilrents": "oil_rents",
    "remittances": "remittances",
}

# Columns that have natural lower bounds (non-negative)
BOUNDED_COLUMNS = {
    "inflation": (0.0, None),
    "unemployment": (0.0, 50.0),
    "exchange_rate": (0.0, None),
    "broad_money": (0.0, None),
    "exports": (0.0, None),
    "imports": (0.0, None),
    "oil_rents": (0.0, None),
    "remittances": (0.0, None),
}

TARGET_COLUMN = "inflation"


def create_output_dirs(output_dir: str) -> None:
    """Create output directory structure."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "shards"), exist_ok=True)
    print(f"  Output directory: {os.path.abspath(output_dir)}/")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: DATA CLEANING & HARMONIZATION
# ══════════════════════════════════════════════════════════════════════════════

def phase1_clean(input_csv: str, output_dir: str) -> pl.DataFrame:
    """
    Clean and harmonize the raw CSV:
    - Strip whitespace from column names
    - Rename to snake_case
    - Cast all to Float64
    - Sort by year, verify no nulls
    """
    print("\n" + "=" * 70)
    print("PHASE 1: Data Cleaning & Harmonization")
    print("=" * 70)

    # Read raw CSV
    df = pl.read_csv(input_csv)
    print(f"  Raw shape: {df.shape}")
    print(f"  Raw columns: {df.columns}")

    # Strip whitespace from column names
    df = df.rename({col: col.strip() for col in df.columns})

    # Apply snake_case renaming
    df = df.rename(COLUMN_RENAME_MAP)
    print(f"  Renamed columns: {df.columns}")

    # Cast all columns to Float64
    df = df.cast({col: pl.Float64 for col in df.columns})

    # Sort by year
    df = df.sort("year")

    # Verify no nulls
    null_counts = df.null_count()
    total_nulls = sum(null_counts.row(0))
    assert total_nulls == 0, f"Found {total_nulls} null values!"
    print(f"  Null check: PASSED (0 nulls)")

    # Save
    out_path = os.path.join(output_dir, "cleaned_data.csv")
    df.write_csv(out_path)
    print(f"  Saved: {out_path} — {df.shape[0]} rows × {df.shape[1]} cols")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: TEMPORAL UPSAMPLING (CUBIC SPLINE INTERPOLATION)
# ══════════════════════════════════════════════════════════════════════════════

def phase2_upsample(df: pl.DataFrame, output_dir: str) -> pl.DataFrame:
    """
    Upsample annual data to monthly using cubic spline interpolation.
    36 annual rows → 432 monthly rows (or adjusted by upsample_freq_months).
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Temporal Upsampling (Cubic Spline)")
    print("=" * 70)

    freq = CONFIG["upsample_freq_months"]
    years = df["year"].to_numpy()
    n_years = len(years)

    # Generate monthly time points between first and last year
    # E.g., 1986.0, 1986.0833, 1986.1667, ..., 2021.0
    year_start, year_end = years[0], years[-1]
    months_per_year = 12 // freq
    total_months = int((year_end - year_start) * months_per_year) + 1
    t_monthly = np.linspace(year_start, year_end, total_months)

    print(f"  Annual points: {n_years}")
    print(f"  Frequency: every {freq} month(s)")
    print(f"  Upsampled points: {total_months}")

    # Feature columns (everything except year)
    feature_cols = [c for c in df.columns if c != "year"]
    interpolated_data = {"year": t_monthly}

    for col in feature_cols:
        values = df[col].to_numpy()
        cs = CubicSpline(years, values)
        interp_values = cs(t_monthly)

        # Clip bounded columns
        if col in BOUNDED_COLUMNS:
            lo, hi = BOUNDED_COLUMNS[col]
            if lo is not None:
                interp_values = np.maximum(interp_values, lo)
            if hi is not None:
                interp_values = np.minimum(interp_values, hi)

        interpolated_data[col] = interp_values

    df_monthly = pl.DataFrame(interpolated_data)

    # Add a proper date column for human-readable intermediate CSVs
    dates = pl.Series("date", [
        dt_date(int(year_start) + (m // 12), (m % 12) + 1, 1)
        for m in range(total_months)
    ])
    df_monthly = df_monthly.with_columns(dates)

    # Save
    out_path = os.path.join(output_dir, "monthly_interpolated.csv")
    df_monthly.write_csv(out_path)
    print(f"  Saved: {out_path} — {df_monthly.shape[0]} rows × {df_monthly.shape[1]} cols")

    # Print sample
    print(f"\n  Sample (first 5 rows):")
    print(df_monthly.head(5))

    return df_monthly


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: SYNTHETIC VOLUME GENERATION (GAUSSIAN JITTERING)
# ══════════════════════════════════════════════════════════════════════════════

def phase3_jitter(df_monthly: pl.DataFrame, output_dir: str) -> pl.DataFrame:
    """
    Generate synthetic samples via Gaussian jittering.
    Each monthly row is replicated N times with small random noise added.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: Synthetic Volume Generation (Jittering)")
    print("=" * 70)

    target_rows = CONFIG["target_total_rows"]
    alpha = CONFIG["jitter_alpha"]
    seed = CONFIG["random_seed"]
    rng = np.random.default_rng(seed)

    n_monthly = df_monthly.shape[0]
    copies_per_row = max(1, target_rows // n_monthly)
    actual_total = n_monthly * copies_per_row

    print(f"  Monthly rows: {n_monthly}")
    print(f"  Copies per row: {copies_per_row}")
    print(f"  Target total: {target_rows}, Actual total: {actual_total}")
    print(f"  Jitter alpha: {alpha}")

    feature_cols = [c for c in df_monthly.columns if c not in ("year", "date")]

    # Compute column standard deviations for noise scaling
    col_stds = {}
    for col in feature_cols:
        col_stds[col] = df_monthly[col].std()

    # Build the synthetic dataset
    # We'll keep a source_month_idx for traceability and lag computation
    years_arr = df_monthly["year"].to_numpy()
    feature_arrays = {col: df_monthly[col].to_numpy() for col in feature_cols}

    synthetic_data = {
        "source_month_idx": np.repeat(np.arange(n_monthly), copies_per_row),
        "year": np.repeat(years_arr, copies_per_row),
    }

    for col in feature_cols:
        base = np.repeat(feature_arrays[col], copies_per_row)
        noise = rng.normal(0, alpha * col_stds[col], size=len(base))
        jittered = base + noise

        # Clip bounded columns
        if col in BOUNDED_COLUMNS:
            lo, hi = BOUNDED_COLUMNS[col]
            if lo is not None:
                jittered = np.maximum(jittered, lo)
            if hi is not None:
                # Use dynamic upper bound: max observed × 1.2
                effective_hi = hi if hi else float(feature_arrays[col].max()) * 1.2
                jittered = np.minimum(jittered, effective_hi)

        synthetic_data[col] = jittered

    df_synthetic = pl.DataFrame(synthetic_data)

    # Reconstruct the date column from source_month_idx (dates are not jittered)
    date_map = pl.DataFrame({
        "source_month_idx": np.arange(n_monthly),
        "date": df_monthly["date"],
    })
    df_synthetic = df_synthetic.join(date_map, on="source_month_idx", how="left")

    # Shuffle rows
    df_synthetic = df_synthetic.sample(fraction=1.0, shuffle=True, seed=seed)

    # Save
    out_path = os.path.join(output_dir, "synthetic_dataset.csv")
    df_synthetic.write_csv(out_path)
    print(f"  Saved: {out_path} — {df_synthetic.shape[0]} rows × {df_synthetic.shape[1]} cols")

    # Distribution check
    print(f"\n  Distribution check (mean ± std):")
    print(f"  {'Column':<20} {'Original':>20} {'Synthetic':>20}")
    print(f"  {'─' * 20} {'─' * 20} {'─' * 20}")
    for col in feature_cols:
        orig_mean = df_monthly[col].mean()
        orig_std = df_monthly[col].std()
        syn_mean = df_synthetic[col].mean()
        syn_std = df_synthetic[col].std()
        print(f"  {col:<20} {orig_mean:>8.2f} ± {orig_std:<8.2f} {syn_mean:>8.2f} ± {syn_std:<8.2f}")

    return df_synthetic


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: PREPROCESSING & SHARDING
# ══════════════════════════════════════════════════════════════════════════════

def phase4_preprocess_and_shard(df_synthetic: pl.DataFrame, df_monthly: pl.DataFrame, output_dir: str) -> None:
    """
    1. Compute lag features (based on source_month_idx ordering)
    2. Temporal train/test split
    3. Z-score normalize using training set statistics
    4. Shard training data into K partitions
    """
    print("\n" + "=" * 70)
    print("PHASE 4: Preprocessing & Sharding")
    print("=" * 70)

    num_lags = CONFIG["num_lags"]
    num_shards = CONFIG["num_shards"]
    train_ratio = CONFIG["train_ratio"]
    seed = CONFIG["random_seed"]

    # ── Step 1: Compute lag features ──────────────────────────────────────
    # Lags are derived from the ORIGINAL monthly series (pre-jitter) so all
    # synthetic copies of month M share identical, consistent lag values.
    print(f"\n  Step 1: Computing {num_lags} lag feature(s)...")

    # Use the original (un-jittered) monthly inflation for lag computation
    inflation_by_month = df_monthly["inflation"].to_numpy()
    n_months = len(inflation_by_month)

    # Create lag arrays for each source_month_idx
    lag_map = {}
    for lag in range(1, num_lags + 1):
        lag_values = np.full(n_months, np.nan)
        lag_values[lag:] = inflation_by_month[:-lag]
        lag_map[f"inflation_lag{lag}"] = lag_values

    # Build a mapping dataframe: source_month_idx → lag values
    lag_df_data = {"source_month_idx": np.arange(n_months)}
    lag_df_data.update(lag_map)
    lag_df = pl.DataFrame(lag_df_data)

    # Join lags onto synthetic data
    df_synthetic = df_synthetic.join(lag_df, on="source_month_idx", how="left")

    # CRITICAL FIX: np.nan creates float NaN, NOT Polars null.
    # Polars' drop_nulls() only drops null, not NaN. Convert first.
    lag_col_names = [f"inflation_lag{lag}" for lag in range(1, num_lags + 1)]
    df_synthetic = df_synthetic.with_columns([
        pl.col(c).fill_nan(None) for c in lag_col_names
    ])

    # Drop rows where lags are null (first `num_lags` months)
    rows_before = df_synthetic.shape[0]
    df_synthetic = df_synthetic.drop_nulls(subset=lag_col_names)
    rows_after = df_synthetic.shape[0]
    print(f"  Dropped {rows_before - rows_after} rows with null lags")
    print(f"  Dataset: {rows_after} rows × {df_synthetic.shape[1]} cols")

    # ── Step 2: Temporal train/test split ─────────────────────────────────
    print(f"\n  Step 2: Temporal train/test split ({train_ratio:.0%} / {1 - train_ratio:.0%})...")

    unique_months = sorted(df_synthetic["source_month_idx"].unique().to_list())
    cutoff_idx = int(len(unique_months) * train_ratio)
    train_months = set(unique_months[:cutoff_idx])
    test_months = set(unique_months[cutoff_idx:])

    df_train = df_synthetic.filter(pl.col("source_month_idx").is_in(train_months))
    df_test = df_synthetic.filter(pl.col("source_month_idx").is_in(test_months))

    print(f"  Train months: {len(train_months)}, Test months: {len(test_months)}")
    print(f"  Train rows: {df_train.shape[0]}, Test rows: {df_test.shape[0]}")

    # ── Step 3: Z-score normalization ─────────────────────────────────────
    print(f"\n  Step 3: Z-score normalization...")

    # Feature columns = everything except year, source_month_idx, target
    meta_cols = ["year", "source_month_idx", "date"]
    feature_cols = [c for c in df_synthetic.columns if c not in meta_cols and c != TARGET_COLUMN]

    # Compute mean/std from TRAINING set only
    scaler_params = {}
    for col in feature_cols:
        mu = df_train[col].mean()
        sigma = df_train[col].std()
        # Guard against zero std
        if sigma == 0 or sigma is None:
            sigma = 1.0
        scaler_params[col] = {"mean": float(mu), "std": float(sigma)}

    # Also normalize the target
    target_mu = df_train[TARGET_COLUMN].mean()
    target_sigma = df_train[TARGET_COLUMN].std()
    if target_sigma == 0 or target_sigma is None:
        target_sigma = 1.0
    scaler_params[TARGET_COLUMN] = {"mean": float(target_mu), "std": float(target_sigma)}

    # Apply normalization
    def normalize(df: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        for col in feature_cols + [TARGET_COLUMN]:
            mu = scaler_params[col]["mean"]
            sigma = scaler_params[col]["std"]
            exprs.append(((pl.col(col) - mu) / sigma).alias(col))
        return df.with_columns(exprs)

    df_train_norm = normalize(df_train)
    df_test_norm = normalize(df_test)

    # Save scaler params
    scaler_path = os.path.join(output_dir, "scaler_params.json")
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"  Scaler params saved: {scaler_path}")

    # ── Step 4: Split features / target and save ──────────────────────────
    print(f"\n  Step 4: Saving train/test splits...")

    X_train = df_train_norm.select(feature_cols)
    y_train = df_train_norm.select(TARGET_COLUMN)
    X_test = df_test_norm.select(feature_cols)
    y_test = df_test_norm.select(TARGET_COLUMN)

    X_train.write_csv(os.path.join(output_dir, "X_train.csv"))
    y_train.write_csv(os.path.join(output_dir, "y_train.csv"))
    X_test.write_csv(os.path.join(output_dir, "X_test.csv"))
    y_test.write_csv(os.path.join(output_dir, "y_test.csv"))

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # ── Step 5: Shard training data ───────────────────────────────────────
    print(f"\n  Step 5: Sharding into {num_shards} partitions...")

    # Shuffle training data before sharding
    X_train_shuffled = X_train.sample(fraction=1.0, shuffle=True, seed=seed)
    y_train_shuffled = y_train.sample(fraction=1.0, shuffle=True, seed=seed)

    # Combine for sharding, then split
    train_combined = pl.concat([X_train_shuffled, y_train_shuffled], how="horizontal")

    shard_size = len(train_combined) // num_shards
    for k in range(num_shards):
        start = k * shard_size
        end = start + shard_size if k < num_shards - 1 else len(train_combined)
        shard = train_combined.slice(start, end - start)
        shard_path = os.path.join(output_dir, "shards", f"shard_{k}.csv")
        shard.write_csv(shard_path)
        print(f"  Shard {k}: {shard.shape[0]} rows → {shard_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — Summary")
    print("=" * 70)
    print(f"  Features:       {feature_cols}")
    print(f"  Target:         {TARGET_COLUMN}")
    print(f"  Train samples:  {X_train.shape[0]}")
    print(f"  Test samples:   {X_test.shape[0]}")
    print(f"  Shards:         {num_shards} × ~{shard_size} rows")
    print(f"  Output dir:     {os.path.abspath(output_dir)}/")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  DSG Data Pipeline — Inflation Forecasting in Pakistan             ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Config: K={CONFIG['num_shards']} shards, "
          f"~{CONFIG['target_total_rows']:,} synthetic rows, "
          f"α={CONFIG['jitter_alpha']}          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Setup
    base_dir = Path(__file__).parent
    input_csv = str(base_dir / CONFIG["input_csv"])
    output_dir = str(base_dir / CONFIG["output_dir"])
    create_output_dirs(output_dir)

    # Execute pipeline
    df_clean = phase1_clean(input_csv, output_dir)
    df_monthly = phase2_upsample(df_clean, output_dir)
    df_synthetic = phase3_jitter(df_monthly, output_dir)
    phase4_preprocess_and_shard(df_synthetic, df_monthly, output_dir)


if __name__ == "__main__":
    main()
