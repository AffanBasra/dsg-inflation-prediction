"""
DSG Data Pipeline — Inflation Forecasting in Pakistan
Transforms raw annual macroeconomic data into a large-scale, normalized,
sharded dataset for Distributed Subgradient (DSG) optimization.
"""

import json
import os
from pathlib import Path
from datetime import date as dt_date

import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline

CONFIG = {
    "input_csv": "Data for Inflation Forecasting in Pakistan.csv",
    "output_dir": "output",
    "upsample_freq_months": 1,
    "target_total_rows": 2_000_000,        # <-- INCREASED TO 2 MILLION FOR THREAD SCALING
    "jitter_alpha": 0.02,
    "random_seed": 42,
    "num_shards": 4,
    "num_lags": 3,
    "train_ratio": 0.80,
}

COLUMN_RENAME_MAP = {
    "Year": "year", "Inflation": "inflation", "Exchange Rate": "exchange_rate",
    "GDPGrowth": "gdp_growth", "Unemployment": "unemployment", "BroadMoney": "broad_money",
    "Exports": "exports", "Imports": "imports", "Oilrents": "oil_rents", "remittances": "remittances",
}

BOUNDED_COLUMNS = {
    "inflation": (0.0, None), "unemployment": (0.0, 50.0), "exchange_rate": (0.0, None),
    "broad_money": (0.0, None), "exports": (0.0, None), "imports": (0.0, None),
    "oil_rents": (0.0, None), "remittances": (0.0, None),
}

TARGET_COLUMN = "inflation"

def create_output_dirs(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "shards"), exist_ok=True)

def phase1_clean(input_csv: str, output_dir: str) -> pl.DataFrame:
    df = pl.read_csv(input_csv)
    df = df.rename({col: col.strip() for col in df.columns})
    df = df.rename(COLUMN_RENAME_MAP)
    df = df.cast({col: pl.Float64 for col in df.columns})
    df = df.sort("year")
    df.write_csv(os.path.join(output_dir, "cleaned_data.csv"))
    return df

def phase2_upsample(df: pl.DataFrame, output_dir: str) -> pl.DataFrame:
    freq = CONFIG["upsample_freq_months"]
    years = df["year"].to_numpy()
    year_start, year_end = years[0], years[-1]
    total_months = int((year_end - year_start) * (12 // freq)) + 1
    t_monthly = np.linspace(year_start, year_end, total_months)

    feature_cols = [c for c in df.columns if c != "year"]
    interpolated_data = {"year": t_monthly}

    for col in feature_cols:
        cs = CubicSpline(years, df[col].to_numpy())
        interp_values = cs(t_monthly)
        if col in BOUNDED_COLUMNS:
            lo, hi = BOUNDED_COLUMNS[col]
            if lo is not None: interp_values = np.maximum(interp_values, lo)
            if hi is not None: interp_values = np.minimum(interp_values, hi)
        interpolated_data[col] = interp_values

    df_monthly = pl.DataFrame(interpolated_data)
    dates = pl.Series("date", [dt_date(int(year_start) + (m // 12), (m % 12) + 1, 1) for m in range(total_months)])
    df_monthly = df_monthly.with_columns(dates)
    df_monthly.write_csv(os.path.join(output_dir, "monthly_interpolated.csv"))
    return df_monthly

def phase3_jitter(df_monthly: pl.DataFrame, output_dir: str) -> pl.DataFrame:
    target_rows = CONFIG["target_total_rows"]
    alpha = CONFIG["jitter_alpha"]
    rng = np.random.default_rng(CONFIG["random_seed"])
    n_monthly = df_monthly.shape[0]
    copies_per_row = max(1, target_rows // n_monthly)

    feature_cols = [c for c in df_monthly.columns if c not in ("year", "date")]
    col_stds = {col: df_monthly[col].std() for col in feature_cols}
    feature_arrays = {col: df_monthly[col].to_numpy() for col in feature_cols}

    synthetic_data = {
        "source_month_idx": np.repeat(np.arange(n_monthly), copies_per_row),
        "year": np.repeat(df_monthly["year"].to_numpy(), copies_per_row),
    }

    for col in feature_cols:
        base = np.repeat(feature_arrays[col], copies_per_row)
        noise = rng.normal(0, alpha * col_stds[col], size=len(base))
        jittered = base + noise
        if col in BOUNDED_COLUMNS:
            lo, hi = BOUNDED_COLUMNS[col]
            if lo is not None: jittered = np.maximum(jittered, lo)
            if hi is not None: jittered = np.minimum(jittered, hi if hi else float(feature_arrays[col].max()) * 1.2)
        synthetic_data[col] = jittered

    df_synthetic = pl.DataFrame(synthetic_data)
    date_map = pl.DataFrame({"source_month_idx": np.arange(n_monthly), "date": df_monthly["date"]})
    df_synthetic = df_synthetic.join(date_map, on="source_month_idx", how="left")
    df_synthetic = df_synthetic.sample(fraction=1.0, shuffle=True, seed=CONFIG["random_seed"])
    df_synthetic.write_csv(os.path.join(output_dir, "synthetic_dataset.csv"))
    return df_synthetic

def phase4_preprocess_and_shard(df_synthetic: pl.DataFrame, df_monthly: pl.DataFrame, output_dir: str) -> None:
    num_lags = CONFIG["num_lags"]
    inflation_by_month = df_monthly["inflation"].to_numpy()
    lag_map = {}
    for lag in range(1, num_lags + 1):
        lag_values = np.full(len(inflation_by_month), np.nan)
        lag_values[lag:] = inflation_by_month[:-lag]
        lag_map[f"inflation_lag{lag}"] = lag_values

    lag_df = pl.DataFrame({"source_month_idx": np.arange(len(inflation_by_month)), **lag_map})
    df_synthetic = df_synthetic.join(lag_df, on="source_month_idx", how="left")
    
    lag_col_names = [f"inflation_lag{lag}" for lag in range(1, num_lags + 1)]
    df_synthetic = df_synthetic.with_columns([pl.col(c).fill_nan(None) for c in lag_col_names])
    df_synthetic = df_synthetic.drop_nulls(subset=lag_col_names)

    unique_months = sorted(df_synthetic["source_month_idx"].unique().to_list())
    cutoff_idx = int(len(unique_months) * CONFIG["train_ratio"])
    train_months = set(unique_months[:cutoff_idx])
    
    df_train = df_synthetic.filter(pl.col("source_month_idx").is_in(train_months))
    df_test = df_synthetic.filter(pl.col("source_month_idx").is_in(set(unique_months[cutoff_idx:])))

    meta_cols = ["year", "source_month_idx", "date"]
    feature_cols = [c for c in df_synthetic.columns if c not in meta_cols and c != TARGET_COLUMN]

    scaler_params = {}
    for col in feature_cols + [TARGET_COLUMN]:
        sigma = df_train[col].std()
        scaler_params[col] = {"mean": float(df_train[col].mean()), "std": float(sigma if sigma and sigma > 0 else 1.0)}

    def normalize(df):
        exprs = [((pl.col(col) - scaler_params[col]["mean"]) / scaler_params[col]["std"]).alias(col) for col in feature_cols + [TARGET_COLUMN]]
        return df.with_columns(exprs)

    df_train_norm, df_test_norm = normalize(df_train), normalize(df_test)

    with open(os.path.join(output_dir, "scaler_params.json"), "w") as f:
        json.dump(scaler_params, f, indent=2)

    X_train, y_train = df_train_norm.select(feature_cols), df_train_norm.select(TARGET_COLUMN)
    X_test, y_test = df_test_norm.select(feature_cols), df_test_norm.select(TARGET_COLUMN)

    X_train.write_csv(os.path.join(output_dir, "X_train.csv"))
    y_train.write_csv(os.path.join(output_dir, "y_train.csv"))
    X_test.write_csv(os.path.join(output_dir, "X_test.csv"))
    y_test.write_csv(os.path.join(output_dir, "y_test.csv"))

    train_combined = pl.concat([X_train.sample(fraction=1.0, shuffle=True, seed=CONFIG["random_seed"]), 
                                y_train.sample(fraction=1.0, shuffle=True, seed=CONFIG["random_seed"])], how="horizontal")

    num_shards = CONFIG["num_shards"]
    shard_size = len(train_combined) // num_shards
    for k in range(num_shards):
        start = k * shard_size
        end = start + shard_size if k < num_shards - 1 else len(train_combined)
        train_combined.slice(start, end - start).write_csv(os.path.join(output_dir, "shards", f"shard_{k}.csv"))

def main():
    base_dir = Path(__file__).parent
    output_dir = str(base_dir / CONFIG["output_dir"])
    create_output_dirs(output_dir)
    df_clean = phase1_clean(str(base_dir / CONFIG["input_csv"]), output_dir)
    df_monthly = phase2_upsample(df_clean, output_dir)
    df_synthetic = phase3_jitter(df_monthly, output_dir)
    phase4_preprocess_and_shard(df_synthetic, df_monthly, output_dir)

if __name__ == "__main__":
    main()