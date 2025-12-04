import pandas as pd
import numpy as np
from typing import Union
from pathlib import Path


def entropy(values: Union[np.ndarray, list]) -> float:
    """Compute Shannon entropy of a 1-D sequence of discrete values.

    Returns 0 for empty input or when all values are identical.
    """
    if values is None:
        return 0.0
    arr = np.asarray(values)
    if arr.size == 0:
        return 0.0
    # drop nans when calculating entropy
    arr = arr[~pd.isna(arr)]
    if arr.size == 0:
        return 0.0
    vals, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def _safe_hex_to_int(x):
    """Convert x (hex string or int) to integer."""
    if pd.isna(x):
        return np.nan
    try:
        s = str(x).strip()
        # allow optional 0x prefix
        return int(s, 16)
    except Exception:
        try:
            # maybe it's already decimal
            return int(float(s))
        except Exception:
            return np.nan


def preprocess_can(file: str, keep_raw: bool = False, remove_error_flags: bool = True, verbose: bool = False) -> pd.DataFrame:
    """Preprocess a CSV CAN dump into a dataframe with payload features.

    Expected input columns (no header): timestamp, ID, DLC, b0..b7, flag

    Parameters
    - file: path to CSV file
    - keep_raw: if True, keep original byte columns as strings; otherwise they are converted to ints
    - remove_error_flags: if True, remove lines with flag='R' (CAN error frames)
    - verbose: if True, print statistics about dropped rows

    Returns a pandas DataFrame with added features: payload_sum, payload_mean, payload_std, entropy, iat
    """
    df = pd.read_csv(
        file,
        header=None,
        names=["timestamp", "ID", "DLC"] + [f"b{i}" for i in range(8)] + ["flag"],
        dtype=str,
        na_values=["", "NA", "nan", None],
        keep_default_na=True,
    )


    # converter timestamp para float (se possível)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    # converter ID e bytes de hex para int de forma robusta
    # ID
    df["ID_raw"] = df["ID"]
    df["ID"] = df["ID"].apply(_safe_hex_to_int)

    byte_cols = [f"b{i}" for i in range(8)]
    if not keep_raw:
        # applymap over subset for robust conversion
        df[byte_cols] = df[byte_cols].applymap(_safe_hex_to_int)

    # calcular estatísticas do payload (desconsiderando NaNs)
    df["payload_sum"] = df[byte_cols].sum(axis=1, skipna=True)
    df["payload_mean"] = df[byte_cols].mean(axis=1, skipna=True)
    # use population std (ddof=0) to be deterministic
    df["payload_std"] = df[byte_cols].std(axis=1, ddof=0, skipna=True)
    df["entropy"] = df[byte_cols].apply(lambda r: entropy(r.values), axis=1)

    # tempo entre mensagens do mesmo ID (IAT) — preencher NaN com 0 quando não há anterior
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["iat"] = df.groupby("ID")["timestamp"].diff().fillna(0)
    df["flag"] = df["flag"].map(lambda x: 0 if x == "R" else 1) # Flag : T or R, T represents injected message while R represents normal message

    return df


def process_can_directory(input_dir: str, output_dir: str = "results", output_format: str = "csv") -> None:
    """Process all CAN CSV files in input_dir and save preprocessed data to output_dir.

    Parameters
    - input_dir: directory containing CSV files to process
    - output_dir: directory where results will be saved
    - output_format: 'csv' or 'parquet' (default: 'csv')
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    # output_path.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        try:
            df = preprocess_can(str(csv_file))
            output_name = csv_file.stem
            
            if output_format == "parquet":
                output_file = output_path / f"{output_name}_processed.parquet"
                df.to_parquet(output_file, index=False)
            else:  # csv
                output_file = output_path / f"{output_name}_processed.csv"
                df.to_csv(output_file, index=False)
            
            print(f"  → Saved to {output_file}")
        except Exception as e:
            print(f"  ✗ Error processing {csv_file.name}: {e}")


if __name__ == "__main__":
    process_can_directory("./car_hacking_dataset", output_dir="./results_car_hacking_dataset", output_format="csv")
    # Example: process a directory of CAN files
    # process_can_directory("./data", output_dir="./results", output_format="csv")
    pass
