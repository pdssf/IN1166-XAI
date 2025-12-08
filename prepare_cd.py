import pandas as pd
import numpy as np
from pathlib import Path




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
        df["flag"] = df["flag"].map(lambda x: 0 if x == "R" else 1) # Flag : T or R, T represents injected message while R represents normal message

    return df


def process_can_directory(input_dir: str, output_dir: str = "results") -> None:
    """Process all CAN CSV files in input_dir and save preprocessed data to output_dir.

    Parameters
    - input_dir: directory containing CSV files to process
    - output_dir: directory where results will be saved
    - output_format: 'csv' or 'parquet' (default: 'csv')
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        try:
            df = preprocess_can(str(csv_file))
            output_name = csv_file.stem
            output_file = output_path / f"{output_name}_processed.csv"
            df.to_csv(output_file, index=False)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    process_can_directory("./car_hacking_dataset", output_dir="./results_car_hacking_dataset", output_format="csv")
    pass
