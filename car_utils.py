import pandas as pd
import numpy as np
from collections import Counter
from math import log2

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

CAR_LOCATION = './car_hacking_dataset/'

def entropy(values):
    if len(values) == 0:
        return 0
    counter = Counter(values)
    total = len(values)
    return -sum((count/total) * log2(count/total) for count in counter.values())


def hex_to_bytes(hex_string):
    """Converte '0A FF 23 ...' ou '0AFF23...' em lista de ints"""
    hex_string = hex_string.replace(" ", "").strip()
    return [int(hex_string[i:i+2], 16) for i in range(0, len(hex_string), 2)]


# ---------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------

def extract_can_features(df, window=50):
    """
    df precisa ter:
        - Timestamp (float ou int)
        - ID (hex string ou int)
        - Data (payload hex string)

    retorna um dataframe tabular com features
    """
    
    # Normaliza ID → int
    df["ID"] = df["ID"].astype(str).apply(lambda x: int(x, 16))

    # Bytes (0 a 7)
    byte_cols = [f"byte_{i}" for i in range(8)]
    df[byte_cols] = df["Data"].apply(lambda x: pd.Series(hex_to_bytes(x)))

    # Timestamp delta (tempo entre frames)
    df["timestamp_delta"] = df["Timestamp"].diff().fillna(0)

    # Contagem acumulada por ID
    df["id_count"] = df.groupby("ID").cumcount()

    # Frequência instantânea por ID (janela deslizante)
    df["id_freq"] = (
        df.groupby("ID")["Timestamp"].rolling(window)
          .apply(lambda x: len(x) / (x.max() - x.min() + 1e-9), raw=False)
          .reset_index(level=0, drop=True)
    ).fillna(0)

    # Entropia do payload (por pacote)
    df["payload_entropy"] = df[byte_cols].apply(lambda row: entropy(row.values), axis=1)

    # Entropia por ID (janela deslizante)
    df["entropy_by_id"] = (
        df.groupby("ID")["payload_entropy"]
          .rolling(window)
          .apply(lambda x: entropy(x), raw=False)
          .reset_index(level=0, drop=True)
    ).fillna(0)

    # Estatísticas temporais por janela
    for col in byte_cols:
        df[f"{col}_mean"] = df[col].rolling(window).mean()
        df[f"{col}_std"] = df[col].rolling(window).std()
        df[f"{col}_min"] = df[col].rolling(window).min()
        df[f"{col}_max"] = df[col].rolling(window).max()

    df = df.fillna(0)

    return df


df = pd.read_csv(f"{CAR_LOCATION}DoS_attack.csv", CAR_LOCATION)

df.columns = ["Timestamp", "ID", "Data"]  # depende do dataset

features = extract_can_features(df, window=50)

print(features.head())