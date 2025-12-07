import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



def read_chd_dataset() -> pd.DataFrame:
    """Read the CHD dataset from a CSV file."""
    files = [
        './results_car_hacking_dataset/Fuzzy_dataset_processed.csv',
        './results_car_hacking_dataset/gear_dataset_processed.csv',
        './results_car_hacking_dataset/DoS_dataset_processed.csv',
        './results_car_hacking_dataset/RPM_dataset_processed.csv'
    ]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.drop(columns=['ID_raw'], inplace=True)
    return df

def normalize_features(X: pd.DataFrame) -> pd.DataFrame:
    """Normalize features to the range [0, 1]."""
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled

def load_malicious_data(return_x_y: bool = True) -> pd.DataFrame:
    df = read_chd_dataset()
    malicious_data = df[df['flag'] == 1]

    if return_x_y:
        X_malicious = malicious_data.drop(columns=['flag'])
        y_malicious = malicious_data['flag']
        return X_malicious, y_malicious
    return malicious_data

def load_benign_data(return_x_y: bool = True) -> pd.DataFrame:
    df = read_chd_dataset()
    benign_data = df[df['flag'] == 0]

    if return_x_y:
        X_benign = benign_data.drop(columns=['flag'])
        y_benign = benign_data['flag']
        return X_benign, y_benign
    return benign_data    


def preprocess_chd_dataset(df: pd.DataFrame):
    """Preprocess the CHD dataset."""
    df = df.dropna()  # Remove missing values
    X = df.drop(columns=['flag'])
    y = df['flag'] # 1 -> attack, 0 -> normal

    return X, y


def split_dataset(X: pd.DataFrame, y: pd.Series, train_size: float = 0.6, val_size: float = 0.2, test_size: float = 0.2, random_state: int = 42):
    """Split the dataset into training, validation, and testing sets.
    """

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class balance
    )
    val_ratio = val_size / (train_size + val_size)  # = 0.2/0.8 = 0.25
    x_train, x_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_temp  # Maintain class balance
    )
    
    return x_train, x_val, y_train, y_val, X_test, y_test


def load_and_preprocess_chd() -> dict:
    """Load and preprocess the CHD dataset.
    """
    df = read_chd_dataset()
    X, y = preprocess_chd_dataset(df)
    x_train, x_val, y_train, y_val, X_test, y_test = split_dataset(X, y)
    
    scaler = MinMaxScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
    x_val = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    data = {}
    data['X_train'] = x_train
    data['Y_train'] = y_train
    data['X_test'] = X_test
    data['Y_test'] = y_test
    data['X_val'] = x_val
    data['Y_val'] = y_val
    data['scaler'] = scaler  # Save scaler for later use if needed


    return data