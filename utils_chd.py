import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



def read_chd_dataset(file_path: str = './results_car_hacking_dataset/DoS_dataset_processed.csv') -> pd.DataFrame:
    """Read the CHD dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def normalize_features(X: pd.DataFrame) -> pd.DataFrame:
    """Normalize features to the range [0, 1]."""
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled


def preprocess_chd_dataset(df: pd.DataFrame):
    """Preprocess the CHD dataset."""
    # Example preprocessing steps
    df = df.dropna()  # Remove missing values
    X = df.drop(columns=['flag'])
    y = df['flag'] # 1 -> attack, 0 -> normal

    return X, y


def split_dataset(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Split the dataset into training and testing sets."""
    X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return x_train, x_val, y_train, y_val, X_test, y_test