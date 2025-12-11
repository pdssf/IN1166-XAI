import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import numpy as np
from utility_funcs import compute_performance_stats

def read_chd_dataset() -> pd.DataFrame:
    """Read the CHD dataset from a CSV file."""
    files = [
        './results_car_hacking_dataset/Fuzzy_dataset_processed.csv',
        './results_car_hacking_dataset/gear_dataset_processed.csv',
        './results_car_hacking_dataset/DoS_dataset_processed.csv',
        './results_car_hacking_dataset/RPM_dataset_processed.csv'
    ]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sample(frac=0.1, random_state=10).reset_index(drop=True)
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
        X_malicious = normalize_features(malicious_data.drop(columns=['flag']))
        y_malicious = malicious_data['flag']
        return X_malicious, y_malicious
    return malicious_data

def load_benign_data(return_x_y: bool = True) -> pd.DataFrame:
    df = read_chd_dataset()
    benign_data = df[df['flag'] == 0]

    if return_x_y:
        X_benign = normalize_features(benign_data.drop(columns=['flag']))
        y_benign = benign_data['flag']
        return X_benign, y_benign
    return benign_data    


def preprocess_chd_dataset(df: pd.DataFrame):
    """Preprocess the CHD dataset."""
    df = df.dropna()  # Remove missing values
    X = df.drop(columns=['flag'])
    y = df['flag'] # 1 -> attack, 0 -> normal

    return X, y


def split_dataset(X: pd.DataFrame, y: pd.Series,  test_size: float = 0.2, random_state: int = 42):
    """Split the dataset into training, and testing sets.
    """

    X, X_test, y, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class balance
    )
    
    return X, X_test, y, y_test

def apply_scaling(scaler: MinMaxScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the given scaler to the dataset."""
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    return X_scaled

def load_and_preprocess_chd() -> dict:
    """Load and preprocess the CHD dataset.
    """
    df = read_chd_dataset()
    X, y = preprocess_chd_dataset(df)
    
    # Split: 80% train, 20% test (for compatibility with current code)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # Fit scaler ONLY on training data to prevent data leakage
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = apply_scaling(scaler, X_test)

    data = {}
    data['X_train'] = X_train.to_numpy()
    data['Y_train'] = y_train.to_numpy()
    data['X_test'] = X_test.to_numpy()
    data['Y_test'] = y_test.to_numpy()
    data['scaler'] = scaler
    data['train_normal_locs'] = np.where(data['Y_train'] == 0)[0]
    data['feature_names'] = list(X_train.columns)

    return data

def AE_anomaly_detection(autoencoder, train_data, test_data, y_test=None):    
    """Main function that applies autoencoder to the train and test sets and calculates performance of detector.
    
    Parameters:
    - autoencoder: trained autoencoder model
    - train_data: training data (used to compute threshold)
    - test_data: test data (used to detect anomalies)
    - y_test: ground truth labels for test data (0=normal, 1=attack) for performance evaluation
    
    Returns:
    - performance_summary: dict with performance metrics (or None if y_test not provided)
    - pred_anomaly_locs: indices of predicted anomalies
    - error_threshold: threshold used for anomaly detection
    """
    
    # To find anomalies, first compute Reconstruction Error on training data
    encoded_train = autoencoder.encoder.predict(train_data)
    decoded_train = autoencoder.decoder.predict(encoded_train)
    mse_train = np.mean(np.abs(train_data - decoded_train), axis=1)
    
    # Calculate error threshold based on training RE distribution (95th percentile)
    error_threshold = np.percentile(mse_train, 95)
    
    # Now compute the RE across the test points
    encoded_test = autoencoder.encoder.predict(test_data)
    decoded_test = autoencoder.decoder.predict(encoded_test)
    mse_test = np.mean(np.abs(test_data - decoded_test), axis=1)
    
    # Get indices of all test points above the threshold (predicted anomalies)
    pred_anomaly_locs = np.where(mse_test > error_threshold)[0]
    y_pred = np.zeros(len(test_data),)
    y_pred[pred_anomaly_locs] = 1

    assert len(y_pred) == len(y_test), "Length of predictions and ground truth do not match."
    # Calculate performance metrics if ground truth labels are provided
    if y_test is not None:
        y_pred = np.zeros(len(test_data))
        y_pred[pred_anomaly_locs] = 1
        performance_summary = compute_performance_stats(y_test, y_pred)
    else:
        performance_summary = None
    
    return performance_summary, pred_anomaly_locs, error_threshold


def get_overall_metrics(y_true, y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  acc = (tp+tn)/(tp+tn+fp+fn)
  tpr = tp/(tp+fn)
  fpr = fp/(fp+tn)
  precision = tp/(tp+fp)
  f1 = (2*tpr*precision)/(tpr+precision)
  return {'acc':acc,'tpr':tpr,'fpr':fpr,'precision':precision,'f1-score':f1}