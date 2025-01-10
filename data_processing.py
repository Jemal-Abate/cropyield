import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    numeric_columns = ['GWETPROF', 'GWETTOP', 'GWETROOT', 'CLOUD_AMT', 'TS', 'PS', 'RH2M', 'QV2M', 'PRECTOTCORR',
                       'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'WS2M']
    data = data[numeric_columns].dropna()

    # Remove outliers using Z-scores
    z_scores = np.abs((data - data.mean()) / data.std())
    threshold = 3
    data = data[(z_scores <= threshold).all(axis=1)]

    # Scale the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled, scaler, data

# Function to prepare data for LSTM
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, :])
        y.append(data[i + time_steps, :])
    return np.array(X), np.array(y)

# Handle outliers
def fill_outliers_with_median(df, threshold=3):
    for column in df.columns:
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        outliers = abs(z_scores) > threshold
        df.loc[outliers, column] = df[column].median()
    return df