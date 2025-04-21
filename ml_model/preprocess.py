import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_and_preprocess_data(file_path, sequence_length=50, max_rul=125):
    # Load and name columns
    df = pd.read_csv(file_path, sep=" ", header=None)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = ['unit', 'time'] + [f'op_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]

    # Calculate RUL
    df['RUL'] = df.groupby('unit')['time'].transform('max') - df['time']
    df['RUL'] = df['RUL'].clip(upper=max_rul)

    # Scale sensor features
    features = [col for col in df.columns if 'sensor_' in col]
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    # Build sequences
    X, y = [], []
    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit]
        for i in range(len(unit_df) - sequence_length):
            seq = unit_df.iloc[i:i+sequence_length][features].values
            label = unit_df.iloc[i+sequence_length]['RUL']
            X.append(seq)
            y.append(label)

    X, y = np.array(X), np.array(y)

    # Save to data/processed/
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_train.npy", X)
    np.save("data/processed/y_train.npy", y)

    print("Preprocessing complete")
    print("X_train shape:", X.shape)
    print("y_train shape:", y.shape)

if __name__ == "__main__":
    load_and_preprocess_data("data/raw/train_FD001.txt")
