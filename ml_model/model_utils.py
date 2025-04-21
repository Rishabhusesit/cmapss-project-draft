from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def scale_input(sensor_sequence):
    return scaler.transform(sensor_sequence)

def fit_scaler(dataframe, features):
    dataframe[features] = scaler.fit_transform(dataframe[features])
    return dataframe
