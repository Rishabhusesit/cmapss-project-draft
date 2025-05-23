import numpy as np
import tensorflow as tf
import os
from ml_model.model_utils import scale_input

# Paths to saved models
lstm_path = "ml_model/models/lstm_rul.h5"
ae_path = "ml_model/models/autoencoder.h5"

# Quick checks to make sure model files exist
if not os.path.exists(lstm_path):
    raise FileNotFoundError(f"Model not found: {lstm_path}")

if not os.path.exists(ae_path):
    raise FileNotFoundError(f"Model not found: {ae_path}")

# Load models
lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
ae_model = tf.keras.models.load_model(ae_path)

# Predict remaining useful life
def predict_rul(sensor_data):
    data = np.array(sensor_data).reshape(1, 50, -1)
    prediction = lstm_model.predict(data)
    return float(prediction[0][0])

# Detect anomalies using autoencoder
def detect_anomaly(sensor_data):
    sensor_data = np.array(sensor_data)
    scaled = scale_input(sensor_data)
    reconstructed = ae_model.predict(scaled)
    mse = np.mean((scaled - reconstructed) ** 2)
    return mse

# import numpy as np
# import tensorflow as tf
# from ml_model.model_utils import scale_input
# import os

# # Load models once at startup
# lstm_path = "ml_model/models/lstm_converted.keras"
# ae_path = "ml_model/models/autoencoder_converted.keras"

# assert os.path.exists(lstm_path), f"❌ File not found: {lstm_path}"


# lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
# ae_model = tf.keras.models.load_model(ae_path)

# def predict_rul(sensor_data):
#     """
#     Predict Remaining Useful Life from a 50-step time-series.
#     Input: List of 50 rows, each with 21 features
#     Output: Predicted RUL (float)
#     """
#     data = np.array(sensor_data).reshape(1, 50, -1)
#     prediction = lstm_model.predict(data)
#     return float(prediction[0][0])

# def detect_anomaly(sensor_data):
#     """
#     Detect anomaly score using Autoencoder reconstruction error.
#     Input: 2D sensor data (50x21)
#     Output: MSE anomaly score
#     """
#     sensor_data = np.array(sensor_data)
#     scaled = scale_input(sensor_data)
#     reconstructed = ae_model.predict(scaled)
#     mse = np.mean((scaled - reconstructed) ** 2)
#     return mse
