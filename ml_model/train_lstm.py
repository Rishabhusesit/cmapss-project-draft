import numpy as np
import tensorflow as tf
import os

# Load preprocessed data
X = np.load("data/processed/X_train.npy")
y = np.load("data/processed/y_train.npy")

# Define LSTM Model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # RUL is a single value
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X, y, epochs=10, batch_size=64, validation_split=0.1)

# Save the trained model
os.makedirs("ml_model/models", exist_ok=True)
model.save("ml_model/models/lstm_rul.h5")

print("Model training complete. Saved to lstm_rul.h5")
