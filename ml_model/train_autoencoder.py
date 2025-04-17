import numpy as np
import tensorflow as tf
import os

# Load training data
X = np.load("data/processed/X_train.npy")

# Flatten time series: (samples, time, features) → (samples, time * features)
X_flat = X.reshape((X.shape[0], -1))

# Define autoencoder architecture
input_dim = X_flat.shape[1]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(input_dim)  # Output layer (same shape as input)
])

model.compile(optimizer='adam', loss='mse')

# Train the autoencoder
model.fit(X_flat, X_flat, epochs=20, batch_size=128, validation_split=0.1)

# Save the model
os.makedirs("ml_model/models", exist_ok=True)
model.save("ml_model/models/autoencoder.h5")

print("Autoencoder training complete. Model saved to autoencoder.h5")
