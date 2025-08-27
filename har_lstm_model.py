import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Simulate dummy training data (replace with real features)
# X = (samples, timesteps, features)
X = np.random.rand(500, 30, 33)   # 500 samples, 30 timesteps, 33 features
y = np.random.randint(0, 3, 500)  # 3 activity classes

# One-hot encode labels
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 33)),
    Dropout(0.5),
    LSTM(32),
    Dropout(0.5),
    Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train quickly (just for testing)
model.fit(X, y, epochs=5, batch_size=16, verbose=1)

# Save model
model.save("har_lstm_model.h5")
print("✅ HAR model saved as har_lstm_model.h5")
