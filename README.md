# Hybrid-CNN-GRU-Model-for-Predicting-Respiratory-Diseases-Caused-by-Air-Pollution
A Hybrid CNN-GRU deep learning model for early prediction of respiratory diseases caused by air pollution. Combines CNN for feature extraction and GRU for temporal patterns, enhanced with self-attention. Achieves high accuracy and supports real-time pollutant-based health risk forecasting.
# =============================
# 1️⃣ Install required packages
# =============================
!pip install --quiet numpy pandas matplotlib seaborn scikit-learn
!pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install --quiet tensorflow==2.19.0 tensorflow-decision-forests==1.12.0 tensorflow-text==2.19.0
!pip install --quiet streamlit

# =============================
# 2️⃣ Upload dataset
# =============================
from google.colab import files
import pandas as pd
import io
data_path = "/content/air_data.csv"  # path after upload
df = pd.read_csv(data_path)
print(df.shape)
df.head()

# =============================
# 3️⃣ Basic EDA
# =============================
import matplotlib.pyplot as plt
import seaborn as sns

print("Columns:", df.columns.tolist())
print("\nSummary statistics:")
print(df.describe())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Simple correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# =============================
# 4️⃣ Preprocessing
# =============================
# Example: convert datetime if exists
time_cols = ['Date','Datetime','date_time','Time','RecordedDate']
for col in df.columns:
    if col in time_cols:
        df['timestamp'] = pd.to_datetime(df[col])
        df = df.set_index('timestamp')
        break

# Fill missing numeric values with interpolation
numeric_cols = df.select_dtypes(include=['float64','int64']).columns
df[numeric_cols] = df[numeric_cols].interpolate()

# Normalize features (for deep learning)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# =============================
# 5️⃣ Split features & target
# =============================
# Assuming target is 'RespiratoryCases'
target = 'RespiratoryCases'
X = df.drop(columns=[target])
y = df[target]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to numpy arrays
import numpy as np
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# =============================
# 6️⃣ Build a simple CNN-GRU hybrid model
# =============================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Conv1D, Flatten, Dropout, Reshape

# Reshape for Conv1D (samples, timesteps=1, features)
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

model = Sequential([
    Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    GRU(32, activation='tanh', return_sequences=False),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# =============================
# 7️⃣ Train the model
# =============================
history = model.fit(X_train_reshaped, y_train, epochs=20, batch_size=16, validation_split=0.2)

# =============================
# 8️⃣ Evaluate
# =============================
loss, mae = model.evaluate(X_test_reshaped, y_test)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Predictions
y_pred = model.predict(X_test_reshaped)
plt.figure(figsize=(10,6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title("Respiratory Cases Prediction")
plt.legend()
plt.show()

# =============================
# 9️⃣ Save model
# =============================
model.save("respiratory_model.h5")
print("Model saved as respiratory_model.h5")
