# Practice Lab - Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor


# =====================================================
# Training dataset
# Features:
# [Body Weight (kg), Squat 1RM (kg), Bench Press 1RM (kg)]
# Target:
# [Deadlift 1RM (kg)]
# =====================================================

X_train = np.array([
    [65, 90, 60],
    [70, 110, 70],
    [73, 115, 75],
    [75, 125, 80],
    [80, 140, 90],
    [82, 145, 95],
    [85, 150, 100],
    [88, 160, 105],
    [90, 170, 110],
    [93, 172, 112],
    [95, 180, 115],
    [98, 185, 120],
    [100, 200, 130],
    [105, 210, 140],
    [110, 220, 145]
], dtype=np.float64)

y_train = np.array([
    120, 140, 145, 160, 180,
    185, 195, 205, 215, 220,
    230, 235, 245, 265, 275
], dtype=np.float64)

feature_names = [
    "Body Weight",
    "Squat",
    "Bench Press"
]

print(f"Dataset loaded successfully.")
print(f"Feature matrix shape: {X_train.shape}")
print(f"Target vector shape: {y_train.shape}")


# =====================================================
# Feature Scaling
# Standardize the input features before training
# =====================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)


# =====================================================
# Model Training
# =====================================================

model = SGDRegressor(random_state=42)
model.fit(X_scaled, y_train)


# =====================================================
# Prediction for a new athlete
# =====================================================

new_sample = np.array([[84, 144, 98]], dtype=np.float64)

# Scale the new sample using the same scaler
new_sample_scaled = scaler.transform(new_sample)

# Predict the deadlift 1RM
prediction = model.predict(new_sample_scaled)

print("\nPrediction")
print("--------------------------------")
print("Body Weight : 84 kg")
print("Squat       : 144 kg")
print("Bench Press : 98 kg")
print(f"Estimated Deadlift: {prediction[0]:.2f} kg")


# =====================================================
# Model Evaluation
# Real values vs Predicted values
# =====================================================

train_predictions = model.predict(X_scaled)

plt.figure(figsize=(6, 6))

plt.scatter(y_train, train_predictions)

# Perfect prediction reference line
plt.plot(
    [y_train.min(), y_train.max()],
    [y_train.min(), y_train.max()],
    "r--"
)

plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Deadlift (kg)")
plt.ylabel("Predicted Deadlift (kg)")
plt.grid(alpha=0.3)

plt.show()