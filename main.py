import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_fwb(x, w, b):
    return np.dot(x, w) + b

def calculate_loss(x, w, y, b):
    m = x.shape[0]
    f_wb = calculate_fwb(x, w, b)
    return np.mean((f_wb - y) ** 2)

def derivative(x, y, w, b):
    m = x.shape[0]
    f_wb = calculate_fwb(x, w, b)
    error = f_wb - y
    df_dw = (np.dot(error, x)) / m
    df_db = np.sum(error) / m
    return df_dw, df_db

def gradient_descent(x, y, w, b, learning_rate, iterations):
    loss_history = []
    for i in range(iterations):
        df_dw, df_db = derivative(x, y, w, b)
        w = w - learning_rate * df_dw
        b = b - learning_rate * df_db
        if i % 500 == 0:
            loss = calculate_loss(x, w, y, b)
            loss_history.append(loss)
            print(f"Iteration {i}: Loss = {loss:.2f}")
    return w, b, loss_history

# Data loading and preprocessing
df = pd.read_csv('housing.csv')
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].mean())
df_small = df.sample(n=5000, random_state=42)

# Optionally encode 'ocean_proximity' if needed
x = df_small.drop(['median_house_value', "ocean_proximity"], axis=1)
y = df_small["median_house_value"]

x_train = x.to_numpy(dtype=np.float64)
y_train = y.to_numpy(dtype=np.float64)

# Feature normalization (per feature)
mean_x = np.mean(x_train, axis=0)
std_x = np.std(x_train, axis=0)
x_scaled = (x_train - mean_x) / std_x

# Target normalization (optional, for stability)
mean_y = np.mean(y_train)
std_y = np.std(y_train)
y_scaled = (y_train - mean_y) / std_y

m, n = x_train.shape
w = np.zeros(n)
b = 0
learning_rate = 0.01
iterations = 20000

w_final, b_final, losses = gradient_descent(x_scaled, y_scaled, w, b, learning_rate, iterations)

print(f"final weights: {w_final}, final bias value: {b_final}")

y_pred_scaled = np.dot(x_scaled, w_final) + b_final
y_pred = y_pred_scaled * std_y + mean_y

plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_pred, alpha=0.3, color='blue')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Predicted vs Actual Values")
plt.grid(True)
plt.show()