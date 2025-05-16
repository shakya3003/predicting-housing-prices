import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def calculate_fwb(x,w,b):
    f_wb = np.dot(x,w) + b
    return f_wb
def calculate_loss(x,w,y,b,m):
    total_loss = 0
    for i in range(m):
        f_wb_i = calculate_fwb(x[i],w,b)
        err = (f_wb_i - y[i])**2
        total_loss += err
    total_loss /= m
    return total_loss
def derivative(x,y,w,b,m):
    df_dw = np.zeros_like(w)
    df_db = 0
    for i in range(m):
        f_wb_i = (calculate_fwb(x[i],w,b))
        error = f_wb_i - y[i]
        df_dw += error*x[i]
        df_db += error
    df_dw /= m
    df_db /= m
    return df_dw, df_db
loss_history = []
def gradient_descent(x,y,w,b,learning_rate,iterations):
    m = x.shape[0]
    for i in range(iterations):
        df_dw, df_db = derivative(x,y,w,b,m)
        w = w - learning_rate * df_dw
        b = b - learning_rate * df_db
    if i % 500 == 0:
        loss = calculate_loss(x, y, w, b, m)
        loss_history.append(loss)
        print(f"Iteration {i}: Loss = {loss:.2f}")

    return w, b, loss_history



df = pd.read_csv('housing.csv')
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)

df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].mean())


#print(df.head())
#print("\n")
df_small = df.sample(n=5000, random_state=42)
x = df_small.drop(['median_house_value',"ocean_proximity"], axis=1)
y = df_small["median_house_value"]
#print(x.head())
#print("\n")
#print(y.head())
x_train = x.to_numpy(dtype=np.float64)
y_train = y.to_numpy(dtype=np.float64)
mean_x = np.mean(x_train)
std_x = np.std(x_train)
mean_y = np.mean(y_train)
std_y = np.std(y_train)
x_scaled =  (x_train - mean_x) / std_x
y_scaled = (y_train - mean_y) / std_y
m,n = x_train.shape
b = 0
#print("\n")
#print(m,n)
w=np.zeros(n)
learning_rate = 0.0001
iterations = 10000
w_final, b_final, losses = gradient_descent(x_scaled, y_scaled, w, b, learning_rate, iterations)

print(f"final weights: {w_final},final bias values: {b_final} ")
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
