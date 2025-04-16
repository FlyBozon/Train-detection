import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()
filename = askopenfilename(filetypes=[("CSV files", "*.csv")])

df = pd.read_csv(filename)

timestamps = df['timestamp'].values
distances = df['distance_m'].values
T = np.mean(np.diff(timestamps))

measurements = distances.reshape(-1, 1)

A = np.array([
    [1, T],
    [0, 1]
])
C = np.array([
    [1, 0]
])
Q = np.array([
    [0.01, 0],
    [0, 0.1]
])
R = np.array([[0.1]])
G = np.eye(2)

def kalman_1d(measurements, x_init, P_init, A, C, Q, R, G):
    n = len(measurements)
    x_est = np.zeros((n, 2))
    P = np.zeros((n, 2, 2))
    x_est[0] = x_init
    P[0] = P_init
    for k in range(1, n):
        x_pred = A @ x_est[k - 1]
        P_pred = A @ P[k - 1] @ A.T + G @ Q @ G.T
        y_pred = C @ x_pred
        e = measurements[k] - y_pred
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)
        x_est[k] = x_pred + (K @ e).flatten()
        P[k] = (np.eye(2) - K @ C) @ P_pred
    return x_est, P

x_init = np.array([measurements[0, 0], 0])
P_init = np.eye(2)

x_est, P_est = kalman_1d(measurements, x_init, P_init, A, C, Q, R, G)

plt.figure(figsize=(10, 5))
plt.plot(timestamps - timestamps[0], distances, 'x-', label='Pomiary')
plt.plot(timestamps - timestamps[0], x_est[:, 0], 'g-', label='Pozycja (Kalman)')
plt.title('Estymacja pozycji filtrem Kalmana')
plt.xlabel('Czas [s]')
plt.ylabel('Odległość [m]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
