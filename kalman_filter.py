import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt


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

def pid_1d(measurements, Kp=1.0, Ki=0.0, Kd=0.0, dt=0.01):
    n = len(measurements)
    filtered = np.zeros(n)
    integral = 0.0
    previous_error = 0.0

    filtered[0] = measurements[0, 0]  # Initialize first value

    for k in range(1, n):
        error = measurements[k, 0] - filtered[k - 1]
        integral += error * dt
        derivative = (error - previous_error) / dt

        control = Kp * error + Ki * integral + Kd * derivative

        filtered[k] = filtered[k - 1] + control

        previous_error = error

    return filtered

def poly_approximation(timestamps, measurements, degree=3):
    """
    Fit a polynomial of given degree to the measurements.
    """
    t = timestamps - timestamps[0]  # start from 0 for better fitting
    y = measurements.flatten()

    coeffs = np.polyfit(t, y, degree)
    poly_func = np.poly1d(coeffs)

    approx = poly_func(t)

    return approx, poly_func


def spline_interpolation(timestamps, measurements, kind='cubic'):
    """
    Interpolate measurements using spline (or linear) interpolation.
    """
    t = timestamps - timestamps[0]
    y = measurements.flatten()

    interp_func = interp1d(t, y, kind=kind, fill_value="extrapolate")
    interpolated = interp_func(t)

    return interpolated, interp_func

def local_poly_approximation(timestamps, measurements, window_size=1.0, degree=3):
    """
    Locally approximate measurements using polynomial fitting over sliding windows.
    window_size is in seconds.
    """
    t = timestamps - timestamps[0]
    y = measurements.flatten()

    approx = np.zeros_like(y)

    for i in range(len(t)):
        # find window
        mask = np.abs(t - t[i]) <= window_size / 2
        if np.sum(mask) > degree:
            coeffs = np.polyfit(t[mask], y[mask], degree)
            poly_func = np.poly1d(coeffs)
            approx[i] = poly_func(t[i])
        else:
            approx[i] = y[i]  # fallback to original if too few points

    return approx

def lowpass_filter(data, cutoff_freq, fs, order=3):
    """
    Apply a low-pass Butterworth filter.
    cutoff_freq: cutoff frequency (Hz)
    fs: sampling frequency (Hz)
    """
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, data)
    return filtered

def highpass_filter(data, cutoff_freq, fs, order=3):
    """
    Apply a high-pass Butterworth filter.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered = filtfilt(b, a, data)
    return filtered

def alpha_filter(data, alpha=0.1):
    """
    Apply a simple exponential (alpha) filter to the data.
    alpha: smoothing factor (0.0 - 1.0)
           closer to 0.0 -> smoother but slower
           closer to 1.0 -> faster reaction but noisier
    """
    filtered = np.zeros_like(data)
    filtered[0] = data[0]
    for i in range(1, len(data)):
        filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i-1]
    return filtered


x_init = np.array([measurements[0, 0], 0])
P_init = np.eye(2)

x_est, P_est = kalman_1d(measurements, x_init, P_init, A, C, Q, R, G)

filtered_pid = pid_1d(measurements, Kp=0.5, Ki=0.1, Kd=0.05, dt=T)

approx_poly, poly_func = poly_approximation(timestamps, measurements, degree=5)
interp_spline, interp_func = spline_interpolation(timestamps, measurements, kind='cubic')
approx_local_poly = local_poly_approximation(timestamps, measurements, window_size=1.0, degree=3)

fs = 1.0 / T  # Sampling frequency (from your timestamp delta)

# Low-pass to smooth small noise
filtered_low = lowpass_filter(approx_local_poly, cutoff_freq=2.0, fs=fs, order=3)

# High-pass to remove slow drift (optional)
filtered_high = highpass_filter(approx_local_poly, cutoff_freq=0.1, fs=fs, order=3)


# After local polynomial approximation
approx_local_poly = local_poly_approximation(timestamps, measurements, window_size=1.0, degree=3)

# Then apply smart alpha filter
smart_filtered = alpha_filter(approx_local_poly, alpha=0.05)  # smaller alpha = slower but smoother


plt.figure(figsize=(10, 5))
plt.plot(timestamps - timestamps[0], distances, 'x-', label='Pomiary')
plt.plot(timestamps - timestamps[0], x_est[:, 0], 'g-', label='Kalman')
#plt.plot(timestamps - timestamps[0], filtered_pid, 'r-', label='PID')
#plt.plot(timestamps - timestamps[0], approx_poly, 'm-', label='Aproksymacja (Polynom)')
#plt.plot(timestamps - timestamps[0], interp_spline, 'b--', label='Interpolacja (Spline)')
plt.plot(timestamps - timestamps[0], approx_local_poly, 'c-', label='Aproksymacja lokalna (Polynom)')
plt.title('Porównanie filtracji, aproksymacji i interpolacji')
plt.xlabel('Czas [s]')
plt.ylabel('Odległość [m]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""
plt.figure(figsize=(10, 5))
plt.plot(timestamps - timestamps[0], distances, 'x-', label='Pomiary')
plt.plot(timestamps - timestamps[0], approx_local_poly, 'c-', label='Aproksymacja lokalna')
plt.plot(timestamps - timestamps[0], filtered_low, 'm-', label='Po Aproksymacji + Low-pass')
plt.plot(timestamps - timestamps[0], filtered_high, 'y--', label='Po Aproksymacji + High-pass')
plt.title('Aproksymacja + Filtracja')
plt.xlabel('Czas [s]')
plt.ylabel('Odległość [m]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(timestamps - timestamps[0], distances, 'x-', label='Pomiary')
plt.plot(timestamps - timestamps[0], approx_local_poly, 'c-', label='Aproksymacja lokalna')
plt.plot(timestamps - timestamps[0], smart_filtered, 'r-', label='Aproksymacja + Smart Alpha')
plt.title('Aproksymacja + inteligentne filtrowanie')
plt.xlabel('Czas [s]')
plt.ylabel('Odległość [m]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""
plt.tight_layout()
plt.show()
