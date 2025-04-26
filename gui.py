import tkinter as tk
from tkinter import filedialog
import serial
import threading
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

# Serial configuration
SERIAL_PORT = '/dev/ttyUSB0'  # Change it to your port
BAUD_RATE = 115200

# ====== FILTER FUNCTIONS ======
def kalman_filter(timestamps, measurements):
    T = np.mean(np.diff(timestamps))
    A = np.array([[1, T], [0, 1]])
    C = np.array([[1, 0]])
    Q = np.array([[0.01, 0], [0, 0.1]])
    R = np.array([[0.1]])
    G = np.eye(2)
    x_init = np.array([measurements[0], 0])
    P_init = np.eye(2)

    n = len(measurements)
    x_est = np.zeros((n, 2))
    P = np.zeros((n, 2, 2))
    x_est[0] = x_init
    P[0] = P_init

    for k in range(1, n):
        x_pred = A @ x_est[k-1]
        P_pred = A @ P[k-1] @ A.T + G @ Q @ G.T
        y_pred = C @ x_pred
        e = measurements[k] - y_pred
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)
        x_est[k] = x_pred + (K @ e).flatten()
        P[k] = (np.eye(2) - K @ C) @ P_pred
    return x_est[:, 0]

def pid_filter(measurements, Kp=0.5, Ki=0.1, Kd=0.05, dt=0.05):
    n = len(measurements)
    filtered = np.zeros(n)
    integral = 0.0
    previous_error = 0.0
    filtered[0] = measurements[0]

    for k in range(1, n):
        error = measurements[k] - filtered[k-1]
        integral += error * dt
        derivative = (error - previous_error) / dt
        control = Kp * error + Ki * integral + Kd * derivative
        filtered[k] = filtered[k-1] + control
        previous_error = error
    return filtered

def alpha_filter(data, alpha=0.05):
    filtered = np.zeros_like(data)
    filtered[0] = data[0]
    for i in range(1, len(data)):
        filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i-1]
    return filtered

# ====== GUI CLASS ======
class RadarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Radar Measurement & Filtering")
        self.running = False
        self.data = []

        # Serial connection
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        except Exception as e:
            print(f"Error opening serial port: {e}")
            self.ser = None

        # GUI Layout
        self.fig, self.ax = plt.subplots()
        self.line_raw, = self.ax.plot([], [], 'k.', label="Raw Data")
        self.line_filtered, = self.ax.plot([], [], 'r-', label="Filtered")
        self.ax.legend()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        # Buttons
        self.start_button = tk.Button(root, text="Start", command=self.start_reading)
        self.start_button.pack(side=tk.LEFT)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_reading)
        self.stop_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(root, text="Save Data", command=self.save_data)
        self.save_button.pack(side=tk.LEFT)

        self.load_button = tk.Button(root, text="Load Data", command=self.load_data)
        self.load_button.pack(side=tk.LEFT)

        self.filter_var = tk.StringVar(value="alpha")
        self.filter_menu = tk.OptionMenu(root, self.filter_var, "none", "alpha", "pid", "kalman")
        self.filter_menu.pack(side=tk.LEFT)

    def start_reading(self):
        if not self.running and self.ser:
            self.running = True
            threading.Thread(target=self.read_serial).start()

    def stop_reading(self):
        self.running = False

    def read_serial(self):
        while self.running:
            try:
                line = self.ser.readline().decode().strip()
                if line and "dis=" in line:
                    try:
                        value_str = line.split('dis=')[1]
                        distance = float(value_str)
                        timestamp = time.time()
                        self.data.append((timestamp, distance))
                        self.update_plot()
                    except ValueError:
                        print(f"Could not parse line: {line}")
            except Exception as e:
                print(f"Error reading serial: {e}")
            time.sleep(0.05)

    def update_plot(self):
        if not self.data:
            return
        times, distances = zip(*self.data)
        times = np.array(times)
        distances = np.array(distances)

        self.line_raw.set_data(times - times[0], distances)

        filtered = self.apply_filter(times, distances)
        self.line_filtered.set_data(times - times[0], filtered)

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def apply_filter(self, timestamps, measurements):
        method = self.filter_var.get()
        if method == "none":
            return measurements
        elif method == "alpha":
            return alpha_filter(measurements, alpha=0.05)
        elif method == "pid":
            T = np.mean(np.diff(timestamps))
            return pid_filter(measurements, dt=T)
        elif method == "kalman":
            return kalman_filter(timestamps, measurements)
        else:
            return measurements

    def save_data(self):
        if not self.data:
            print("No data to save.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("CSV files", "*.csv")],
                                                initialfile=f"radar_data_{int(time.time())}.csv")
        if filename:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "distance_m"])
                writer.writerows(self.data)
            print(f"Data saved to {filename}")

    def load_data(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            df = pd.read_csv(filename)
            timestamps = df['timestamp'].values
            distances = df['distance_m'].values
            self.data = list(zip(timestamps, distances))
            self.update_plot()


if __name__ == "__main__":
    root = tk.Tk()
    app = RadarApp(root)
    root.mainloop()
