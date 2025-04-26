import tkinter as tk
from tkinter import filedialog
import serial
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv

# Serial configuration
SERIAL_PORT = '/dev/ttyUSB1'  # Change it to your port (e.g., '/dev/ttyUSB0' on Linux)
BAUD_RATE = 115200

class RadarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Radar Test Interface")

        # Start/Stop flags
        self.running = False
        self.data = []

        # Serial connection
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        except Exception as e:
            print(f"Error opening serial port: {e}")
            self.ser = None

        # Set up plot
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        # Buttons
        self.start_button = tk.Button(root, text="Start", command=self.start_reading)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = tk.Button(root, text="Stop", command=self.stop_reading)
        self.stop_button.pack(side=tk.LEFT)
        self.save_button = tk.Button(root, text="Save Data", command=self.save_data)
        self.save_button.pack(side=tk.LEFT)

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
        self.line.set_data(times, distances)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

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


if __name__ == "__main__":
    root = tk.Tk()
    app = RadarApp(root)
    root.mainloop()
