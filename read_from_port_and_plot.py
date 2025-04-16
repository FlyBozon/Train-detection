import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import time
from collections import deque

PORT = '/dev/ttyUSB0' #COM dla Windows
BAUD = 115200
ser = serial.Serial(PORT, BAUD, timeout=0.5)

MAX_POINTS = 100
data = deque([0]*MAX_POINTS, maxlen=MAX_POINTS)

pattern = re.compile(r'dis=(\d+(\.\d+)?)')

last_time = time.time()
intervals = deque(maxlen=10)

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_ylim(0, 10)
ax.set_xlim(0, MAX_POINTS)
ax.set_title('Radar HLK-LD1125H - pomiar odległości')
ax.set_xlabel('Czas (próbki)')
ax.set_ylabel('Odległość (m)')

def update(frame):
    global last_time

    try:
        line_bytes = ser.readline()
        line_str = line_bytes.decode('utf-8', errors='ignore').strip()

        match = pattern.search(line_str)
        if match:
            distance = float(match.group(1))
            data.append(distance)

            now = time.time()
            dt = now - last_time
            last_time = now
            intervals.append(dt)

            avg_freq = 1.0 / (sum(intervals) / len(intervals)) if intervals else 0.0

            print(f"{line_str}  |  Δt: {dt:.3f}s  |  Śr. freq: {avg_freq:.2f} Hz")
        else:
            print(f"{line_str}")
    except Exception as e:
        print(f"Błąd: {e}")

    line.set_data(range(len(data)), list(data))
    return line,

ani = animation.FuncAnimation(fig, update, blit=True, interval=100)
plt.show()
