import numpy as np
import matplotlib.pyplot as plt
import sys

YMAX = 15000
YTICKS = list(range(0, YMAX + 1, 2500))

LABELS = ["Red", "Green", "Blue"]
COLORS = ["red", "green", "blue"]

def readraw(filename, width, height):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape((height, width, 3))

def channelHist(img, channel):
    return np.histogram(
        img[:, :, channel].flatten(),
        bins=256,
        range=(0, 255)
    )[0]

if len(sys.argv) != 5:
    print("Usage:")
    print("  python3 plot_p3_before_after.py before.raw after.raw width height")
    sys.exit(1)

before_path = sys.argv[1]
after_path  = sys.argv[2]
width  = int(sys.argv[3])
height = int(sys.argv[4])

before = readraw(before_path, width, height)
after  = readraw(after_path,  width, height)

plt.figure(figsize=(10, 8))

for i in range(3):
    before_hist = channelHist(before, i)
    after_hist  = channelHist(after,  i)

    axis1 = plt.subplot(3, 2, 2*i + 1)
    axis1.bar(range(256), before_hist, color=COLORS[i], width=1.0)
    axis1.set_title(f"{LABELS[i]} Channel - Before AWB")
    axis1.set_xlim(0, 255)
    axis1.set_ylim(0, YMAX)
    axis1.set_yticks(YTICKS)
    axis1.set_xlabel("Intensity")
    axis1.set_ylabel("Count")

    
    axis2 = plt.subplot(3, 2, 2*i + 2)
    axis2.bar(range(256), after_hist, color=COLORS[i], width=1.0)
    axis2.set_title(f"{LABELS[i]} Channel - After AWB")
    axis2.set_xlim(0, 255)
    axis2.set_ylim(0, YMAX)
    axis2.set_yticks(YTICKS)
    axis2.set_xlabel("Intensity")
    axis2.set_ylabel("Count")

plt.tight_layout()
plt.show()
