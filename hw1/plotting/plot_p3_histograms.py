import numpy as np
import matplotlib.pyplot as plt
import sys

def read_raw_rgb(filename, width, height):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = data.reshape((height, width, 3))
    return img

if len(sys.argv) != 4:
    print("Usage: python3 plot_p3_histograms.py image.raw width height")
    sys.exit(1)

filename = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])

img = read_raw_rgb(filename, width, height)

colors = ['r', 'g', 'b']
labels = ['Red', 'Green', 'Blue']

plt.figure(figsize=(10,6))
for i in range(3):
    plt.hist(img[:,:,i].flatten(), bins=256, range=(0,255),
             color=colors[i], alpha=0.5, label=labels[i])

plt.legend()
plt.xlabel("Intensity")
plt.ylabel("Pixel Count")
plt.title("RGB Channel Histograms")
plt.show()
