import numpy as np
import matplotlib.pyplot as plt
import sys

def read_raw_gray(filename, width, height):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape((height, width))

if len(sys.argv) != 4:
    print("Usage: python3 plot_p1b_histograms.py image.raw width height")
    sys.exit(1)

filename = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])

img = read_raw_gray(filename, width, height)

plt.figure(figsize=(8,5))
plt.hist(img.flatten(), bins=256, range=(0,255))
plt.title("Histogram")
plt.xlabel("Intensity")
plt.ylabel("Pixel Count")
plt.show()
