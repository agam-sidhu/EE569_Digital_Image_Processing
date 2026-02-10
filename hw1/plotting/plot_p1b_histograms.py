#
#(1) Name: Agam Sidhu
#(2) USC ID: 3027948957
#(3) USC Email: agamsidh@usc.edu
#(4) Submission Date: February 1, 2026
#EE569 HW1 - Problem 1(b): Histogram Manipulation 
#
import numpy as np
import matplotlib.pyplot as plt
import sys

def readraw(filename, width, height):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape((height, width))

if len(sys.argv) != 4:
    print("Usage: python3 plot_p1b_histograms.py image.raw width height")
    sys.exit(1)

filename = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])

img = readraw(filename, width, height)

plt.figure(figsize=(8,5))
plt.hist(img.flatten(), bins=256, range=(0,255))
plt.title("Histogram")
plt.xlabel("Intensity")
plt.ylabel("Pixel Count")
plt.show()
