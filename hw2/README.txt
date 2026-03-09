EE569 Homework #2
Name: Agam Sidhu
USC ID: 3027948957
USC Email: agamsidh@usc.edu
Submission Date: February 22, 2026

SYSTEM INFORMATION

Operating System: macOS (Apple Silicon)
C++ Compiler: g++
MATLAB Version: MATLAB Online / R2025b


GENERAL NOTES
- No input arguments are required for MATLAB scripts.
- Image dimensions are hardcoded within each MATLAB script.
- All input image filenames are specified inside each script.
- Output files are automatically written to their corresponding output folders.
- No RAW or image files are included in the submission ZIP file.
- All quantitative evaluation metrics (precision, recall, F-measure) are computed within p1d.m.
- PR curves and evaluation plots are generated automatically by the evaluation script.

PROBLEM 1(a) – C++ Implementation
File: p1a.cpp

Compile:
g++ -o p1a p1a.cpp

Run:
./p1a input.raw output.raw width height [parameters]

Example:
./p1a hw2/images/Bird.raw hw2/outputs/p1a/output.raw 481 321 ...

PROBLEM 1(b) – C++ Canny Edge Detector

File: p1b.cpp

Compile:
g++ -o p1b p1b.cpp

Run:
./p1b input.raw output.raw width height lowThreshold highThreshold sigma

Example:
./p1b hw2/images/Deer.raw hw2/outputs/p1b/deer_canny.raw 481 321 40 120 1

Outputs are written as RAW files.

MATLAB DIRECTORY REQUIREMENTS

Before running MATLAB scripts, ensure the following directory
structure and files are present:

Problem 1(c) – Structured Edge


Working Directory:
hw2/src/p1

Required folders and files inside p1:

- p1c.m
- edges/               (Structured Edge toolbox)
- input/
    ├── Bird.jpg
    ├── Deer.jpg

The script automatically:
- Loads pretrained model from edges/models/forest/modelBsds
- Creates an "output" folder
- Saves:
    Bird_SE_prob.png
    Bird_SE_bin_Txx.png
    Deer_SE_prob.png
    Deer_SE_bin_Txx.png

Problem 1(d) – Edge Evaluation

Working Directory:
hw2/src/p1

Required folders:

- gt/ (Ground truth .mat files)
- edges/ (Structured Edge toolbox)
- eval_out_p1d_manual/ (if used)
- Output edge maps generated from p1c

The script:
- Evaluates Sobel, Canny, and Structured Edge results
- Computes precision, recall, and F-measure
- Generates PR curves
To generate F-measure vs threshold plots:

Run:
plot_p1d("Bird")
plot_p1d("Deer")
or
plot_p1d("ALL")

This script:
- Reads saved CSV threshold tables
- Generates F-measure vs threshold plots
- Saves plots in eval_out_p1d_manual/plots

Problem 2(a) – Dithering
Working Directory:
hw2/src/p2

Required files:
- p2a.m
- Reflection.raw

Output:
- p2a_fixed.raw
- p2a_random.raw
- p2a_I2.raw
- p2a_I8.raw
- p2a_I32.raw
- Threshold matrix images (T2.png, T8.png, T32.png)

Problem 2(b) – Error Diffusion

Working Directory:
hw2/src/p2

Required files:
- p2b.m
- Reflection.raw

Outputs saved to p2b_outputs folder.

Problem 3(a) – Separable Color Diffusion

Working Directory:
hw2/src/p3

Required files:
- p3a.m
- Flowers.raw

Outputs saved to p3a_outputs.

Problem 3(b) – MBVQ Diffusion

Working Directory:
hw2/src/p3

Required files:
- p3b.m
- getNearestVertex.m
- Flowers.raw

Outputs saved to p3b_outputs.