EE569 Homework Assignment #1
Date: February 1, 2026
Name: Agam Sidhu
USC ID: 3027948957
Email: agamsidh@usc.edu

System Information
Operating System:
- macOS (Apple Silicon)

Compiler:
- clang++ / g++ with C++17 support
Python Version:
- Python 3.x (used only for plotting)


Directory Structure
hw1/
├── src/
│   ├── p1/        (Problem 1 source code)
│   ├── p2/        (Problem 2 source code)
│   └── p3/        (Problem 3 source code)
├── plotting/      (Python scripts for histogram visualization)
└── README.txt


Compilation Instructions

All programs are compiled using g++ (or clang++) with C++17.

Examples:

Problem 1(a):
g++ -O2 -std=c++17 src/p1/p1a/p1a.cpp -o p1a

Problem 1(b):
g++ -O2 -std=c++17 src/p1/p1b/p1b_A.cpp -o p1b_A
g++ -O2 -std=c++17 src/p1/p1b/p1b_B.cpp -o p1b_B

Problem 1(c):
g++ -O2 -std=c++17 src/p1/p1c/p1c_A.cpp -o p1c_A
g++ -O2 -std=c++17 src/p1/p1c/p1c_B.cpp -o p1c_B
g++ -O2 -std=c++17 src/p1/p1c/p1c_clahe.cpp -o p1c_clahe

Problem 2:
g++ -O2 -std=c++17 src/p2/p2a.cpp -o p2a
g++ -O2 -std=c++17 src/p2/p2b.cpp -o p2b
g++ -O2 -std=c++17 src/p2/p2c.cpp -o p2c
g++ -O2 -std=c++17 src/p2/p2d.cpp -o p2d

Problem 3:
g++ -O2 -std=c++17 src/p3/p3.cpp -o p3


Execution Instructions

Each program is executed from the command line using raw image inputs
provided by the course.

Examples:

Problem 1(b) Method A:
./p1b_A airplane.raw airplane_methodA.raw hist_original.csv hist_A.csv tf_A.csv

Problem 3 (Auto White Balancing):
./p3 sea.raw sea_awb.raw 768 512

Plotting Instructions (Python)
Python scripts are provided to visualize histograms used in the report.

Problem 1(b) Histogram Visualization:
python3 plotting/plot_p1b_histograms.py airplane_methodA.raw 1024 1024
python3 plotting/plot_p1b_histograms.py airplane_methodB.raw 1024 1024

Problem 3 Auto White Balancing Histogram Comparison:
python3 plotting/plot_p3_before_after.py sea.raw sea_awb.raw 768 512


Notes

- Input images are not included in this submission.
- Output images are not included; results are shown in the report.
- Python is used only for plotting and visualization.
- No Makefiles or executables are included, per assignment instructions.
