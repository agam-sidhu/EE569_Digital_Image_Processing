EE569 Homework #3
Name: Agam Sidhu
USC ID: 3027948957
USC Email: agamsidh@usc.edu
Submission Date: March 15, 2026

SYSTEM INFORMATION

Operating System: macOS (Apple Silicon)
C++ Compiler: g++
Library Requirement:
- OpenCV 4.x is required for Problem 2 only


DIRECTORY STRUCTURE

hw3/
├── src/
│   ├── Image_hw3/
│   │   ├── input RAW/PNG files
│   │   └── outputs/
│   │       ├── p1/
│   │       ├── p2/
│   │       └── p3/
│   ├── p1/
│   ├── p2/
│   └── p3/
├── hw3submission/
└── README.txt


GENERAL NOTES

- All programs are written in C++.
- Problems 1, 3(a), and 3(b) use only the C++ standard library.
- Problem 2 uses OpenCV for SIFT feature extraction and match visualization.
- Output directories must exist before running each program.
- Input images used in the commands below are located in src/Image_hw3/.
- Generated RAW and PNG outputs are stored under src/Image_hw3/outputs/.
- No Makefile is included in this submission.


PROBLEM 1 - Geometric Image Modification

Working Directory:
hw3/src/p1

File:
- p1.cpp

Compile:
g++ -O2 -std=c++17 p1.cpp -o p1

Run:
./p1 input.raw output.raw width height type arc

Notes:
- type = 0 performs the square-to-star mapping
- type = 1 performs the reverse mapping
- Bird.raw and Cat.raw are RGB images with size 800 x 800

Examples:
./p1 ../Image_hw3/Bird.raw ../Image_hw3/outputs/p1/Bird_star.raw 800 800 0 128
./p1 ../Image_hw3/outputs/p1/Bird_star.raw ../Image_hw3/outputs/p1/Bird_recovered.raw 800 800 1 128
./p1 ../Image_hw3/Cat.raw ../Image_hw3/outputs/p1/Cat_star.raw 800 800 0 128
./p1 ../Image_hw3/outputs/p1/Cat_star.raw ../Image_hw3/outputs/p1/Cat_recovered.raw 800 800 1 128


PROBLEM 2 - Homographic Transformation and Image Stitching

Working Directory:
hw3/src/p2

File:
- p2.cpp

Compile:
g++ -O2 -std=c++17 p2.cpp -o p2 $(pkg-config --cflags --libs opencv4)

Run:
./p2 left.raw middle.raw right.raw panorama.raw width height match_left.png match_right.png points_left.txt points_right.txt

Notes:
- Broad_left.raw, Broad_middle.raw, and Broad_right.raw are RGB images with size 600 x 400
- This program writes one panorama RAW file, two match-visualization PNG files, and two matched-point text files

Example:
./p2 ../Image_hw3/Broad_left.raw ../Image_hw3/Broad_middle.raw ../Image_hw3/Broad_right.raw ../Image_hw3/outputs/p2/Broad_panorama.raw 600 400 ../Image_hw3/outputs/p2/left_middle_matches.png ../Image_hw3/outputs/p2/right_middle_matches.png ../Image_hw3/outputs/p2/left_middle_points.txt ../Image_hw3/outputs/p2/right_middle_points.txt


PROBLEM 3(a) - Basic Morphological Thinning

Working Directory:
hw3/src/p3

File:
- p3a.cpp

Compile:
g++ -O2 -std=c++17 p3a.cpp -o p3a

Run:
./p3a input.raw output_prefix width height maxIterations

Notes:
- The program automatically writes:
  output_prefix_binary.raw
  output_prefix_thin_iter20.raw
  output_prefix_thin_final.raw
- Jar.raw, Moon.raw, Spring.raw, and Star.raw are grayscale images with size 512 x 512

Examples:
./p3a ../Image_hw3/Jar.raw ../Image_hw3/outputs/p3/jar 512 512 100
./p3a ../Image_hw3/Moon.raw ../Image_hw3/outputs/p3/moon 512 512 100
./p3a ../Image_hw3/Spring.raw ../Image_hw3/outputs/p3/spring 512 512 100
./p3a ../Image_hw3/Star.raw ../Image_hw3/outputs/p3/star 512 512 100


PROBLEM 3(b) - Shape Detection and Counting

Working Directory:
hw3/src/p3

File:
- p3b.cpp

Compile:
g++ -O2 -std=c++17 p3b.cpp -o p3b

Run:
./p3b input.raw output_binary.raw width height

Notes:
- This program binarizes the input image, labels connected components, estimates Euler number, counts holes, and classifies detected objects as rectangles or circles
- It prints the counting summary to the terminal
- If the input image is already a grayscale shape image, the output RAW file stores the binarized version used for analysis

Example:
./p3b ../Image_hw3/outputs/p3/star_binary.raw ../Image_hw3/outputs/p3/shapes/star_binary_analysis.raw 445 445


SUBMISSION NOTES

- Source files are included under the corresponding problem folders in src/.
- Output images shown in the report are stored in src/Image_hw3/outputs/.
- The PDF report is included in hw3submission/.
