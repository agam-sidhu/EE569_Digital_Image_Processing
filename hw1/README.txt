# EE569 Homework Assignment #1
# Date: February 1, 2026
# Name: Agam Sidhu
# ID: 3027948957
# Email: agamsidh@usc.edu
# Operating System: macOS (Apple Silicon)
# Compiler: g++ (Apple clang / GNU g++)
# This README is written in makefile syntax.
# Run using:
#     make -f README.txt
# NOTE:
# - Input images are provided by the course and are NOT included.
# - Output .raw files ARE included as required for grading and
#   correspond exactly to the figures shown in the report.

CC=g++
LN=g++
CFLAGS=-O2 -std=c++17

ALL: p1a p1b p1c p2a p2b p2c p2d p3


# Problem 1(a): Bilinear Demosaicing

p1a:
	@echo "Problem 1(a): Bilinear Demosaicing"
	$(CC) $(CFLAGS) src/p1/p1a.cpp -o bin/p1a
	@echo "Output:"
	@echo "outputs/sailboats_bilinear.raw (512x768 RGB)"
	./bin/p1a Image_hw1/sailboats_cfa.raw outputs/sailboats_bilinear.raw 512 768


# Problem 1(b): Histogram Manipulation

p1b:
	@echo "Problem 1(b): Histogram Manipulation"
	$(CC) $(CFLAGS) src/p1/p1b.cpp -o bin/p1b
	@echo "Outputs:"
	@echo "outputs/airplane_methodA.raw"
	@echo "outputs/airplane_methodB.raw"
	./bin/p1b Image_hw1/airplane.raw outputs/airplane_methodA.raw 1024 1024 A
	./bin/p1b Image_hw1/airplane.raw outputs/airplane_methodB.raw 1024 1024 B

# Problem 1(c): CLAHE
p1c:
	@echo "Problem 1(c): CLAHE"
	$(CC) $(CFLAGS) src/p1/p1c.cpp -o bin/p1c `pkg-config --cflags --libs opencv4`
	@echo "Output:"
	@echo "outputs/towers_clahe.raw"
	./bin/p1c Image_hw1/towers.raw outputs/towers_clahe.raw

########################################
# Problem 2(a): Linear Denoising
########################################
p2a:
	@echo "Problem 2(a): Linear Denoising"
	$(CC) $(CFLAGS) src/p2/p2a.cpp -o bin/p2a
	@echo "Outputs:"
	@echo "outputs/p2a_uniform_3x3.raw"
	@echo "outputs/p2a_gaussian_5x5.raw"
	./bin/p2a Image_hw1/flower_gray.raw Image_hw1/flower_gray_noisy.raw outputs/p2a_uniform_3x3.raw 768 512 uniform 3
	./bin/p2a Image_hw1/flower_gray.raw Image_hw1/flower_gray_noisy.raw outputs/p2a_gaussian_5x5.raw 768 512 gauss 5 1.2
# Problem 2(b): Bilateral Filtering

p2b:
	@echo "Problem 2(b): Bilateral Filtering"
	$(CC) $(CFLAGS) src/p2/p2b.cpp -o bin/p2b
	@echo "Outputs:"
	@echo "outputs/p2b_sigS2_sigC50.raw"
	@echo "outputs/p2b_sigS2_sigC75.raw"
	./bin/p2b Image_hw1/flower_gray.raw Image_hw1/flower_gray_noisy.raw outputs/p2b_sigS2_sigC50.raw 768 512 5 2.0 50.0
	./bin/p2b Image_hw1/flower_gray.raw Image_hw1/flower_gray_noisy.raw outputs/p2b_sigS2_sigC75.raw 768 512 5 2.0 75.0

# Problem 2(c): Non-Local Means

p2c:
	@echo "Problem 2(c): Non-Local Means"
	$(CC) $(CFLAGS) src/p2/p2c.cpp -o bin/p2c
	@echo "Outputs:"
	@echo "outputs/p2c_h10.raw"
	@echo "outputs/p2c_h35.raw"
	./bin/p2c Image_hw1/flower_gray.raw Image_hw1/flower_gray_noisy.raw outputs/p2c_h10.raw 768 512 10
	./bin/p2c Image_hw1/flower_gray.raw Image_hw1/flower_gray_noisy.raw outputs/p2c_h35.raw 768 512 35


# Problem 2(d): Color Image Denoising

p2d:
	@echo "Problem 2(d): Color Image Denoising"
	$(CC) $(CFLAGS) src/p2/p2d.cpp -o bin/p2d
	@echo "Outputs:"
	@echo "outputs/p2d_median_gaussian.raw"
	@echo "outputs/p2d_median_nlm.raw"
	./bin/p2d Image_hw1/flower.raw Image_hw1/flower_noisy.raw outputs/p2d_median_gaussian.raw 768 512 gaussian
	./bin/p2d Image_hw1/flower.raw Image_hw1/flower_noisy.raw outputs/p2d_median_nlm.raw 768 512 nlm


# Problem 3: Auto White Balancing

p3:
	@echo "Problem 3: Auto White Balancing"
	$(CC) $(CFLAGS) src/p3/p3.cpp -o bin/p3
	@echo "Output:"
	@echo "outputs/sea_awb.raw"
	./bin/p3 Image_hw1/sea.raw outputs/sea_awb.raw 768 512

Plotting (Python):

Problem 1(b):
python3 plotting/plot_p1b_histograms.py outputs/airplane_methodA.raw 1024 1024
python3 plotting/plot_p1b_histograms.py outputs/airplane_methodB.raw 1024 1024

Problem 3:
python3 plotting/plot_p3_histograms.py outputs/sea_awb.raw 768 512
