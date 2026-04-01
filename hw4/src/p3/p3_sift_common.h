/* EE569 Homework #4
 * Name: Agam Sidhu
 * USC ID: 3027948957
 * USC Email: agamsidh@usc.edu
 * Submission Date: [fill]
 * Shared Helper for Problem 3
 */

#ifndef EE569_HW4_P3_SIFT_COMMON_H
#define EE569_HW4_P3_SIFT_COMMON_H

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

static inline void ensureDir(const string& path)
{
    mkdir(path.c_str(), 0755);
}

static inline cv::Mat readRawRGBImage(const string& filename,
                                      int width,
                                      int height)
{
    int byteCount = width * height * 3;
    vector<unsigned char> buffer(byteCount, 0);

    ifstream file(filename.c_str(), ios::binary);
    if (!file) {
        cerr << "Error: cannot open input file " << filename << endl;
        exit(1);
    }

    file.read(reinterpret_cast<char*>(buffer.data()), byteCount);
    if (!file) {
        cerr << "Error: failed to read input file " << filename << endl;
        exit(1);
    }
    file.close();

    cv::Mat image(height, width, CV_8UC3);
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = (row * width + col) * 3;
            unsigned char r = buffer[idx];
            unsigned char g = buffer[idx + 1];
            unsigned char b = buffer[idx + 2];
            image.at<cv::Vec3b>(row, col) = cv::Vec3b(b, g, r);
        }
    }

    return image;
}

static inline void extractSIFT(const cv::Mat& image,
                               vector<cv::KeyPoint>& keypoints,
                               cv::Mat& descriptors)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
}

#endif
