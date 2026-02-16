// p1b.cpp - Canny Edge Detector using OpenCV (EE569 HW2)
// Author: Agam Sidhu

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

//Read raw image function
static void readraw(const string& filename,
                    vector<uint8_t>& buffer,
                    int width,
                    int height,
                    int channels)
{
    int byteCount = width * height * channels;
    buffer.resize(byteCount);

    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: cannot open input file\n";
        exit(1);
    }
    file.read(reinterpret_cast<char*>(buffer.data()), byteCount);
    file.close();
}
//Write raw image function
static void writeraw(const string& filename,
                     const vector<uint8_t>& buffer)
{
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: cannot open output file\n";
        exit(1);
    }
    file.write(reinterpret_cast<const char*>(buffer.data()),
               buffer.size());
    file.close();
}
//Clamp function
static inline int clamp(int x, int low, int high) {
    if (x < low) {
        return low;
    } else if (x > high) {
        return high;
    } else {
        return x;
    }
}
//Getting base name function
static string getBaseName(const string& filePath) {
    string fileName = filePath;

    size_t lastSlash = fileName.find_last_of("/\\");
    if (lastSlash != string::npos) {
        fileName = fileName.substr(lastSlash + 1);
    }

    size_t lastDot = fileName.find_last_of('.');
    if (lastDot != string::npos) {
        fileName = fileName.substr(0, lastDot);
    }
    return fileName;
}

int main(int argc, char* argv[])
{
    if (argc < 8) {
        cerr << "Usage:\n"
             << "  " << argv[0]
             << " <input_rgb_raw> <output> <width> <height> <low> <high> <sigmaVal>\n\n"
             << "Example:\n"
             << "  " << argv[0]
             << " hw2/images/Bird.raw hw2/outputs/p1/p1b 481 321 50 150 1.0\n";
        return 1;
    }

    const string input = argv[1];
    const string output = argv[2];
    const int width = atoi(argv[3]);
    const int height = atoi(argv[4]);
    const int low = atoi(argv[5]);
    const int high = atoi(argv[6]);
    const double sigmaVal = atof(argv[7]); 

    if (width <= 0 || height <= 0) {
        cerr << "Error: invalid image dimensions.\n";
        return 1;
    }
    if (low < 0 || high < 0 || low >= high) {
        cerr << "Error, Wrong Canny threshold.\n";
        return 1;
    }
    vector<uint8_t> rgb;
    readraw(input, rgb, width, height, 3);

    Mat gray(height, width, CV_8UC1);
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = 3 * (row * width + col);

            uint8_t R = rgb[idx];
            uint8_t G = rgb[idx + 1];
            uint8_t B = rgb[idx + 2];
            int Y = (int)lround(0.2989 * R + 0.5870 * G + 0.1140 * B);
            gray.at<uint8_t>(row, col) = (uint8_t)clamp(Y, 0, 255);
        }
    }

    Mat cEdges;
    Canny(gray, cEdges, low, high, 3, true);
    vector<uint8_t> edge(width * height, 255);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            if (cEdges.at<uint8_t>(row, col) != 0) {
                edge[row * width + col] = 0;
            }
        }
    }

    const string baseName = getBaseName(input);
    const string out =
        output + "/" + baseName +
        "_canny_L" + to_string(low) +
        "_H" + to_string(high) + ".raw";
    writeraw(out, edge);

    cout << "Wrote: " << out << endl;
    cout << "Canny thresholds: L=" << low
         << ", H=" << high
         << ", sigma=" << sigmaVal << endl;

    return 0;
}