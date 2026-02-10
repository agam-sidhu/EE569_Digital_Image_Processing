/*
(1) Name: Agam Sidhu
(2) USC ID: 3027948957
(3) USC Email: agamsidh@usc.edu
(4) Submission Date: February 1, 2026
EE569 HW1 - Problem 1(a): Bilinear Demosaicing
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <string>
#include <cstdlib>

using namespace std;

//Clamp function
static inline int clamp(int x, int low, int high) {
    return max(low, min(high, x));
}
//Clamp gray value 
static inline uint8_t clampGray(const vector<uint8_t>& img,
                    int width,
                    int height,
                    int row,
                    int col) {
    row = clamp(row, 0, height - 1);
    col = clamp(col, 0, width - 1);
    return img[row * width + col];
}

//Read raw image function
static void readraw(const string& filename,
                    vector<uint8_t>& buffer,
                    int width,
                    int height,
                    int channels)
{
    const int byteCount = width * height * channels;
    buffer.resize(byteCount);

    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: cannot open input file " << filename << endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(buffer.data()), byteCount);
    if (!file) {
        cerr << "Error: failed to read expected bytes from " << filename << endl;
        exit(1);
    }
    file.close();
}

//Write raw image function
static void writeraw(const string& filename,
                     const vector<uint8_t>& buffer)
{
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: cannot open output file " << filename << endl;
        exit(1);
    }
    file.write(reinterpret_cast<const char*>(buffer.data()),
               static_cast<streamsize>(buffer.size()));
    if (!file) {
        cerr << "Error: failed to write output file " << filename << endl;
        exit(1);
    }
    file.close();
}
//Check color functions
static inline bool isRed(int row, int col) {
    return (row % 2 == 0) && (col % 2 == 1);
}

static inline bool isBlue(int row, int col) {
    return (row % 2 == 1) && (col % 2 == 0);
}

static inline bool isGreen(int row, int col) {
    return !isRed(row, col) && !isBlue(row, col);
}

//Main function
int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " input_cfa.raw output_rgb.raw width height\n";
        return 1;
    }

    const string inputPath  = argv[1];
    const string outputPath = argv[2];
    const int width = atoi(argv[3]);
    const int height = atoi(argv[4]);

    if (width <= 0 || height <= 0) {
        cerr << "Error: width and height must be positive.\n";
        return 1;
    }

    vector<uint8_t> cfa;
    readraw(inputPath, cfa, width, height, 1);
    vector<uint8_t> rgb(3 * width * height, 0);
    // Set pxels w lambda
    auto setPixel = [&](int row, int col, int R, int G, int B) {
        const int idx = (row * width + col) * 3;
        rgb[idx + 0] = static_cast<uint8_t>(clamp(R, 0, 255));
        rgb[idx + 1] = static_cast<uint8_t>(clamp(G, 0, 255));
        rgb[idx + 2] = static_cast<uint8_t>(clamp(B, 0, 255));
    };

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int R, G, B;

            if (isRed(row, col)) { // Red pixel
                R = clampGray(cfa, width, height, row, col);
                G = (clampGray(cfa,width,height,row-1,col) + clampGray(cfa,width,height,row+1,col) + clampGray(cfa,width,height,row,col-1) + clampGray(cfa,width,height,row,col+1)) / 4;
                B = (clampGray(cfa,width,height,row-1,col-1) + clampGray(cfa,width,height,row-1,col+1) + clampGray(cfa,width,height,row+1,col-1) + clampGray(cfa,width,height,row+1,col+1)) / 4;

            } else if (isBlue(row, col)) { //` Blue pixel
                B = clampGray(cfa, width, height, row, col);
                G = (clampGray(cfa,width,height,row-1,col) + clampGray(cfa,width,height,row+1,col) + clampGray(cfa,width,height,row,col-1) + clampGray(cfa,width,height,row,col+1)) / 4;
                R = (clampGray(cfa,width,height,row-1,col-1) + clampGray(cfa,width,height,row-1,col+1) + clampGray(cfa,width,height,row+1,col-1) + clampGray(cfa,width,height,row+1,col+1)) / 4;

            } else { // Green pixel
                G = clampGray(cfa, width, height, row, col);
                if ((row % 2 == 0) && (col % 2 == 0)) {
                    R = (clampGray(cfa,width,height,row, col-1) + clampGray(cfa,width,height,row, col+1)) / 2;
                    B = (clampGray(cfa,width,height,row-1,col) + clampGray(cfa,width,height,row+1,col)) / 2;
                } else { 
                    B = (clampGray(cfa,width,height,row, col-1) + clampGray(cfa,width,height,row, col+1)) / 2;
                    R = (clampGray(cfa,width,height,row-1,col) + clampGray(cfa,width,height,row+1,col)) / 2;
                }
            }
            setPixel(row, col, R, G, B);
        }
    }
    writeraw(outputPath, rgb);
    return 0;
}
