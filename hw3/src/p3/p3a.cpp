/* EE569 Homework #3
 * Name: Agam Sidhu
 * USC ID: 3027948957
 * USC Email: agamsidh@usc.edu
 * Submission Date: March 15, 2026
 * Problem 3(a): Basic morphological process implementation 
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
static inline uint8_t clampGray(const vector<uint8_t>& img, int width, int height, int row, int col) {
    row = clamp(row, 0, height - 1);
    col = clamp(col, 0, width - 1);
    return img[row * width + col];
}
//Read raw image function
static void readraw(const string& filename, vector<uint8_t>& buffer, int width, int height, int channels)
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
static void writeraw(const string& filename, const vector<uint8_t>& buffer)
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

//Function to convert into raw from binary
static vector<uint8_t> toRaw(const vector<uint8_t>& bi) {
    size_t n = bi.size();
    vector<uint8_t> output(n, 0);
    for (size_t i = 0; i < n; i++) {
        if (bi[i] == 1) {
        output[i] = 255; 
        } else {
            output[i] = 0;  
        }
    }
    return output;
}

//Function to binarize image
static vector<uint8_t> toBinary(const vector<uint8_t>& image, int width, int height)
{
    //gets the max pixel valye
    uint8_t fMax = 0;
    for (size_t i = 0; i < image.size(); i++) {
        fMax = max(fMax, image[i]);
    }
    //hreshold value (1/2 the max)
    const double tVal = 0.5 * static_cast<double>(fMax);
    vector<uint8_t> biOut(width * height, 0);

    //check to see if greater than threshold
    for (int i = 0; i < width * height; i++) {
         // 1 if above else 0 
         bool above = image[i] > tVal; 
        if (above) {
            biOut[i] = 1; 
        } else {
            biOut[i] = 0;
        }
    }
    return biOut;
}

//Function to get binary pixel 
static inline uint8_t getBP(const vector<uint8_t>& image, int width, int height, int row, int col)
{
    //sees if out of bounds
    if (row < 0 || row >= height || col < 0 || col >= width) {
        return 0;
    }
    uint8_t pix = image[row * width + col];
    return pix;
}

//Function to count nonzero neightbors (8-neighborhood)
static int neighborCount(const vector<uint8_t>& image, int width, int height, int row, int col)
{
    //all 8 neighbors in order (p2 to p9)
    int p2 = getBP(image, width, height, row - 1, col);
    int p3 = getBP(image, width, height, row - 1, col + 1);
    int p4 = getBP(image, width, height, row, col + 1);
    int p5 = getBP(image, width, height, row + 1, col + 1);
    int p6 = getBP(image, width, height, row + 1, col);
    int p7 = getBP(image, width, height, row + 1, col - 1);
    int p8 = getBP(image, width, height, row, col - 1);
    int p9 = getBP(image, width, height, row - 1, col - 1);
    //sum of total neighbors (nonzero)
    int sum = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
    return sum;
}

//Function to count transitions from 0/1 
static int countTrans(const vector<uint8_t>& image, int width, int height, int row, int col)
{
    //get neightbors in order (p2 to p9)
    int p2 = getBP(image, width, height, row - 1, col);
    int p3 = getBP(image, width, height, row - 1, col + 1);
    int p4 = getBP(image, width, height, row, col + 1);
    int p5 = getBP(image, width, height, row + 1, col + 1);
    int p6 = getBP(image, width, height, row + 1, col);
    int p7 = getBP(image, width, height, row + 1, col - 1);
    int p8 = getBP(image, width, height, row, col - 1);
    int p9 = getBP(image, width, height, row - 1, col - 1);

    //add p2 to end to complete cycle
    int seq[9] = {p2, p3, p4, p5, p6, p7, p8, p9, p2};
    int count = 0;
    //count transitions (from 0 ->1)
    for (int i = 0; i < 8; i++) {
        if (seq[i] == 0 && seq[i + 1] == 1) {
            count++;
        }
    }
    return count;
}

//Function to run one iteration of the thinning slgo
static bool thinIter(vector<uint8_t>& image, int width, int height)
{
    bool altered = false;
    vector<int> del;

    //iteration 1
    del.clear();
    for (int row = 1; row < height - 1; row++) {
        for (int col = 1; col < width - 1; col++) {
            if (image[row * width + col] == 0) continue;
            //get neighbors
            int p2 = getBP(image, width, height, row - 1, col);
            int p4 = getBP(image, width, height, row, col + 1);
            int p6 = getBP(image, width, height, row + 1, col);
            int p8 = getBP(image, width, height, row, col - 1);
            //neighbor count & transition count
            int neighNum = neighborCount(image, width, height, row, col);
            int trans = countTrans(image, width, height, row, col);
            //check conditions
            bool isTrans = trans == 1;
            bool isNeigh = neighNum >= 2 && neighNum <= 6;
            bool isZeroProd1 = (p2 * p4 * p6) == 0;
            bool isZeroProd2 = (p4 * p6 * p8) == 0;
            if (isTrans && isNeigh && isZeroProd1 && isZeroProd2) {
                del.push_back(row * width + col); // highlight to delete
            }
        }
    }

    for (size_t i = 0; i < del.size(); i++) {
        image[del[i]] = 0; // delete the pixel
        altered = true;
    }

    //iteration 2
    del.clear();
    for (int row = 1; row < height - 1; row++) {
        for (int col = 1; col < width - 1; col++) {
            if (image[row * width + col] == 0) continue;
            //get neighbors
            int p2 = getBP(image, width, height, row - 1, col);
            int p4 = getBP(image, width, height, row, col + 1);
            int p6 = getBP(image, width, height, row + 1, col);
            int p8 = getBP(image, width, height, row, col - 1);

            //neighbor count & transition count
            int neighNum = neighborCount(image, width, height, row, col);
            int trans = countTrans(image, width, height, row, col);
            //check conditions
            bool isTrans = trans == 1;
            bool isNeigh = neighNum >= 2 && neighNum <= 6;
            bool isZeroProd1 = (p2 * p4 * p8) == 0;
            bool isZeroProd2 = (p2 * p6 * p8) == 0; 
            if (isTrans && isNeigh && isZeroProd1 && isZeroProd2) {
                del.push_back(row * width + col);
            }
        }
    }

    for (size_t i = 0; i < del.size(); i++) {
        image[del[i]] = 0; // delete the pixel
        altered = true;
    }

    return altered;
}

//Main function
int main(int argc, char* argv[]) {
    if (argc != 6) {
        cerr << "Usage: " << argv[0]
             << " input.raw output_prefix width height maxIterations\n";
        cerr << "Example: " << argv[0]
             << " Jar.raw outputs/jar 512 512 100\n";
        return 1;
    }
    //parse input args
    const string input = argv[1];
    const string output = argv[2];
    const int width = atoi(argv[3]);
    const int height = atoi(argv[4]);
    const int maxIter = atoi(argv[5]);

    vector<uint8_t> image;
    readraw(input, image, width, height, 1);

    //convert to binary & show result
    vector<uint8_t> binary = toBinary(image, width, height);
    writeraw(output + "_binary.raw", toRaw(binary));

    vector<int> checkPoint = {20};
    //Runs algo until convergence/max iteration
    for (int step = 1; step <= maxIter; step++) {
        vector<uint8_t> prev = binary;
        thinIter(binary, width, height);

        //if checkpoint (output results)
        if (find(checkPoint.begin(), checkPoint.end(), step) != checkPoint.end()) {
            writeraw(output + "_thin_iter" + to_string(step) + ".raw",
                    toRaw(binary));
        }
        //convergence check
        if (binary == prev) {
            cout << "Reached convergence at: " << step << endl;
            writeraw(output + "_thin_final.raw", toRaw(binary));
            return 0;
        }
    }
    writeraw(output + "_thin_final.raw", toRaw(binary));
    return 0;
}
