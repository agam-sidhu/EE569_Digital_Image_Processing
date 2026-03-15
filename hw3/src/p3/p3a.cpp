/* EE569 Homework #3
 * Name: Agam Sidhu
 * USC ID: 3027948957
 * USC Email: agamsidh@usc.edu
 * Submission Date: March 15, 2026
 * Problem 3(a): Geometric Image Modification - Star Warping
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

//Binarize image using threshold 0.5 * Fmax
static vector<uint8_t> binarizeImage(const vector<uint8_t>& gray, int width, int height)
{
    uint8_t fmax = 0;
    for (size_t i = 0; i < gray.size(); i++) {
        fmax = max(fmax, gray[i]);
    }

    const double thresh = 0.5 * static_cast<double>(fmax);
    vector<uint8_t> binary(width * height, 0);

    for (int i = 0; i < width * height; i++) {
        binary[i] = (gray[i] > thresh) ? 1 : 0;
    }
    return binary;
}

//Convert binary {0,1} to raw {0,255}
static vector<uint8_t> binaryToRaw(const vector<uint8_t>& binary) {
    vector<uint8_t> out(binary.size(), 0);
    for (size_t i = 0; i < binary.size(); i++) {
        out[i] = binary[i] ? 255 : 0;
    }
    return out;
}

//Get binary pixel
static inline uint8_t getBin(const vector<uint8_t>& img, int width, int height, int row, int col)
{
    if (row < 0 || row >= height || col < 0 || col >= width) {
        return 0;
    }
    return img[row * width + col];
}

//Number of nonzero neighbors in 8-neighborhood
static int countNeighbors(const vector<uint8_t>& img, int width, int height, int row, int col)
{
    int p2 = getBin(img, width, height, row - 1, col);
    int p3 = getBin(img, width, height, row - 1, col + 1);
    int p4 = getBin(img, width, height, row, col + 1);
    int p5 = getBin(img, width, height, row + 1, col + 1);
    int p6 = getBin(img, width, height, row + 1, col);
    int p7 = getBin(img, width, height, row + 1, col - 1);
    int p8 = getBin(img, width, height, row, col - 1);
    int p9 = getBin(img, width, height, row - 1, col - 1);

    return p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
}

//Number of 0->1 transitions in ordered neighborhood
static int countTransitions(const vector<uint8_t>& img,
                            int width,
                            int height,
                            int row,
                            int col)
{
    int p2 = getBin(img, width, height, row - 1, col);
    int p3 = getBin(img, width, height, row - 1, col + 1);
    int p4 = getBin(img, width, height, row,     col + 1);
    int p5 = getBin(img, width, height, row + 1, col + 1);
    int p6 = getBin(img, width, height, row + 1, col);
    int p7 = getBin(img, width, height, row + 1, col - 1);
    int p8 = getBin(img, width, height, row,     col - 1);
    int p9 = getBin(img, width, height, row - 1, col - 1);

    int seq[9] = {p2, p3, p4, p5, p6, p7, p8, p9, p2};
    int A = 0;
    for (int i = 0; i < 8; i++) {
        if (seq[i] == 0 && seq[i + 1] == 1) {
            A++;
        }
    }
    return A;
}

//One full Zhang-Suen thinning iteration
static bool thinningIteration(vector<uint8_t>& img,
                              int width,
                              int height)
{
    bool changed = false;
    vector<int> toDelete;

    //Subiteration 1
    toDelete.clear();
    for (int row = 1; row < height - 1; row++) {
        for (int col = 1; col < width - 1; col++) {
            if (img[row * width + col] == 0) continue;

            int p2 = getBin(img, width, height, row - 1, col);
            int p4 = getBin(img, width, height, row,     col + 1);
            int p6 = getBin(img, width, height, row + 1, col);
            int p8 = getBin(img, width, height, row,     col - 1);

            int B = countNeighbors(img, width, height, row, col);
            int A = countTransitions(img, width, height, row, col);

            if (A == 1 &&
                B >= 2 && B <= 6 &&
                (p2 * p4 * p6) == 0 &&
                (p4 * p6 * p8) == 0) {
                toDelete.push_back(row * width + col);
            }
        }
    }

    for (size_t i = 0; i < toDelete.size(); i++) {
        img[toDelete[i]] = 0;
        changed = true;
    }

    //Subiteration 2
    toDelete.clear();
    for (int row = 1; row < height - 1; row++) {
        for (int col = 1; col < width - 1; col++) {
            if (img[row * width + col] == 0) continue;

            int p2 = getBin(img, width, height, row - 1, col);
            int p4 = getBin(img, width, height, row,     col + 1);
            int p6 = getBin(img, width, height, row + 1, col);
            int p8 = getBin(img, width, height, row,     col - 1);

            int B = countNeighbors(img, width, height, row, col);
            int A = countTransitions(img, width, height, row, col);

            if (A == 1 &&
                B >= 2 && B <= 6 &&
                (p2 * p4 * p8) == 0 &&
                (p2 * p6 * p8) == 0) {
                toDelete.push_back(row * width + col);
            }
        }
    }

    for (size_t i = 0; i < toDelete.size(); i++) {
        img[toDelete[i]] = 0;
        changed = true;
    }

    return changed;
}

//Main function
int main(int argc, char* argv[]) {
    if (argc != 6) {
        cerr << "Usage: " << argv[0]
             << " input.raw output_prefix width height maxIterations\n";
        cerr << "Example: " << argv[0]
             << " Jar.raw outputs/jar 512 512 30\n";
        return 1;
    }

    const string inputPath = argv[1];
    const string outputPrefix = argv[2];
    const int width = atoi(argv[3]);
    const int height = atoi(argv[4]);
    const int maxIterations = atoi(argv[5]);

    if (width <= 0 || height <= 0 || maxIterations <= 0) {
        cerr << "Error: width, height, and maxIterations must be positive.\n";
        return 1;
    }

    vector<uint8_t> gray;
    readraw(inputPath, gray, width, height, 1);

    vector<uint8_t> binary = binarizeImage(gray, width, height);
    writeraw(outputPrefix + "_binary.raw", binaryToRaw(binary));

    vector<int> saveIters = {1, 5, 10, 15, 20};

    for (int iter = 1; iter <= maxIterations; iter++) {
        vector<uint8_t> prevBinary = binary;
        thinningIteration(binary, width, height);

        if (find(saveIters.begin(), saveIters.end(), iter) != saveIters.end()) {
            writeraw(outputPrefix + "_thin_iter" + to_string(iter) + ".raw",
                     binaryToRaw(binary));
        }

        if (binary == prevBinary) {
            cout << "Converged at iteration: " << iter << endl;
            writeraw(outputPrefix + "_thin_final.raw", binaryToRaw(binary));
            return 0;
        }
    }

    writeraw(outputPrefix + "_thin_final.raw", binaryToRaw(binary));
    cout << "Reached max iterations without earlier convergence.\n";
    return 0;
}
