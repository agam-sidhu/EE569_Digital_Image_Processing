// p1a.cpp - Sobel Edge Detector (EE569 HW2 P1a)
// Name:
// USC ID:
// USC Email:
// Submission Date:

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

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

//Normalize Function
static vector<uint8_t> normalizeMinMax(const vector<float>& input) {
    float minVal = input[0];
    float maxVal = input[0];

    for (float value : input) {
        if (value < minVal){
            minVal = value; 
        } 
        if (value > maxVal){
            maxVal = value;
        }
    }
    vector<uint8_t> output(input.size(), 0);

    if (fabs(maxVal - minVal) < 1e-12f) {
        return output;
    }
    for (size_t i = 0; i < input.size(); i++) {
        float norm = (input[i] - minVal) / (maxVal - minVal);
        int pixelVal = (int)lroundf(norm * 255.0f);
        output[i] = (uint8_t)clamp(pixelVal, 0, 255);
    }
    return output;
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        cerr << "Usage:\n"
             << "  " << argv[0] << " <input_rgb_raw> <output> <width> <height> <percent>\n\n"
             << "Example:\n"
             << "  " << argv[0] << " hw2/images/Bird.raw hw2/outputs/p1/p1a 481 321 20\n";
        return 1;
    }

    const string input = argv[1];
    const string output = argv[2];
    const int width  = atoi(argv[3]);
    const int height = atoi(argv[4]);
    const float percent = (float)atof(argv[5]);

    if (width <= 0 || height <= 0) {
        cerr << "Error: invalid width or height.\n";
        return 1;
    }
    if (percent < 0.0f || percent > 100.0f) {
        cerr << "Error: Percent is not in [0,100].\n";
        return 1;
    }
    vector<uint8_t> rgb;
    readraw(input, rgb, width, height, 3);

    const int numPixels = width * height;

    vector<uint8_t> gray(numPixels, 0);
    for (int i = 0; i < numPixels; i++) {
        const uint8_t R = rgb[3*i + 0];
        const uint8_t G = rgb[3*i + 1];
        const uint8_t B = rgb[3*i + 2];
        float y = 0.2989f * (float)R + 0.5870f * (float)G + 0.1140f * (float)B;
        int yi = (int)lroundf(y);
        gray[i] = (uint8_t)clamp(yi, 0, 255);
    }
    const int xVal[3][3] = {
        { -1,  0,  1},
        { -2,  0,  2},
        { -1,  0,  1}
    };
    const int yVal[3][3] = {
        {  1,  2,  1},
        {  0,  0,  0},
        { -1, -2, -1}
    };

    vector<float> gx(numPixels, 0.0f), gy(numPixels, 0.0f), mag(numPixels, 0.0f);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int xSum = 0;
            int ySum = 0;
            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    int xx = clamp(col + i, 0, width - 1);
                    int yy = clamp(row + j, 0, height - 1);
                    uint8_t pixel = gray[yy * width + xx];
                    xSum += xVal[j+1][i+1] * (int)pixel;
                    ySum += yVal[j+1][i+1] * (int)pixel;
                }
            }
            int idx = row * width + col;
            gx[idx] = (float)xSum;
            gy[idx] = (float)ySum;
            mag[idx] = sqrtf(gx[idx]*gx[idx] + gy[idx]*gy[idx]);
        }
    }

    vector<uint8_t> gxNorm = normalizeMinMax(gx);
    vector<uint8_t> gyNorm = normalizeMinMax(gy);
    vector<uint8_t> magNorm = normalizeMinMax(mag);

    const int T = (int)lroundf((percent / 100.0f) * 255.0f);
    vector<uint8_t> edge(numPixels, 255);
    for (int i = 0; i < numPixels; i++) {
        if ((int)magNorm[i] >= T) edge[i] = 0;
        else edge[i] = 255;
    }
    const string stem = getBaseName(input);

    writeraw(output + "/" + stem + "_gray.raw", gray);
    writeraw(output + "/" + stem + "_gx.raw", gxNorm);
    writeraw(output + "/" + stem + "_gy.raw", gyNorm);
    writeraw(output + "/" + stem + "_mag.raw", magNorm);

    int t = (int)lroundf(percent);
    writeraw(output + "/" + stem + "_edge_T" + to_string(t) + ".raw", edge);
    cout << "Wrote outputs to: " << output << "\n";
    cout << "Threshold (percent): " << percent << " -> T=" << T << " on [0,255] magnitude\n";
    return 0;
}