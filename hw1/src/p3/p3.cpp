/*
Name:
USC ID:
USC Email:
Submission Date:

EE569 HW1 - Problem 3: Auto White Balancing (Grey World)
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdlib>

using namespace std;
// Clamp function
static inline unsigned char clamp(double x) {
    if (x < 0.0) return 0;
    if (x > 255.0) return 255;
    return static_cast<unsigned char>(x + 0.5);
}

//Read raw image function
void readraw(const string& filename,
             vector<unsigned char>& buffer,
             int width,
             int height,
             int channels)
{
    int byteCount = width * height * channels;
    buffer.resize(byteCount);

    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Can't open input file " << filename << endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(buffer.data()), byteCount);

    if (!file) {
        cerr << "Error: failed to read " << filename << endl;
        exit(1);
    }

    file.close();
}

//Write raw image function
void writeraw(const string& filename,
              const vector<unsigned char>& buffer)
{
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: cannot open output file " << filename << endl;
        exit(1);
    }

    file.write(reinterpret_cast<const char*>(buffer.data()),
               static_cast<streamsize>(buffer.size()));

    if (!file) {
        cerr << "Error: failed to write raw file " << filename << endl;
        exit(1);
    }

    file.close();
}

void writeHisto(const string& filename,
                       const vector<int>& bef_R,
                       const vector<int>& bef_G,
                       const vector<int>& bef_B,
                       const vector<int>& aft_R,
                       const vector<int>& aft_G,
                       const vector<int>& aft_B)
{
    ofstream file(filename);
    if (!file) {
        cerr << "Error: Can't open histogra " << filename << endl;
        exit(1);
    }
    file << "value,R_before,G_before,B_before,R_after,G_after,B_after\n";
    for (int i = 0; i < 256; i++) {
        file << i << ","
             << bef_R[i] << ","
             << bef_G[i] << ","
             << bef_B[i] << ","
             << aft_R[i] << ","
             << aft_G[i] << ","
             << aft_B[i] << "\n";
    }
    file.close();
}
// Function to get histogram of an RGB image
void getHisto(const vector<unsigned char>& img,
                      vector<int>& r,
                      vector<int>& g,
                      vector<int>& b)
{
    r.assign(256, 0);
    g.assign(256, 0);
    b.assign(256, 0);

    int pixelCount = img.size() / 3;
    for (int i = 0; i < pixelCount; i++) {
        r[img[3*i + 0]]++;
        g[img[3*i + 1]]++;
        b[img[3*i + 2]]++;
    }
}
// Main function
int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0]
             << " input.raw output.raw width height" << endl;
        return 1;
    }

    const string inputPath  = argv[1];
    const string outputPath = argv[2];
    const int width  = atoi(argv[3]);
    const int height = atoi(argv[4]);

    if (width <= 0 || height <= 0) {
        cerr << "Error: width & height needs to be positive." << endl;
        return 1;
    }

    const int channels = 3;
    const int dim = width * height;

    vector<unsigned char> input;
    readraw(inputPath, input, width, height, channels);
    vector<unsigned char> output(input.size());
    vector<int> bef_R, bef_G, bef_B;
    getHisto(input, bef_R, bef_G, bef_B);

    double rSum = 0.0;
    double gSum = 0.0;
    double bSum = 0.0;

    for (int i = 0; i < dim; i++) {
        rSum += input[3*i + 0];
        gSum += input[3*i + 1];
        bSum += input[3*i + 2];
    }

    const double rMean = rSum / static_cast<double>(dim);
    const double gMean = gSum / static_cast<double>(dim);
    const double bMean = bSum / static_cast<double>(dim);
    const double mean  = (rMean + gMean + bMean) / 3.0;

    const double eps = 1e-12;
    double rVal, gVal, bVal;
    if (rMean > eps) {
        rVal = mean / rMean;
    } else {
        rVal = 1.0;
    }

    if (gMean > eps) {
        gVal = mean / gMean;
    } else {
        gVal = 1.0;
    }

    if (bMean > eps) {
        bVal = mean / bMean;
    } else {
        bVal = 1.0;
    }

    for (int i = 0; i < dim; i++) {
        unsigned char R = input[3*i + 0];
        unsigned char G = input[3*i + 1];
        unsigned char B = input[3*i + 2];

        output[3*i + 0] = clamp(rVal * static_cast<double>(R));
        output[3*i + 1] = clamp(gVal * static_cast<double>(G));
        output[3*i + 2] = clamp(bVal * static_cast<double>(B));
    }

    writeraw(outputPath, output);
    vector<int> aft_R, aft_G, aft_B;
    getHisto(output, aft_R, aft_G, aft_B);
    /* 
    writeHisto(
        "../../Image_hw1/outputs/p3_histogram.csv",
        bef_R, bef_G, bef_B,
        aft_R,  aft_G,  aft_B
    );
    */

    double rsum = 0.0, gsum = 0.0, bsum = 0.0;
    for (int i = 0; i < dim; i++) {
        rsum += output[3*i + 0];
        gsum += output[3*i + 1];
        bsum += output[3*i + 2];
    }

    const double rmean = rsum / static_cast<double>(dim);
    const double gmean = gsum / static_cast<double>(dim);
    const double bmean = bsum / static_cast<double>(dim);

    cout << fixed << setprecision(4);
    cout << "Before AWB means: muR=" << rMean
         << ", muG=" << gMean
         << ", muB=" << bMean << endl;
    cout << "Target gray mean: mu=" << mean << endl;
    cout << "Gains: aR=" << rVal
         << ", aG=" << gVal
         << ", aB=" << bVal << endl;
    cout << "After AWB means:  muR=" << rmean
         << ", muG=" << gmean
         << ", muB=" << bmean << endl;
    return 0;
}
