/*
 * Name: Agam Sidhu
 * USC ID: XXXXXXXXXX
 * Email: agamsidhu@usc.edu
 * Submission Date: January 29, 2026
 *
 * EE569 Digital Image Processing
 * Homework #1
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <cstdlib>

using namespace std;

//Read raw image function
void readraw(const string& filename,
             vector<unsigned char>& buffer,
             int width,
             int height,
             int channels)
{
    int totalBytes = width * height * channels;
    buffer.resize(totalBytes);

    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Can't open input file " << filename << endl;
        exit(1);
    }

    file.read(reinterpret_cast<char*>(buffer.data()), totalBytes);

    if (!file) {
        cerr << "Error: failed to read" << filename << endl;
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

    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());

    if (!file) {
        cerr << "Error: failed to write raw file " << filename << endl;
        exit(1);
    }

    file.close();
}

//Function to calculate PSNR
double calc_psnr(const vector<unsigned char>& og,
                 const vector<unsigned char>& output,
                 int width,
                 int height)
{
    double mse = 0.0;
    int pixelCount= width * height;

    for (int i = 0; i < pixelCount; i++) {
        double diff = og[i] - output[i];
        mse += diff * diff;
    }
    mse /= pixelCount;
    if (mse == 0.0) {
        return 1e9;
    }
    return 10.0 * log10((255.0 * 255.0) / mse);
}


//Clamp function
inline int clamp(int val, int low, int high) {
    if (val < low) {
        return low;
    }
    if (val > high) {
        return high;
    }
    return val;
}

//NLM Denoising function
vector<unsigned char> nlm(const vector<unsigned char>& input,
                          int width,
                          int height,
                          double h,
                          int windowSize,
                          int searchWindow,
                          double a)
{
    int patchRad = windowSize / 2;
    int searchRad = searchWindow / 2;

    vector<unsigned char> output(width * height, 0);

    double h2 = h * h;
    if (h2 <= 0.0) {
        return input;
    }

    if (a <= 0.0) {
        a = 1.0;
    }
    double a2 = a * a;

    vector<double> Ga(windowSize * windowSize, 0.0);

    // ✅ add gsum
    double gsum = 0.0;
    for (int prOffset = -patchRad; prOffset <= patchRad; prOffset++) {
        for (int pcOffset = -patchRad; pcOffset <= patchRad; pcOffset++) {
            double g = exp(-(prOffset * prOffset + pcOffset * pcOffset) / (2.0 * a2));
            Ga[(prOffset + patchRad) * windowSize + (pcOffset + patchRad)] = g;
            gsum += g;
        }
    }
    if (gsum <= 0.0) gsum = 1.0;

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double wsum = 0.0;
            double weightedSum = 0.0;

            for (int rowOffset = -searchRad; rowOffset <= searchRad; rowOffset++) {
                int yNbr = clamp(r + rowOffset, 0, height - 1);

                for (int colOffset = -searchRad; colOffset <= searchRad; colOffset++) {
                    int xNbr = clamp(c + colOffset, 0, width - 1);

                    // ✅ avoid self patch dominating
                    if (yNbr == r && xNbr == c) continue;

                    double patchDist = 0.0;

                    for (int prOffset = -patchRad; prOffset <= patchRad; prOffset++) {
                        int pRow1 = clamp(r + prOffset, 0, height - 1);
                        int pRow2 = clamp(yNbr + prOffset, 0, height - 1);

                        for (int pcOffset = -patchRad; pcOffset <= patchRad; pcOffset++) {
                            int pCol1 = clamp(c + pcOffset, 0, width - 1);
                            int pCol2 = clamp(xNbr + pcOffset, 0, width - 1);

                            double pix1 = double(input[pRow1 * width + pCol1]);
                            double pix2 = double(input[pRow2 * width + pCol2]);
                            double diff = pix1 - pix2;

                            double g = Ga[(prOffset + patchRad) * windowSize + (pcOffset + patchRad)];
                            patchDist += g * diff * diff;
                        }
                    }

                    // ✅ normalize distance
                    patchDist /= gsum;

                    double w = exp(-patchDist / h2);
                    double neighbor = double(input[yNbr * width + xNbr]);
                    wsum += w;
                    weightedSum += w * neighbor;
                }
            }

            int outputVal;
            if (wsum > 0.0) {
                outputVal = (int)round(weightedSum / wsum);
            } else {
                outputVal = (int)input[r * width + c];
            }
            output[r * width + c] = (unsigned char)clamp(outputVal, 0, 255);
        }
    }

    return output;
}




//Main function
int main(int argc, char** argv) {
    if (argc < 10) {
        cout << "Usage:\n"
             << argv[0]
             << " og.raw noisy.raw out.raw width height h windowSize searchWindow a\n";
        return 1;
    }

    string ogPath = argv[1];
    string noise = argv[2];
    string outputPath = argv[3];

    int width = stoi(argv[4]);
    int height = stoi(argv[5]);

    double h = stod(argv[6]);
    int windowSize = stoi(argv[7]); 
    int searchWindow = stoi(argv[8]);
    double a = stod(argv[9]);
    if (a <= 0.0) {
        cerr << "Error: a must be > 0\n";
        return 1;
    }   

    if (windowSize <= 0 || (windowSize % 2 == 0) ||
        searchWindow <= 0 || (searchWindow % 2 == 0)) {
        cerr << "Error: windowSize and searchWindow is not positive and odd" << endl;
        return 1;
    }
    if (h <= 0.0) {
        cerr << "Error: h must be greater than 0" << endl;
        return 1;
    }

    vector<unsigned char> og;
    vector<unsigned char> no;

    readraw(ogPath, og, width, height, 1);
    readraw(noise, no, width, height, 1);

    vector<unsigned char> denoise = nlm(no, width, height, h, windowSize, searchWindow, a);
    writeraw(outputPath, denoise);

    double psnr = calc_psnr(og, denoise, width, height);
    cout << "PSNR = " << psnr << " dB" << endl;

    return 0;
}
