/*
(1) Name: Agam Sidhu
(2) USC ID: 3027948957
(3) USC Email: agamsidh@usc.edu
(4) Submission Date: February 1, 2026
EE569 HW1 - Problem 2(b): Bilateral filtering 
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

    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());

    if (!file) {
        cerr << "Error: failed to write raw file " << filename << endl;
        exit(1);
    }

    file.close();
}

//Clamp function
int clamp(int val, int low, int high) {
    if (val < low) return low;
    if (val > high) return high;
    return val;
}

//Function to calculate PSNR
double calc_psnr(const vector<unsigned char>& og,
                 const vector<unsigned char>& output,
                 int width,
                 int height)
{
    double mse = 0.0;
    int pixelCount = width * height;

    for (int i = 0; i < pixelCount; i++) {
        double diff = double(og[i]) - double(output[i]);
        mse += diff * diff;
    }
    mse /= pixelCount;
    if (mse == 0.0) {
        return 1e9;
    }
    return 10.0 * log10((255.0 * 255.0) / mse);
}
//Bilateral filter function
vector<unsigned char> bilat(const vector<unsigned char>& input,
                                     int width,
                                     int height,
                                     int k,
                                     double sigs,
                                     double sigc)
{
    if (k <= 0 || (k % 2 == 0)) {
        cerr << "Error: k is not positive and odd" << endl;
        exit(1);
    }
    if (sigs <= 0 || sigc <= 0) {
        cerr << "Error: sigs and sigc must be greater than 0" << endl;
        exit(1);
    }

    int r = k / 2;
    vector<unsigned char> output(width * height, 0);

    vector<double> spat(k * k, 0.0);
    double s_denom = 2.0 * sigs * sigs;

    for (int i = -r; i <= r; i++) {
        for (int j = -r; j <= r; j++) {
            spat[(i + r) * k + (j + r)] = exp(-(i * i + j * j) / s_denom);
        }
    }

    double c_denom = 2.0 * sigc * sigc;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            double centerVal = double(input[y * width + x]);
            double wsum = 0.0;
            double weightedSum = 0.0;

            for (int i = -r; i<= r; i++) {
                int yNbr = clamp(y + i, 0, height - 1);
                for (int j = -r; j <= r; j++) {
                    int xNbr = clamp(x + j, 0, width - 1);

                    double neighbor = double(input[yNbr * width + xNbr]);
                    double delta = neighbor - centerVal;

                    double weightedS = spat[(i + r) * k + (j + r)];
                    double weightedC = exp(-(delta * delta) / c_denom);
                    double w = weightedS * weightedC;

                    wsum += w;
                    weightedSum += w * neighbor;
                }
            }
            int outputVal;
            if (wsum > 0.0) {
                outputVal = int(round(weightedSum / wsum));
            } else {
                outputVal = int(centerVal);
            }
            output[y * width + x] = (unsigned char) clamp(outputVal, 0, 255);
        }
    }
    return output;
}

//Main function
int main(int argc, char** argv) {
    if (argc < 9) {
        cout << "Usage:\n  " << argv[0]
             << " og.raw noisy.raw out.raw width height k sigs sigc\n";
        return 1;
    }

    string ogPath  = argv[1];
    string noise = argv[2];
    string outputPath   = argv[3];

    int width = stoi(argv[4]);
    int height = stoi(argv[5]);
    int k = stoi(argv[6]);
    double sigs = stod(argv[7]);
    double sigc = stod(argv[8]);

    vector<unsigned char> og;
    vector<unsigned char> noisy;

    readraw(ogPath, og, width, height, 1);
    readraw(noise, noisy, width, height, 1);

    vector<unsigned char> filt = bilat(noisy, width, height, k, sigs, sigc);
    writeraw(outputPath, filt);
    double psnr = calc_psnr(og, filt, width, height);
    cout << "PSNR = " << psnr << " dB" << endl;
    return 0;
}
