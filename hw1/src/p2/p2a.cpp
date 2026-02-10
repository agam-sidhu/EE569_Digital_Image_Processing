/*
(1) Name: Agam Sidhu
(2) USC ID: 3027948957
(3) USC Email: agamsidh@usc.edu
(4) Submission Date: February 1, 2026
EE569 HW1 - Problem 2(a): Basic denoising methods 
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
    int pixelCount = width * height;

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
//Uniform filter creator
vector<double> uniform(int k) {
    int size = k * k;
    double val = 1.0 / size;
    return vector<double>(size, val);
}
//Gaussian filter creator
vector<double> gaussian(int k, double sigma) {
    int halfSize = k / 2;
    vector<double> gaussKernel(k * k);
    double weightSum = 0.0;

    for (int y = -halfSize; y <= halfSize; y++) {
        for (int x = -halfSize; x <= halfSize; x++) {
            double dist = x * x + y * y;
            double denom = 2.0 * sigma * sigma;
            double val = exp(-dist / (denom));
            gaussKernel[(y + halfSize) * k + (x + halfSize)] = val;
            weightSum += val;
        }
    }
    for (double& val : gaussKernel){
        val  /= weightSum;
    }
    return gaussKernel;
}

//Convolution function
vector<unsigned char> convolve(const vector<unsigned char>& input,
                               const vector<double>& kernel,
                               int width,
                               int height,
                               int k)
{
    int halfSize = k / 2;
    vector<unsigned char> output(width * height);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double weightedSum = 0.0;
            for (int rowOffset = -halfSize; rowOffset <= halfSize; rowOffset++) {
                for (int x = -halfSize; x <= halfSize; x++) {

                    int nbrr = clamp(r + rowOffset, 0, height - 1);
                    int nbrc = clamp(c + x, 0, width - 1);

                    int kernRow = rowOffset + halfSize;
                    int kernCol = x + halfSize;

                    double w = kernel[kernRow * k + kernCol];
                    unsigned char pixel = input[nbrr * width + nbrc];

                    weightedSum += w * pixel;
                }
            }
            int norm= (int)round(weightedSum);
            output[r * width + c] = (unsigned char)clamp(norm, 0, 255);
        }
    }

    return output;
}

//Main function
int main(int argc, char** argv) {
    if (argc < 8) {
        cout << "Usage:\n"
             << argv[0]
             << " og.raw noisy.raw out.raw width height uniform|gauss k [sigma]\n";
        return 1;
    }

    string ogPath = argv[1];
    string noise = argv[2];
    string outputPath = argv[3];

    int width = stoi(argv[4]);
    int height = stoi(argv[5]);
    string filter = argv[6];
    int k = stoi(argv[7]);

    vector<unsigned char> og;
    vector<unsigned char> no;

    readraw(ogPath, og, width, height, 1);
    readraw(noise, no, width, height, 1);

    vector<double> kernel;
    if (filter == "uniform") {
        kernel = uniform(k);
    } else if (filter == "gauss") {
        if (argc < 9) {
            cout << "Usage (Gaussian):\n";
            cout << argv[0] << " og.raw noisy.raw out.raw width height gauss k sigma\n";
            return 1;
        }
        double sig = stod(argv[8]);
        kernel = gaussian(k, sig);
    } else {
        cout << "Must be 'uniform' or 'gauss'\n";
        return 1;
    }

    vector<unsigned char> finalFilt = convolve(no, kernel, width, height, k);
    writeraw(outputPath, finalFilt);

    double psnr = calc_psnr(og, finalFilt, width, height);
    cout << "PSNR = " << psnr << " dB" << endl;
    return 0;
}
