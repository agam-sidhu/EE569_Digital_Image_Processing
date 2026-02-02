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
#include <algorithm>
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

//Clamp function
inline int clamp(int val, int low, int high) {
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
    int countPixels = width * height;

    for (int i = 0; i < countPixels; i++) {
        double diff = double(og[i]) - double(output[i]);
        mse += diff * diff;
    }

    mse /= countPixels;
    if (mse == 0.0) {
        return 1e9;
    }

    return 10.0 * log10((255.0 * 255.0) / mse);
}


//Function to calculate PSNR (RGB)
void calc_psnr_color(const vector<unsigned char>& og,
                   const vector<unsigned char>& output,
                   int width,
                   int height,
                   double& psnrR,
                   double& psnrG,
                   double& psnrB,
                   double& avg_psnr)
{
    double mseR = 0.0;
    double mseG = 0.0;
    double mseB = 0.0;
    int pixelCount = width * height;

    for (int i = 0; i < pixelCount; i++) {
        int id = i * 3;

        double diffR = double(og[id + 0]) - double(output[id + 0]);
        double diffG = double(og[id + 1]) - double(output[id + 1]);
        double diffB = double(og[id + 2]) - double(output[id + 2]);
        mseR += diffR * diffR;
        mseG += diffG * diffG;
        mseB += diffB * diffB;
    }

    mseR /= pixelCount;
    mseG /= pixelCount;
    mseB /= pixelCount;

    auto mse_psnr = [](double mse) -> double {
        if (mse <= 0.0){
            return 1e9;
        }
        return 10.0 * log10((255.0 * 255.0) / mse);
    };

    psnrR = mse_psnr(mseR);
    psnrG = mse_psnr(mseG);
    psnrB = mse_psnr(mseB);
    avg_psnr = (psnrR + psnrG + psnrB) / 3.0;
}

//Median filter function for RGB
vector<unsigned char> median(const vector<unsigned char>& input,
                                 int width,
                                 int height,
                                 int k)
{
    if (k <= 0 || (k % 2 == 0)) {
        cerr << "Error: medianK is not positive and odd" << endl;
        exit(1);
    }

    int halfSize = k / 2;
    vector<unsigned char> output(width * height * 3, 0);

    vector<unsigned char> window;
    window.reserve(k * k);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {

            for (int ch = 0; ch < 3; ch++) {
                window.clear();
                for (int i = -halfSize; i <= halfSize; i++) {
                    int nbrr = clamp(row + i, 0, height - 1);
                    for (int j = -halfSize; j <= halfSize; j++) {
                        int nbrc = clamp(col + j, 0, width - 1);
                        int id = (nbrr * width + nbrc) * 3 + ch;
                        window.push_back(input[id]);
                    }
                }
                int mid = (int)window.size() / 2;
                nth_element(window.begin(), window.begin() + mid, window.end());
                unsigned char med = window[mid];

                int outIdx = (row * width + col) * 3 + ch;
                output[outIdx] = med;
            }
        }
    }
    return output;
}

//Gaussian kernel creator
vector<double> gaussKernel(int k, double sigma)
{
    if (k <= 0 || (k % 2 == 0)) {
        cerr << "Error: gaussK is not positive and odd" << endl;
        exit(1);
    }
    if (sigma <= 0.0) {
        cerr << "Error: sigma must be greater thab 0" << endl;
        exit(1);
    }

    int halfSize = k / 2;
    vector<double> kernel(k * k, 0.0);

    double denom = 2.0 * sigma * sigma;
    double sum = 0.0;

    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            double dist = i * i + j * j;
            double val = exp(-dist / denom);
            kernel[(i + halfSize) * k + (j + halfSize)] = val;
            sum += val;
        }
    }
    for (double& v : kernel) {
        v /= sum;
    }

    return kernel;
}
//Gaussian filter function for RGB
vector<unsigned char> gaussColor(const vector<unsigned char>& input,
                                int width,
                                int height,
                                int k,
                                double sigma)
{
    vector<double> kernel = gaussKernel(k, sigma);
    int halfSize = k / 2;

    vector<unsigned char> output(width * height * 3, 0);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            for (int ch = 0; ch < 3; ch++) {

                double sum = 0.0;   // ✅ accumulator in double

                for (int i = -halfSize; i <= halfSize; i++) {
                    int nbrr = clamp(row + i, 0, height - 1);
                    for (int j = -halfSize; j <= halfSize; j++) {
                        int nbrc = clamp(col + j, 0, width - 1);

                        double w = kernel[(i + halfSize) * k + (j + halfSize)];
                        int id = (nbrr * width + nbrc) * 3 + ch;

                        sum += w * double(input[id]);
                    }
                }

                int outputIdx = (row * width + col) * 3 + ch;
                int outputVal = (int)round(sum);   
                output[outputIdx] = (unsigned char)clamp(outputVal, 0, 255);
            }
        }
    }
    return output;
}
//NLM Denoising function for RGB
vector<unsigned char> nlm(const vector<unsigned char>& input,
                              int width,
                              int height,
                              double h,
                              int windowSize,
                              int searchWindow)
{
    if (windowSize <= 0 || (windowSize % 2 == 0) ||
        searchWindow <= 0 || (searchWindow % 2 == 0)) {
        cerr << "Error: windowSize and searchWindow must be positive and odd" << endl;
        exit(1);
    }
    if (h <= 0.0) {
        cerr << "Error: h must be > 0" << endl;
        exit(1);
    }
    int patchRad = windowSize / 2;
    int searchRad = searchWindow / 2;

    vector<unsigned char> output(width * height * 3, 0);
    double h2 = h * h;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {

            double weightSum = 0.0;
            double sumR = 0.0;
            double sumG = 0.0;
            double sumB = 0.0;

            for (int rowOffset = -searchRad; rowOffset <= searchRad; rowOffset++) {
                int nbrRow = clamp(row + rowOffset, 0, height - 1);

                for (int colOffset = -searchRad; colOffset <= searchRad; colOffset++) {
                    int nbrCol = clamp(col + colOffset, 0, width - 1);

                    // patch distance between (row,col) and (nbrRow,nbrCol)
                    double patchDist = 0.0;

                    for (int prOffset = -patchRad; prOffset <= patchRad; prOffset++) {
                        int r1 = clamp(row + prOffset, 0, height - 1);
                        int r2 = clamp(nbrRow + prOffset, 0, height - 1);

                        for (int pcOffset = -patchRad; pcOffset <= patchRad; pcOffset++) {
                            int c1 = clamp(col + pcOffset, 0, width - 1);
                            int c2 = clamp(nbrCol + pcOffset, 0, width - 1);

                            int idx1 = (r1 * width + c1) * 3;
                            int idx2 = (r2 * width + c2) * 3;

                            for (int ch = 0; ch < 3; ch++) {
                                double a = double(input[idx1 + ch]);
                                double b = double(input[idx2 + ch]);
                                double diff = a - b;
                                patchDist += diff * diff;
                            }
                        }
                    }
                    double w = exp(-patchDist / h2);
                    int nbrIdx = (nbrRow * width + nbrCol) * 3;
                    sumR += w * double(input[nbrIdx + 0]);
                    sumG += w * double(input[nbrIdx + 1]);
                    sumB += w * double(input[nbrIdx + 2]);
                    weightSum += w;
                }
            }

            int outputIdx = (row * width + col) * 3;

            int outR;
            int outG;
            int outB;
            if (weightSum > 0.0) {
                outR = (int)round(sumR / weightSum);
                outG = (int)round(sumG / weightSum);
                outB = (int)round(sumB / weightSum);
            } else {
                outR = (int)input[outputIdx + 0];
                outG = (int)input[outputIdx + 1];
                outB = (int)input[outputIdx + 2];
            }
            output[outputIdx + 0] = (unsigned char)clamp(outR, 0, 255);
            output[outputIdx + 1] = (unsigned char)clamp(outG, 0, 255);
            output[outputIdx + 2] = (unsigned char)clamp(outB, 0, 255);
        }
    }

    return output;
}

//Main function
int main(int argc, char** argv) {
    if (argc < 9) {
        cout << "Usage:\n"
             << "  " << argv[0] << " og.raw noisy.raw out.raw width height medianK gauss gaussK sigma\n"
             << "  " << argv[0] << " og.raw noisy.raw out.raw width height medianK nlm h windowSize searchWindow\n";
        return 1;
    }
    string ogPath  = argv[1];
    string noise = argv[2];
    string outputPath   = argv[3];

    int width = stoi(argv[4]);
    int height = stoi(argv[5]);

    int medianK = stoi(argv[6]);
    string filter = argv[7];

    if (medianK <= 0 || (medianK % 2 == 0)) {
        cerr << "Error: medianK is not positive and odd" << endl;
        return 1;
    }

    vector<unsigned char> og;
    vector<unsigned char> no;

    readraw(ogPath, og, width, height, 3);
    readraw(noise, no, width, height, 3);

    
    vector<unsigned char> med = median(no, width, height, medianK);
    vector<unsigned char> out;

    if (filter == "gauss") {
        if (argc < 10) {
            cerr << "Error: Gaussian mode needs gaussK sigma" << endl;
            return 1;
        }

        int gaussK = stoi(argv[8]);
        double sigma = stod(argv[9]);

        if (gaussK <= 0 || (gaussK % 2 == 0)) {
            cerr << "Error: gaussK is not positive and odd" << endl;
            return 1;
        }
        if (sigma <= 0.0) {
            cerr << "Error: sigma must be greater than 0" << endl;
            return 1;
        }

        // Step 2: Gaussian (Gaussian noise)
        out = gaussColor(med, width, height, gaussK, sigma);

    } else if (filter == "nlm") {
        if (argc < 11) {
            cerr << "Error: NLM mode needs h windowSize searchWindow" << endl;
            return 1;
        }

        double h = stod(argv[8]);
        int windowSize = stoi(argv[9]);
        int searchWindow = stoi(argv[10]);

        // Step 2: NLM (fine grain)
        out = nlm(med, width, height, h, windowSize, searchWindow);

    } else {
        cerr << "Error: method must be 'gauss' or 'nlm'" << endl;
        return 1;
    }

    writeraw(outputPath, out);

    double psnrR, psnrG, psnrB, avg_psnr;
    calc_psnr_color(og, out, width, height, psnrR, psnrG, psnrB, avg_psnr);

    cout << "PSNR_R = " << psnrR << " dB" << endl;
    cout << "PSNR_G = " << psnrG << " dB" << endl;
    cout << "PSNR_B = " << psnrB << " dB" << endl;
    cout << "PSNR_avg = " << avg_psnr << " dB" << endl;

    return 0;
}
