// File: hw1/src/p1/p1b/p1b_A.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <string>

using namespace std;

// Read raw image function 
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

// Write raw image function 
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

// Clamp function
static inline int clamp(int x, int low, int high) {
    return max(low, min(high, x));
}

// Function to write histogram
static void writeHisto(const string& filename, const vector<long long>& hist) {
    ofstream out(filename);
    if (!out) { cerr << "Error: cannot open " << filename << endl; exit(1); }
    out << "intensity,count\n";
    for (int k = 0; k < 256; ++k) out << k << "," << hist[k] << "\n";
    out.close();
}

// Function to write transfer function
static void writeTrans(const string& filename, const vector<int>& T) {
    ofstream out(filename);
    if (!out) { cerr << "Error: cannot open " << filename << endl; exit(1); }
    out << "intensity,mapped\n";
    for (int k = 0; k < 256; ++k) out << k << "," << T[k] << "\n";
    out.close();
}
// Main function
int main(int argc, char* argv[]) {
    if (argc != 6) {
        cerr << "Usage: p1b_A input_airplane.raw output_A.raw hist_original.csv hist_A.csv tf_A.csv\n";
        return 1;
    }

    const string input = argv[1];
    const string output = argv[2];
    const string histOgCSV = argv[3];
    const string histOutCSV  = argv[4];
    const string tfCSV = argv[5];

    const int width = 1024;
    const int height = 1024;
    const int channels = 1;
    const int dim = width * height;
    const int level = 256;

    vector<uint8_t> img;
    readraw(input, img, width, height, channels);


    vector<long long> hist(level, 0);
    for (int i = 0; i < dim; ++i) hist[img[i]]++;

    vector<long long> cdf(level, 0);
    cdf[0] = hist[0];
    for (int k = 1; k < level; ++k) cdf[k] = cdf[k - 1] + hist[k];
 
    long long cdf_min = 0;
    for (int k = 0; k < level; ++k) {
        if (hist[k] != 0) { cdf_min = cdf[k]; break; }
    }

    vector<int> T(level, 0);
    for (int k = 0; k < level; ++k) {
        if (dim == (int)cdf_min) {
            T[k] = 0;
        } else {
            double val = (double)(cdf[k] - cdf_min) / (double)(dim - cdf_min);
            int mapped = (int)lround((level - 1) * val);
            T[k] = clamp(mapped, 0, 255);
        }
    }

    vector<uint8_t> out(dim);
    for (int i = 0; i < dim; ++i) out[i] = (uint8_t)T[img[i]];

    writeraw(output, out);
    vector<long long> histO(level, 0);
    for (int i = 0; i < dim; ++i) histO[out[i]]++;

    writeHisto(histOgCSV, hist);
    writeHisto(histOutCSV,  histO);
    writeTrans(tfCSV, T);

    auto [minIt, maxIt] = minmax_element(out.begin(), out.end());
    cout << "p1b_A done.\n";
    cout << "Output min: " << (int)(*minIt) << " max: " << (int)(*maxIt) << "\n";
    cout << "Wrote: " << output << "\n";
    cout << "CSVs: " << histOgCSV << ", " << histOutCSV << ", " << tfCSV << "\n";
    return 0;
}
