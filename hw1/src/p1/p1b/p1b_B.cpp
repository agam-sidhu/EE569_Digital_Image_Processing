/*
(1) Name: Agam Sidhu
(2) USC ID: 3027948957
(3) USC Email: agamsidh@usc.edu
(4) Submission Date: February 1, 2026
EE569 HW1 - Problem 1(b): Histogram Manipulation 
*/
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <algorithm>
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
//Function to write the CDF
static void writeCD(const string& filename,
                        const vector<long long>& cdf,
                        long long dim)
{
    ofstream out(filename);
    if (!out) { cerr << "Error: cannot open " << filename << endl; exit(1); }
    out << "intensity,cdf\n";
    for (int k = 0; k < 256; ++k) {
        double p = (double)cdf[k] / (double)dim;  
        out << k << "," << p << "\n";
    }
    out.close();
}

// Main function
int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: p1b_B input_airplane.raw output_B.raw cdf_before.csv cdf_after.csv\n";
        return 1;
    }

    const string input = argv[1];
    const string output = argv[2];
    const string before= argv[3];
    const string after  = argv[4];

    const int width = 1024;
    const int height = 1024;
    const int channels = 1;
    const int dim = width * height;
    const int level = 256;

    vector<uint8_t> img;
    vector<long long> histogram(level, 0);
    vector<long long> cdf(level, 0), cdfLow(level, 0);
    vector<long long> check(level, 0);
    vector<uint8_t> out(dim);

    readraw(input, img, width, height, channels);
    for (int i = 0; i < dim; ++i) histogram[img[i]]++;

    cdf[0] = histogram[0];
    for (int intensity = 1; intensity < level; ++intensity) {
        cdf[intensity] = cdf[intensity - 1] + histogram[intensity];
    }

    for (int intensity = 0; intensity < level; ++intensity) {
        cdfLow[intensity] = cdf[intensity] - histogram[intensity];
    }

    for (int idx = 0; idx < dim; ++idx) {
        const int intensity = static_cast<int>(img[idx]);
        const long long rank = cdfLow[intensity] + check[intensity];
        check[intensity]++;
        int mapped = static_cast<int>((static_cast<long long>(level) * rank) / dim);
        if (mapped >= level) mapped = level - 1; 
        out[idx] = static_cast<uint8_t>(mapped);
    }
    writeraw(output, out);

    vector<long long> histO(level, 0);
    for (int i = 0; i < dim; ++i) histO[out[i]]++;

    vector<long long> cdfO(level, 0);
    cdfO[0] = histO[0];
    for (int k = 1; k < level; ++k) cdfO[k] = cdfO[k - 1] + histO[k];

    writeCD(before, cdf, dim);
    writeCD(after,  cdfO, dim);

    auto [minIt, maxIt] = minmax_element(out.begin(), out.end());
    cout << "p1b_B done.\n";
    cout << "Output min: " << (int)(*minIt) << " max: " << (int)(*maxIt) << "\n";
    cout << "Wrote: " << output << "\n";
    cout << "CDF CSVs: " << before << ", " << after << "\n";
    return 0;
}
