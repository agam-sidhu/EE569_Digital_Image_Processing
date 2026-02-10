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
//1Function to write histogram
static void writeHisto(const string& filename, const vector<long long>& hist) {
    ofstream out(filename);
    if (!out) {
        cerr << "Error: cannot open CSV output file " << filename << endl;
        exit(1);
    }
    out << "intensity,count\n";
    for (int k = 0; k < 256; ++k) {
        out << k << "," << hist[k] << "\n";
    }
    out.close();
}

// Main function
int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: p1b_hist input_airplane.raw hist_original.csv\n";
        return 1;
    }

    const string input = argv[1];
    const string histogram = argv[2];
    const int width = 1024;
    const int height = 1024;
    const int channels = 1;
    const int dim = width * height;

    vector<uint8_t> img;
    vector<long long> hist(256, 0);
    readraw(input, img, width, height, channels);

    for (int idx = 0; idx < dim; ++idx) {
        const int intensity = static_cast<int>(img[idx]);
        hist[intensity]++;
    }
    writeHisto(histogram, hist);
    auto [minIt, maxIt] = minmax_element(img.begin(), img.end());
    cout << "p1b_hist done.\n";
    cout << "Min intensity: " << (int)(*minIt) << "  Max intensity: " << (int)(*maxIt) << "\n";
    cout << "Wrote: " << histogram << "\n";
    return 0;
}
