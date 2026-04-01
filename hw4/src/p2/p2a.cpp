/* EE569 Homework #4
 * Name: Agam Sidhu
 * USC ID: 3027948957
 * USC Email: agamsidh@usc.edu
 * Submission Date: [fill]
 * Problem 2(a): Basic Texture Segmentation
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <string>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

struct KMeansResultFlat {
    vector<int> labels;
    vector<double> centroids;
};

static void ensureDir(const string& path)
{
    mkdir(path.c_str(), 0755);
}

//Clamp function
static inline int clamp(int x, int low, int high) {
    return max(low, min(high, x));
}

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

//Function to get gray pixel value while keeping coordinates in range
static double getPixGray(const vector<double>& img,
                         int width, int height,
                         int row, int col)
{
    row = clamp(row, 0, height - 1);
    col = clamp(col, 0, width - 1);
    return img[row * width + col];
}

//Function to convert the grayscale img to a double vec
static void readGray(const string& filename,
                                vector<double>& image,
                                int width,
                                int height)
{
    vector<uint8_t> buff;
    readraw(filename, buff, width, height, 1);

    //rezie the img vector & convert pixel vals -> double
    image.resize(width * height);
    for (int i = 0; i < width * height; i++) {
        image[i] = static_cast<double>(buff[i]);
    }
}

//Subtract image mean intensity 
static void meanSub(vector<double>& image)
{

    double res = 0.0;
    //Computes the sum of all pix vals
    for (size_t i = 0; i < image.size(); i++) {
        res += image[i];
    }
    //gets the mean intensity 
    double meanIntenisty = res / static_cast<double>(image.size());

    for (size_t i = 0; i < image.size(); i++) {
        //subtract the mean intensity from each pixel val
        image[i] -= meanIntenisty;
    }
}

//Function to convolve the 5 x 5 kernel 
static void convolveFive(const vector<double>& image,
                        vector<double>& res,
                        int width,
                        int height,
                        const double kernel[5][5])
{
    //intialize output 
    res.assign(width * height, 0.0);

    //go through each pixel location 
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            double cSum = 0.0;

            //apply 5 x 5 kernel that is centered at (row,col)
            for (int kernRow = -2; kernRow <= 2; kernRow++) {
                for (int kernCol = -2; kernCol <= 2; kernCol++) {
                    //pixel val w/ clamp
                    double pixVal = getPixGray(image, width, height, row + kernRow, col + kernCol);
                    //multiply with the kernel weight
                    cSum += pixVal * kernel[kernRow + 2][kernCol + 2];
                }
            }
            //get curr pixel index & stores solution there
            int idx = row * width + col;
            res[idx] = cSum;
        }
    }
}

//Generates the 25 laws filters 
static void genLFs(vector< vector< vector<double> > >& filters,
                             vector<string>& filterNames)
{
    //Setup the 1D kernels for L5, E5, S5, W5, R5
    const double L5[5] = { 1,  4,  6,  4,  1};
    const double E5[5] = {-1, -2,  0,  2,  1};
    const double S5[5] = {-1,  0,  2,  0, -1};
    const double W5[5] = {-1,  2,  0, -2,  1};
    const double R5[5] = { 1, -4,  6, -4,  1};

    //Stores kernel pointers + names 
    const double* oneDKern[5] = {L5, E5, S5, W5, R5};
    const string kerns[5] = {"L5", "E5", "S5", "W5", "R5"};

    //clearing output containers
    filters.clear();
    filterNames.clear();

    for (int kernRow = 0; kernRow < 5; kernRow++) {
        for (int kernCol = 0; kernCol < 5; kernCol++) {
            //simple intialize one 5x5 filter
            vector< vector<double> > kernel(5, vector<double>(5, 0.0));

            //computer outer product 
            for (int r = 0; r < 5; r++) {
                for (int c = 0; c < 5; c++) {
                    kernel[r][c] = oneDKern[kernRow][r] * oneDKern[kernCol][c];
                }
            }
            //save filter + name
            filters.push_back(kernel);
            filterNames.push_back(kerns[kernRow] + kerns[kernCol]);
        }
    }
}

static vector<double> buildIntegral(const vector<double>& data,
                                    int width,
                                    int height)
{
    vector<double> integral((width + 1) * (height + 1), 0.0);

    for (int row = 1; row <= height; row++) {
        double rowSum = 0.0;
        for (int col = 1; col <= width; col++) {
            rowSum += fabs(data[(row - 1) * width + (col - 1)]);
            integral[row * (width + 1) + col] =
                integral[(row - 1) * (width + 1) + col] + rowSum;
        }
    }

    return integral;
}

static double rectMean(const vector<double>& integral,
                       int width,
                       int height,
                       int row0,
                       int col0,
                       int row1,
                       int col1)
{
    row0 = clamp(row0, 0, height - 1);
    col0 = clamp(col0, 0, width - 1);
    row1 = clamp(row1, 0, height - 1);
    col1 = clamp(col1, 0, width - 1);

    if (row1 < row0) {
        swap(row0, row1);
    }
    if (col1 < col0) {
        swap(col0, col1);
    }

    int top = row0;
    int left = col0;
    int bottom = row1 + 1;
    int right = col1 + 1;

    double sum =
        integral[bottom * (width + 1) + right] -
        integral[top * (width + 1) + right] -
        integral[bottom * (width + 1) + left] +
        integral[top * (width + 1) + left];

    double area = static_cast<double>((row1 - row0 + 1) * (col1 - col0 + 1));
    return sum / area;
}

static void buildNormalizedLawsFeatures(const vector<double>& image,
                                        int width,
                                        int height,
                                        int windowSize,
                                        vector<double>& features,
                                        int& featDim)
{
    vector<double> zeroMean = image;
    meanSub(zeroMean);

    vector< vector< vector<double> > > filters;
    vector<string> filterNames;
    genLFs(filters, filterNames);

    int numPixels = width * height;
    int halfWindow = windowSize / 2;

    vector<double> baseEnergy(numPixels, 1.0);
    vector< vector<double> > energyMaps(24, vector<double>(numPixels, 0.0));

    for (size_t idx = 0; idx < filters.size(); idx++) {
        double kernel[5][5];
        for (int r = 0; r < 5; r++) {
            for (int c = 0; c < 5; c++) {
                kernel[r][c] = filters[idx][r][c];
            }
        }

        vector<double> result;
        convolveFive(zeroMean, result, width, height, kernel);
        vector<double> integral = buildIntegral(result, width, height);

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int idxPix = row * width + col;
                double energy = rectMean(integral,
                                         width,
                                         height,
                                         row - halfWindow,
                                         col - halfWindow,
                                         row + halfWindow,
                                         col + halfWindow);

                if (idx == 0) {
                    baseEnergy[idxPix] = energy;
                } else {
                    energyMaps[idx - 1][idxPix] = energy;
                }
            }
        }
    }

    featDim = 24;
    features.assign(numPixels * featDim, 0.0);
    for (int p = 0; p < numPixels; p++) {
        double denom = baseEnergy[p];
        if (denom < 1e-8) {
            denom = 1e-8;
        }

        for (int d = 0; d < featDim; d++) {
            features[p * featDim + d] = energyMaps[d][p] / denom;
        }
    }
}

static KMeansResultFlat runKMeansFlat(const vector<double>& data,
                                      int n,
                                      int dim,
                                      int k,
                                      int maxIter)
{
    KMeansResultFlat result;
    result.labels.assign(n, 0);
    result.centroids.assign(k * dim, 0.0);

    for (int c = 0; c < k; c++) {
        int sample = (c * n) / k;
        for (int d = 0; d < dim; d++) {
            result.centroids[c * dim + d] = data[sample * dim + d];
        }
    }

    for (int iter = 0; iter < maxIter; iter++) {
        bool changed = false;
        vector<double> sums(k * dim, 0.0);
        vector<int> counts(k, 0);

        for (int i = 0; i < n; i++) {
            int bestCluster = 0;
            double bestDist = numeric_limits<double>::max();

            for (int c = 0; c < k; c++) {
                double dist = 0.0;
                for (int d = 0; d < dim; d++) {
                    double diff = data[i * dim + d] - result.centroids[c * dim + d];
                    dist += diff * diff;
                }

                if (dist < bestDist) {
                    bestDist = dist;
                    bestCluster = c;
                }
            }

            if (result.labels[i] != bestCluster) {
                result.labels[i] = bestCluster;
                changed = true;
            }

            counts[bestCluster]++;
            for (int d = 0; d < dim; d++) {
                sums[bestCluster * dim + d] += data[i * dim + d];
            }
        }

        for (int c = 0; c < k; c++) {
            if (counts[c] == 0) {
                int sample = (c * n) / k;
                for (int d = 0; d < dim; d++) {
                    result.centroids[c * dim + d] = data[sample * dim + d];
                }
                continue;
            }

            for (int d = 0; d < dim; d++) {
                result.centroids[c * dim + d] =
                    sums[c * dim + d] / static_cast<double>(counts[c]);
            }
        }

        if (!changed) {
            break;
        }
    }

    return result;
}

static vector<uint8_t> labelsToGray(const vector<int>& labels,
                                    int width,
                                    int height,
                                    const vector<double>& centroids,
                                    int dim,
                                    int k)
{
    vector<int> order(k, 0);
    for (int i = 0; i < k; i++) {
        order[i] = i;
    }

    sort(order.begin(), order.end(),
         [&](int a, int b) {
             return centroids[a * dim] < centroids[b * dim];
         });

    vector<int> remap(k, 0);
    for (int i = 0; i < k; i++) {
        remap[order[i]] = i;
    }

    vector<uint8_t> gray(width * height, 0);
    for (int i = 0; i < width * height; i++) {
        int idx = remap[labels[i]];
        gray[i] = static_cast<uint8_t>((255 * idx) / max(1, k - 1));
    }

    return gray;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <hw4_root_path> [window_size]\n";
        return 1;
    }

    string hw4Root = argv[1];
    int windowSize = 31;
    if (argc >= 3) {
        windowSize = atoi(argv[2]);
        if (windowSize < 3) {
            windowSize = 3;
        }
        if (windowSize % 2 == 0) {
            windowSize++;
        }
    }

    const int width = 512;
    const int height = 512;
    const int k = 6;

    string inputFile = hw4Root + "/EE569_2026Spring_HW4_materials/Mosaic.raw";
    string outputDir = hw4Root + "/outputs/p2";
    string outputFile = outputDir + "/p2a_segmented.raw";

    ensureDir(hw4Root + "/outputs");
    ensureDir(outputDir);

    vector<double> image;
    readGray(inputFile, image, width, height);

    vector<double> features;
    int featDim = 0;
    buildNormalizedLawsFeatures(image, width, height, windowSize, features, featDim);

    int numPixels = width * height;
    KMeansResultFlat result = runKMeansFlat(features, numPixels, featDim, k, 20);

    vector<uint8_t> segmented = labelsToGray(result.labels,
                                             width,
                                             height,
                                             result.centroids,
                                             featDim,
                                             k);

    writeraw(outputFile, segmented);

    cout << "Saved segmented mosaic to " << outputFile << endl;
    return 0;
}
