/* EE569 Homework #4
 * Name: Agam Sidhu
 * USC ID: 3027948957
 * USC Email: agamsidh@usc.edu
 * Submission Date: [fill]
 * Shared Helper for Problem 2
 */

#ifndef EE569_HW4_P2_TEXTURE_COMMON_H
#define EE569_HW4_P2_TEXTURE_COMMON_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

using namespace std;

struct KMeansResultFlat {
    vector<int> labels;
    vector<double> centroids;
};

static inline int clampInt(int x, int low, int high)
{
    return max(low, min(high, x));
}

static inline void ensureDir(const string& path)
{
    mkdir(path.c_str(), 0755);
}

static inline void readraw(const string& filename,
                           vector<uint8_t>& buffer,
                           int width,
                           int height,
                           int channels)
{
    int byteCount = width * height * channels;
    buffer.resize(byteCount);

    ifstream file(filename.c_str(), ios::binary);
    if (!file) {
        cerr << "Error: cannot open input file " << filename << endl;
        exit(1);
    }

    file.read(reinterpret_cast<char*>(buffer.data()), byteCount);
    if (!file) {
        cerr << "Error: failed to read bytes from " << filename << endl;
        exit(1);
    }

    file.close();
}

static inline void writeraw(const string& filename,
                            const vector<uint8_t>& buffer)
{
    ofstream file(filename.c_str(), ios::binary);
    if (!file) {
        cerr << "Error: cannot open output file " << filename << endl;
        exit(1);
    }

    file.write(reinterpret_cast<const char*>(buffer.data()),
               static_cast<streamsize>(buffer.size()));
    if (!file) {
        cerr << "Error: failed to write bytes to " << filename << endl;
        exit(1);
    }

    file.close();
}

static inline void readGray(const string& filename,
                            vector<double>& image,
                            int width,
                            int height)
{
    vector<uint8_t> buffer;
    readraw(filename, buffer, width, height, 1);

    image.resize(width * height);
    for (int i = 0; i < width * height; i++) {
        image[i] = static_cast<double>(buffer[i]);
    }
}

static inline double getPixGray(const vector<double>& image,
                                int width,
                                int height,
                                int row,
                                int col)
{
    row = clampInt(row, 0, height - 1);
    col = clampInt(col, 0, width - 1);
    return image[row * width + col];
}

static inline void meanSub(vector<double>& image)
{
    double sum = 0.0;
    for (size_t i = 0; i < image.size(); i++) {
        sum += image[i];
    }

    double meanVal = sum / static_cast<double>(image.size());
    for (size_t i = 0; i < image.size(); i++) {
        image[i] -= meanVal;
    }
}

static inline void convolveFive(const vector<double>& image,
                                vector<double>& result,
                                int width,
                                int height,
                                const double kernel[5][5])
{
    result.assign(width * height, 0.0);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            double sum = 0.0;
            for (int kr = -2; kr <= 2; kr++) {
                for (int kc = -2; kc <= 2; kc++) {
                    double pix = getPixGray(image, width, height, row + kr, col + kc);
                    sum += pix * kernel[kr + 2][kc + 2];
                }
            }
            result[row * width + col] = sum;
        }
    }
}

static inline void genLFs(vector< vector< vector<double> > >& filters)
{
    const double L5[5] = {1, 4, 6, 4, 1};
    const double E5[5] = {-1, -2, 0, 2, 1};
    const double S5[5] = {-1, 0, 2, 0, -1};
    const double W5[5] = {-1, 2, 0, -2, 1};
    const double R5[5] = {1, -4, 6, -4, 1};

    const double* kernels[5] = {L5, E5, S5, W5, R5};
    filters.clear();

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            vector< vector<double> > kernel(5, vector<double>(5, 0.0));
            for (int r = 0; r < 5; r++) {
                for (int c = 0; c < 5; c++) {
                    kernel[r][c] = kernels[i][r] * kernels[j][c];
                }
            }
            filters.push_back(kernel);
        }
    }
}

static inline vector<double> buildIntegral(const vector<double>& data,
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

static inline double rectMean(const vector<double>& integral,
                              int width,
                              int height,
                              int row0,
                              int col0,
                              int row1,
                              int col1)
{
    row0 = clampInt(row0, 0, height - 1);
    col0 = clampInt(col0, 0, width - 1);
    row1 = clampInt(row1, 0, height - 1);
    col1 = clampInt(col1, 0, width - 1);

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

static inline void buildNormalizedLawsFeatures(const vector<double>& image,
                                               int width,
                                               int height,
                                               int windowSize,
                                               vector<double>& features,
                                               int& featDim)
{
    vector<double> zeroMean = image;
    meanSub(zeroMean);

    vector< vector< vector<double> > > filters;
    genLFs(filters);

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

        vector<double> response;
        convolveFive(zeroMean, response, width, height, kernel);
        vector<double> integral = buildIntegral(response, width, height);

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

static inline void calcMeanStdFlat(const vector<double>& data,
                                   int n,
                                   int dim,
                                   vector<double>& mean,
                                   vector<double>& stdv)
{
    mean.assign(dim, 0.0);
    stdv.assign(dim, 0.0);

    for (int i = 0; i < n; i++) {
        for (int d = 0; d < dim; d++) {
            mean[d] += data[i * dim + d];
        }
    }

    for (int d = 0; d < dim; d++) {
        mean[d] /= static_cast<double>(n);
    }

    for (int i = 0; i < n; i++) {
        for (int d = 0; d < dim; d++) {
            double diff = data[i * dim + d] - mean[d];
            stdv[d] += diff * diff;
        }
    }

    for (int d = 0; d < dim; d++) {
        stdv[d] = sqrt(stdv[d] / static_cast<double>(n));
        if (stdv[d] < 1e-12) {
            stdv[d] = 1.0;
        }
    }
}

static inline void normalizeFlat(vector<double>& data,
                                 int n,
                                 int dim,
                                 const vector<double>& mean,
                                 const vector<double>& stdv)
{
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < dim; d++) {
            data[i * dim + d] = (data[i * dim + d] - mean[d]) / stdv[d];
        }
    }
}

static inline KMeansResultFlat runKMeansFlat(const vector<double>& data,
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

static inline vector<uint8_t> labelsToGray(const vector<int>& labels,
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

static inline vector< vector<double> > calcCovariance(const vector<double>& data,
                                                      int n,
                                                      int dim)
{
    vector< vector<double> > cov(dim, vector<double>(dim, 0.0));
    vector<double> mean(dim, 0.0);

    for (int i = 0; i < n; i++) {
        for (int d = 0; d < dim; d++) {
            mean[d] += data[i * dim + d];
        }
    }

    for (int d = 0; d < dim; d++) {
        mean[d] /= static_cast<double>(n);
    }

    for (int i = 0; i < n; i++) {
        for (int r = 0; r < dim; r++) {
            for (int c = 0; c < dim; c++) {
                cov[r][c] += (data[i * dim + r] - mean[r]) *
                             (data[i * dim + c] - mean[c]);
            }
        }
    }

    double denom = (n > 1) ? static_cast<double>(n - 1) : 1.0;
    for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++) {
            cov[r][c] /= denom;
        }
    }

    return cov;
}

static inline vector< vector<double> > identityMat(int n)
{
    vector< vector<double> > id(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        id[i][i] = 1.0;
    }
    return id;
}

static inline void jacobiDecomp(const vector< vector<double> >& input,
                                vector<double>& eigenvalues,
                                vector< vector<double> >& eigenvectors)
{
    int n = static_cast<int>(input.size());
    vector< vector<double> > mat = input;
    eigenvectors = identityMat(n);

    int maxIter = 100 * n * n;
    double eps = 1e-10;

    for (int iter = 0; iter < maxIter; iter++) {
        int p = 0;
        int q = 1;
        double maxOffDiag = 0.0;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double val = fabs(mat[i][j]);
                if (val > maxOffDiag) {
                    maxOffDiag = val;
                    p = i;
                    q = j;
                }
            }
        }

        if (maxOffDiag < eps) {
            break;
        }

        double app = mat[p][p];
        double aqq = mat[q][q];
        double apq = mat[p][q];

        double phi = 0.5 * atan2(2.0 * apq, aqq - app);
        double cs = cos(phi);
        double sn = sin(phi);

        for (int k = 0; k < n; k++) {
            if (k != p && k != q) {
                double mkp = mat[k][p];
                double mkq = mat[k][q];
                mat[k][p] = cs * mkp - sn * mkq;
                mat[p][k] = mat[k][p];
                mat[k][q] = sn * mkp + cs * mkq;
                mat[q][k] = mat[k][q];
            }
        }

        mat[p][p] = cs * cs * app - 2.0 * cs * sn * apq + sn * sn * aqq;
        mat[q][q] = sn * sn * app + 2.0 * cs * sn * apq + cs * cs * aqq;
        mat[p][q] = 0.0;
        mat[q][p] = 0.0;

        for (int k = 0; k < n; k++) {
            double vkp = eigenvectors[k][p];
            double vkq = eigenvectors[k][q];
            eigenvectors[k][p] = cs * vkp - sn * vkq;
            eigenvectors[k][q] = sn * vkp + cs * vkq;
        }
    }

    eigenvalues.assign(n, 0.0);
    for (int i = 0; i < n; i++) {
        eigenvalues[i] = mat[i][i];
    }
}

static inline void sortEigen(vector<double>& eigenvalues,
                             vector< vector<double> >& eigenvectors)
{
    int dim = static_cast<int>(eigenvalues.size());
    vector<int> order(dim, 0);
    for (int i = 0; i < dim; i++) {
        order[i] = i;
    }

    sort(order.begin(), order.end(),
         [&](int a, int b) {
             return eigenvalues[a] > eigenvalues[b];
         });

    vector<double> sortedValues(dim, 0.0);
    vector< vector<double> > sortedVecs(dim, vector<double>(dim, 0.0));

    for (int i = 0; i < dim; i++) {
        sortedValues[i] = eigenvalues[order[i]];
        for (int r = 0; r < dim; r++) {
            sortedVecs[r][i] = eigenvectors[r][order[i]];
        }
    }

    eigenvalues = sortedValues;
    eigenvectors = sortedVecs;
}

static inline vector<double> projectDataFlat(const vector<double>& data,
                                             int n,
                                             int inputDim,
                                             const vector< vector<double> >& eigenvectors,
                                             int outDim)
{
    vector<double> projected(n * outDim, 0.0);

    for (int i = 0; i < n; i++) {
        for (int pc = 0; pc < outDim; pc++) {
            double sum = 0.0;
            for (int d = 0; d < inputDim; d++) {
                sum += data[i * inputDim + d] * eigenvectors[d][pc];
            }
            projected[i * outDim + pc] = sum;
        }
    }

    return projected;
}

#endif
