/* EE569 Homework #4
 * Name: Agam Sidhu
 * USC ID: 3027948957
 * USC Email: agamsidh@usc.edu
 * Submission Date: [fill]
 * Problem 2(b): Advanced Texture Segmentation
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <string>
#include <cstdlib>
#include <cmath>
#include <map>
#include <queue>
#include <limits>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/core.hpp>

using namespace std;

struct KMeans {
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

//Function make the inegral img of the abs filter response
static vector<double> calcInt(const vector<double>& result,
                                    int width,
                                    int height)
{
    vector<double> integral((width + 1) * (height + 1), 0.0);

    //makes image row by row
    for (int row = 1; row <= height; row++) {
        double rowSum = 0.0;

        for (int col = 1; col <= width; col++) {
            //gets the absolute response in row
            rowSum += fabs(result[(row - 1) * width + (col - 1)]);
            int idx = row * (width + 1) + col;
            int idxTop = (row - 1) * (width + 1) + col;
            integral[idx] = integral[idxTop] + rowSum;
        }
    }

    return integral;
}

//Function to calculate the mean abs energy 
static double calcAvg(const vector<double>& integral,
                       int width,
                       int height,
                       int intialRow,
                       int intialCol,
                       int finalRow,
                       int finalCol)
{
    //clamps the coordinates to keep them in valid range
    int iRow = clamp(intialRow, 0, height - 1);
    int iCol = clamp(intialCol, 0, width - 1);
    int fRow = clamp(finalRow, 0, height - 1);
    int fCol = clamp(finalCol, 0, width - 1);
    //swap to make sure initial <= final 
    if (fRow < iRow) {
        swap(iRow, fRow);
    }
    if (fCol < iCol) {
        swap(iCol, fCol);
    }
    //integral img coordinates
    int top = iRow;
    int left = iCol;
    int bottom = fRow + 1;
    int right = fCol + 1;

    double sum =
        integral[bottom * (width + 1) + right] -
        integral[top * (width + 1) + right] -
        integral[bottom * (width + 1) + left] +
        integral[top * (width + 1) + left];
    double area = static_cast<double>((fRow - iRow + 1) * (fCol - iCol + 1));
    return sum / area;
}

//Function to get normalized Law's Features for each pixel
static void getNormLF(const vector<double>& image,
                                        int width,
                                        int height,
                                        int windowSize,
                                        vector<double>& feat,
                                        int& featDim)
{
    //subtract the mean intensity from the image to get zero mean img
    vector<double> zeroMean = image;
    meanSub(zeroMean);

    //makes the 25 laws filters
    vector< vector< vector<double> > > filters;
    vector<string> filterNames;
    genLFs(filters, filterNames);

    int numPixels = width * height;
    int halfSize = windowSize / 2;
    //l5l5 = normalization factor
    vector<double> energyNorm(numPixels, 1.0);
    vector< vector<double> > energyMaps(24, vector<double>(numPixels, 0.0));
    featDim = static_cast<int>(filters.size()) - 1;

    //runs through each Law's filter
    for (size_t idx = 0; idx < filters.size(); idx++) {
        double kern[5][5];
        //copies filter into kernel array
        for (int r = 0; r < 5; r++) {
            for (int c = 0; c < 5; c++) {
                kern[r][c] = filters[idx][r][c];
            }
        }

        // convolve img w/ filter & get the integral img of abs response
        vector<double> res;
        convolveFive(zeroMean, res, width, height, kern);
        vector<double> integral = calcInt(res, width, height);

        //calculates the avg energy for each pixel
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int pixIdx = row * width + col;
                double energy = calcAvg(integral,
                                         width,
                                         height,
                                         row - halfSize,
                                         col - halfSize,
                                         row + halfSize,
                                         col + halfSize);
                
                //uses the l5l5 energy for normalization
                if (idx == 0) {
                    energyNorm[pixIdx] = energy;
                } else {
                    energyMaps[idx - 1][pixIdx] = energy;
                }
            }
        }
    }
    //need to flatten into numPixels * featDim
    feat.assign(numPixels * featDim, 0.0);
    for (int p = 0; p < numPixels; p++) {
        //prevents dividing by 0 
        double denom = energyNorm[p];
        if (denom < 1e-8) {
            denom = 1e-8;
        }
        for (int dim = 0; dim < featDim; dim++) {
            int idx = p * featDim + dim;
            feat[idx] = energyMaps[dim][p] / denom;
        }
    }
}

//Function to compute mean/std vec for flattened feature data
static void calcMeanStdFlat(const vector<double>& data,
                            int numSamples,
                            int dimNum,
                            vector<double>& mean,
                            vector<double>& std)
{
    //assigns mean & std
    mean.assign(dimNum, 0.0);
    std.assign(dimNum, 0.0);

    //calculates mean
    for (int i = 0; i < numSamples; i++) {
        for (int dim = 0; dim < dimNum; dim++) {
            mean[dim] += data[i * dimNum + dim];
        }
    }
    for (int d = 0; d < dimNum; d++) {
        mean[d] /= static_cast<double>(numSamples);
    }

    //calculates std
    for (int i = 0; i < numSamples; i++) {
        for (int d = 0; d < dimNum; d++) {
            double diff = data[i * dimNum + d] - mean[d];
            std[d] += diff * diff;
        }
    }
    //prevent divide by 0
    for (int d = 0; d < dimNum; d++) {
        std[d] = sqrt(std[d] / static_cast<double>(numSamples));
        if (std[d] < 1e-12) {
            std[d] = 1.0;
        }
    }
}

//Function to normalize flattened data (apply z-score norm)
static void normFlat(vector<double>& data,
                          int numSamples,
                          int dimNum,
                          const vector<double>& mean,
                          const vector<double>& std)
{
    //applies the z score
    for (int i = 0; i < numSamples; i++) {
        for (int dim = 0; dim < dimNum; dim++) {
            int idx = i * dimNum + dim;
            data[idx] = (data[idx] - mean[dim]) / std[dim];
        }
    }
}

//Funcion to convert vector into mat
static cv::Mat vecToMat(const vector<double>& data,
                        int numSamples,
                        int dimNum)
{
    cv::Mat matData(numSamples, dimNum, CV_64F);
    for (int i = 0; i < numSamples; i++) {
        for (int d = 0; d < dimNum; d++) {
            matData.at<double>(i, d) = data[i * dimNum + d];
        }
    }
    return matData;
}

//Function to convert data into vector
static vector<double> matToValVec(const cv::Mat& data)
{
    vector<double> val(data.rows * data.cols, 0.0);
    for (int row = 0; row < data.rows; row++) {
        for (int col = 0; col < data.cols; col++) {
            val[row * data.cols + col] = data.at<double>(row, col);
        }
    }
    return val;
}

//Function to run Kmeans algo on feature data
static KMeans kmRunner(const vector<double>& featData,
                                      int numSamples,
                                      int featDim,
                                      int clusterK,
                                      int maxVal)
{
    //set up labels + centroids 
    KMeans result;
    result.labels.assign(numSamples, 0);
    result.centroids.assign(clusterK * featDim, 0.0);

    //intializes the centroids by evenly sampling spaces
    for (int c = 0; c < clusterK; c++) {
        int sIdx= (c * numSamples) / clusterK;
        for (int d = 0; d < featDim; d++) {
            int idx = c * featDim + d;
            int featIdx = sIdx * featDim + d;
            result.centroids[idx] = featData[featIdx];
        }
    }
    //loop to run Kmeans until convergence or we hit max value (iterations)
    for (int iter = 0; iter < maxVal; iter++) {
        bool altered = false;
        //stores feature sums + counts for each cluster
        vector<double> cSum(clusterK * featDim, 0.0);
        vector<int> cCount(clusterK, 0);

        //goes through each sample and assigns to nearest centroid
        for (int i = 0; i < numSamples; i++) {
            int bestCluster = 0;
            double bestD = numeric_limits<double>::max();

            for (int c = 0; c < clusterK; c++) {
                double dist = 0.0;
                for (int d = 0; d < featDim; d++) {
                    int idx = i * featDim + d;
                    double diff = featData[idx] - result.centroids[c * featDim + d];
                    dist += diff * diff;
                }
                if (dist < bestD) {
                    bestD = dist;
                    bestCluster = c;
                }
            }
            //update label (if cluster changed)
            if (result.labels[i] != bestCluster) {
                result.labels[i] = bestCluster;
                altered = true;
            }
            //get sum for centroid update
            cCount[bestCluster]++;
            for (int d = 0; d < featDim; d++) {
                cSum[bestCluster * featDim + d] += featData[i * featDim + d];
            }
        }
        //recalculate centroid
        for (int c = 0; c < clusterK; c++) {
            if (cCount[c] == 0) {
                int sample = (c * numSamples) / clusterK;
                for (int dim = 0; dim < featDim; dim++) {
                    int idx = c * featDim + dim;
                    result.centroids[idx] = featData[sample * featDim + dim];
                }
                continue;
            }

            for (int dim = 0; dim < featDim; dim++) {
                int idx = c* featDim + dim;
                result.centroids[idx] = cSum[idx] / static_cast<double>(cCount[c]);
            }
        }
        //in case no sample changed cluster
        if (!altered) {
            break;
        }
    }

    return result;
}

//Function to convert cluster into grayscale
static vector<uint8_t> labelsToGray(const vector<int>& labels,
                                    int width,
                                    int height,
                                    const vector<double>& centroid,
                                    int featDim,
                                    int clusterK)
{
    //stores cluster order indices
    vector<int> orderIdx(clusterK, 0);
    for (int i = 0; i < clusterK; i++) {
        orderIdx[i] = i;
    }
    //sorts using centroid feature
    sort(orderIdx.begin(), orderIdx.end(),
         [&](int a, int b) {
             return centroid[a * featDim] < centroid[b * featDim];
         });
    //maps og cluster id -> sorted grayscale rank
    vector<int> remap(clusterK, 0);
    for (int i = 0; i < clusterK; i++) {
        remap[orderIdx[i]] = i;
    }
    //makes output img
    vector<uint8_t> grey(width * height, 0);
    for (int i = 0; i < width * height; i++) {
        int idx = remap[labels[i]];
        grey[i] = static_cast<uint8_t>((255 * idx) / max(1, clusterK - 1));
    }

    return grey;
}

//Function to run majority filtering
static void majorityFilter(vector<int>& labels,
                           int width,
                           int height,
                           int k,
                           int windowSize,
                           int numPasses)
{
    //halves the window size
    int halfWindow = windowSize / 2;
    //through each pass
    for (int pass = 0; pass < numPasses; pass++) {
        vector<int> filt = labels;
        //goes trhrough each pixel
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {

                vector<int> count(k, 0);
                for (int i = row - halfWindow; i <= row + halfWindow; i++) {
                    for (int  j = col - halfWindow; j <= col + halfWindow; j++) {
                        int r = clamp(i, 0, height - 1);
                        int c = clamp(j, 0, width - 1);
                        int idx = r * width + c;
                        count[labels[idx]]++;
                    }
                }
                // looks to find best label
                int idx = row * width + col;
                int current = labels[idx];
                int bestLabel = current;
                int bestCount = count[current];
                //sets best label
                for (int label = 0; label < k; label++) {
                    if (count[label] > bestCount) {
                        bestCount = count[label];
                        bestLabel = label;
                    }
                }
                idx = row * width + col;
                filt[idx] = bestLabel;
            }
        }

        labels = filt;
    }
}
//Function to calculate centroids
static vector<double> calcCentorids(const vector<int>& label,
                                                 const vector<double>& data,
                                                 int numSample,
                                                 int dimNum,
                                                 int k)
{
    vector<double> centroid(k * dimNum, 0.0);
    vector<int> count(k, 0);
    //runs through each sample
    for (int i = 0; i < numSample; i++) {
        int lb = label[i];
        count[lb]++;
        for (int dim = 0; dim < dimNum; dim++) {
            int idx = lb * dimNum + dim;
            centroid[idx] += data[i * dimNum + dim];
        }
    }
    //calculates the centroid
    for (int c = 0; c < k; c++) {
        if (count[c] == 0) {
            continue;
        }
        for (int dim = 0; dim < dimNum; dim++) {
            int idx = c * dimNum + dim;
            centroid[idx] /= static_cast<double>(count[c]);
        }
    }

    return centroid;
}
//Function to remove the small components
static void removeMini(vector<int>& labels,
                                  int width,
                                  int height,
                                  int minSize)
{
    vector<int> seen(width * height, 0);
     //4-connected neighborhood directions
    const int rowDir[4] = {-1, 1, 0, 0};
    const int colDir[4] = {0, 0, -1, 1};

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int intialIdx = row * width + col;
            //skip if already seen 
            if (seen[intialIdx]) {
                continue;
            }
            int ogLB = labels[intialIdx];
            queue<int> pixelQueue;
            vector<int> comp;
            map<int, int> nCount; // neighbor count

            seen[intialIdx] = 1;
            pixelQueue.push(intialIdx);

            while (!pixelQueue.empty()) {
                int idx = pixelQueue.front();
                pixelQueue.pop();
                comp.push_back(idx);

                int r = idx / width;
                int c = idx % width;

                //checks the 4 connected neighbors
                for (int dir = 0; dir < 4; dir++) {
                    int nextRow = r + rowDir[dir];
                    int nextCol = c + colDir[dir];
                    if (nextRow < 0 || nextRow >= height || nextCol < 0 || nextCol >= width) {
                        continue;
                    }
                    ///expand component if neighbor has same label
                    int nextId = nextRow * width + nextCol;
                    if (labels[nextId] == ogLB) {
                        if (!seen[nextId]) {
                            seen[nextId] = 1;
                            pixelQueue.push(nextId);
                        }
                    } else {
                        //count neighboring labls around this component
                        nCount[labels[nextId]]++;
                    }
                }
            }
            // keep comp if it is large enough
            if (static_cast<int>(comp.size()) >= minSize || nCount.empty()) {
                continue;
            }

            //find the neighboring label that appears most often
            int bestLB = ogLB;
            int bestCount = -1;
            for (map<int, int>::iterator it = nCount.begin(); it != nCount.end(); ++it) {
                if (it->second > bestCount) {
                    bestCount = it->second;
                    bestLB = it->first;
                }
            }
            //replace small component
            for (size_t i = 0; i < comp.size(); i++) {
                labels[comp[i]] = bestLB;
            }
        }
    }
}

static void refineBounds(vector<int>& labels,
                                 int width,
                                 int height,
                                 int k,
                                 const vector<double>& data,
                                 int dimNum)
{

    vector<double> centroids = calcCentorids(labels, data, width * height, dimNum, k);
    vector<int> refinedLB = labels;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = row * width + col;
            int curr = labels[idx];
            bool isBoundary = false;
            vector<int> candidate;
            vector<int> seen(k, 0);

            //looks at 3x3 window around the pixel
            for (int wRow = row - 1; wRow <= row + 1; wRow++) {
                for (int wCol = col - 1; wCol <= col + 1; wCol++) {
                    int r = clamp(wRow, 0, height - 1);
                    int c = clamp(wCol, 0, width - 1);
                    int nLB = labels[r * width + c]; //neighbor labels
                    //if any neighbor has a different label -> current pixel is on a boundary
                    if (nLB != curr) {
                        isBoundary = true;
                    }
                    //stores unique neighbor labels
                    if (!seen[nLB]) {
                        seen[nLB] = 1;
                        candidate.push_back(nLB);
                    }
                }
            }
            //skips any non boundary pixels
            if (!isBoundary) {
                continue;
            }
            //set boundary pixel to the nearest centroid
            int bestLB = curr;
            double bestDist = numeric_limits<double>::max();
            for (size_t i = 0; i < candidate.size(); i++) {
                int label = candidate[i];
                double dist = 0.0;
                for (int d = 0; d < dimNum; d++) {
                    int diffID = idx * dimNum + d; //data index
                    double diff = data[diffID] - centroids[label * dimNum + d];
                    dist += diff * diff;
                }
                //sets centroid that is closest
                if (dist < bestDist) {
                    bestDist = dist;
                    bestLB = label;
                }
            }

            refinedLB[idx] = bestLB;
        }
    }

    labels = refinedLB;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <hw4_root_path> [window_size]\n";
        return 1;
    }

    string hw4Root = argv[1];

    //default window size for Law's energy comp
    int windowSize = 31;
    vector<double> image;
    if (argc >= 3) {
        windowSize = atoi(argv[2]);

        if (windowSize < 3) {
            windowSize = 3;
        }

        if (windowSize % 2 == 0) {
            windowSize++;
        }
    }
    //default params
    const int width = 512;
    const int height = 512;
    const int clusterK = 6;
    const int maxVal = 20;
    const int refinedDim = 3;
    //setting up input/output directory
    string inputFile = hw4Root + "/EE569_2026Spring_HW4_materials/Mosaic.raw";
    string outRoot = hw4Root + "/outputs";
    string outDir = outRoot + "/p2";
    string pcaFile = outDir + "/p2b_pca_kmeans.raw";
    string postFile = outDir + "/p2b_postprocess.raw";
    string boundFile = outDir + "/p2b_boundary_refined.raw";
    string outFile = outDir + "/p2b_final.raw";

    //make output directories
    ensureDir(outRoot);
    ensureDir(outDir);

    //read the mosaic image
    readGray(inputFile, image, width, height);

    //extract normalized local Law's features for each pixel
    vector<double> feat;
    int featDim = 0;
    getNormLF(image, width, height, windowSize, feat, featDim);

    int numSamples = width * height;
    vector<double> featMean;
    vector<double> featStd;
    calcMeanStdFlat(feat, numSamples, featDim, featMean, featStd);
    normFlat(feat, numSamples, featDim, featMean, featStd);

    //convert the feature vec into Mat (for PCA)
    cv::Mat featMat = vecToMat(feat, numSamples, featDim);
    //run PCA 
    cv::PCA pca(featMat, cv::Mat(), cv::PCA::DATA_AS_ROW, refinedDim);
    cv::Mat proMat = pca.project(featMat);
    //flatten the projected PCA data back into vector form
    vector<double> proj = matToValVec(proMat);
    //runs k means
    KMeans result = kmRunner(proj, numSamples, refinedDim, clusterK, maxVal);
    vector<int> labels = result.labels;

    //Saves segmentation result
    vector<uint8_t> imgSeg = labelsToGray(labels,
                                          width,
                                          height,
                                          result.centroids,
                                          refinedDim,
                                          clusterK);
    writeraw(pcaFile, imgSeg);

    //Merge small components (sstep 2)
    majorityFilter(labels, width, height, clusterK, 3, 1);
    removeMini(labels, width, height, 96);

    //Save the result after the post-processing stage
    vector<double> postCent = calcCentorids(labels,
                                                              proj,
                                                              numSamples,
                                                              refinedDim,
                                                              clusterK);
    imgSeg = labelsToGray(labels,
                          width,
                          height,
                          postCent,
                          refinedDim,
                          clusterK);
    writeraw(postFile, imgSeg);

    //Uses projected data to refine bounds (step 3)
    refineBounds(labels, width, height, clusterK, proj, refinedDim);

    //Save the result right after boundary refinement
    vector<double> boundCents = calcCentorids(labels,
                                                                  proj,
                                                                  numSamples,
                                                                  refinedDim,
                                                                  clusterK);
    imgSeg = labelsToGray(labels,
                          width,
                          height,
                          boundCents,
                          refinedDim,
                          clusterK);
    writeraw(boundFile, imgSeg);

    majorityFilter(labels, width, height, clusterK, 3, 1);

    //Save the final results
    vector<double> finalCent = calcCentorids(labels,
                                                               proj,
                                                               numSamples,
                                                               refinedDim,
                                                               clusterK);
    imgSeg = labelsToGray(labels,
                          width,
                          height,
                          finalCent,
                          refinedDim,
                          clusterK);

    writeraw(outFile, imgSeg);

    cout << "Saved PCA + K-means result to " << pcaFile << endl;
    cout << "Saved post-processed result to " << postFile << endl;
    cout << "Saved boundary-refined result to " << boundFile << endl;
    cout << "Saved final improved segmented mosaic to " << outFile << endl;
    return 0;
}
