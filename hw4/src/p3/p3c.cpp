/* EE569 Homework #4
 * Name: Agam Sidhu
 * USC ID: 3027948957
 * USC Email: agamsidh@usc.edu
 * Submission Date: [fill]
 * Problem 3(c): Bag of Words
 */

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "p3_sift_common.h"

using namespace std;

struct KMeans {
    vector<int> labels;
    vector<double> centroids;
};

//struct to store img names
struct ImagePair {
    string leftName;
    string rightName;
};

//struct to store img metadata (name + filepath)
struct ImgData {
    string name;
    string path;
};

//Struct to store extracted feature descriptors
struct ImgFeatData {
    string imgName;
    cv::Mat desc;
};

//Function to calculate the distance between descriptor and centroid
static double calcDescriptDist(const cv::Mat& desc1,
                               int row1,
                               const vector<double>& centroids,
                               int featDim,
                               int clusterIdx)
{
    double sum = 0.0;
    for (int c = 0; c < desc1.cols; c++) {
        double diff = static_cast<double>(desc1.at<float>(row1, c)) -
                      centroids[clusterIdx * featDim + c];
        sum += diff * diff;
    }
    double dist = sqrt(sum);
    return dist;
}

//Function to calculate the L2 distance bettwen 2 histograms
static double calcHistDist(const vector<double>& hist1,
                           const vector<double>& hist2)
{
    double sum = 0.0;
    //calculates the squared dist
    for (size_t i = 0; i < hist1.size(); i++) {
        double diff = hist1[i] - hist2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

//Function to stack all img descriptors -> 1 matrix
static cv::Mat stackDesc(const vector<ImgFeatData>& featSet)
{
    int totalRows = 0;
    int featDim = 0;
    //calculates the total # of descriptors rows + feature dimension
    for (size_t i = 0; i < featSet.size(); i++) {
        totalRows += featSet[i].desc.rows;
        if (!featSet[i].desc.empty()) {
            featDim = featSet[i].desc.cols;
        }
    }
    //returns empty matrix if no descriptors found
    if (totalRows == 0 || featDim == 0) {
        return cv::Mat();
    }
    //store all descriptors
    cv::Mat descStore(totalRows, featDim, CV_32F);
    int rowStart = 0;
    //copy descriptors into stacked matrix
    for (size_t idx = 0; idx < featSet.size(); idx++) {
        if (featSet[idx].desc.empty()) {
            continue;
        }

        featSet[idx].desc.copyTo(
            descStore.rowRange(rowStart, rowStart + featSet[idx].desc.rows));
        rowStart += featSet[idx].desc.rows;
    }

    return descStore;
}

//Function to flatten descriptor matrix into feature data
static vector<double> matToFeatData(const cv::Mat& data)
{
    vector<double> featData(data.rows * data.cols, 0.0);

    for (int row = 0; row < data.rows; row++) {
        for (int col = 0; col < data.cols; col++) {
            featData[row * data.cols + col] = static_cast<double>(data.at<float>(row, col));
        }
    }

    return featData;
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
                for (int dim = 0; dim < featDim; dim++) {
                    int idx = i * featDim + dim;
                    double diff = featData[idx] - result.centroids[c * featDim + dim];
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

//Function to find the closest centroid match
static int findBestMatch(const cv::Mat& queryDesc,
                         const vector<double>& centroids,
                         int featDim,
                         int clusterK,
                         double& bestDist)
{
    //intialize the best distance + index
    bestDist = numeric_limits<double>::max();
    int bestIdx = -1;

    //checks the query descriptor vs every centroid
    for (int idx = 0; idx < clusterK; idx++) {
        // compute the distance from query desc to centroid
        double currDist = calcDescriptDist(queryDesc,
                                           0,
                                           centroids,
                                           featDim,
                                           idx);
        if (currDist < bestDist) {
            bestDist = currDist;
            bestIdx = idx;
        }
    }

    return bestIdx;
}
//Function to build a normalized histogram 
static vector<double> makeNormHist(const cv::Mat& desc,
                                               const vector<double>& centroids,
                                               int featDim,
                                               int clusterK)
{
    //intializes the historgram (w/ one bin per visual)
    vector<double> hist(clusterK, 0.0);
    //returns empty hist if no descriptor
    if (desc.empty()) {
        return hist;
    }
    //sets each descriptor to its closest centroid
    for (int row = 0; row < desc.rows; row++) {
        cv::Mat queryDesc = desc.row(row);
        double bestDist = 0.0;
        int bestCluster = findBestMatch(queryDesc, centroids, featDim, clusterK, bestDist);
        hist[bestCluster] += 1.0;
    }

    //normalize the histogram by total # of descriptors
    for (size_t i = 0; i < hist.size(); i++) {
        hist[i] /= static_cast<double>(desc.rows);
    }

    return hist;
}
//Function to write the histogram -> CSV
static void writeToCSV(const string& filename,
                              const vector<ImgFeatData>& featSet,
                              const vector< vector<double> >& histograms)
{
    ofstream file(filename.c_str());
    if (!file) {
        cerr << "Error: cannot open histogram CSV file " << filename << endl;
        exit(1);
    }
    //header of csv
    file << "filename";
    for (int bin = 0; bin < 8; bin++) {
        file << ",bin" << bin;
    }
    file << "\n";
    //writes 1 histogram per image
    for (size_t binIdx = 0; binIdx < featSet.size(); binIdx++) {
        file << featSet[binIdx].imgName;
        for (size_t bin = 0; bin < histograms[binIdx].size(); bin++) {
            file << "," << fixed << setprecision(10) << histograms[binIdx][bin];
        }
        file << "\n";
    }

    file.close();
}
//Function to write comparison distance -> txt file 
static void writeCompare(const string& filename,
                                double cat3Cat1,
                                double cat3Cat2,
                                double cat3Dog1)
{
    ofstream file(filename.c_str());
    if (!file) {
        cerr << "Error: cannot open comparison text file " << filename << endl;
        exit(1);
    }

    file << fixed << setprecision(10);
    file << "Problem 3(c) Bag of Words Comparison\n";
    file << "----------------------------------------\n";
    file << "Cat_3 vs Cat_1: " << cat3Cat1 << "\n";
    file << "Cat_3 vs Cat_2: " << cat3Cat2 << "\n";
    file << "Cat_3 vs Dog_1: " << cat3Dog1 << "\n";

    file.close();
}

//Main Function
int main(int argc, char* argv[])
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <hw4_root_path>\n";
        return 1;
    }

    string hw4Root = argv[1];
    string matDir = hw4Root + "/EE569_2026Spring_HW4_materials";
    string outDir = hw4Root + "/outputs/p3";

    //make output directories
    ensureDir(hw4Root + "/outputs");
    ensureDir(outDir);

    const int width = 600;
    const int height = 400;
    const int clusterK = 8;
    const int maxIter = 100;

    //store the image set
    vector<ImgData> imgSet;
    imgSet.push_back({"Cat_1.raw", matDir + "/Cat_1.raw"});
    imgSet.push_back({"Cat_2.raw", matDir + "/Cat_2.raw"});
    imgSet.push_back({"Cat_3.raw", matDir + "/Cat_3.raw"});
    imgSet.push_back({"Dog_1.raw", matDir + "/Dog_1.raw"});

    //store the comparison pairs
    vector<ImagePair> pairs;
    pairs.push_back({"Cat_3.raw", "Cat_1.raw"});
    pairs.push_back({"Cat_3.raw", "Cat_2.raw"});
    pairs.push_back({"Cat_3.raw", "Dog_1.raw"});

    vector<ImgFeatData> featSet; //stores descriptor data

    //extracts the SIFT keypoints + descriptors
    for (size_t idx = 0; idx < imgSet.size(); idx++) {
        cv::Mat img = readrawRGB(imgSet[idx].path, width, height);

        vector<cv::KeyPoint> kp;
        cv::Mat desc;
        extractSIFT(img, kp, desc);

        ImgFeatData curr;
        curr.imgName = imgSet[idx].name;
        curr.desc = desc;
        featSet.push_back(curr);

        cout << "Extracted " << desc.rows
             << " descriptors from " << imgSet[idx].name << endl;
    }
    //stacks all descriptors -> 1 matrix
    cv::Mat allDesc = stackDesc(featSet);
    if (allDesc.empty()) {
        cerr << "Error: no SIFT descriptors were extracted." << endl;
        return 1;
    }

    //build the 8-word codebook using all descriptors
    int numSamples = allDesc.rows;
    int featDim = allDesc.cols;
    vector<double> featData = matToFeatData(allDesc);
    KMeans result = kmRunner(featData,
                             numSamples,
                             featDim,
                             clusterK,
                             maxIter);

    //build one normalized histogram for each image
    vector< vector<double> > histSet;
    map<string, int> imgIndex;
    for (size_t idx = 0; idx < featSet.size(); idx++) {
        histSet.push_back(makeNormHist(featSet[idx].desc,
                                                      result.centroids,
                                                      featDim,
                                                      clusterK));
        imgIndex[featSet[idx].imgName] = static_cast<int>(idx);
    }
    //computes the histogram disances between image pairs
    vector<double> pairDist;
    for (size_t pIdx = 0; pIdx < pairs.size(); pIdx++) {
        double dist = calcHistDist(histSet[imgIndex[pairs[pIdx].leftName]],
                                   histSet[imgIndex[pairs[pIdx].rightName]]);
        pairDist.push_back(dist);
    }

    double cat3Cat1 = pairDist[0];
    double cat3Cat2 = pairDist[1];
    double cat3Dog1 = pairDist[2];

    string outFile = outDir + "/p3c_bow_histograms.csv";
    string summaryFile = outDir + "/p3c_bow_comparison.txt";
    //write the histogram CSV
    writeToCSV(outFile, featSet, histSet);
    writeCompare(summaryFile, cat3Cat1, cat3Cat2, cat3Dog1);
    //print summary
    cout << fixed << setprecision(10);
    for (size_t i = 0; i < pairs.size(); i++) {
        cout << pairs[i].leftName.substr(0, pairs[i].leftName.size() - 4)
             << " vs "
             << pairs[i].rightName.substr(0, pairs[i].rightName.size() - 4)
             << ": "
             << pairDist[i] << endl;
    }
    cout << "\nGenerated files:" << endl;
    cout << outFile << endl;
    cout << summaryFile << endl;

    return 0;
}
