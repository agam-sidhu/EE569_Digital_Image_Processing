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

struct ImageSpec {
    string name;
    string path;
};

struct ImageFeatureData {
    string name;
    cv::Mat descriptors;
};

static double rowToCentroidDist(const cv::Mat& data,
                                int row,
                                const vector<double>& centroid)
{
    double sum = 0.0;
    for (int c = 0; c < data.cols; c++) {
        double diff = static_cast<double>(data.at<float>(row, c)) - centroid[c];
        sum += diff * diff;
    }
    return sqrt(sum);
}

static double histL2Distance(const vector<double>& histA,
                             const vector<double>& histB)
{
    double sum = 0.0;
    for (size_t i = 0; i < histA.size(); i++) {
        double diff = histA[i] - histB[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

static cv::Mat stackDescriptors(const vector<ImageFeatureData>& imageData)
{
    int totalRows = 0;
    int cols = 0;

    for (size_t i = 0; i < imageData.size(); i++) {
        totalRows += imageData[i].descriptors.rows;
        if (!imageData[i].descriptors.empty()) {
            cols = imageData[i].descriptors.cols;
        }
    }

    if (totalRows == 0 || cols == 0) {
        return cv::Mat();
    }

    cv::Mat allDescriptors(totalRows, cols, CV_32F);
    int startRow = 0;

    for (size_t i = 0; i < imageData.size(); i++) {
        if (imageData[i].descriptors.empty()) {
            continue;
        }

        imageData[i].descriptors.copyTo(
            allDescriptors.rowRange(startRow, startRow + imageData[i].descriptors.rows));
        startRow += imageData[i].descriptors.rows;
    }

    return allDescriptors;
}

static vector< vector<double> > initCentroids(const cv::Mat& data, int k)
{
    vector< vector<double> > centroids(k, vector<double>(data.cols, 0.0));

    for (int cluster = 0; cluster < k; cluster++) {
        int row = (cluster * data.rows) / k;
        if (row >= data.rows) {
            row = data.rows - 1;
        }

        for (int c = 0; c < data.cols; c++) {
            centroids[cluster][c] = static_cast<double>(data.at<float>(row, c));
        }
    }

    return centroids;
}

static void runKMeans(const cv::Mat& data,
                      int k,
                      int maxIter,
                      vector<int>& assignments,
                      vector< vector<double> >& centroids)
{
    assignments.assign(data.rows, 0);
    centroids = initCentroids(data, k);

    for (int iter = 0; iter < maxIter; iter++) {
        bool changed = false;

        vector< vector<double> > sums(k, vector<double>(data.cols, 0.0));
        vector<int> counts(k, 0);

        for (int row = 0; row < data.rows; row++) {
            int bestCluster = 0;
            double bestDist = numeric_limits<double>::max();

            for (int cluster = 0; cluster < k; cluster++) {
                double dist = rowToCentroidDist(data, row, centroids[cluster]);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestCluster = cluster;
                }
            }

            if (assignments[row] != bestCluster) {
                assignments[row] = bestCluster;
                changed = true;
            }

            counts[bestCluster]++;
            for (int c = 0; c < data.cols; c++) {
                sums[bestCluster][c] += static_cast<double>(data.at<float>(row, c));
            }
        }

        for (int cluster = 0; cluster < k; cluster++) {
            if (counts[cluster] == 0) {
                int row = (cluster * data.rows) / k;
                if (row >= data.rows) {
                    row = data.rows - 1;
                }
                for (int c = 0; c < data.cols; c++) {
                    centroids[cluster][c] = static_cast<double>(data.at<float>(row, c));
                }
                continue;
            }

            for (int c = 0; c < data.cols; c++) {
                centroids[cluster][c] = sums[cluster][c] /
                                        static_cast<double>(counts[cluster]);
            }
        }

        if (!changed) {
            break;
        }
    }
}

static int nearestCentroid(const cv::Mat& descriptors,
                           int row,
                           const vector< vector<double> >& centroids)
{
    int bestCluster = 0;
    double bestDist = numeric_limits<double>::max();

    for (size_t cluster = 0; cluster < centroids.size(); cluster++) {
        double dist = rowToCentroidDist(descriptors, row, centroids[cluster]);
        if (dist < bestDist) {
            bestDist = dist;
            bestCluster = static_cast<int>(cluster);
        }
    }

    return bestCluster;
}

static vector<double> buildNormalizedHistogram(const cv::Mat& descriptors,
                                               const vector< vector<double> >& centroids)
{
    vector<double> hist(centroids.size(), 0.0);
    if (descriptors.empty()) {
        return hist;
    }

    for (int row = 0; row < descriptors.rows; row++) {
        int cluster = nearestCentroid(descriptors, row, centroids);
        hist[cluster] += 1.0;
    }

    for (size_t i = 0; i < hist.size(); i++) {
        hist[i] /= static_cast<double>(descriptors.rows);
    }

    return hist;
}

static void writeHistogramCSV(const string& filename,
                              const vector<ImageFeatureData>& imageData,
                              const vector< vector<double> >& histograms)
{
    ofstream file(filename.c_str());
    if (!file) {
        cerr << "Error: cannot open histogram CSV file " << filename << endl;
        exit(1);
    }

    file << "filename";
    for (int bin = 0; bin < 8; bin++) {
        file << ",bin" << bin;
    }
    file << "\n";

    for (size_t i = 0; i < imageData.size(); i++) {
        file << imageData[i].name;
        for (size_t bin = 0; bin < histograms[i].size(); bin++) {
            file << "," << fixed << setprecision(10) << histograms[i][bin];
        }
        file << "\n";
    }

    file.close();
}

static void writeComparisonText(const string& filename,
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

int main(int argc, char* argv[])
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <hw4_root_path>\n";
        return 1;
    }

    string hw4Root = argv[1];
    string materialDir = hw4Root + "/EE569_2026Spring_HW4_materials";
    string outputDir = hw4Root + "/outputs/p3";

    ensureDir(hw4Root + "/outputs");
    ensureDir(outputDir);

    const int width = 600;
    const int height = 400;
    const int k = 8;
    const int maxIter = 100;

    vector<ImageSpec> images;
    images.push_back({"Cat_1.raw", materialDir + "/Cat_1.raw"});
    images.push_back({"Cat_2.raw", materialDir + "/Cat_2.raw"});
    images.push_back({"Cat_3.raw", materialDir + "/Cat_3.raw"});
    images.push_back({"Dog_1.raw", materialDir + "/Dog_1.raw"});

    vector<ImageFeatureData> imageData;

    // Extract SIFT descriptors for every image.
    for (size_t i = 0; i < images.size(); i++) {
        cv::Mat image = readRawRGBImage(images[i].path, width, height);

        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        extractSIFT(image, keypoints, descriptors);

        ImageFeatureData curr;
        curr.name = images[i].name;
        curr.descriptors = descriptors;
        imageData.push_back(curr);

        cout << "Extracted " << descriptors.rows
             << " descriptors from " << images[i].name << endl;
    }

    cv::Mat allDescriptors = stackDescriptors(imageData);
    if (allDescriptors.empty()) {
        cerr << "Error: no SIFT descriptors were extracted." << endl;
        return 1;
    }

    // Build the 8-word codebook using all descriptors.
    vector<int> assignments;
    vector< vector<double> > centroids;
    runKMeans(allDescriptors, k, maxIter, assignments, centroids);

    // Build one normalized histogram for each image.
    vector< vector<double> > histograms;
    map<string, int> imageIndex;
    for (size_t i = 0; i < imageData.size(); i++) {
        histograms.push_back(buildNormalizedHistogram(imageData[i].descriptors, centroids));
        imageIndex[imageData[i].name] = static_cast<int>(i);
    }

    double cat3Cat1 = histL2Distance(histograms[imageIndex["Cat_3.raw"]],
                                     histograms[imageIndex["Cat_1.raw"]]);
    double cat3Cat2 = histL2Distance(histograms[imageIndex["Cat_3.raw"]],
                                     histograms[imageIndex["Cat_2.raw"]]);
    double cat3Dog1 = histL2Distance(histograms[imageIndex["Cat_3.raw"]],
                                     histograms[imageIndex["Dog_1.raw"]]);

    string histFile = outputDir + "/p3c_bow_histograms.csv";
    string compareFile = outputDir + "/p3c_bow_comparison.txt";

    writeHistogramCSV(histFile, imageData, histograms);
    writeComparisonText(compareFile, cat3Cat1, cat3Cat2, cat3Dog1);

    cout << fixed << setprecision(10);
    cout << "Cat_3 vs Cat_1: " << cat3Cat1 << endl;
    cout << "Cat_3 vs Cat_2: " << cat3Cat2 << endl;
    cout << "Cat_3 vs Dog_1: " << cat3Dog1 << endl;
    cout << "\nGenerated files:" << endl;
    cout << histFile << endl;
    cout << compareFile << endl;

    return 0;
}
