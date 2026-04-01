/* EE569 Homework #4
 * Name: Agam Sidhu
 * USC ID: 3027948957
 * USC Email: agamsidh@usc.edu
 * Submission Date: [fill]
 * Problem 3(b): Image Matching
 */

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/features2d.hpp>

#include "p3_sift_common.h"

using namespace std;

struct PairSpec {
    string leftName;
    string rightName;
    string leftPath;
    string rightPath;
    string prefix;
};

static double descriptorDistance(const cv::Mat& descriptorsA,
                                 int rowA,
                                 const cv::Mat& descriptorsB,
                                 int rowB)
{
    double sum = 0.0;
    for (int c = 0; c < descriptorsA.cols; c++) {
        double diff = descriptorsA.at<float>(rowA, c) - descriptorsB.at<float>(rowB, c);
        sum += diff * diff;
    }
    return sqrt(sum);
}

static int findBestDescriptorMatch(const cv::Mat& queryDescriptor,
                                   const cv::Mat& searchDescriptors,
                                   double& bestDistance)
{
    bestDistance = numeric_limits<double>::max();
    int bestIdx = -1;

    for (int i = 0; i < searchDescriptors.rows; i++) {
        double sum = 0.0;
        for (int c = 0; c < searchDescriptors.cols; c++) {
            double diff = queryDescriptor.at<float>(0, c) - searchDescriptors.at<float>(i, c);
            sum += diff * diff;
        }

        double dist = sqrt(sum);
        if (dist < bestDistance) {
            bestDistance = dist;
            bestIdx = i;
        }
    }

    return bestIdx;
}

static vector<cv::DMatch> ratioMatches(const cv::Mat& desc1,
                                       const cv::Mat& desc2)
{
    vector< vector<cv::DMatch> > knnMatches;
    vector<cv::DMatch> goodMatches;

    if (desc1.empty() || desc2.empty()) {
        return goodMatches;
    }

    cv::BFMatcher matcher(cv::NORM_L2);
    matcher.knnMatch(desc1, desc2, knnMatches, 2);

    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i].size() < 2) {
            continue;
        }

        if (knnMatches[i][0].distance < 0.75f * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    sort(goodMatches.begin(), goodMatches.end(),
         [](const cv::DMatch& a, const cv::DMatch& b) {
             return a.distance < b.distance;
         });

    return goodMatches;
}

static void saveMatchImage(const cv::Mat& imageA,
                           const vector<cv::KeyPoint>& keypointsA,
                           const cv::Mat& imageB,
                           const vector<cv::KeyPoint>& keypointsB,
                           const vector<cv::DMatch>& matches,
                           const string& outputFile,
                           int maxShow)
{
    vector<cv::DMatch> shownMatches = matches;
    if (static_cast<int>(shownMatches.size()) > maxShow) {
        shownMatches.resize(maxShow);
    }

    cv::Mat matchView;
    cv::drawMatches(imageA,
                    keypointsA,
                    imageB,
                    keypointsB,
                    shownMatches,
                    matchView,
                    cv::Scalar::all(-1),
                    cv::Scalar::all(-1),
                    vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imwrite(outputFile, matchView);
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

    vector<PairSpec> pairs;
    pairs.push_back({"Cat_1", "Cat_3", materialDir + "/Cat_1.raw", materialDir + "/Cat_3.raw", "cat1_cat3"});
    pairs.push_back({"Cat_3", "Cat_2", materialDir + "/Cat_3.raw", materialDir + "/Cat_2.raw", "cat3_cat2"});
    pairs.push_back({"Dog_1", "Cat_3", materialDir + "/Dog_1.raw", materialDir + "/Cat_3.raw", "dog1_cat3"});
    pairs.push_back({"Cat_1", "Dog_1", materialDir + "/Cat_1.raw", materialDir + "/Dog_1.raw", "cat1_dog1"});

    ofstream summary((outputDir + "/p3b_summary.txt").c_str());
    if (!summary) {
        cerr << "Error: cannot open p3b summary output file." << endl;
        return 1;
    }

    summary << "Problem 3(b) Matching Summary\n";
    summary << "----------------------------------------\n";

    // Special match for the largest-scale keypoint in Cat_1 against Cat_3.
    {
        cv::Mat cat1 = readRawRGBImage(materialDir + "/Cat_1.raw", width, height);
        cv::Mat cat3 = readRawRGBImage(materialDir + "/Cat_3.raw", width, height);

        vector<cv::KeyPoint> kp1;
        vector<cv::KeyPoint> kp3;
        cv::Mat desc1;
        cv::Mat desc3;
        extractSIFT(cat1, kp1, desc1);
        extractSIFT(cat3, kp3, desc3);

        if (!kp1.empty() && !kp3.empty() && !desc1.empty() && !desc3.empty()) {
            int largestIdx = 0;
            for (size_t i = 1; i < kp1.size(); i++) {
                if (kp1[i].size > kp1[largestIdx].size) {
                    largestIdx = static_cast<int>(i);
                }
            }

            cv::Mat queryDescriptor = desc1.row(largestIdx);
            double bestDistance = 0.0;
            int bestIdx = findBestDescriptorMatch(queryDescriptor, desc3, bestDistance);

            vector<cv::DMatch> singleMatch;
            if (bestIdx >= 0) {
                singleMatch.push_back(cv::DMatch(largestIdx, bestIdx, static_cast<float>(bestDistance)));
                saveMatchImage(cat1,
                               kp1,
                               cat3,
                               kp3,
                               singleMatch,
                               outputDir + "/p3b_cat1_cat3_largest_scale_match.png",
                               1);

                summary << fixed << setprecision(4);
                summary << "Largest-scale keypoint in Cat_1\n";
                summary << "Cat_1: x = " << kp1[largestIdx].pt.x
                        << ", y = " << kp1[largestIdx].pt.y
                        << ", size = " << kp1[largestIdx].size
                        << ", angle = " << kp1[largestIdx].angle << "\n";
                summary << "Closest point in Cat_3: x = " << kp3[bestIdx].pt.x
                        << ", y = " << kp3[bestIdx].pt.y
                        << ", size = " << kp3[bestIdx].size
                        << ", angle = " << kp3[bestIdx].angle
                        << ", descriptor distance = " << bestDistance << "\n\n";
            }
        }
    }

    for (size_t i = 0; i < pairs.size(); i++) {
        cv::Mat imageA = readRawRGBImage(pairs[i].leftPath, width, height);
        cv::Mat imageB = readRawRGBImage(pairs[i].rightPath, width, height);

        vector<cv::KeyPoint> keypointsA;
        vector<cv::KeyPoint> keypointsB;
        cv::Mat descriptorsA;
        cv::Mat descriptorsB;
        extractSIFT(imageA, keypointsA, descriptorsA);
        extractSIFT(imageB, keypointsB, descriptorsB);

        vector<cv::DMatch> matches = ratioMatches(descriptorsA, descriptorsB);
        string outputFile = outputDir + "/p3b_" + pairs[i].prefix + "_matches.png";
        saveMatchImage(imageA, keypointsA, imageB, keypointsB, matches, outputFile, 80);

        summary << pairs[i].leftName << " vs " << pairs[i].rightName << "\n";
        summary << "Left keypoints: " << keypointsA.size() << "\n";
        summary << "Right keypoints: " << keypointsB.size() << "\n";
        summary << "Ratio-test matches: " << matches.size() << "\n";
        if (!matches.empty()) {
            summary << fixed << setprecision(4)
                    << "Best match distance: " << matches[0].distance << "\n";
        }
        summary << "\n";

        cout << "Saved matches for " << pairs[i].leftName
             << " vs " << pairs[i].rightName << endl;
    }

    summary.close();
    return 0;
}
