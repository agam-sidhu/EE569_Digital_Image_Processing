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

struct ImagePair {
    string leftName;
    string rightName;
    string leftPath;
    string rightPath;
    string prefix;
};
//Function to calculate the distance between 2 desciptors
static double calcDescriptDist(const cv::Mat& desc1,
                                 int row1,
                                 const cv::Mat& desc2,
                                 int row2)
{
    double sum = 0.0;
    for (int c = 0; c < desc1.cols; c++) {
        double diff = desc1.at<float>(row1, c) - desc2.at<float>(row2, c);
        sum += diff * diff;
    }
    double dist = sqrt(sum);
    return dist;
}
//Function to find the closest descriptor match 
static int findBestMatch(const cv::Mat& queryDesc,
                                   const cv::Mat& searchDesc,
                                   double& bestDist)
{
    //intialize the best distance + index
    bestDist = numeric_limits<double>::max();
    int bestIdx = -1;

    //checks the query descriptor vs every search descriptor
    for (int idx = 0; idx < searchDesc.rows; idx++) {
        double distSum = 0.0;
        for (int c = 0; c < searchDesc.cols; c++) {
            double diff = queryDesc.at<float>(0, c) - searchDesc.at<float>(idx, c);
            distSum += diff * diff;
        }
        // compute the distance from query desc to search desc
        double currDist = sqrt(distSum);
        if (currDist < bestDist) {
            bestDist = currDist;
            bestIdx = idx;
        }
    }

    return bestIdx;
}
//Function to get matches through Lowe's ratio test
static vector<cv::DMatch> ratioMatches(const cv::Mat& desc1,
                                       const cv::Mat& desc2)
{
    vector< vector<cv::DMatch> > knnMatches;
    vector<cv::DMatch> match;

    //incase a descriptor is empty
    if (desc1.empty() || desc2.empty()) {
        return match;
    }

    //use a brute force matcher w/ L2 distance
    cv::BFMatcher matcher(cv::NORM_L2);

    matcher.knnMatch(desc1, desc2, knnMatches, 2);

    //keeps matches that pass lowe's test
    for (size_t idx = 0; idx < knnMatches.size(); idx++) {
        if (knnMatches[idx].size() < 2) {
            continue;
        }

        const cv::DMatch& bestMatch = knnMatches[idx][0];
        const cv::DMatch& nextBest = knnMatches[idx][1];

        if (bestMatch.distance < 0.75f * nextBest.distance) {
            match.push_back(bestMatch);
        }
    }
    //sorts the matches by distance (strong appear first)
    sort(match.begin(), match.end(),
         [](const cv::DMatch& a, const cv::DMatch& b) {
             return a.distance < b.distance;
         });

    return match;
}
//Function to save a match 
static void saveMatches(const cv::Mat& img1,
                           const vector<cv::KeyPoint>& kp1,
                           const cv::Mat& img2,
                           const vector<cv::KeyPoint>& kp2,
                           const vector<cv::DMatch>& matches,
                           const string& outputFile,
                           int maxShow)
{
    //copy matches + limits how many shown
    vector<cv::DMatch> show = matches;
    if (static_cast<int>(show.size()) > maxShow) {
        show.resize(maxShow);
    }

    cv::Mat imgMatch;
    cv::drawMatches(img1,
                    kp1,
                    img2,
                    kp2,
                    show,
                    imgMatch,
                    cv::Scalar::all(-1),
                    cv::Scalar::all(-1),
                    vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imwrite(outputFile, imgMatch);
}
//Main Function
int main(int argc, char* argv[])
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <hw4_root_path>\n";
        return 1;
    }


    string hw4Root = argv[1];
    string inputDir = hw4Root + "/EE569_2026Spring_HW4_materials";
    string outputDir = hw4Root + "/outputs/p3";
    //makes the output repo
    ensureDir(hw4Root + "/outputs");
    ensureDir(outputDir);

    const int width = 600;
    const int height = 400;

    //store the image pairs
    vector<ImagePair> pairs;
    pairs.push_back({"Cat_1", "Cat_3", inputDir + "/Cat_1.raw", inputDir + "/Cat_3.raw", "cat1_cat3"});
    pairs.push_back({"Cat_3", "Cat_2", inputDir + "/Cat_3.raw", inputDir + "/Cat_2.raw", "cat3_cat2"});
    pairs.push_back({"Dog_1", "Cat_3", inputDir + "/Dog_1.raw", inputDir + "/Cat_3.raw", "dog1_cat3"});
    pairs.push_back({"Cat_1", "Dog_1", inputDir + "/Cat_1.raw", inputDir + "/Dog_1.raw", "cat1_dog1"});
    //open summary utput file
    ofstream summary((outputDir + "/p3b_summary.txt").c_str());
    if (!summary) {
        cerr << "Error: cannot open p3b summary output file." << endl;
        return 1;
    }

    summary << "Problem 3(b) Matching Summary\n";
    summary << "----------------------------------------\n";

    // Special match for the largest-scale keypoint in Cat_1 vs Cat_3
    {
        cv::Mat cat1 = readrawRGB(inputDir + "/Cat_1.raw", width, height);
        cv::Mat cat3 = readrawRGB(inputDir + "/Cat_3.raw", width, height);

        vector<cv::KeyPoint> kp1;
        vector<cv::KeyPoint> kp3;
        cv::Mat desc1;
        cv::Mat desc3;
        //extracts the SIFT keypoints + descriptors
        extractSIFT(cat1, kp1, desc1);
        extractSIFT(cat3, kp3, desc3);

        if (!kp1.empty() && !kp3.empty() && !desc1.empty() && !desc3.empty()) {
            //finds largest scale keypoint in Cat 1
            int lIdx = 0;
            for (size_t idx = 1; idx < kp1.size(); idx++) {
                if (kp1[idx].size > kp1[lIdx].size) {
                    lIdx = static_cast<int>(idx);
                }
            }
            //match the descriptor to closest descriptor in Cat 3
            cv::Mat queryDesc = desc1.row(lIdx);
            double bestDist = 0.0;
            int bestIdx = findBestMatch(queryDesc, desc3, bestDist);
            vector<cv::DMatch> oneMatch;
            if (bestIdx >= 0) {
                oneMatch.push_back(cv::DMatch(lIdx, bestIdx, static_cast<float>(bestDist)));
                saveMatches(cat1,
                               kp1,
                               cat3,
                               kp3,
                               oneMatch,
                               outputDir + "/p3b_cat1_cat3_largest_scale_match.png",
                               1);

                summary << fixed << setprecision(4);
                summary << "Largest Scale keypoint in Cat_1\n";
                summary << "Cat_1: x = " << kp1[lIdx].pt.x
                        << ", y = " << kp1[lIdx].pt.y
                        << ", size = " << kp1[lIdx].size
                        << ", angle = " << kp1[lIdx].angle << "\n";
                summary << "Closest Point in Cat_3: x = " << kp3[bestIdx].pt.x
                        << ", y = " << kp3[bestIdx].pt.y
                        << ", size = " << kp3[bestIdx].size
                        << ", angle = " << kp3[bestIdx].angle
                        << ", descriptor distance = " << bestDist << "\n\n";
            }
        }
    }
    //runs ratio test for all image pair
    for (size_t i = 0; i < pairs.size(); i++) {
        cv::Mat img1 = readrawRGB(pairs[i].leftPath, width, height);
        cv::Mat img2 = readrawRGB(pairs[i].rightPath, width, height);

        vector<cv::KeyPoint> kp1;
        vector<cv::KeyPoint> kp2;
        cv::Mat desc1;
        cv::Mat desc2;
        //extracts the SIFt features from both imgs
        extractSIFT(img1, kp1, desc1);
        extractSIFT(img2, kp2, desc2);
        //gets the matches using lowe's ratio test
        vector<cv::DMatch> matches = ratioMatches(desc1, desc2);
        string outputFile = outputDir + "/p3b_" + pairs[i].prefix + "_matches.png";
        saveMatches(img1, kp1, img2, kp2, matches, outputFile, 80);

        //writes the summary 
        summary << pairs[i].leftName << " vs " << pairs[i].rightName << "\n";
        summary << "Left keypoints: " << kp1.size() << "\n";
        summary << "Right keypoints: " << kp2.size() << "\n";
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
