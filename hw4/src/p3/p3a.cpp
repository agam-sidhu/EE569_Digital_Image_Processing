/* EE569 Homework #4
 * Name: Agam Sidhu
 * USC ID: 3027948957
 * USC Email: agamsidh@usc.edu
 * Submission Date: [fill]
 * Problem 3(a): Salient Point Descriptor
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/features2d.hpp>

#include "p3_sift_common.h"

using namespace std;

struct ImageSpec {
    string name;
    string path;
};

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

    vector<ImageSpec> images;
    images.push_back({"Cat_1", materialDir + "/Cat_1.raw"});
    images.push_back({"Cat_2", materialDir + "/Cat_2.raw"});
    images.push_back({"Cat_3", materialDir + "/Cat_3.raw"});
    images.push_back({"Dog_1", materialDir + "/Dog_1.raw"});

    ofstream summary((outputDir + "/p3a_summary.txt").c_str());
    if (!summary) {
        cerr << "Error: cannot open p3a summary output file." << endl;
        return 1;
    }

    summary << "Problem 3(a) SIFT Summary\n";
    summary << "----------------------------------------\n";

    for (size_t i = 0; i < images.size(); i++) {
        cv::Mat image = readRawRGBImage(images[i].path, width, height);

        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        extractSIFT(image, keypoints, descriptors);

        cv::Mat keypointView;
        cv::drawKeypoints(image,
                          keypoints,
                          keypointView,
                          cv::Scalar(0, 255, 0),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        string outputImage = outputDir + "/p3a_" + images[i].name + "_keypoints.png";
        cv::imwrite(outputImage, keypointView);

        summary << images[i].name << "\n";
        summary << "Keypoint count: " << keypoints.size() << "\n";

        if (!keypoints.empty()) {
            size_t bestIdx = 0;
            for (size_t k = 1; k < keypoints.size(); k++) {
                if (keypoints[k].size > keypoints[bestIdx].size) {
                    bestIdx = k;
                }
            }

            summary << fixed << setprecision(4)
                    << "Largest keypoint: x = " << keypoints[bestIdx].pt.x
                    << ", y = " << keypoints[bestIdx].pt.y
                    << ", size = " << keypoints[bestIdx].size
                    << ", angle = " << keypoints[bestIdx].angle << "\n";
        }

        summary << "\n";
        cout << "Saved keypoint visualization for " << images[i].name << endl;
    }

    summary.close();
    return 0;
}
