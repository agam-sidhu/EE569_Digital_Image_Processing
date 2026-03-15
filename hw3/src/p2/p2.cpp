/* EE569 Homework #3
 * Name: Agam Sidhu
 * USC ID: 3027948957
 * USC Email: agamsidh@usc.edu
 * Submission Date: March 15, 2026
 * Problem 2: Homographic Transformation and Image Stitching
 */


#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <string>
#include <cstdlib>
#include <cmath>
#include <random>
#include <limits>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

//Clamp function
static inline int clamp(int x, int low, int high) {
    return max(low, min(high, x));
}
//Clamp to double value function
static inline double clampDouble(double x, double low, double high) {
    return max(low, min(high, x));
}

//Read raw image function
static void readraw(const string& filename, vector<uint8_t>& buffer, int width, int height, int channels)
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
static void writeraw(const string& filename, const vector<uint8_t>& buffer)
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
// Covert raw data RGB into the OPENCV BGR
static Mat toBGR(const vector<uint8_t> &raw, int width, int height)
{
    Mat transformed(height, width, CV_8UC3);
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = (row * width + col) * 3;
            Vec3b& channel = transformed.at<Vec3b>(row, col);
            channel[0] = raw[idx + 2]; //B val
            channel[1] = raw[idx + 1]; // G val 
            channel[2] = raw[idx + 0]; // R val 
        }
    }
    return transformed;
}

// Convert OpenCV BGR image back to raw RGB data
static vector<uint8_t> toRaw(const Mat &img)
{
    vector<uint8_t> raw(img.rows * img.cols * 3);

    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            int val = row * img.cols + col;
            int idx = val * 3;
            const Vec3b& channel = img.at<Vec3b>(row, col);

            raw[idx + 0] = channel[2]; // R val
            raw[idx + 1] = channel[1]; // G val
            raw[idx + 2] = channel[0]; // B val
        }
    }

    return raw;
}

// Get one color value with boundary clamping
static inline uint8_t getPixelValue(const Mat& img, int row, int col, int ch)
{
    row = clamp(row, 0, img.rows - 1);
    col = clamp(col, 0, img.cols - 1);
    ch  = clamp(ch, 0, 2);

    return img.at<Vec3b>(row, col)[ch];
}

// Bilinear interpolation for one channel
static double bilinearInterp(const Mat& img, double row, double col, int ch)
{
    row = clampDouble(row, 0.0, static_cast<double>(img.rows - 1));
    col = clampDouble(col, 0.0, static_cast<double>(img.cols - 1));

    int r0 = static_cast<int>(floor(row));
    int c0 = static_cast<int>(floor(col));
    int r1 = clamp(r0 + 1, 0, img.rows - 1);
    int c1 = clamp(c0 + 1, 0, img.cols - 1);

    double dr = row - r0;
    double dc = col - c0;

    double v00 = static_cast<double>(getPixelValue(img, r0, c0, ch));
    double v01 = static_cast<double>(getPixelValue(img, r0, c1, ch));
    double v10 = static_cast<double>(getPixelValue(img, r1, c0, ch));
    double v11 = static_cast<double>(getPixelValue(img, r1, c1, ch));

    double top = v00 * (1.0 - dc) + v01 * dc;
    double bottom = v10 * (1.0 - dc) + v11 * dc;

    return top * (1.0 - dr) + bottom * dr;
}

// 3x3 homography matrix
struct HMatrix
{
    double h[3][3];
};

// Multiply two homography matrices
static HMatrix multiplyH(const HMatrix& H1, const HMatrix& H2)
{
    HMatrix result{};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result.h[i][j] = 0.0;
            for (int k = 0; k < 3; k++) {
                result.h[i][j] += H1.h[i][k] * H2.h[k][j];
            }
        }
    }

    return result;
}

// Inverse of a 3x3 homography matrix
static HMatrix inverseH(const HMatrix& H)
{
    HMatrix inv{};

    double a = H.h[0][0], b = H.h[0][1], c = H.h[0][2];
    double d = H.h[1][0], e = H.h[1][1], f = H.h[1][2];
    double g = H.h[2][0], h = H.h[2][1], i = H.h[2][2];

    double det = a * (e * i - f * h)
               - b * (d * i - f * g)
               + c * (d * h - e * g);

    if (fabs(det) < 1e-12) {
        cerr << "Error: homography matrix is singular." << endl;
        exit(1);
    }

    inv.h[0][0] =  (e * i - f * h) / det;
    inv.h[0][1] = -(b * i - c * h) / det;
    inv.h[0][2] =  (b * f - c * e) / det;

    inv.h[1][0] = -(d * i - f * g) / det;
    inv.h[1][1] =  (a * i - c * g) / det;
    inv.h[1][2] = -(a * f - c * d) / det;

    inv.h[2][0] =  (d * h - e * g) / det;
    inv.h[2][1] = -(a * h - b * g) / det;
    inv.h[2][2] =  (a * e - b * d) / det;

    return inv;
}

// Apply homography to one point
static Point2d applyH(const HMatrix& H, const Point2d& pt)
{
    double x = pt.x;
    double y = pt.y;

    double X = H.h[0][0] * x + H.h[0][1] * y + H.h[0][2];
    double Y = H.h[1][0] * x + H.h[1][1] * y + H.h[1][2];
    double W = H.h[2][0] * x + H.h[2][1] * y + H.h[2][2];

    if (fabs(W) < 1e-12) {
        return Point2d(0.0, 0.0);
    }

    return Point2d(X / W, Y / W);
}

// Solve 8x8 system using Gaussian elimination
static bool solveLinear8x8(double A[8][9], double x[8])
{
    for (int col = 0; col < 8; col++) {
        int pivotRow = col;

        for (int row = col + 1; row < 8; row++) {
            if (fabs(A[row][col]) > fabs(A[pivotRow][col])) {
                pivotRow = row;
            }
        }

        if (fabs(A[pivotRow][col]) < 1e-12) {
            return false;
        }

        if (pivotRow != col) {
            for (int k = col; k < 9; k++) {
                swap(A[col][k], A[pivotRow][k]);
            }
        }

        double pivotVal = A[col][col];
        for (int k = col; k < 9; k++) {
            A[col][k] /= pivotVal;
        }

        for (int row = 0; row < 8; row++) {
            if (row == col) continue;

            double factor = A[row][col];
            for (int k = col; k < 9; k++) {
                A[row][k] -= factor * A[col][k];
            }
        }
    }

    for (int i = 0; i < 8; i++) {
        x[i] = A[i][8];
    }

    return true;
}

// Estimate homography using 4 point pairs
static bool estimateHomography4pt(const vector<Point2d>& srcPts,
                                  const vector<Point2d>& dstPts,
                                  HMatrix& H)
{
    if (srcPts.size() != 4 || dstPts.size() != 4) {
        return false;
    }

    double A[8][9] = {0.0};

    for (int i = 0; i < 4; i++) {
        double x = srcPts[i].x;
        double y = srcPts[i].y;
        double u = dstPts[i].x;
        double v = dstPts[i].y;

        A[2 * i][0] = x;
        A[2 * i][1] = y;
        A[2 * i][2] = 1.0;
        A[2 * i][3] = 0.0;
        A[2 * i][4] = 0.0;
        A[2 * i][5] = 0.0;
        A[2 * i][6] = -u * x;
        A[2 * i][7] = -u * y;
        A[2 * i][8] = u;

        A[2 * i + 1][0] = 0.0;
        A[2 * i + 1][1] = 0.0;
        A[2 * i + 1][2] = 0.0;
        A[2 * i + 1][3] = x;
        A[2 * i + 1][4] = y;
        A[2 * i + 1][5] = 1.0;
        A[2 * i + 1][6] = -v * x;
        A[2 * i + 1][7] = -v * y;
        A[2 * i + 1][8] = v;
    }

    double sol[8];
    if (!solveLinear8x8(A, sol)) {
        return false;
    }

    H.h[0][0] = sol[0];
    H.h[0][1] = sol[1];
    H.h[0][2] = sol[2];
    H.h[1][0] = sol[3];
    H.h[1][1] = sol[4];
    H.h[1][2] = sol[5];
    H.h[2][0] = sol[6];
    H.h[2][1] = sol[7];
    H.h[2][2] = 1.0;

    return true;
}

// Refine homography using all inliers
static bool refineHomographyLS(const vector<Point2d>& srcPts,
                               const vector<Point2d>& dstPts,
                               const vector<int>& inlierIdx,
                               HMatrix& H)
{
    if (inlierIdx.size() < 4) {
        return false;
    }

    int n = static_cast<int>(inlierIdx.size());
    Mat A(2 * n, 8, CV_64F);
    Mat b(2 * n, 1, CV_64F);

    for (int k = 0; k < n; k++) {
        int idx = inlierIdx[k];

        double x = srcPts[idx].x;
        double y = srcPts[idx].y;
        double u = dstPts[idx].x;
        double v = dstPts[idx].y;

        A.at<double>(2 * k, 0) = x;
        A.at<double>(2 * k, 1) = y;
        A.at<double>(2 * k, 2) = 1.0;
        A.at<double>(2 * k, 3) = 0.0;
        A.at<double>(2 * k, 4) = 0.0;
        A.at<double>(2 * k, 5) = 0.0;
        A.at<double>(2 * k, 6) = -u * x;
        A.at<double>(2 * k, 7) = -u * y;
        b.at<double>(2 * k, 0) = u;

        A.at<double>(2 * k + 1, 0) = 0.0;
        A.at<double>(2 * k + 1, 1) = 0.0;
        A.at<double>(2 * k + 1, 2) = 0.0;
        A.at<double>(2 * k + 1, 3) = x;
        A.at<double>(2 * k + 1, 4) = y;
        A.at<double>(2 * k + 1, 5) = 1.0;
        A.at<double>(2 * k + 1, 6) = -v * x;
        A.at<double>(2 * k + 1, 7) = -v * y;
        b.at<double>(2 * k + 1, 0) = v;
    }

    Mat sol;
    if (!solve(A, b, sol, DECOMP_SVD)) {
        return false;
    }

    H.h[0][0] = sol.at<double>(0, 0);
    H.h[0][1] = sol.at<double>(1, 0);
    H.h[0][2] = sol.at<double>(2, 0);
    H.h[1][0] = sol.at<double>(3, 0);
    H.h[1][1] = sol.at<double>(4, 0);
    H.h[1][2] = sol.at<double>(5, 0);
    H.h[2][0] = sol.at<double>(6, 0);
    H.h[2][1] = sol.at<double>(7, 0);
    H.h[2][2] = 1.0;

    return true;
}

// Compute reprojection error
static double getReprojError(const HMatrix& H, const Point2d& srcPt, const Point2d& dstPt)
{
    Point2d mappedPt = applyH(H, srcPt);
    double dx = mappedPt.x - dstPt.x;
    double dy = mappedPt.y - dstPt.y;

    return sqrt(dx * dx + dy * dy);
}

// Estimate homography using RANSAC
static bool estimateHomographyRANSAC(const vector<Point2d>& srcPts,
                                     const vector<Point2d>& dstPts,
                                     HMatrix& bestH,
                                     vector<int>& bestInliers,
                                     int numIter = 3000,
                                     double thresh = 3.0)
{
    if (srcPts.size() != dstPts.size() || srcPts.size() < 4) {
        return false;
    }

    mt19937 rng(569);
    uniform_int_distribution<int> dist(0, static_cast<int>(srcPts.size()) - 1);

    int maxInliers = -1;
    double minErr = numeric_limits<double>::max();

    for (int iter = 0; iter < numIter; iter++) {
        vector<int> pickIdx;
        while (pickIdx.size() < 4) {
            int r = dist(rng);
            if (find(pickIdx.begin(), pickIdx.end(), r) == pickIdx.end()) {
                pickIdx.push_back(r);
            }
        }

        vector<Point2d> srcSample(4), dstSample(4);
        for (int i = 0; i < 4; i++) {
            srcSample[i] = srcPts[pickIdx[i]];
            dstSample[i] = dstPts[pickIdx[i]];
        }

        HMatrix Htemp;
        if (!estimateHomography4pt(srcSample, dstSample, Htemp)) {
            continue;
        }

        vector<int> inliers;
        double totalErr = 0.0;

        for (int i = 0; i < static_cast<int>(srcPts.size()); i++) {
            double err = getReprojError(Htemp, srcPts[i], dstPts[i]);
            if (err < thresh) {
                inliers.push_back(i);
                totalErr += err;
            }
        }

        if (static_cast<int>(inliers.size()) > maxInliers ||
            (static_cast<int>(inliers.size()) == maxInliers && totalErr < minErr)) {
            maxInliers = static_cast<int>(inliers.size());
            minErr = totalErr;
            bestInliers = inliers;
            bestH = Htemp;
        }
    }

    if (maxInliers < 4) {
        return false;
    }

    return refineHomographyLS(srcPts, dstPts, bestInliers, bestH);
}

// Find SIFT matches between two images
static void getSIFTMatches(const Mat& img1,
                           const Mat& img2,
                           vector<Point2d>& pts1,
                           vector<Point2d>& pts2,
                           vector<KeyPoint>& kp1,
                           vector<KeyPoint>& kp2,
                           vector<DMatch>& goodMatches)
{
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    Ptr<SIFT> sift = SIFT::create(3000);
    Mat desc1, desc2;

    sift->detectAndCompute(gray1, noArray(), kp1, desc1);
    sift->detectAndCompute(gray2, noArray(), kp2, desc2);

    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knnMatches;
    matcher.knnMatch(desc1, desc2, knnMatches, 2);

    const float ratioThresh = 0.75f;

    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i].size() < 2) continue;

        const DMatch& m1 = knnMatches[i][0];
        const DMatch& m2 = knnMatches[i][1];

        if (m1.distance < ratioThresh * m2.distance) {
            goodMatches.push_back(m1);
            pts1.push_back(kp1[m1.queryIdx].pt);
            pts2.push_back(kp2[m1.trainIdx].pt);
        }
    }
}

// Save image showing selected matches
static void saveMatchFigure(const Mat& img1,
                            const Mat& img2,
                            const vector<KeyPoint>& kp1,
                            const vector<KeyPoint>& kp2,
                            const vector<DMatch>& matches,
                            const string& outName,
                            int maxPts = 50)
{
    vector<DMatch> drawList = matches;

    sort(drawList.begin(), drawList.end(),
         [](const DMatch& a, const DMatch& b) {
             return a.distance < b.distance;
         });

    if (static_cast<int>(drawList.size()) > maxPts) {
        drawList.resize(maxPts);
    }

    Mat outImg;
    drawMatches(img1, kp1, img2, kp2, drawList, outImg,
                Scalar::all(-1), Scalar::all(-1), vector<char>(),
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imwrite(outName, outImg);
}

// Save point pairs into txt file
static void savePointPairs(const vector<Point2d>& srcPts,
                           const vector<Point2d>& dstPts,
                           const vector<int>& inliers,
                           const string& outName,
                           int maxPts = 20)
{
    ofstream fout(outName);
    if (!fout) {
        cerr << "Warning: cannot open file " << outName << endl;
        return;
    }

    fout << "Index\tSrcX\tSrcY\tDstX\tDstY\n";

    int n = min(static_cast<int>(inliers.size()), maxPts);
    for (int i = 0; i < n; i++) {
        int idx = inliers[i];
        fout << idx << "\t"
             << srcPts[idx].x << "\t"
             << srcPts[idx].y << "\t"
             << dstPts[idx].x << "\t"
             << dstPts[idx].y << "\n";
    }

    fout.close();
}

// Warp one image into panorama canvas and accumulate values
static void warpAndAdd(const Mat& srcImg,
                       const HMatrix& H_src_to_pano,
                       Mat& sumImg,
                       Mat& countImg)
{
    HMatrix H_pano_to_src = inverseH(H_src_to_pano);

    for (int i = 0; i < sumImg.rows; i++) {
        for (int j = 0; j < sumImg.cols; j++) {
            Point2d srcPt = applyH(H_pano_to_src, Point2d(j, i));
            double x = srcPt.x;
            double y = srcPt.y;

            if (x >= 0.0 && x <= srcImg.cols - 1.0 &&
                y >= 0.0 && y <= srcImg.rows - 1.0) {
                Vec3d& sumPixel = sumImg.at<Vec3d>(i, j);

                for (int ch = 0; ch < 3; ch++) {
                    sumPixel[ch] += bilinearInterp(srcImg, y, x, ch);
                }

                countImg.at<double>(i, j) += 1.0;
            }
        }
    }
}

// Get transformed corner positions
static vector<Point2d> getWarpedCorners(const HMatrix& H, int width, int height)
{
    vector<Point2d> corners;

    corners.push_back(applyH(H, Point2d(0, 0)));
    corners.push_back(applyH(H, Point2d(width - 1, 0)));
    corners.push_back(applyH(H, Point2d(0, height - 1)));
    corners.push_back(applyH(H, Point2d(width - 1, height - 1)));

    return corners;
}

// Main function
int main(int argc, char* argv[])
{
    if (argc != 11) {
        cerr << "Usage: " << argv[0]
             << " left.raw middle.raw right.raw panorama.raw"
             << " width height match_left.png match_right.png points_left.txt points_right.txt"
             << endl;
        return 1;
    }

    string leftName = argv[1];
    string midName = argv[2];
    string rightName = argv[3];
    string panoName = argv[4];

    int width = atoi(argv[5]);
    int height = atoi(argv[6]);

    string leftMatchName = argv[7];
    string rightMatchName = argv[8];
    string leftPtsName = argv[9];
    string rightPtsName = argv[10];

    if (width <= 0 || height <= 0) {
        cerr << "Error: width and height must be positive." << endl;
        return 1;
    }

    vector<uint8_t> leftRaw, midRaw, rightRaw;
    readraw(leftName, leftRaw, width, height, 3);
    readraw(midName, midRaw, width, height, 3);
    readraw(rightName, rightRaw, width, height, 3);

    Mat leftImg = toBGR(leftRaw, width, height);
    Mat midImg = toBGR(midRaw, width, height);
    Mat rightImg = toBGR(rightRaw, width, height);

    // Left to middle matching
    vector<Point2d> leftPts, midPts1;
    vector<KeyPoint> kpLeft, kpMid1;
    vector<DMatch> matchLeftMid;
    getSIFTMatches(leftImg, midImg, leftPts, midPts1, kpLeft, kpMid1, matchLeftMid);

    // Right to middle matching
    vector<Point2d> rightPts, midPts2;
    vector<KeyPoint> kpRight, kpMid2;
    vector<DMatch> matchRightMid;
    getSIFTMatches(rightImg, midImg, rightPts, midPts2, kpRight, kpMid2, matchRightMid);

    if (leftPts.size() < 4 || rightPts.size() < 4) {
        cerr << "Error: not enough matched points." << endl;
        return 1;
    }

    HMatrix H_left_mid, H_right_mid;
    vector<int> inliersLeft, inliersRight;

    if (!estimateHomographyRANSAC(leftPts, midPts1, H_left_mid, inliersLeft)) {
        cerr << "Error: failed to estimate left-to-middle homography." << endl;
        return 1;
    }

    if (!estimateHomographyRANSAC(rightPts, midPts2, H_right_mid, inliersRight)) {
        cerr << "Error: failed to estimate right-to-middle homography." << endl;
        return 1;
    }

    saveMatchFigure(leftImg, midImg, kpLeft, kpMid1, matchLeftMid, leftMatchName);
    saveMatchFigure(rightImg, midImg, kpRight, kpMid2, matchRightMid, rightMatchName);
    savePointPairs(leftPts, midPts1, inliersLeft, leftPtsName);
    savePointPairs(rightPts, midPts2, inliersRight, rightPtsName);

    // Middle image is the reference
    HMatrix I{};
    I.h[0][0] = 1.0; I.h[0][1] = 0.0; I.h[0][2] = 0.0;
    I.h[1][0] = 0.0; I.h[1][1] = 1.0; I.h[1][2] = 0.0;
    I.h[2][0] = 0.0; I.h[2][1] = 0.0; I.h[2][2] = 1.0;

    vector<Point2d> allCorners;
    vector<Point2d> leftCorners = getWarpedCorners(H_left_mid, width, height);
    vector<Point2d> midCorners = getWarpedCorners(I, width, height);
    vector<Point2d> rightCorners = getWarpedCorners(H_right_mid, width, height);

    allCorners.insert(allCorners.end(), leftCorners.begin(), leftCorners.end());
    allCorners.insert(allCorners.end(), midCorners.begin(), midCorners.end());
    allCorners.insert(allCorners.end(), rightCorners.begin(), rightCorners.end());

    double minX = numeric_limits<double>::max();
    double minY = numeric_limits<double>::max();
    double maxX = -numeric_limits<double>::max();
    double maxY = -numeric_limits<double>::max();

    for (const auto& pt : allCorners) {
        minX = min(minX, pt.x);
        minY = min(minY, pt.y);
        maxX = max(maxX, pt.x);
        maxY = max(maxY, pt.y);
    }

    // Shift all warped images so panorama starts at (0,0)
    HMatrix T{};
    T.h[0][0] = 1.0; T.h[0][1] = 0.0; T.h[0][2] = -minX;
    T.h[1][0] = 0.0; T.h[1][1] = 1.0; T.h[1][2] = -minY;
    T.h[2][0] = 0.0; T.h[2][1] = 0.0; T.h[2][2] = 1.0;

    HMatrix H_left_pano = multiplyH(T, H_left_mid);
    HMatrix H_mid_pano = multiplyH(T, I);
    HMatrix H_right_pano = multiplyH(T, H_right_mid);

    int panoWidth = static_cast<int>(ceil(maxX - minX + 1.0));
    int panoHeight = static_cast<int>(ceil(maxY - minY + 1.0));

    Mat sumImg(panoHeight, panoWidth, CV_64FC3, Scalar(0, 0, 0));
    Mat countImg(panoHeight, panoWidth, CV_64F, Scalar(0));

    warpAndAdd(leftImg, H_left_pano, sumImg, countImg);
    warpAndAdd(midImg, H_mid_pano, sumImg, countImg);
    warpAndAdd(rightImg, H_right_pano, sumImg, countImg);

    Mat panoImg(panoHeight, panoWidth, CV_8UC3, Scalar(0, 0, 0));

    for (int i = 0; i < panoHeight; i++) {
        for (int j = 0; j < panoWidth; j++) {
            double cnt = countImg.at<double>(i, j);

            if (cnt > 0.0) {
                Vec3d sumPixel = sumImg.at<Vec3d>(i, j);
                Vec3b& outPixel = panoImg.at<Vec3b>(i, j);

                for (int ch = 0; ch < 3; ch++) {
                    outPixel[ch] = static_cast<uint8_t>(
                        clamp(static_cast<int>(round(sumPixel[ch] / cnt)), 0, 255)
                    );
                }
            }
        }
    }

    vector<uint8_t> panoRaw = toRaw(panoImg);
    writeraw(panoName, panoRaw);

    cout << "Panorama size: " << panoWidth << " x " << panoHeight << endl;
    cout << "Left-Middle matches: " << leftPts.size()
         << ", inliers: " << inliersLeft.size() << endl;
    cout << "Right-Middle matches: " << rightPts.size()
         << ", inliers: " << inliersRight.size() << endl;

    return 0;
}