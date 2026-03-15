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
// Function to convert raw data to BGR format
static Mat toBGR(const vector<uint8_t> &image, int width, int height)
{
    Mat transformed(height, width, CV_8UC3);
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int val = row * width + col;
            int i = val * 3;
            Vec3b& channel = transformed.at<Vec3b>(row, col);
            channel[0] = image[i + 2]; //B val
            channel[1] = image[i + 1]; // G val 
            channel[2] = image[i + 0]; // R val 
        }
    }
    return transformed;
}

// Convert image into raw RGB
static vector<uint8_t> toRaw(const Mat &image)
{
    vector<uint8_t> output(image.rows * image.cols * 3);
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            int val = row * image.cols + col;
            int i = val * 3;
            const Vec3b& channel = image.at<Vec3b>(row, col);

            output[i + 0] = channel[2]; // R val
            output[i + 1] = channel[1]; // G val
            output[i + 2] = channel[0]; // B val
        }
    }
    return output;
}

// FUnction that gets the pixel val from the image
static inline uint8_t getPix(const Mat& input, int row, int col, int channel)
{
    row = clamp(row, 0, input.rows - 1);
    col = clamp(col, 0, input.cols - 1);
    channel = clamp(channel, 0, 2);
    return input.at<Vec3b>(row, col)[channel];
}

// Bilinear interpolation for one channel
static double biInter(const Mat& image, double r, double c, int ch)
{
    r = clampDouble(r, 0.0, static_cast<double>(image.rows - 1));
    c = clampDouble(c, 0.0, static_cast<double>(image.cols - 1));

    //Get coordintes
    int rowLow = static_cast<int>(floor(r)); 
    int colLow = static_cast<int>(floor(c));
    int rowHigh = clamp(rowLow + 1, 0, image.rows - 1); 
    int colHigh = clamp(colLow + 1, 0, image.cols - 1); 

    double distRow = r - rowLow; 
    double distCol = c - colLow; 

    //Pixel values 
    double tl = static_cast<double>(getPix(image, rowLow, colLow, ch)); 
    double tr = static_cast<double>(getPix(image, rowLow, colHigh, ch)); 
    double bl = static_cast<double>(getPix(image, rowHigh, colLow, ch)); 
    double br = static_cast<double>(getPix(image, rowHigh, colHigh, ch));

    //neightbor weights
    double weightL = 1.0 - distCol;  
    double weightR = distCol;        
    //blend top + bottom 
    double top = tl * weightL + tr * weightR; 
    double bot = bl * weightL + br * weightR;
    double wTop = 1.0 - distRow;  
    return top * wTop + bot * distRow; 
}

struct HMat {
    double mat[3][3];
};

// Function to multiple 2 homography matrices
static HMat matrixMult(const HMat& mat1, const HMat& mat2)
{
    HMat result{};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double sum = 0.0;          
            for (int x = 0; x < 3; x++) {
                double mult = mat1.mat[i][x] * mat2.mat[x][j]; 
                sum += mult;                             
            }
            result.mat[i][j] = sum; 
        }
    }
    return result;
}

// Function to invert the matrix
static HMat invertMat(const HMat& homo)
{
    HMat out{};
    //setting up matrix
    double a = homo.mat[0][0]; 
    double b = homo.mat[0][1]; 
    double c = homo.mat[0][2];
    double d = homo.mat[1][0]; 
    double e = homo.mat[1][1]; 
    double f = homo.mat[1][2];
    double g = homo.mat[2][0];
    double h = homo.mat[2][1];
    double i = homo.mat[2][2];

    double upper = a * (e * i - f * h);
    double mid = b * (d * i - f * g);
    double low = c * (d * h - e * g);
    double determinant = upper - mid + low;
    //invert each row
    //row1
    out.mat[0][0] = (e * i - f * h) / determinant;
    out.mat[0][1] = -(b * i - c * h) / determinant;
    out.mat[0][2] = (b * f - c * e) / determinant;
    //row2
    out.mat[1][0] = -(d * i - f * g) / determinant;
    out.mat[1][1] = (a * i - c * g) / determinant;
    out.mat[1][2] = -(a * f - c * d) / determinant;
    //row3
    out.mat[2][0] = (d * h - e * g) / determinant;
    out.mat[2][1]= -(a * h - b * g) / determinant;
    out.mat[2][2] = (a * e - b * d) / determinant;
    return out;
}

// Function to apply homography to a point 
static Point2d applyHomo(const HMat& homo, const Point2d& p)
{
    ///get the point coordinates
    double x = p.x;
    double y = p.y;

    double num1 = homo.mat[0][0] * x + homo.mat[0][1] * y + homo.mat[0][2];
    double num2 = homo.mat[1][0] * x + homo.mat[1][1] * y + homo.mat[1][2];
    double denom = homo.mat[2][0] * x + homo.mat[2][1] * y + homo.mat[2][2];

    if (fabs(denom) < 1e-12) {
        return Point2d(0.0, 0.0);
    }
    //normalize da values
    double normX = num1 / denom; 
    double normY = num2 / denom; 
    return Point2d(normX, normY);
}

//Function to solvr 8 by 8 system
static bool solveEight(double matrix[8][9], double result[8])
{
    for (int col = 0; col < 8; col++) {
        int pivot = col;
        //get row with largest value in col
        for (int row = col + 1; row < 8; row++) {
            if (fabs(matrix[row][col]) > fabs(matrix[pivot][col])) {
                pivot = row;
            }
        }
        //Check for singular matrix -> no solution
        if (fabs(matrix[pivot][col]) < 1e-12) {
            return false;
        }

        //swap curr row with pivot row if necessary
        if (pivot != col) {
            for (int k = col; k < 9; k++) {
                swap(matrix[col][k], matrix[pivot][k]);
            }
        }
        //normalize pivot row 
        double pivVal = matrix[col][col];
        for (int k = col; k < 9; k++) {
            matrix[col][k] /= pivVal;
        }
        for (int row = 0; row < 8; row++) {
            if (row == col) continue;

            double scale = matrix[row][col];
            for (int k = col; k < 9; k++) {
                matrix[row][k] -= scale * matrix[col][k];
            }
        }
    }
    for (int i = 0; i < 8; i++) {
        result[i] = matrix[i][8];
    }
    return true;
}

// Function to estimate homography 
static bool estHomo(const vector<Point2d>& src,
                                  const vector<Point2d>& dist,
                                  HMat& matrix)
{
    if (src.size() != 4 || dist.size() != 4) {
        return false;
    }

    double res[8][9] = {0.0};

    for (int i = 0; i < 4; i++) {
        double x = src[i].x;
        double y = src[i].y;
        double u = dist[i].x;
        double v = dist[i].y;
        //setup eq system
        res[2 * i][0] = x;
        res[2 * i][1] = y;
        res[2 * i][2] = 1.0;
        res[2 * i][3] = 0.0;
        res[2 * i][4] = 0.0;
        res[2 * i][5] = 0.0;
        res[2 * i][6] = -u * x;
        res[2 * i][7] = -u * y;
        res[2 * i][8] = u;
        res[2 * i + 1][0] = 0.0;
        res[2 * i + 1][1] = 0.0;
        res[2 * i + 1][2] = 0.0;
        res[2 * i + 1][3] = x;
        res[2 * i + 1][4] = y;
        res[2 * i + 1][5] = 1.0;
        res[2 * i + 1][6] = -v * x;
        res[2 * i + 1][7] = -v * y;
        res[2 * i + 1][8] = v;
    }

    double out[8];
    if (!solveEight(res, out)) {
        return false;
    }
    //fill out the homography matrix
    matrix.mat[0][0] = out[0], matrix.mat[0][1] = out[1], matrix.mat[0][2] = out[2];
    matrix.mat[1][0] = out[3], matrix.mat[1][1] = out[4], matrix.mat[1][2] = out[5];
    matrix.mat[2][0] = out[6], matrix.mat[2][1] = out[7], matrix.mat[2][2] = 1.0;

    return true;
}

// Refine homography using all inliers
static bool refineHomographyLS(const vector<Point2d>& srcPts,
                               const vector<Point2d>& dstPts,
                               const vector<int>& inlierIdx,
                               HMat& H)
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

    H.mat[0][0] = sol.at<double>(0, 0);
    H.mat[0][1] = sol.at<double>(1, 0);
    H.mat[0][2] = sol.at<double>(2, 0);
    H.mat[1][0] = sol.at<double>(3, 0);
    H.mat[1][1] = sol.at<double>(4, 0);
    H.mat[1][2] = sol.at<double>(5, 0);
    H.mat[2][0] = sol.at<double>(6, 0);
    H.mat[2][1] = sol.at<double>(7, 0);
    H.mat[2][2] = 1.0;

    return true;
}

// Compute reprojection error
static double getReprojError(const HMat& H, const Point2d& srcPt, const Point2d& dstPt)
{
    Point2d mappedPt = applyHomo(H, srcPt);
    double dx = mappedPt.x - dstPt.x;
    double dy = mappedPt.y - dstPt.y;

    return sqrt(dx * dx + dy * dy);
}

// Estimate homography using RANSAC
static bool estimateHomographyRANSAC(const vector<Point2d>& srcPts,
                                     const vector<Point2d>& dstPts,
                                     HMat& bestH,
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

        HMat Htemp;
        if (!estHomo(srcSample, dstSample, Htemp)) {
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
                       const HMat& H_src_to_pano,
                       Mat& sumImg,
                       Mat& countImg)
{
    HMat H_pano_to_src = invertMat(H_src_to_pano);

    for (int i = 0; i < sumImg.rows; i++) {
        for (int j = 0; j < sumImg.cols; j++) {
            Point2d srcPt = applyHomo(H_pano_to_src, Point2d(j, i));
            double x = srcPt.x;
            double y = srcPt.y;

            if (x >= 0.0 && x <= srcImg.cols - 1.0 &&
                y >= 0.0 && y <= srcImg.rows - 1.0) {
                Vec3d& sumPixel = sumImg.at<Vec3d>(i, j);

                for (int ch = 0; ch < 3; ch++) {
                    sumPixel[ch] += biInter(srcImg, y, x, ch);
                }

                countImg.at<double>(i, j) += 1.0;
            }
        }
    }
}

// Get transformed corner positions
static vector<Point2d> getWarpedCorners(const HMat& H, int width, int height)
{
    vector<Point2d> corners;

    corners.push_back(applyHomo(H, Point2d(0, 0)));
    corners.push_back(applyHomo(H, Point2d(width - 1, 0)));
    corners.push_back(applyHomo(H, Point2d(0, height - 1)));
    corners.push_back(applyHomo(H, Point2d(width - 1, height - 1)));

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

    HMat H_left_mid, H_right_mid;
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
    HMat I{};
    I.mat[0][0] = 1.0; I.mat[0][1] = 0.0; I.mat[0][2] = 0.0;
    I.mat[1][0] = 0.0; I.mat[1][1] = 1.0; I.mat[1][2] = 0.0;
    I.mat[2][0] = 0.0; I.mat[2][1] = 0.0; I.mat[2][2] = 1.0;

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
    HMat T{};
    T.mat[0][0] = 1.0; T.mat[0][1] = 0.0; T.mat[0][2] = -minX;
    T.mat[1][0] = 0.0; T.mat[1][1] = 1.0; T.mat[1][2] = -minY;
    T.mat[2][0] = 0.0; T.mat[2][1] = 0.0; T.mat[2][2] = 1.0;

    HMat H_left_pano = matrixMult(T, H_left_mid);
    HMat H_mid_pano = matrixMult(T, I);
    HMat H_right_pano = matrixMult(T, H_right_mid);

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