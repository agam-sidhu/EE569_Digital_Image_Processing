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
    if (fabs(determinant) < 1e-12) {
        cerr << "Error: SINGULAR matrix." << endl;
        exit(1);
    }
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

// Function to compute reprojection error
static double repoError(const HMat& matrix, const Point2d& src, const Point2d& dist)
{
    Point2d mappedPt = applyHomo(matrix, src);
    double xDist = mappedPt.x - dist.x;
    double yDist = mappedPt.y - dist.y;
    double error = sqrt(xDist * xDist + yDist * yDist);
    return error;
}

// Function to refine the homography 
static bool refineHomo(const vector<Point2d>& src,
                               const vector<Point2d>& dist,
                               const vector<int>& lnIdx,
                               HMat& matrix)
{
    if (lnIdx.size() < 4) {
        return false;
    }

    int n = static_cast<int>(lnIdx.size());
    int twoN = 2 * n;
    Mat sysMat(twoN, 8, CV_64F);
    Mat rightVec(twoN, 1, CV_64F);

    for (int k = 0; k < n; k++) {
        int id = lnIdx[k];
        double x = src[id].x;
        double y = src[id].y;
        double u = dist[id].x;
        double v = dist[id].y;

        //fill out system matrix
        sysMat.at<double>(2 * k, 0) = x;
        sysMat.at<double>(2 * k, 1) = y;
        sysMat.at<double>(2 * k, 2) = 1.0;
        sysMat.at<double>(2 * k, 3) = 0.0;
        sysMat.at<double>(2 * k, 4) = 0.0;
        sysMat.at<double>(2 * k, 5) = 0.0;
        sysMat.at<double>(2 * k, 6) = -u * x;
        sysMat.at<double>(2 * k, 7) = -u * y;
        rightVec.at<double>(2 * k, 0) = u;

        sysMat.at<double>(2 * k + 1, 0) = 0.0;
        sysMat.at<double>(2 * k + 1, 1) = 0.0;
        sysMat.at<double>(2 * k + 1, 2) = 0.0;
        sysMat.at<double>(2 * k + 1, 3) = x;
        sysMat.at<double>(2 * k + 1, 4) = y;
        sysMat.at<double>(2 * k + 1, 5) = 1.0;
        sysMat.at<double>(2 * k + 1, 6) = -v * x;
        sysMat.at<double>(2 * k + 1, 7) = -v * y;
        rightVec.at<double>(2 * k + 1, 0) = v;
    }

    Mat result;
    if (!solve(sysMat, rightVec, result, DECOMP_SVD)) {
        return false;
    }

    //row by row fill out of matrix
    matrix.mat[0][0] = result.at<double>(0, 0);
    matrix.mat[0][1] = result.at<double>(1, 0);
    matrix.mat[0][2] = result.at<double>(2, 0);
    matrix.mat[1][0] = result.at<double>(3, 0);
    matrix.mat[1][1] = result.at<double>(4, 0);
    matrix.mat[1][2] = result.at<double>(5, 0);
    matrix.mat[2][0] = result.at<double>(6, 0);
    matrix.mat[2][1] = result.at<double>(7, 0);
    matrix.mat[2][2] = 1.0;

    return true;
}
// Estimate homography using RANSAC
static bool estHomoRANSAC(const vector<Point2d>& src,
                                     const vector<Point2d>& dst,
                                     HMat& optHomo,
                                     vector<int>& optLn,
                                     int iter = 3000,
                                     double tVal = 3.0)
{
    if (src.size() != dst.size() || src.size() < 4) {
        return false;
    }

    mt19937 rng(569);
    uniform_int_distribution<int> dist(0, static_cast<int>(src.size()) - 1);
    //Best inline count for now
    int maxLn = -1;
    double minVal = numeric_limits<double>::max();

    for (int i = 0; i < iter; i++) {
        vector<int> pId;
        while (pId.size() < 4) {
            int r = dist(rng);
            if (find(pId.begin(), pId.end(), r) == pId.end()) {
                pId.push_back(r);
            }
        }
        //make sample point from picked indices
        vector<Point2d> sampSrc(4), sampDist(4);
        for (int j = 0; j < 4; j++) {
            sampSrc[j] = src[pId[j]];
            sampDist[j] = dst[pId[j]];
        }
        //estimate the homography (from sample)
        HMat temp;
        if (!estHomo(sampSrc, sampDist, temp)) {
            continue;
        }
        //get inliners count
        vector<int> ln;
        double total = 0.0;
        for (int k = 0; k < static_cast<int>(src.size()); k++) {
            double err = repoError(temp, src[k], dst[k]);
            if (err < tVal) {
                ln.push_back(k);
                total += err;
            }
        }
        int num = static_cast<int>(ln.size());
        bool moreLn = num > maxLn;
        bool sameLn = (num == maxLn && total < minVal);
        if (moreLn || sameLn) {
            maxLn = num;
            minVal = total;
            optLn = ln;
            optHomo = temp;
        }
    }

    if (maxLn < 4) {
        return false;
    }

    return refineHomo(src, dst, optLn, optHomo);
}

//Function to find SIFT matches
static void getSIFT(const Mat& image1,
                           const Mat& image2,
                           vector<Point2d>& p1,
                           vector<Point2d>& p2,
                           vector<KeyPoint>& key1,
                           vector<KeyPoint>& key2,
                           vector<DMatch>& match)
{
    //convert to grayscale
    Mat g1, g2;
    cvtColor(image1, g1, COLOR_BGR2GRAY);
    cvtColor(image2, g2, COLOR_BGR2GRAY);

    //detect and compute SIFT features
    Ptr<SIFT> sift = SIFT::create(3000);
    Mat val1, val2;
    sift->detectAndCompute(g1, noArray(), key1, val1);
    sift->detectAndCompute(g2, noArray(), key2, val2);
    //knn runner with k=2 
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knnOut;
    matcher.knnMatch(val1, val2, knnOut, 2);

    const float tRatio = 0.75f;
    for (size_t i = 0; i < knnOut.size(); i++) {
        if (knnOut[i].size() < 2) continue;
        //best & 2nd best match
        const DMatch& mat1 = knnOut[i][0];
        const DMatch& mat2 = knnOut[i][1];

        //save results 
        if (mat1.distance < tRatio * mat2.distance) {
            match.push_back(mat1);
            p1.push_back(key1[mat1.queryIdx].pt);
            p2.push_back(key2[mat1.trainIdx].pt);
        }
    }
}

// Function to save the matched points
static void savePair(const vector<Point2d>& src,
                           const vector<Point2d>& dist,
                           const vector<int>& ln,
                           const string& outFileName,
                           int maxP = 20)
{
    ofstream fout(outFileName);
    if (!fout) {
        cerr << "Warning: Can't Open file " << outFileName << endl;
        return;
    }
    //header
    fout << "Index\tSrcX\tSrcY\tDstX\tDstY\n";

    int n = min(static_cast<int>(ln.size()), maxP);
    for (int i = 0; i < n; i++) {
        int id = ln[i];
        fout << id << "\t"
             << src[id].x << "\t"
             << src[id].y << "\t"
             << dist[id].x << "\t"
             << dist[id].y << "\n";
    }
    fout.close();
}
// Function to save the match gigs 
static void saveFig(const Mat& image1,
                            const Mat& image2,
                            const vector<KeyPoint>& key1,
                            const vector<KeyPoint>& key2,
                            const vector<DMatch>& matches,
                            const string& outFileName,
                            int maxP = 50)
{
    vector<DMatch> matchList = matches;
    //sor tmatches 
    sort(matchList.begin(), matchList.end(),
         [](const DMatch& a, const DMatch& b) {
             return a.distance < b.distance;
         });
    //keep only top maxP matches
    if (static_cast<int>(matchList.size()) > maxP) {
        matchList.resize(maxP);
    }

    //draw matches
    Mat output;
    drawMatches(image1, key1, image2, key2, matchList, output,
                Scalar::all(-1), Scalar::all(-1), vector<char>(),
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imwrite(outFileName, output);
}
//Function to warp corners (w/ homography)
static vector<Point2d> warpCorner(const HMat& matrix, int width, int height)
{
    vector<Point2d> crn;
    crn.push_back(applyHomo(matrix, Point2d(0, 0)));
    crn.push_back(applyHomo(matrix, Point2d(width - 1, 0)));
    crn.push_back(applyHomo(matrix, Point2d(0, height - 1)));
    crn.push_back(applyHomo(matrix, Point2d(width - 1, height - 1)));

    return crn;
}

// Function to warp and add the image to the panorama
static void warpAdd(const Mat& image,
                       const HMat& homoFwd,
                       Mat& buffSum,
                       Mat& buffCount)
{
    //invert the homography 
    HMat homoInv = invertMat(homoFwd);

    for (int i = 0; i < buffSum.rows; i++) {
        for (int j = 0; j < buffSum.cols; j++) {
            Point2d src = applyHomo(homoInv, Point2d(j, i));
            double x = src.x;
            double y = src.y;
            //check if point is inbounds
            bool inBounds = x >= 0.0 && x <= image.cols - 1.0 &&
                            y >= 0.0 && y <= image.rows - 1.0;
            if (!inBounds) continue;
            //get pixel val
            Vec3d& pix = buffSum.at<Vec3d>(i, j);
            for (int c = 0; c < 3; c++) {
                pix[c] += biInter(image, y, x, c); 
            }

            buffCount.at<double>(i, j) += 1.0;
        }
    }
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
    //parse the input args 
    string leftPath = argv[1];
    string midPath = argv[2];
    string rightPath = argv[3];
    string panoPath = argv[4];
    //parse w*h dims
    int width = atoi(argv[5]);
    int height = atoi(argv[6]);
    //parse output file paths
    string leftMatch=argv[7];
    string rightMatch = argv[8];
    string leftPoint = argv[9];
    string rightPoint = argv[10];

    //Load images
    vector<uint8_t> leftRaw, midRaw, rightRaw;
    readraw(leftPath, leftRaw, width, height, 3);
    readraw(midPath, midRaw, width, height, 3);
    readraw(rightPath, rightRaw, width, height, 3);

    //convert to BGR format
    Mat leftImg = toBGR(leftRaw, width, height);
    Mat midImg = toBGR(midRaw, width, height);
    Mat rightImg = toBGR(rightRaw, width, height);

    // Left to mid match
    vector<Point2d> leftPt, midPoint;
    vector<KeyPoint> keyL, keyM;
    vector<DMatch> lmMatch;
    getSIFT(leftImg, midImg, leftPt, midPoint, keyL, keyM, lmMatch);

    // Right to mid match
    vector<Point2d> rightPt, midPoint2;
    vector<KeyPoint> keyR, keyM2;
    vector<DMatch> rmMatch;
    getSIFT(rightImg, midImg, rightPt, midPoint2, keyR, keyM2, rmMatch);

    if (leftPt.size() < 4 || rightPt.size() < 4) {
        cerr << "Error: not enough matched points." << endl;
        return 1;
    }

    //estimate homographies with RANSAC
    HMat hLM, hRM;
    vector<int> lnL, lnR;

    if (!estHomoRANSAC(leftPt, midPoint, hLM, lnL)) {
        cerr << "Error: failed left to mid." << endl;
        return 1;
    }

    if (!estHomoRANSAC(rightPt, midPoint2, hRM, lnR)) {
        cerr << "Error: failed right to mid ." << endl;
        return 1;
    }
    //Save figures + point tables
    saveFig(leftImg, midImg, keyL, keyM, lmMatch, leftMatch);
    saveFig(rightImg, midImg, keyR, keyM2, rmMatch, rightMatch);
    savePair(leftPt, midPoint, lnL, leftPoint);
    savePair(rightPt, midPoint2, lnR, rightPoint);

    // Create identity homography (middle image)
    HMat identity{};
    identity.mat[0][0] = 1.0; identity.mat[0][1] = 0.0; identity.mat[0][2] = 0.0;
    identity.mat[1][0] = 0.0; identity.mat[1][1] = 1.0; identity.mat[1][2] = 0.0;
    identity.mat[2][0] = 0.0; identity.mat[2][1] = 0.0; identity.mat[2][2] = 1.0;

    //Get the warped corners of 3 images (to find bounds)
    vector<Point2d> totalCorner;
    vector<Point2d> lc = warpCorner(hLM, width, height);
    vector<Point2d> mc = warpCorner(identity, width, height);
    vector<Point2d> rc = warpCorner(hRM, width, height);
    //Warp corners to find panorama bounds
    totalCorner.insert(totalCorner.end(), lc.begin(), lc.end());
    totalCorner.insert(totalCorner.end(), mc.begin(), mc.end());
    totalCorner.insert(totalCorner.end(), rc.begin(), rc.end());
    //find min/max bounds 
    double minX = numeric_limits<double>::max();
    double minY = numeric_limits<double>::max();
    double maxX = -numeric_limits<double>::max();
    double maxY = -numeric_limits<double>::max();

    for (const auto& p : totalCorner) {
        minX = min(minX, p.x), maxX = max(maxX, p.x); //we are updating x bounds
        minY = min(minY, p.y), maxY = max(maxY, p.y); //we are updating y bounds
    }

    //tranlsation matrix to shift panorama to origin
    HMat shiftMatrix{};
    shiftMatrix.mat[0][0] = 1.0; shiftMatrix.mat[0][1] = 0.0; shiftMatrix.mat[0][2] = -minX;
    shiftMatrix.mat[1][0] = 0.0; shiftMatrix.mat[1][1] = 1.0; shiftMatrix.mat[1][2] = -minY;
    shiftMatrix.mat[2][0] = 0.0; shiftMatrix.mat[2][1] = 0.0; shiftMatrix.mat[2][2] = 1.0;
    //get each homography with the shift applied 
    HMat hLP = matrixMult(shiftMatrix, hLM);
    HMat hMP = matrixMult(shiftMatrix, identity);
    HMat hRP = matrixMult(shiftMatrix, hRM);

    int pWidth = static_cast<int>(ceil(maxX - minX + 1.0));
    int pHeight = static_cast<int>(ceil(maxY - minY + 1.0));
    
    //accumulate imgs into sum & count buffers
    Mat buffSum(pHeight, pWidth, CV_64FC3, Scalar(0, 0, 0));
    Mat buffCount(pHeight, pWidth, CV_64F, Scalar(0));

    warpAdd(leftImg, hLP, buffSum, buffCount);
    warpAdd(midImg, hMP, buffSum, buffCount);
    warpAdd(rightImg, hRP, buffSum, buffCount);

    Mat panoImg(pHeight, pWidth, CV_8UC3, Scalar(0, 0, 0));
    //blend values into final panorama
    for (int i = 0; i < pHeight; i++) {
        for (int j = 0; j < pWidth; j++) {
            double cnt = buffCount.at<double>(i, j);

            if (cnt > 0.0) {
                Vec3d sumPixel = buffSum.at<Vec3d>(i, j);
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
    writeraw(panoPath, panoRaw);

    //Output stats
    cout << "Panorama size: " << pWidth << " x " << pHeight << endl;
    cout << "Left-Middle matches: " << leftPt.size()
         << ", inliers: " << lnL.size() << endl;
    cout << "Right-Middle matches: " << rightPt.size()
         << ", inliers: " << lnR.size() << endl;

    return 0;
}