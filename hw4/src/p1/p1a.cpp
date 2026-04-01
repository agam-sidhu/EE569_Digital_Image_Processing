/* EE569 Homework #4
 * Name: Agam Sidhu
 * USC ID: 3027948957
 * USC Email: agamsidh@usc.edu
 * Submission Date: April 1, 2026
 * Problem 1(a): Texture Classification using Laws' Filters
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <string>
#include <cstdlib>
#include <cmath>
#include <dirent.h>
#include <map>
#include <set>
#include <limits>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/core.hpp>

using namespace std;

//Clamp function
static inline int clamp(int x, int low, int high) {
    return max(low, min(high, x));
}

//Clamp to double value function
static inline double clampDouble(double x, double low, double high) {
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

//Remove extension from filename
static string removeExtra(const string& filename)
{
    size_t rawPos = filename.find_last_of('.');
    if (rawPos == string::npos) {
        return filename;
    }
    return filename.substr(0, rawPos);
}

//Function that returns the file basename
static string getFile(const string& path)
{
    //last account for / & '\\'
    size_t p1 = path.find_last_of('/');
    size_t p2 = path.find_last_of('\\');
    size_t pos = string::npos;

    if (p1 == string::npos && p2 == string::npos) { //if not path separator is found -> we already have file name 
        pos = string::npos;
    } else if (p1 == string::npos) {
        pos = p2; //in case yall use windows
    } else if (p2 == string::npos) {
        pos = p1; // forward slash -> mac verison
    } else {
        pos = max(p1, p2); // extreme case where both are found
    }

    if (pos == string::npos) {
        return path; // if no separator found
    }
    return path.substr(pos + 1);
}

//to loercase function
static string toLowercase(string str)
{
    for (size_t i = 0; i < str.size(); i++) {
        if (str[i] >= 'A' && str[i] <= 'Z') { //goes through each char & if upper -> to lower
            str[i] = static_cast<char>(str[i] - 'A' + 'a');
        }
    }
    return str;
}

//Functio to see if we only have digits in filename 
// KEY FOR TEST DIRECTORY (it has 1.raw, 2.raw etc)
static bool isNumOnly(const string& filepath)
{
    string filename = removeExtra(getFile(filepath));
    if (filename.empty()) { //case if no file name is foudn
        return false;
    } 

    for (size_t i = 0; i < filename.size(); i++) { //ccheck if each char in filename is a digit
        if (!(filename[i] >= '0' && filename[i] <= '9')) {
            return false; 
        }
    }
    return true;
}

//Function to return the texture labels from training directory 
static string extractLabel(const string& filepath)
{
    string label = "";
    string filename = removeExtra(getFile(filepath));

    //get the alphabetic prefix as the label
    for (size_t i = 0; i < filename.size(); i++) {
        char curr = filename[i];
        if ((curr >= 'A' && curr <= 'Z') || (curr >= 'a' && curr <= 'z')) {
            //if upper convert to lower
            if (curr >= 'A' && curr <= 'Z') {
                curr = static_cast<char>(curr - 'A' + 'a');
            }
            label.push_back(curr);
        //stop at first digit if found
        } else if (curr >= '0' && curr <= '9') {
            break;
        }
    }

    //key if no label is found
    if (label.empty()) {
        label = "unknown";
    }
    return label;
}

//Function to list raw files in a directory
static vector<string> listRF(const string& path)
{
    vector<string> files;
    DIR* dir = opendir(path.c_str());

    if (!dir) {
        cerr << "Error: Can't open directory " << path << endl;
        exit(1);
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }

        string lower = toLowercase(name);
        if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".raw") {
            files.push_back(path + "/" + name);
        }
    }

    closedir(dir);
    sort(files.begin(), files.end());
    return files;
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

//Function to computer the mean abs energy of filter response
static double calcEnergy(const vector<double>& res)
{
    double eSum = 0.0;
    //accumlate the abs val of all filter responses
    for (size_t i = 0; i < res.size(); i++) {
        eSum += fabs(res[i]);
    }
    //calc average energy 
    double sol = eSum / static_cast<double>(res.size());
    return sol;
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

//Function to extract the 25 dim Law's features vectors for an img
static vector<double> extractLFs(const vector<double>& image,
                                          int width,
                                          int height,
                                          const vector< vector< vector<double> > >& filters)
{
    //Subtract the mean intensity from image to get zero mean image
    vector<double> zeroMean = image;
    meanSub(zeroMean);

    vector<double> feat(filters.size(), 0.0);

    //Goes through each filter & compute the mean abs energy
    for (size_t idx = 0; idx < filters.size(); idx++) {
        double kernel[5][5];
        //copies the filter into a kernal array
        for (int r = 0; r < 5; r++) {
            for (int c = 0; c < 5; c++) {
                kernel[r][c] = filters[idx][r][c];
            }
        }

        //convole img w/ filter
        vector<double> result;
        convolveFive(zeroMean, result, width, height, kernel);
        //computer the mean abs energy of filter response
        feat[idx] = calcEnergy(result);
    }

    return feat;
}

//Calculate the mean vector for samples
static vector<double> calcMeanVec(const vector< vector<double> >& data)
{
    //just incase we have have empty data
    if (data.empty()) {
        return vector<double>();
    }

    int dNum = static_cast<int>(data[0].size());
    vector<double> mVec(dNum, 0.0);

    //gets the values for each feature dim
    for (size_t idx = 0; idx < data.size(); idx++) {
        for (int dim = 0; dim < dNum; dim++) {
            mVec[dim] += data[idx][dim];
        }
    }

    //divides the # of samples to get mean for each feature dim
    for (int dim = 0; dim < dNum; dim++) {
        mVec[dim] /= static_cast<double>(data.size());
    }

    return mVec;
}

//Compute standard deviation vector across samples
static vector<double> calcStdVec(const vector< vector<double> >& data,
                                       const vector<double>& mean)
{
    int dNum = static_cast<int>(mean.size());
    vector<double> sVec(dNum, 0.0);

    //get squared differences from mean
    for (size_t idx = 0; idx < data.size(); idx++) {
        for (int dim = 0; dim < dNum; dim++) {
            double diff = data[idx][dim] - mean[dim];
            sVec[dim] += diff * diff;
        }
    }

    //Calc standard dev for each feature dim
    for (int dim = 0; dim < dNum; dim++) {
        sVec[dim] = sqrt(sVec[dim] / static_cast<double>(data.size()));
        //just to make sure we dont get zero during norm
        if (sVec[dim] < 1e-12) {
            sVec[dim] = 1.0;
        }
    }

    return sVec;
}

//Function to normalize the data (apply z-score norm)
static void normData(vector< vector<double> >& data,
                          const vector<double>& mean,
                          const vector<double>& std)
{
    //normalizes each feature dim 
    for (size_t idx = 0; idx < data.size(); idx++) {
        for (size_t dim = 0; dim < mean.size(); dim++) {
            data[idx][dim] = (data[idx][dim] - mean[dim]) / std[dim];
        }
    }
}

//Computers the covariance matrix for the data 
static vector< vector<double> > calcCovar(const vector< vector<double> >& data)
{
    int nSamp = static_cast<int>(data.size());
    int dNum = static_cast<int>(data[0].size());

    //calc the mean vector of data
    vector<double> mean = calcMeanVec(data);
    //intialize the covar matrix
    vector< vector<double> > covar(dNum, vector<double>(dNum, 0.0));

    //gets the covariance values
    for (int i = 0; i < nSamp; i++) {
        for (int r = 0; r < dNum; r++) {
            for (int c = 0; c < dNum; c++) {
                covar[r][c] += (data[i][r] - mean[r]) * (data[i][c] - mean[c]);
            }
        }
    }

    //divides by n-1 to get the final covariance matrix
    double deno;
    if (nSamp > 1) {
        deno = static_cast<double>(nSamp - 1);
    } else {
        deno = 1.0; 
    }

    //normalize the covariance matrix
    for (int r = 0; r < dNum; r++) {
        for (int c = 0; c < dNum; c++) {
            covar[r][c] /= deno;
        }
    }

    return covar;
}

//Function to convert vector data into rows
static cv::Mat vecToMat(const vector< vector<double> >& data)
{
    int nSamples = static_cast<int>(data.size());
    int dimNum = static_cast<int>(data[0].size());
    cv::Mat matrix(nSamples, dimNum, CV_64F);

    for (int row = 0; row < nSamples; row++) {
        for (int col = 0; col < dimNum; col++) {
            matrix.at<double>(row, col) = data[row][col];
        }
    }

    return matrix;
}

//Function to convert cv::Mat rows back into vector data
static vector< vector<double> > matToVec(const cv::Mat& matrix)
{
    vector< vector<double> > result(matrix.rows, vector<double>(matrix.cols, 0.0));

    for (int row = 0; row < matrix.rows; row++) {
        for (int col = 0; col < matrix.cols; col++) {
            result[row][col] = matrix.at<double>(row, col);
        }
    }

    return result;
}

//Function to convert PCA eigenvalues into a vector
static vector<double> matToValVec(const cv::Mat& matrix)
{
    vector<double> output(matrix.rows, 0.0);
    for (int row = 0; row < matrix.rows; row++) {
        output[row] = matrix.at<double>(row, 0);
    }
    return output;
}

//Function to get inverse of 3x3 matrix (sets false if not 3x3 or singular)
static bool invertMat(const vector< vector<double> >& mat,
                      vector< vector<double> >& invMat)
{
    //Makes sure that we have 3x3 matrix
    if (mat.size() != 3 || mat[0].size() != 3) {
        return false;
    }

    //Gets trhe matrix elements
    double a = mat[0][0];
    double b = mat[0][1];
    double c = mat[0][2];
    double d = mat[1][0];
    double e = mat[1][1];
    double f = mat[1][2];
    double g = mat[2][0];
    double h = mat[2][1];
    double i = mat[2][2];

    //calculates determinant
    double upper = e * i - f * h;
    double mid = d * i - f * g;
    double lower = d * h - e * g;
    double det = a * upper - b * mid + c * lower;

    //Matrix = singular if our determinant is close to zero -> false
    if (fabs(det) < 1e-12) {
        return false;
    }

    //calculates the inv by adjuate/determinant
    invMat.assign(3, vector<double>(3, 0.0));
    invMat[0][0] = upper / det;
    invMat[0][1] = -(b * i - c * h) / det;
    invMat[0][2] = (b * f - c * e) / det;
    invMat[1][0] = -(mid) / det;
    invMat[1][1] = (a * i - c * g) / det;
    invMat[1][2] = -(a * f - c * d) / det;
    invMat[2][0] = (lower) / det;
    invMat[2][1] = -(a * h - b * g) / det;
    invMat[2][2] = (a * e - b * d) / det;

    return true;
}

//This function will calculate the Mahalanobis distance 
static double calcMahaDist(const vector<double>& x,
                                    const vector<double>& y,
                                    const vector< vector<double> >& icMatrix)
{
    //calculates the difference between vector x &y
    vector<double> diff(3, 0.0);
    for (int i = 0; i < 3; i++) {
        diff[i] = x[i] - y[i];
    }
    //multiplies the inverse covar matrix with the difference vector
    double result[3] = {0.0, 0.0, 0.0};
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            result[r] += icMatrix[r][c] * diff[c];
        }
    }

    //calculates the dot product from the result of matrix & diff vector 
    double sqaured = 0.0;
    for (int k = 0; k < 3; k++) {
        sqaured += diff[k] * result[k];
    }

    //to prvent negative values
    if (sqaured < 0.0) {
        sqaured = 0.0;
    }
    return sqrt(sqaured);
}

//Compute discriminative score for each feature using Fisher style score
static vector<double> calcFDScore(const vector< vector<double> >& trainFeatures,
                                                       const vector<string>& trainLabels)
{
    int featDim = static_cast<int>(trainFeatures[0].size()); //#of feature dims
    vector<double> total = calcMeanVec(trainFeatures); // Total "global" mean vector across all samples

    //this groups sample idx by their class label
    map<string, vector<int> > classIdx;
    for (size_t idx = 0; idx < trainLabels.size(); idx++) {
        classIdx[trainLabels[idx]].push_back(static_cast<int>(idx));
    }

    vector<double> featScore(featDim, 0.0);

    //evaluates 1 feature dim 
    for (int d = 0; d < featDim; d++) {
        double betweenClass = 0.0;
        double inClass = 0.0;

        //go through each class 
        for (map<string, vector<int> >::iterator it = classIdx.begin(); it != classIdx.end(); ++it) {
            const vector<int>& idxs = it->second;
            double mean = 0.0;

            //calculates the mean for current class & feature dim 
            for (size_t k = 0; k < idxs.size(); k++) {
                mean += trainFeatures[idxs[k]][d];
            }
            //takes # of samples so we can get mean
            mean /= static_cast<double>(idxs.size());

            //get the inclass variance
            for (size_t k = 0; k < idxs.size(); k++) {
                double dev = trainFeatures[idxs[k]][d] - mean;
                inClass += dev * dev;
            }

            //get the between class variance
            double devMean = mean - total[d];
            betweenClass += static_cast<double>(idxs.size()) * devMean * devMean;
        }

        //fisher score = between class / in class
        featScore[d] = betweenClass / (inClass + 1e-12);
    }

    return featScore;
}
//FUnction to load test labels
static map<string, string> loadTestLabels(const string& filename)
{
    map<string, string> label;
    ifstream file(filename.c_str());

    //in case we have empty file
    if (!file) {
        return label;
    }

    string idxToken;
    string classLabel;

    while (file >> idxToken >> classLabel) {
        if (!idxToken.empty() && idxToken[idxToken.size() - 1] == '.') {
            idxToken.pop_back();
        }

        //takes index into raw filename 
        string key = idxToken + ".raw";
        label[key] = toLowercase(classLabel);
    }

    file.close();
    return label;
}
//function to write the features to csv
static void writeFCSV(const string& filename,
                             const vector<string>& path,
                             const vector<string>& label,
                             const vector< vector<double> >& features,
                             const vector<string>& featureNames)
{
    ofstream file(filename.c_str());
    if (!file) {
        cerr << "Error: cannot open CSV file " << filename << endl;
        exit(1);
    }

    file << "filename,label";
    for (size_t d = 0; d < featureNames.size(); d++) {
        file << "," << featureNames[d];
    }
    file << "\n";

    //goes through each sample + writes name, label, & feature vals ->csv
    for (size_t i = 0; i < features.size(); i++) {
        file << getFile(path[i]) << "," << label[i];
        for (size_t d = 0; d < features[i].size(); d++) {
            file << "," << fixed << setprecision(10) << features[i][d];
        }
        file << "\n";
    }

    file.close();
}

//Function to project PCA results into csv
static void writePCA(const string& filename,
                          const vector<string>& path,
                          const vector<string>& labels,
                          const vector< vector<double> >& proj)
{
    //writes label filename + pc vals into csv
    ofstream file(filename.c_str());
    if (!file) {
        cerr << "Error: cannot open CSV file " << filename << endl;
        exit(1);
    }

    file << "filename,label,pc1,pc2,pc3\n";
    for (size_t i = 0; i < proj.size(); i++) {
        file << getFile(path[i]) << "," << labels[i]
             << "," << fixed << setprecision(10) << proj[i][0]
             << "," << fixed << setprecision(10) << proj[i][1]
             << "," << fixed << setprecision(10) << proj[i][2] << "\n";
    }
    file.close();
}

//Write text summary
static void summarize(const string& output,
                         const vector<string>& filter,
                         const vector<double>& fScore,
                         const vector<double>& eVals,
                         int trainSamples,
                         int testSamples,
                         bool hasGT,
                         int numErrors,
                         double testError)
{

    ofstream file(output.c_str());
    if (!file) {
        cerr << "Error: cannot open summary file " << output << endl;
        exit(1);
    }

    //finds best & worst feature 
    int bestIdx = 0, worstIdx = 0;
    for (size_t i = 1; i < fScore.size(); i++) {
        if (fScore[i] > fScore[bestIdx]) {
            bestIdx = static_cast<int>(i);
        }
        if (fScore[i] < fScore[worstIdx]) {
            worstIdx = static_cast<int>(i);
        }
    }

    //gets sum of positive eigenvalues
    double eSum = 0.0;
    for (size_t i = 0; i < eVals.size(); i++) {
        if (eVals[i] > 0.0) {
            eSum += eVals[i];
        }
    }

    //calculates the variance ratio for the 3 top PCw
    double pc1 = (eSum > 0.0) ? eVals[0] / eSum : 0.0;
    double pc2 = (eSum > 0.0) ? eVals[1] / eSum : 0.0;
    double pc3 = (eSum > 0.0) ? eVals[2] / eSum : 0.0;

    //writes summary of results into file
    file << "Problem 1(a) Summary\n";
    file << "----------------------------------------\n";
    file << "Training samples: " << trainSamples << "\n";
    file << "Testing samples: " << testSamples << "\n\n";

    file << "Most discriminative feature: "
         << filter[bestIdx] << "  score = " << fScore[bestIdx] << "\n";
    file << "Least discriminative feature (Fisher score): "
         << filter[worstIdx] << "  score = " << fScore[worstIdx] << "\n\n";

    file << "Top 3 PCA EignValues:\n";
    file << "PC1: " << eVals[0] << "  explained ratio = " << pc1 << "\n";
    file << "PC2: " << eVals[1] << "  explained ratio = " << pc2 << "\n";
    file << "PC3: " << eVals[2] << "  explained ratio = " << pc3 << "\n\n";

    file << "Nearest neighbor classification using Mahalanobis distance in 3D PCA space\n";

    if (hasGT) {
        file << "Test errors: " << numErrors << " / " << testSamples << "\n";
        file << "Test error rate: " << testError << "\n";
    } else {
        file << "Test Ground Truth was not provided.\n";
        file << "Only predicted categories are reported for the testing images.\n";
        file << "No numerical test error rate is reported.\n";
    }

    file.close();
}
//Main Function
int main(int argc, char* argv[])
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <hw4_root_path>\n";
        cerr << "Example: " << argv[0] << " ../../\n";
        return 1;
    }

    string hw4Root = argv[1];
    string trRepo = hw4Root + "/EE569_2026Spring_HW4_materials/train";
    string tRepo  = hw4Root + "/EE569_2026Spring_HW4_materials/test";
    string output = hw4Root + "/outputs";
    string outputRepo = hw4Root + "/outputs/p1";
    string tLabelFile = hw4Root + "/EE569_2026Spring_HW4_materials/test/test_label.txt";

    const int width = 128;
    const int height = 128;

    //Get training and testing file lists
    vector<string> trFile = listRF(trRepo);
    vector<string> tFile = listRF(tRepo);

    //incase train + test files are empty
    if (trFile.empty()) {
        cerr << "Error: no training raw files found in " << trRepo << endl;
        return 1;
    }
    if (tFile.empty()) {
        cerr << "Error: no testing raw files found in " << tRepo << endl;
        return 1;
    }

    cout << "Found " << trFile.size() << " training files." << endl;
    cout << "Found " << tFile.size() << " testing files." << endl;

    //loads the ground truth labels 
    map<string, string> tLabelSet = loadTestLabels(tLabelFile);
    bool hasGT = !tLabelSet.empty();

    //checking to see if ground truth labels exist
    if (hasGT) {
        cout << "Loaded optional test label map from: " << tLabelFile << endl;
    } else {
        cout << "No test label map found. Test labels will be treated as unknown." << endl;
    }

    //Generate the 25 Laws filters
    vector< vector< vector<double> > > filters;
    vector<string> filterNames;
    genLFs(filters, filterNames);

    vector< vector<double> > trFeatures;
    vector< vector<double> > tFeatures;
    vector<string> trLabels;
    vector<string> testLabels;

    //Extract training features
    for (size_t i = 0; i < trFile.size(); i++) {
        vector<double> image;
        readGray(trFile[i], image, width, height);
        vector<double> features = extractLFs(image, width, height, filters);

        string label = extractLabel(trFile[i]);

        trFeatures.push_back(features);
        trLabels.push_back(label);

        cout << "Processed train: " << getFile(trFile[i])
             << "  label = " << label << endl;
    }

    //Extract testing features
    for (size_t i = 0; i < tFile.size(); i++) {
        vector<double> image;
        readGray(tFile[i], image, width, height);
        vector<double> features = extractLFs(image, width, height, filters);

        string name = getFile(tFile[i]);
        string label = "unknown";

        if (hasGT) {
            map<string, string>::iterator it = tLabelSet.find(name);
            if (it != tLabelSet.end()) {
                label = it->second;
            }
        } else if (!isNumOnly(tFile[i])) {
            label = extractLabel(tFile[i]);
        }

        tFeatures.push_back(features);
        testLabels.push_back(label);

        cout << "Processed test:  " << name
             << "  label = " << label << endl;
    }

    //Write raw 25D feature vectors
    writeFCSV(outputRepo + "/train_features_25d.csv",
              trFile, trLabels, trFeatures, filterNames);

    writeFCSV(outputRepo + "/test_features_25d.csv",
              tFile, testLabels, tFeatures, filterNames);

    //Compute Fisher discriminant scores on raw training features
    vector<double> fScore = calcFDScore(trFeatures, trLabels);

    //Normalize features using training statistics
    vector<double> trMean = calcMeanVec(trFeatures);
    vector<double> trStd = calcStdVec(trFeatures, trMean);

    vector< vector<double> > normTrFeat = trFeatures;
    vector< vector<double> > normTFeat = tFeatures;

    normData(normTrFeat, trMean, trStd);
    normData(normTFeat, trMean, trStd);

    //PCA using normalized training features
    cv::Mat trainDataMat = vecToMat(normTrFeat);
    cv::Mat testDataMat = vecToMat(normTFeat);
    cv::PCA pca(trainDataMat, cv::Mat(), cv::PCA::DATA_AS_ROW, 3);

    vector<double> eVals = matToValVec(pca.eigenvalues);
    cv::Mat trainPCAMat = pca.project(trainDataMat);
    cv::Mat testPCAMat = pca.project(testDataMat);

    vector< vector<double> > trPCA = matToVec(trainPCAMat);
    vector< vector<double> > tPCA = matToVec(testPCAMat);

    //Write 3D PCA results
    writePCA(outputRepo + "/train_pca_3d.csv", trFile, trLabels, trPCA);
    writePCA(outputRepo + "/test_pca_3d.csv", tFile, testLabels, tPCA);

    //Calulculates covar in 3D PCA space for Mahalanobis distance
    vector< vector<double> > cov3 = calcCovar(trPCA);
    for (int d = 0; d < 3; d++) {
        cov3[d][d] += 1e-6;
    }

    vector< vector<double> > invCov3;
    if (!invertMat(cov3, invCov3)) {
        cerr << "Error: failed to invert 3x3 covariance matrix" << endl;
        return 1;
    }

    //Nearest-neighbor classification
    int numErrors = 0;
    int evalNum = 0;

    ofstream predFile((outputRepo + "/test_predictions.txt").c_str());
    if (!predFile) {
        cerr << "Error: cannot open prediction output file." << endl;
        return 1;
    }

    predFile << "filename,true_label,predicted_label,mahalanobis_distance\n";

    for (size_t i = 0; i < tPCA.size(); i++) {
        double bestDist = numeric_limits<double>::max();
        int bestTrainIdx = -1;

        for (size_t j = 0; j < trPCA.size(); j++) {
            double dist = calcMahaDist(tPCA[i], trPCA[j], invCov3);
            if (dist < bestDist) {
                bestDist = dist;
                bestTrainIdx = static_cast<int>(j);
            }
        }

        string predictedLabel = trLabels[bestTrainIdx];
        string trueLabel = testLabels[i];

        if (trueLabel != "unknown") {
            evalNum++;
            if (predictedLabel != trueLabel) {
                numErrors++;
            }
        }

        predFile << getFile(tFile[i]) << ","
                 << trueLabel << ","
                 << predictedLabel << ","
                 << fixed << setprecision(10) << bestDist << "\n";
    }

    predFile.close();

    double er = 0.0;
    bool canReportError = (evalNum == static_cast<int>(tFile.size()) && evalNum > 0);
    if (canReportError) {
        er = static_cast<double>(numErrors) / static_cast<double>(evalNum);
    }

    summarize(outputRepo + "/summary.txt",
              filterNames,
              fScore,
              eVals,
              static_cast<int>(trFile.size()),
              static_cast<int>(tFile.size()),
              canReportError,
              numErrors,
              er);

    cout << "\nDone.\n";
    cout << "25D features saved to: " << outputRepo << "/train_features_25d.csv and "
         << outputRepo << "/test_features_25d.csv" << endl;
    cout << "3D PCA points saved to: " << outputRepo << "/train_pca_3d.csv and "
         << outputRepo << "/test_pca_3d.csv" << endl;
    cout << "Predictions saved to: " << outputRepo << "/test_predictions.txt" << endl;
    cout << "Summary saved to: " << outputRepo << "/summary.txt" << endl;

    if (canReportError) {
        cout << "Test error rate: " << er << endl;
    } else {
        cout << "No valid groundtruth labels for all test images -> No test error returned." << endl;
    }

    return 0;
}