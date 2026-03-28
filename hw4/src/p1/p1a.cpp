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

//Function that makes the n by n identity matrix
static vector< vector<double> > idMat(int val)
{
    vector< vector<double> > id(val, vector<double>(val, 0.0));
    //diagonals = 1
    for (int i = 0; i < val; i++) {
        id[i][i] = 1.0;
    }
    return id;
}

// Function to run Jacobian Eigenval Decomposition for a symmetric matrix
static void jacobDecomp(const vector< vector<double> >& input,
                                     vector<double>& eVal,
                                     vector< vector<double> >& eVec)
{
    int n = static_cast<int>(input.size());
    vector< vector<double> > mat = input;
    eVec = idMat(n);

    const int squareN = n * n;
    //set up to make sure we dont run forever 
    const int maxRun = 100 * squareN; 
    const double ep = 1e-10;

    for (int i = 0; i < maxRun; i++) {
        int p = 0, q = 1;
        double maxOffD = 0.0;
        //finds the largest off diagonal element in mat
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                double val = fabs(mat[a][b]);
                if (val > maxOffD) {
                    maxOffD = val;
                    p = a;
                    q = b;
                }
            }
        }

        //breaks if we are close enough to diagonal
        if (maxOffD < ep) {
            break;
        }

        //get elements for jacobian rotation
        double matpp = mat[p][p];
        double matqq = mat[q][q];
        double matpq = mat[p][q];

        //calculates the rotation angle phi
        double phiVal = 0.5 * atan2(2.0 * matpq, matqq - matpp);
        double cosVal = cos(phiVal); //cosine val 
        double sinVal = sin(phiVal); // sine val 

        //applies the rotation
        for (int k = 0; k < n; k++) {
            if (k != p && k != q) {
                double kpVal = mat[k][p];
                double kqVal = mat[k][q];

                mat[k][p] = cosVal * kpVal - sinVal * kqVal;
                mat[p][k] = mat[k][p];

                mat[k][q] = sinVal * kpVal + cosVal * kqVal;
                mat[q][k] = mat[k][q];
            }
        }

        //updates our diagonal elements
        double ppNew = cosVal * cosVal * matpp - 2.0 * sinVal * cosVal * matpq + sinVal * sinVal * matqq;
        double qqNew = sinVal * sinVal * matpp + 2.0 * sinVal * cosVal * matpq + cosVal * cosVal * matqq;

        mat[p][p] = ppNew;
        mat[q][q] = qqNew;
        mat[p][q] = 0.0;
        mat[q][p] = 0.0;

        //update the eigenvector matrix
        for (int k = 0; k < n; k++) {
            double kpVal = eVec[k][p];
            double kqVal = eVec[k][q];
            eVec[k][p] = cosVal * kpVal - sinVal * kqVal;
            eVec[k][q] = sinVal * kpVal + cosVal * kqVal;
        }
    }

    //store the eigenvalues on diagonal of mat
    eVal.assign(n, 0.0);
    for (int i = 0; i < n; i++) {
        eVal[i] = mat[i][i];
    }
}

//Sort eigenpairs descending
static void sortEigenPairsDescending(vector<double>& eigenvalues,
                                     vector< vector<double> >& eigenvectors)
{
    int n = static_cast<int>(eigenvalues.size());
    vector<int> indices(n);
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }

    sort(indices.begin(), indices.end(),
         [&](int a, int b) { return eigenvalues[a] > eigenvalues[b]; });

    vector<double> sortedValues(n, 0.0);
    vector< vector<double> > sortedVectors(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        sortedValues[i] = eigenvalues[indices[i]];
        for (int r = 0; r < n; r++) {
            sortedVectors[r][i] = eigenvectors[r][indices[i]];
        }
    }

    eigenvalues = sortedValues;
    eigenvectors = sortedVectors;
}

//Project data using first k eigenvectors
static vector< vector<double> > projectDataPCA(const vector< vector<double> >& data,
                                               const vector< vector<double> >& eigenvectors,
                                               int k)
{
    int n = static_cast<int>(data.size());
    int dim = static_cast<int>(data[0].size());

    vector< vector<double> > projected(n, vector<double>(k, 0.0));

    for (int i = 0; i < n; i++) {
        for (int pc = 0; pc < k; pc++) {
            double sum = 0.0;
            for (int d = 0; d < dim; d++) {
                sum += data[i][d] * eigenvectors[d][pc];
            }
            projected[i][pc] = sum;
        }
    }

    return projected;
}

//3x3 matrix inverse
static bool invert3x3(const vector< vector<double> >& A,
                      vector< vector<double> >& invA)
{
    if (A.size() != 3 || A[0].size() != 3) {
        return false;
    }

    double a = A[0][0], b = A[0][1], c = A[0][2];
    double d = A[1][0], e = A[1][1], f = A[1][2];
    double g = A[2][0], h = A[2][1], i = A[2][2];

    double det = a * (e * i - f * h)
               - b * (d * i - f * g)
               + c * (d * h - e * g);

    if (fabs(det) < 1e-12) {
        return false;
    }

    invA.assign(3, vector<double>(3, 0.0));

    invA[0][0] =  (e * i - f * h) / det;
    invA[0][1] = -(b * i - c * h) / det;
    invA[0][2] =  (b * f - c * e) / det;

    invA[1][0] = -(d * i - f * g) / det;
    invA[1][1] =  (a * i - c * g) / det;
    invA[1][2] = -(a * f - c * d) / det;

    invA[2][0] =  (d * h - e * g) / det;
    invA[2][1] = -(a * h - b * g) / det;
    invA[2][2] =  (a * e - b * d) / det;

    return true;
}

//Mahalanobis distance in 3D
static double mahalanobisDistance3D(const vector<double>& x,
                                    const vector<double>& y,
                                    const vector< vector<double> >& invCov)
{
    vector<double> diff(3, 0.0);
    for (int k = 0; k < 3; k++) {
        diff[k] = x[k] - y[k];
    }

    double temp[3] = {0.0, 0.0, 0.0};
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            temp[r] += invCov[r][c] * diff[c];
        }
    }

    double dist2 = 0.0;
    for (int k = 0; k < 3; k++) {
        dist2 += diff[k] * temp[k];
    }

    if (dist2 < 0.0) {
        dist2 = 0.0;
    }
    return sqrt(dist2);
}

//Compute discriminative score for each feature using Fisher style score
static vector<double> computeFeatureDiscriminantScores(const vector< vector<double> >& trainFeatures,
                                                       const vector<string>& trainLabels)
{
    int dim = static_cast<int>(trainFeatures[0].size());
    vector<double> globalMean = computeMeanVector(trainFeatures);

    map<string, vector<int> > classIndices;
    for (size_t i = 0; i < trainLabels.size(); i++) {
        classIndices[trainLabels[i]].push_back(static_cast<int>(i));
    }

    vector<double> scores(dim, 0.0);

    for (int d = 0; d < dim; d++) {
        double between = 0.0;
        double within = 0.0;

        for (map<string, vector<int> >::iterator it = classIndices.begin(); it != classIndices.end(); ++it) {
            const vector<int>& idxs = it->second;
            double mean = 0.0;

            for (size_t k = 0; k < idxs.size(); k++) {
                mean += trainFeatures[idxs[k]][d];
            }
            mean /= static_cast<double>(idxs.size());

            for (size_t k = 0; k < idxs.size(); k++) {
                double diff = trainFeatures[idxs[k]][d] - mean;
                within += diff * diff;
            }

            double diffMean = mean - globalMean[d];
            between += static_cast<double>(idxs.size()) * diffMean * diffMean;
        }

        scores[d] = between / (within + 1e-12);
    }

    return scores;
}

static map<string, string> loadTestLabelMap(const string& filename)
{
    map<string, string> labelMap;
    ifstream file(filename.c_str());

    if (!file) {
        return labelMap;
    }

    string indexToken;
    string label;

    while (file >> indexToken >> label) {
        if (!indexToken.empty() && indexToken[indexToken.size() - 1] == '.') {
            indexToken.pop_back();
        }

        string key = indexToken + ".raw";
        labelMap[key] = toLowercase(label);
    }

    file.close();
    return labelMap;
}
//Write features to CSV
static void writeFeaturesCSV(const string& filename,
                             const vector<string>& filepaths,
                             const vector<string>& labels,
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

    for (size_t i = 0; i < features.size(); i++) {
        file << getFile(filepaths[i]) << "," << labels[i];
        for (size_t d = 0; d < features[i].size(); d++) {
            file << "," << fixed << setprecision(10) << features[i][d];
        }
        file << "\n";
    }

    file.close();
}

//Write PCA points to CSV
static void writePCA3DCSV(const string& filename,
                          const vector<string>& filepaths,
                          const vector<string>& labels,
                          const vector< vector<double> >& projected)
{
    ofstream file(filename.c_str());
    if (!file) {
        cerr << "Error: cannot open CSV file " << filename << endl;
        exit(1);
    }

    file << "filename,label,pc1,pc2,pc3\n";
    for (size_t i = 0; i < projected.size(); i++) {
        file << getFile(filepaths[i]) << "," << labels[i]
             << "," << fixed << setprecision(10) << projected[i][0]
             << "," << fixed << setprecision(10) << projected[i][1]
             << "," << fixed << setprecision(10) << projected[i][2] << "\n";
    }
    file.close();
}

//Write text summary
static void writeSummary(const string& filename,
                         const vector<string>& filterNames,
                         const vector<double>& fisherScores,
                         const vector<double>& eigenvalues,
                         int numTrain,
                         int numTest,
                         bool hasGroundTruth,
                         int numErrors,
                         double errorRate)
{
    ofstream file(filename.c_str());
    if (!file) {
        cerr << "Error: cannot open summary file " << filename << endl;
        exit(1);
    }

    int bestIdx = 0;
    int worstIdx = 0;
    for (size_t i = 1; i < fisherScores.size(); i++) {
        if (fisherScores[i] > fisherScores[bestIdx]) {
            bestIdx = static_cast<int>(i);
        }
        if (fisherScores[i] < fisherScores[worstIdx]) {
            worstIdx = static_cast<int>(i);
        }
    }

    double eigSum = 0.0;
    for (size_t i = 0; i < eigenvalues.size(); i++) {
        if (eigenvalues[i] > 0.0) {
            eigSum += eigenvalues[i];
        }
    }

    double pc1ratio = (eigSum > 0.0) ? eigenvalues[0] / eigSum : 0.0;
    double pc2ratio = (eigSum > 0.0) ? eigenvalues[1] / eigSum : 0.0;
    double pc3ratio = (eigSum > 0.0) ? eigenvalues[2] / eigSum : 0.0;

    file << "EE569 HW4 Problem 1(a) Summary\n";
    file << "----------------------------------------\n";
    file << "Training samples: " << numTrain << "\n";
    file << "Testing samples: " << numTest << "\n\n";

    file << "Most discriminative feature (Fisher score): "
         << filterNames[bestIdx] << "  score = " << fisherScores[bestIdx] << "\n";
    file << "Least discriminative feature (Fisher score): "
         << filterNames[worstIdx] << "  score = " << fisherScores[worstIdx] << "\n\n";

    file << "Top 3 PCA eigenvalues:\n";
    file << "PC1: " << eigenvalues[0] << "  explained ratio = " << pc1ratio << "\n";
    file << "PC2: " << eigenvalues[1] << "  explained ratio = " << pc2ratio << "\n";
    file << "PC3: " << eigenvalues[2] << "  explained ratio = " << pc3ratio << "\n\n";

    file << "Nearest neighbor classification using Mahalanobis distance in 3D PCA space\n";

    if (hasGroundTruth) {
        file << "Test errors: " << numErrors << " / " << numTest << "\n";
        file << "Test error rate: " << errorRate << "\n";
    } else {
        file << "Test ground truth labels were not provided.\n";
        file << "Therefore, only predicted categories are reported for the testing images.\n";
        file << "No numerical test error rate is reported.\n";
    }

    file.close();
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <hw4_root_path>\n";
        cerr << "Example: " << argv[0] << " ../../\n";
        return 1;
    }

    string hw4Root = argv[1];
    string trainDir = hw4Root + "/EE569_2026Spring_HW4_materials/train";
    string testDir  = hw4Root + "/EE569_2026Spring_HW4_materials/test";
    string outputsDir = hw4Root + "/outputs";
    string outputDir = hw4Root + "/outputs/p1";
    string testLabelMapFile = hw4Root + "/EE569_2026Spring_HW4_materials/test/test_label.txt";

    const int width = 128;
    const int height = 128;


    vector<string> trainFiles = listRF(trainDir);
    vector<string> testFiles = listRF(testDir);

    if (trainFiles.empty()) {
        cerr << "Error: no training raw files found in " << trainDir << endl;
        return 1;
    }
    if (testFiles.empty()) {
        cerr << "Error: no testing raw files found in " << testDir << endl;
        return 1;
    }

    cout << "Found " << trainFiles.size() << " training files." << endl;
    cout << "Found " << testFiles.size() << " testing files." << endl;

    map<string, string> testLabelMap = loadTestLabelMap(testLabelMapFile);
    bool hasGroundTruth = !testLabelMap.empty();

    if (hasGroundTruth) {
        cout << "Loaded optional test label map from: " << testLabelMapFile << endl;
    } else {
        cout << "No test label map found. Test labels will be treated as unknown." << endl;
    }

    vector< vector< vector<double> > > filters;
    vector<string> filterNames;
    buildLawsFilters(filters, filterNames);

    vector< vector<double> > trainFeatures;
    vector< vector<double> > testFeatures;
    vector<string> trainLabels;
    vector<string> testLabels;

    //Extract training features
    for (size_t i = 0; i < trainFiles.size(); i++) {
        vector<double> image;
        readGray(trainFiles[i], image, width, height);
        vector<double> features = extractLawsFeatures(image, width, height, filters);

        string label = extractLabel(trainFiles[i]);

        trainFeatures.push_back(features);
        trainLabels.push_back(label);

        cout << "Processed train: " << getFile(trainFiles[i])
             << "  label = " << label << endl;
    }

    //Extract testing features
    for (size_t i = 0; i < testFiles.size(); i++) {
        vector<double> image;
        readGrayImageDouble(testFiles[i], image, width, height);
        vector<double> features = extractLawsFeatures(image, width, height, filters);

        string base = getFile(testFiles[i]);
        string label = "unknown";

        if (hasGroundTruth) {
            map<string, string>::iterator it = testLabelMap.find(base);
            if (it != testLabelMap.end()) {
                label = it->second;
            }
        } else if (!isNumOnly(testFiles[i])) {
            label = extractLabel(testFiles[i]);
        }

        testFeatures.push_back(features);
        testLabels.push_back(label);

        cout << "Processed test:  " << base
             << "  label = " << label << endl;
    }

    //Save raw 25D features first
    writeFeaturesCSV(outputDir + "/train_features_25d.csv", trainFiles, trainLabels, trainFeatures, filterNames);
    writeFeaturesCSV(outputDir + "/test_features_25d.csv", testFiles, testLabels, testFeatures, filterNames);

    //Discriminative score on raw train features
    vector<double> fisherScores = computeFeatureDiscriminantScores(trainFeatures, trainLabels);

    //Normalize using training statistics
    vector<double> trainMean = computeMeanVector(trainFeatures);
    vector<double> trainStd = computeStdVec(trainFeatures, trainMean);

    vector< vector<double> > trainFeaturesNorm = trainFeatures;
    vector< vector<double> > testFeaturesNorm = testFeatures;
    normalizeData(trainFeaturesNorm, trainMean, trainStd);
    normalizeData(testFeaturesNorm, trainMean, trainStd);

    //PCA using training set only
    vector< vector<double> > cov = computeCovariance(trainFeaturesNorm);
    vector<double> eigenvalues;
    vector< vector<double> > eigenvectors;
    jacobiEigenDecomposition(cov, eigenvalues, eigenvectors);
    sortEigenPairsDescending(eigenvalues, eigenvectors);

    vector< vector<double> > trainPCA = projectDataPCA(trainFeaturesNorm, eigenvectors, 3);
    vector< vector<double> > testPCA = projectDataPCA(testFeaturesNorm, eigenvectors, 3);

    writePCA3DCSV(outputDir + "/train_pca_3d.csv", trainFiles, trainLabels, trainPCA);
    writePCA3DCSV(outputDir + "/test_pca_3d.csv", testFiles, testLabels, testPCA);

    //Covariance in projected 3D train space for Mahalanobis metric
    vector< vector<double> > cov3 = computeCovariance(trainPCA);
    for (int d = 0; d < 3; d++) {
        cov3[d][d] += 1e-6;
    }

    vector< vector<double> > invCov3;
    if (!invert3x3(cov3, invCov3)) {
        cerr << "Error: failed to invert 3x3 covariance matrix for Mahalanobis distance." << endl;
        return 1;
    }

    //Nearest neighbor classification
    int numErrors = 0;
    int numEvaluated = 0;

    ofstream predFile((outputDir + "/test_predictions.txt").c_str());
    if (!predFile) {
        cerr << "Error: cannot open prediction output file." << endl;
        return 1;
    }

    predFile << "filename,true_label,predicted_label,mahalanobis_distance\n";

    for (size_t i = 0; i < testPCA.size(); i++) {
        double bestDist = numeric_limits<double>::max();
        int bestIdx = -1;

        for (size_t j = 0; j < trainPCA.size(); j++) {
            double dist = mahalanobisDistance3D(testPCA[i], trainPCA[j], invCov3);
            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = static_cast<int>(j);
            }
        }

        string predicted = trainLabels[bestIdx];
        string truth = testLabels[i];

        if (truth != "unknown") {
            numEvaluated++;
            if (predicted != truth) {
                numErrors++;
            }
        }

        predFile << getFile(testFiles[i]) << ","
                 << truth << ","
                 << predicted << ","
                 << fixed << setprecision(10) << bestDist << "\n";
    }

    predFile.close();

    double errorRate = 0.0;
    bool canReportError = (numEvaluated == static_cast<int>(testFiles.size()) && numEvaluated > 0);
    if (canReportError) {
        errorRate = static_cast<double>(numErrors) / static_cast<double>(numEvaluated);
    }

    writeSummary(outputDir + "/summary.txt",
                 filterNames,
                 fisherScores,
                 eigenvalues,
                 static_cast<int>(trainFiles.size()),
                 static_cast<int>(testFiles.size()),
                 canReportError,
                 numErrors,
                 errorRate);

    cout << "\nDone.\n";
    cout << "25D features saved to: " << outputDir << "/train_features_25d.csv and test_features_25d.csv" << endl;
    cout << "3D PCA points saved to: " << outputDir << "/train_pca_3d.csv and test_pca_3d.csv" << endl;
    cout << "Predictions saved to: " << outputDir << "/test_predictions.txt" << endl;
    cout << "Summary saved to: " << outputDir << "/summary.txt" << endl;

    if (canReportError) {
        cout << "Test error rate: " << errorRate << endl;
    } else {
        cout << "No valid ground-truth labels for all test images, so no numerical test error rate was reported." << endl;
    }

    return 0;
}