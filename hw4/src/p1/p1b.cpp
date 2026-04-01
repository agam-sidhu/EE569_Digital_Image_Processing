/* EE569 Homework #4
 * Name: Agam Sidhu
 * USC ID: 3027948957
 * USC Email: agamsidh@usc.edu
 * Submission Date: [fill]
 * Problem 1(b): Advanced Texture Classification
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

using namespace std;

struct FeatData { //struct that will hold the feature data
    vector<string> fileName;
    vector<string> labels;
    vector< vector<double> > features;
};

struct KMCluster { //struct that will store cluster results
    vector<int> clusterLabels;
    vector< vector<double> > centroids;
};

struct ClusterResults { // struct that will store the cluster evaluation results
    vector<string> predLabels; // predicted labels
    vector<string> domLabels; //dominant labels
    int numErrors;
    double errorRate;
};

struct SVMModel { // struct that will hold SVM model params
    cv::Ptr<cv::ml::SVM> model;
    map<int, string> intLabelMap;
    map<string, int> strLabelMap;
};

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
//function to sepearate the CSV row 
static vector<string> splitRows(const string& ln)
{
    vector<string> tkns;
    string curr;
    //splits at commas
    for (size_t i = 0; i < ln.size(); i++) {
        if (ln[i] == ',') {
            tkns.push_back(curr);
            curr.clear();
        } else {
            curr.push_back(ln[i]);
        }
    }
    //adds the last field
    tkns.push_back(curr);
    return tkns;
}

//Function to read feature data 
static FeatData readFeat(const string& input)
{
    FeatData dataset;
    //check in case we can't open file
    ifstream file(input.c_str());
    if (!file) {
        cerr << "Error: cannot open CSV file " << input << endl;
        exit(1);
    }

    //makes sure we have non-empty file (also skips header)
    string ln;
    if (!getline(file, ln)) {
        cerr << "Error: empty CSV file " << input << endl;
        exit(1);
    }
    //reads each line 
    while (getline(file, ln)) {
        if (ln.empty()) {
            continue;
        }
        //splits the line into fileName, label, and features
        vector<string> field = splitRows(ln);
        if (field.size() < 3) {
            cerr << "Error: bad CSV row in " << input << endl;
            exit(1);
        }

        //stores the file name, label, and features (in dataset struct)
        dataset.fileName.push_back(field[0]);
        dataset.labels.push_back(toLowercase(field[1]));

        //sets feature values from string -> DOUBLE
        vector<double> featVal;
        for (size_t i = 2; i < field.size(); i++) {
            featVal.push_back(atof(field[i].c_str()));
        }
        dataset.features.push_back(featVal);
    }

    file.close();
    return dataset;
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

//Function to get the squared distance between 2 feature vectors
static double calcSquareDist(const vector<double>& x, const vector<double>& y)
{
    double result = 0.0;
    //adds squared distance for each feature dim
    for (size_t i = 0; i < x.size(); i++) {
        double diff = x[i] - y[i];
        result += diff * diff;
    }
    return result;
}

//Function to convert vector data into cv::Mat rows
static cv::Mat vecToMat(const vector< vector<double> >& data)
{
    int sampleNum = static_cast<int>(data.size());
    int dimNum = static_cast<int>(data[0].size());
    cv::Mat dataMat(sampleNum, dimNum, CV_32F);

    for (int row = 0; row < sampleNum; row++) {
        for (int col = 0; col < dimNum; col++) {
            dataMat.at<float>(row, col) = static_cast<float>(data[row][col]);
        }
    }

    return dataMat;
}
//Function to run the KMeans algo
static KMCluster KMeans(const vector< vector<double> >& features,
                              int clusterK,
                              int maxVal)
{
    //check in case we get empty input
    if (features.empty()) {
        cerr << "Error: Received empty data." << endl;
        exit(1);
    }
    KMCluster outputerCluster;
    int sampleNum = static_cast<int>(features.size());
    int dimNum = static_cast<int>(features[0].size());
    cv::Mat dataMat = vecToMat(features);
    cv::Mat labelMat(sampleNum, 1, CV_32S);

    //intialize the centroids 
    for (int i = 0; i < sampleNum; i++) {
        labelMat.at<int>(i, 0) = (i * clusterK) / sampleNum;
    }

    cv::Mat centerMat;
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                              maxVal,
                              1e-6);
    cv::kmeans(dataMat,
               clusterK,
               labelMat,
               criteria,
               1,
               cv::KMEANS_USE_INITIAL_LABELS,
               centerMat);

    outputerCluster.clusterLabels.assign(sampleNum, 0);
    outputerCluster.centroids.assign(clusterK, vector<double>(dimNum, 0.0));

    for (int i = 0; i < sampleNum; i++) {
        outputerCluster.clusterLabels[i] = labelMat.at<int>(i, 0);
    }

    for (int row = 0; row < clusterK; row++) {
        for (int col = 0; col < dimNum; col++) {
            outputerCluster.centroids[row][col] = static_cast<double>(centerMat.at<float>(row, col));
        }
    }

    return outputerCluster;
}

//Function to calculate the cluster result 
static ClusterResults evalCluster(const vector<int>& cLabel,
                                    const vector<string>& trueLabels,
                                    int clusterK)
{
    ClusterResults result;

    result.predLabels.assign(trueLabels.size(), "unknown");
    result.domLabels.assign(clusterK, "unknown");
    result.errorRate = 0.0;
    result.numErrors = 0;

    //determines dominant label for each cluster
    for (int c = 0; c < clusterK; c++) {
        map<string, int> count;
        for (size_t i = 0; i < cLabel.size(); i++) {
            if (cLabel[i] == c) {
                count[trueLabels[i]]++;
            }
        }
        int maxCount = -1;
        string bLabel = "unknown";
        for (map<string, int>::iterator it = count.begin(); it != count.end(); ++it) {
            if (it->second > maxCount) {
                maxCount = it->second;
                bLabel = it->first;
            }
        }
        result.domLabels[c] = bLabel;
    }

    //converts the cluster to preducted label + counts the errors
    for (size_t i = 0; i < cLabel.size(); i++) {
        result.predLabels[i] = result.domLabels[cLabel[i]];
        if (result.predLabels[i] != trueLabels[i]) {
            result.numErrors++;
        }
    }
    // calculates the error rate
    if (!trueLabels.empty()) {
        result.errorRate = static_cast<double>(result.numErrors) / static_cast<double>(trueLabels.size());
    }

    return result;
}
//Trains a multi class SVM (one v rest approach)
static vector<SVMModel> trainMClass(const vector< vector<double> >& features,
                                              const vector<string>& labels)
{
    vector<SVMModel> mdls(1);
    cv::Mat dataMat = vecToMat(features);
    cv::Mat labelMat(static_cast<int>(labels.size()), 1, CV_32S);
    set<string> uniLabel(labels.begin(), labels.end());

    int currIdx = 0;
    for (set<string>::iterator lbIt = uniLabel.begin(); lbIt != uniLabel.end(); ++lbIt) {
        mdls[0].strLabelMap[*lbIt] = currIdx;
        mdls[0].intLabelMap[currIdx] = *lbIt;
        currIdx++;
    }

    for (size_t i = 0; i < labels.size(); i++) {
        labelMat.at<int>(static_cast<int>(i), 0) = mdls[0].strLabelMap[labels[i]];
    }

    mdls[0].model = cv::ml::SVM::create();
    mdls[0].model->setType(cv::ml::SVM::C_SVC);
    mdls[0].model->setKernel(cv::ml::SVM::LINEAR);
    mdls[0].model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-6));
    mdls[0].model->train(dataMat, cv::ml::ROW_SAMPLE, labelMat);

    return mdls;
}

//Predicts the labels using the multi class SVM models
static vector<string> predMClass(const vector< vector<double> >& features,
                                           const vector<SVMModel>& models)
{   
    vector<string> preds(features.size(), "unknown");
    cv::Mat dataMat = vecToMat(features);

    //goes through the samples & chooses highest scoring classifer 
    for (int i = 0; i < dataMat.rows; i++) {
        float labelNum = models[0].model->predict(dataMat.row(i));
        int predIdx = static_cast<int>(labelNum);
        preds[i] = models[0].intLabelMap.find(predIdx)->second;
    }

    return preds;
}

//Function to calculate the error rate
static double calcER(const vector<string>& truthVal,
                            const vector<string>& predVal,
                            int& eCount)
{
    eCount = 0;
    //counts # of errors
    for (size_t i = 0; i < truthVal.size(); i++) {
        if (truthVal[i] != predVal[i]) {
            eCount++;
        }
    }

    if (truthVal.empty()) {
        return 0.0;
    }
    //calculates the error rate
    double result = static_cast<double>(eCount) / static_cast<double>(truthVal.size());
    return result;
}
//Write Predictions function
static void writePreds(const string& output,
                               const vector<string>& filename,
                               const vector<string>& tLabels,
                               const vector<string>& preds)
{
    ofstream file(output.c_str());
    if (!file) {
        cerr << "Error: cannot open output file " << output << endl;
        exit(1);
    }

    file << "filename,true_label,predicted_label\n";
    for (size_t i = 0; i < filename.size(); i++) {
        file << filename[i] << "," << tLabels[i] << "," << preds[i] << "\n";
    }

    file.close();
}
//Function to write the KMeans results
static void writeKM(const string& output,
                              const vector<string>& filename,
                              const vector<string>& tLabels,
                              const vector<int>& cLabels,
                              const ClusterResults& res)
{
    ofstream file(output.c_str());
    if (!file) {
        cerr << "Error: cannot open output file " << output << endl;
        exit(1);
    }

    file << "filename,true_label,cluster_id,cluster_majority_label,predicted_label\n";
    for (size_t i = 0; i < filename.size(); i++) {
        int cluster = cLabels[i];
        file << filename[i] << ","
             << tLabels[i] << ","
             << cluster << ","
             << res.domLabels[cluster] << ","
             << res.predLabels[i] << "\n";
    }

    file.close();
}
//Write text summary
static void summarize(const string& output,
                         int km25Error,
                         double km25ER,
                         int km3Error,
                         double km3ER,
                         int svm25Error,
                         double svm25ER,
                         int svm3Error,
                         double svm3ER)
{
    ofstream file(output.c_str());
    if (!file) {
        cerr << "Error: cannot open summary file " << output << endl;
        exit(1);
    }

    file << "Problem 1(b) Summary\n";
    file << "----------------------------------------\n";
    file << "Unsupervised K-means on 25-D test features\n";
    file << "Errors: " << km25Error << "\n";
    file << "Error rate: " << km25ER << "\n\n";

    file << "Unsupervised K-means on 3-D PCA test features\n";
    file << "Errors: " << km3Error << "\n";
    file << "Error rate: " << km3ER << "\n\n";

    file << "Supervised one-vs-rest linear SVM on 25-D features\n";
    file << "Errors: " << svm25Error << "\n";
    file << "Error rate: " << svm25ER << "\n\n";

    file << "Supervised one-vs-rest linear SVM on 3-D PCA features\n";
    file << "Errors: " << svm3Error << "\n";
    file << "Error rate: " << svm3ER << "\n";

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
    string input = hw4Root + "/outputs/p1";

    //tr = train, t = test
    FeatData tr25 = readFeat(input + "/train_features_25d.csv");
    FeatData t25 = readFeat(input + "/test_features_25d.csv");
    FeatData tr3 = readFeat(input + "/train_pca_3d.csv");
    FeatData t3 = readFeat(input + "/test_pca_3d.csv");
    //Extra check to make sure we gace the files
    if (tr25.features.empty() || t25.features.empty() ||
        tr3.features.empty() || t3.features.empty()) {
        cerr << "Error: missing p1 feature files. Run p1a first." << endl;
        return 1;
    }
    //# of clusters  = # of class in training dataset
    set<string> uqLabels(tr25.labels.begin(), tr25.labels.end());
    int clusterK = static_cast<int>(uqLabels.size());

    //Normalize the data for kmeans 
    vector< vector<double> > normT25 = t25.features;
    vector< vector<double> > normT3 = t3.features;
    vector<double> meanT25 = calcMeanVec(normT25);
    vector<double> stdT25 = calcStdVec(normT25, meanT25);
    vector<double> meanT3 = calcMeanVec(normT3);
    vector<double> stdT3 = calcStdVec(normT3, meanT3);
    normData(normT25, meanT25, stdT25);
    normData(normT3, meanT3, stdT3);

    //runs Kmeans on 25d & 3d features
    KMCluster kmeans25 = KMeans(normT25, clusterK, 100);
    KMCluster kmeans3 = KMeans(normT3, clusterK, 100);
    //calculates the cluster results
    ClusterResults eval25 = evalCluster(kmeans25.clusterLabels, t25.labels, clusterK);
    ClusterResults eval3 = evalCluster(kmeans3.clusterLabels, t3.labels, clusterK);
    //saves the results 
    writeKM(input + "/p1b_kmeans_test_25d.csv",
                      t25.fileName, t25.labels, kmeans25.clusterLabels, eval25);
    writeKM(input + "/p1b_kmeans_test_3d.csv",
                      t3.fileName, t3.labels, kmeans3.clusterLabels, eval3);

    //Normalize the data for the SVM
    vector< vector<double> > normTr25 = tr25.features;
    vector< vector<double> > t25SVM = t25.features;
    vector<double> meanTr25 = calcMeanVec(normTr25);
    vector<double> stdTr25 = calcStdVec(normTr25, meanTr25);
    normData(normTr25, meanTr25, stdTr25);
    normData(t25SVM, meanTr25, stdTr25);

    vector< vector<double> > normTr3 = tr3.features;
    vector< vector<double> > t3SVM = t3.features;
    vector<double> meanTr3 = calcMeanVec(normTr3);
    vector<double> stdTr3 = calcStdVec(normTr3, meanTr3);
    normData(normTr3, meanTr3, stdTr3);
    normData(t3SVM, meanTr3, stdTr3);

    //train the SVM models
    vector<SVMModel> svm25 = trainMClass(normTr25, tr25.labels);
    vector<SVMModel> svm3 = trainMClass(normTr3, tr3.labels);
    //gets the predictions for SVM models
    vector<string> svm25Pred = predMClass(t25SVM, svm25);
    vector<string> svm3Pred = predMClass(t3SVM, svm3);

    //caclulates the error rates (for SVM)
    int svm25Errors = 0;
    int svm3Errors = 0;
    double svm25Rate = calcER(t25.labels, svm25Pred, svm25Errors);
    double svm3Rate = calcER(t3.labels, svm3Pred, svm3Errors);
    //writes the prediction of SVM models
    writePreds(input + "/p1b_svm_test_25d.csv",
                       t25.fileName, t25.labels, svm25Pred);
    writePreds(input + "/p1b_svm_test_3d.csv",
                       t3.fileName, t3.labels, svm3Pred);
    //writes the summary of the results
    summarize(input + "/p1b_summary.txt",
                 eval25.numErrors, eval25.errorRate,
                 eval3.numErrors, eval3.errorRate,
                 svm25Errors, svm25Rate,
                 svm3Errors, svm3Rate);

    cout << "Saveed K-means and SVM results to " << input << endl;
    return 0;
}
