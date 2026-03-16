/* EE569 Homework #3
 * Name: Agam Sidhu
 * USC ID: 3027948957
 * USC Email: agamsidh@usc.edu
 * Submission Date: March 15, 2026
 * Problem 3(b): Shape detection and counting
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cstdint>
#include <string>
#include <cstdlib>
#include <cmath>
using namespace std;

//Clamp function
static inline int clamp(int x, int low, int high) {
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

//Function to convert into raw from binary
static vector<uint8_t> toRaw(const vector<uint8_t>& bi) {
    size_t n = bi.size();
    vector<uint8_t> output(n, 0);
    for (size_t i = 0; i < n; i++) {
        if (bi[i] == 1) {
            output[i] = 255; //this well set to on
        } else {
            output[i] = 0; // set to off
        }
    }
    return output;
}

//Function to binarize image
static vector<uint8_t> toBinary(const vector<uint8_t>& image, int width, int height)
{
    //find max pixel value
    uint8_t fMax = 0;
    for (size_t i = 0; i < image.size(); i++) {
        fMax = max(fMax, image[i]); //if pizel is large -> update
    }

    double tVal = 0.5 * static_cast<double>(fMax); //half of max
    vector<uint8_t> biOut(width * height, 0);

    for (int i = 0; i < width * height; i++) {
        //check if pixel is aboved our threshold val
        bool above = image[i] > tVal;
        if (above) {
            biOut[i] = 1;
        } else {
            biOut[i] = 0;
        }
    }
    return biOut;
}

//Function to see if pixel is inside bounds
static inline bool inBounds(int row, int col, int width, int height) {
    return row >= 0 && row < height && col >= 0 && col < width;
}

//Store info about each connected component
struct shapeInfo {
    int label;
    int area;
    int minR;
    int maxR;
    int minC;
    int maxC;
    int numHole;
};

//Function that will label all connected compontnets
static vector<shapeInfo> labelComp(const vector<uint8_t>& binary,
                                   int width, int height,
                                   vector<int>& labels)
{
    labels.assign(width * height, 0);
    vector<shapeInfo> comps;
    int currLabel = 0;

    //4 connected neighbors
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = row * width + col;
            if (binary[idx] == 0 || labels[idx] != 0) continue; //skip background or already labeled

            currLabel++;
            queue<pair<int,int>> q;
            q.push({row, col});
            labels[idx] = currLabel;

            //setup info
            shapeInfo info;
            info.label = currLabel;
            info.area = 0;
            info.minR = info.maxR = row;
            info.minC = info.maxC = col;
            info.numHole = 0;

            //flood fill to find all pixels in component
            while (!q.empty()) {
                auto cur = q.front();
                q.pop();

                int r = cur.first;
                int c = cur.second;

                info.area++;
                info.minR = min(info.minR, r);
                info.maxR = max(info.maxR, r);
                info.minC = min(info.minC, c);
                info.maxC = max(info.maxC, c);

                for (int k = 0; k < 4; k++) {
                    int nr  = r + dr[k];
                    int nc  = c + dc[k];
                    if (!inBounds(nr, nc, width, height)) continue;
                    int nid = nr * width + nc;
                    if (binary[nid] == 1 && labels[nid] == 0) {
                        labels[nid] = currLabel;
                        q.push({nr, nc});
                    }
                }
            }
            comps.push_back(info);
        }
    }
    return comps;
}

//Compute euler number using 2x2 quad patterns
//euler = objects - holes
static int eulerN(const vector<uint8_t>& binary, int width, int height)
{
    //counters for the various patterns
    int n1 = 0; //1 white pix
    int n3 = 0;  //3 white pix
    int nd = 0;  //daigonal pair

    for (int r = 0; r < height - 1; r++) {
        for (int c = 0; c < width - 1; c++) {
            //get 2x2 quad pixels
            int tl = binary[r * width + c]; //top left
            int tr = binary[r * width + c + 1];//top right
            int bl = binary[(r+1) * width + c]; //bottom left
            int br = binary[(r+1) * width + c + 1]; //bottom right
            int sum = tl + tr + bl + br;

            bool isOne = (sum == 1); //1 white pixel
            bool isThree = (sum == 3);  //3 white pixels
            bool isDiags = (sum == 2) && ((tl == 1 && br == 1 && tr == 0 && bl == 0) ||  (tr == 1 && bl == 1 && tl == 0 && br == 0)); //diagonal pair

            //update counts on pattern found                       
            if (isOne){
                n1++;
            }
            if (isThree){
                n3++;
            }   
            if (isDiags) {
               nd++; 
            } 
        }
    }
    // formula for 4Connect
    int val = (n1 - n3 + 2*nd) / 4;
    return val;
}

//Function to classify shape as rectangle or circle using fill ratio
static string detectShape(const shapeInfo& comp)
{
    int bh = comp.maxR - comp.minR + 1;
    int bw = comp.maxC - comp.minC + 1;
    double boxArea   = static_cast<double>(bh) * static_cast<double>(bw);
    double fillRatio = static_cast<double>(comp.area) / boxArea;

    //rects shuld have rastio close to 1, circle will have less
    bool isRect = fillRatio > 0.82;
    if (isRect) {
        return "rectangle";
    } else {
        return "circle";
    }
}

//Main function
int main(int argc, char* argv[])
{
    if (argc != 5) {
        cerr << "Usage: " << argv[0]
             << " input.raw output_binary.raw width height\n";
        return 1;
    }

    //parse input args
    string input  = argv[1];
    string output = argv[2];
    int width = atoi(argv[3]);
    int height= atoi(argv[4]);

    if (width <= 0 || height <= 0) {
        cerr << "Error: width and height must be positive." << endl;
        return 1;
    }

    //load images + convert to binary
    vector<uint8_t> image;
    readraw(input, image, width, height, 1);
    vector<uint8_t> binary = toBinary(image, width, height);
    writeraw(output, toRaw(binary)); //save binarized image

    //labels all the white components
    vector<int> labels;
    vector<shapeInfo> components = labelComp(binary, width, height, labels);

    int totalShape = static_cast<int>(components.size()); //object count from connected components

    //compute euler Num
    int euler = eulerN(binary, width, height);

    //holes = objects - euler value
    int totalHoles = totalShape - euler;

    int rectangleCount = 0;
    int circleCount = 0;

    for (size_t i = 0; i < components.size(); i++) {
        //will classify if rectabgle or circle
        string typeShape = detectShape(components[i]);
        if (typeShape == "rectangle") {
            rectangleCount++;
        } else {
            circleCount++;
        }
    }

    //Summary stats
    cout << "Total number of holes: " << totalHoles << endl;
    cout << "Total number of white shapes: "  << totalShape << endl;
    cout << "Total number of white rectangles: "  << rectangleCount << endl;
    cout << "Total number of white circles: " << circleCount << endl;

    //Object-level details
    cout << "\nPer bject details:\n";
    for (size_t i = 0; i < components.size(); i++) {
        int bh = components[i].maxR - components[i].minR + 1;
        int bw = components[i].maxC - components[i].minC + 1;
        double fillRatio = static_cast<double>(components[i].area) /
                           (static_cast<double>(bh) * bw);

        cout << "Object " << (i + 1)
             << ": area=" << components[i].area
             << ", bbox=(" << bw << "x" << bh << ")"
             << ", holes=" << components[i].numHole
             << ", fillRatio=" << fillRatio
             << ", class="<< detectShape(components[i])
             << endl;
    }

    return 0;
}