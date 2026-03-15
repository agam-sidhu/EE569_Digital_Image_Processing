/* EE569 Homework #3
 * Name: Agam Sidhu
 * USC ID: 3027948957
 * USC Email: agamsidh@usc.edu
 * Submission Date: March 15, 2026
 * Problem 1: Geometric Image Modification 
 */

#include <iostream>
#include <fstream>
#include <vector>
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

//Function to get RGB pixel channel values while keeping coordinates 
static uint8_t getPixChannel(const vector<uint8_t>& img,
                               int width, int height,
                               int y, int x, int ch)
{
    y = clamp(y, 0, height - 1); //clamp row
    x = clamp(x, 0, width - 1); //clam column
    ch = clamp(ch, 0, 2); // clamp channel indexes
    int index = (y * width + x) * 3 + ch; 
    return img[index];
}

//Function to apply bilinear interpolstion to one RGB channel
static double biInter(const vector<uint8_t>& img, int width, int height, double row, double col, int channel){
    row = clampDouble(row, 0.0, static_cast<double>(height - 1));
    col = clampDouble(col, 0.0, static_cast<double>(width - 1));
    const int x = static_cast<int>(floor(row));
    const int y = static_cast<int>(floor(col));
    const int x1 = clamp(x + 1, 0, height - 1);
    const int y1 = clamp(y + 1, 0, width - 1);

    //the distance inside the cells 
    const double rowOff = row - x;
    const double colOff = col - y;
    
    //Get the pixel values of the nighbors
    //top values that are interpolated in left+right  
    
    const double topL = static_cast<double>(getPixChannel(img, width, height, x, y, channel));
    const double topR = static_cast<double>(getPixChannel(img, width, height, x, y1, channel));
    
    //bottom values that are interpolated in left+right
    const double botL = static_cast<double>(getPixChannel(img, width, height, x1, y, channel));
    const double botR = static_cast<double>(getPixChannel(img, width, height, x1, y1, channel));
   
    const double top = topL * (1.0 - colOff) + topR * colOff;
    const double bot = botL * (1.0 - colOff) + botR * colOff;
    //Run interpolation in top/bottom direction
    return top * (1.0 - rowOff) + bot * rowOff;
}

//Function that get star scaling factor
static double starShift(double theta, double halfSize, double arc){
    double beta = arc / halfSize; 
    beta = clampDouble(beta, 0.0, 0.99); //to keep scale positive
    const double c = cos(2.0 * theta);//shrinks on axes more than diafonals
    double final = 1.0 - beta * c * c;
    return final;
}

//Function to map the img into star shape
static void forward(const vector<uint8_t>& input,
                        vector<uint8_t>& output,
                        int width,
                        int height,
                        double arc)
{
    output.assign(width * height * 3, 0);

    const double cx = (width - 1) / 2.0;
    const double cy = (height - 1) / 2.0;
    const double halfSize = min(width, height) / 2.0;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            const double x = col - cx;
            const double y = row - cy;
            const double theta = atan2(y, x);
            const double scale = starShift(theta, halfSize, arc);

            //Check whether output pixel lies inside the star region
            const double squareRadius = max(abs(x), abs(y));
            if (squareRadius > halfSize * scale) {
                continue; // leave black
            }

            //Backward mapping from star image to square image
            const double srcX = x / scale;
            const double srcY = y / scale;
            const double srcCol = srcX + cx;
            const double srcRow = srcY + cy;

            const int outIdx = (row * width + col) * 3;
            for (int ch = 0; ch < 3; ch++) {
                const double val = biInter(input, width, height, srcRow, srcCol, ch);
                output[outIdx + ch] = static_cast<uint8_t>(clamp(static_cast<int>(round(val)), 0, 255));
            }
        }
    }
}

//Function to map img back into square shape
static void reverse(const vector<uint8_t>& input,
                        vector<uint8_t>& output,
                        int width,
                        int height,
                        double arc)
{
    output.assign(width * height * 3, 0);
    //center image coords
    const double cx = (width - 1) / 2.0;
    const double cy = (height - 1) / 2.0;
    const double halfSize = min(width, height) / 2.0;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            const double x = col - cx;
            const double y = row - cy;
            const double theta = atan2(y, x); //get polar angle
            const double scale = starShift(theta, halfSize, arc); //get scale factor
            //map to star 
            const double tostarX = x * scale;
            const double tostarY = y * scale;
            //convert back to img coords
            const double imgCol = tostarX + cx;
            const double imgRow = tostarY + cy;
            const int opId = (row * width + col) * 3;
            for (int ch = 0; ch < 3; ch++) {
                const double val = biInter(input, width, height, imgRow, imgCol, ch);
                output[opId + ch] = static_cast<uint8_t>(clamp(static_cast<int>(round(val)), 0, 255));
            }
        }
    }
}

//Main function
int main(int argc, char* argv[]) {

    string inputFile = argv[1];;
    string outputFile= argv[2];;
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);
    int type = atoi(argv[5]);
    double arc = atof(argv[6]);
    vector<uint8_t> input;
    vector<uint8_t> output;

    readraw(inputFile, input, width, height, 3);

    if (type == 0) {
        forward(input, output, width, height, arc);
    } else {
        reverse(input, output, width, height, arc);
    }

    writeraw(outputFile, output);
    return 0;
}