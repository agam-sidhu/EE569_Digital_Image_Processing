/* EE569 Homework #3
 * Name: Agam Sidhu
 * USC ID: 3027948957
 * USC Email: agamsidh@usc.edu
 * Submission Date: March 15, 2026
 * Problem 3(b): Connected Component Labeling
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

//Binarize image using threshold 0.5 * Fmax
static vector<uint8_t> binarizeImage(const vector<uint8_t>& gray,
                                     int width,
                                     int height)
{
    uint8_t fmax = 0;
    for (size_t i = 0; i < gray.size(); i++) {
        fmax = max(fmax, gray[i]);
    }

    const double thresh = 0.5 * static_cast<double>(fmax);
    vector<uint8_t> binary(width * height, 0);

    for (int i = 0; i < width * height; i++) {
        binary[i] = (gray[i] > thresh) ? 1 : 0;
    }
    return binary;
}

//Convert binary {0,1} to raw {0,255}
static vector<uint8_t> binaryToRaw(const vector<uint8_t>& binary) {
    vector<uint8_t> out(binary.size(), 0);
    for (size_t i = 0; i < binary.size(); i++) {
        out[i] = binary[i] ? 255 : 0;
    }
    return out;
}

struct ComponentInfo {
    int label;
    int area;
    int minRow;
    int maxRow;
    int minCol;
    int maxCol;
    int holeCount;
};

static inline bool inside(int row, int col, int width, int height) {
    return row >= 0 && row < height && col >= 0 && col < width;
}

//Label white connected components
static vector<ComponentInfo> labelWhiteComponents(const vector<uint8_t>& binary,
                                                  int width,
                                                  int height,
                                                  vector<int>& labels)
{
    labels.assign(width * height, 0);
    vector<ComponentInfo> comps;
    int currentLabel = 0;

    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = row * width + col;
            if (binary[idx] == 0 || labels[idx] != 0) continue;

            currentLabel++;
            queue<pair<int,int>> q;
            q.push({row, col});
            labels[idx] = currentLabel;

            ComponentInfo info;
            info.label = currentLabel;
            info.area = 0;
            info.minRow = info.maxRow = row;
            info.minCol = info.maxCol = col;
            info.holeCount = 0;

            while (!q.empty()) {
                auto cur = q.front();
                q.pop();

                int r = cur.first;
                int c = cur.second;
                int id = r * width + c;

                info.area++;
                info.minRow = min(info.minRow, r);
                info.maxRow = max(info.maxRow, r);
                info.minCol = min(info.minCol, c);
                info.maxCol = max(info.maxCol, c);

                for (int k = 0; k < 4; k++) {
                    int nr = r + dr[k];
                    int nc = c + dc[k];
                    if (!inside(nr, nc, width, height)) continue;

                    int nid = nr * width + nc;
                    if (binary[nid] == 1 && labels[nid] == 0) {
                        labels[nid] = currentLabel;
                        q.push({nr, nc});
                    }
                }
            }

            comps.push_back(info);
        }
    }
    return comps;
}

//Count black holes inside one object bounding box
static int countHolesForComponent(const vector<uint8_t>& binary,
                                  int width,
                                  int height,
                                  const ComponentInfo& comp)
{
    int r0 = max(0, comp.minRow);
    int r1 = min(height - 1, comp.maxRow);
    int c0 = max(0, comp.minCol);
    int c1 = min(width - 1, comp.maxCol);

    const int bh = r1 - r0 + 1;
    const int bw = c1 - c0 + 1;

    vector<uint8_t> sub(bh * bw, 0);

    //Object area becomes 1, background becomes 0
    for (int r = r0; r <= r1; r++) {
        for (int c = c0; c <= c1; c++) {
            sub[(r - r0) * bw + (c - c0)] = binary[r * width + c];
        }
    }

    //Visited for background black regions
    vector<int> vis(bh * bw, 0);
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};

    auto insideSub = [&](int r, int c) {
        return r >= 0 && r < bh && c >= 0 && c < bw;
    };

    //Flood external black regions touching box boundary
    queue<pair<int,int>> q;
    for (int r = 0; r < bh; r++) {
        for (int c = 0; c < bw; c++) {
            if (r != 0 && r != bh - 1 && c != 0 && c != bw - 1) continue;
            int idx = r * bw + c;
            if (sub[idx] == 0 && vis[idx] == 0) {
                vis[idx] = 1;
                q.push({r, c});
                while (!q.empty()) {
                    auto cur = q.front();
                    q.pop();
                    for (int k = 0; k < 4; k++) {
                        int nr = cur.first + dr[k];
                        int nc = cur.second + dc[k];
                        if (!insideSub(nr, nc)) continue;
                        int nid = nr * bw + nc;
                        if (sub[nid] == 0 && vis[nid] == 0) {
                            vis[nid] = 1;
                            q.push({nr, nc});
                        }
                    }
                }
            }
        }
    }

    //Remaining black regions are holes
    int holes = 0;
    for (int r = 0; r < bh; r++) {
        for (int c = 0; c < bw; c++) {
            int idx = r * bw + c;
            if (sub[idx] == 0 && vis[idx] == 0) {
                holes++;
                vis[idx] = 1;
                q.push({r, c});
                while (!q.empty()) {
                    auto cur = q.front();
                    q.pop();
                    for (int k = 0; k < 4; k++) {
                        int nr = cur.first + dr[k];
                        int nc = cur.second + dc[k];
                        if (!insideSub(nr, nc)) continue;
                        int nid = nr * bw + nc;
                        if (sub[nid] == 0 && vis[nid] == 0) {
                            vis[nid] = 1;
                            q.push({nr, nc});
                        }
                    }
                }
            }
        }
    }

    return holes;
}

//Classify object as rectangle or circle
static string classifyShape(const ComponentInfo& comp)
{
    const int boxH = comp.maxRow - comp.minRow + 1;
    const int boxW = comp.maxCol - comp.minCol + 1;
    const double boxArea = static_cast<double>(boxH) * static_cast<double>(boxW);
    const double fillRatio = static_cast<double>(comp.area) / boxArea;

    //Rectangles tend to fill their box much more
    //Circles tend to have fill ratio near pi/4 ~ 0.785
    if (fillRatio > 0.86) {
        return "rectangle";
    } else {
        return "circle";
    }
}

//Main function
int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0]
             << " input.raw output_binary.raw width height\n";
        return 1;
    }

    const string inputPath = argv[1];
    const string binaryOutPath = argv[2];
    const int width = atoi(argv[3]);
    const int height = atoi(argv[4]);

    if (width <= 0 || height <= 0) {
        cerr << "Error: width and height must be positive.\n";
        return 1;
    }

    vector<uint8_t> gray;
    readraw(inputPath, gray, width, height, 1);

    vector<uint8_t> binary = binarizeImage(gray, width, height);
    writeraw(binaryOutPath, binaryToRaw(binary));

    vector<int> labels;
    vector<ComponentInfo> comps = labelWhiteComponents(binary, width, height, labels);

    int totalObjects = static_cast<int>(comps.size());
    int totalHoles = 0;
    int rectangleCount = 0;
    int circleCount = 0;

    for (size_t i = 0; i < comps.size(); i++) {
        comps[i].holeCount = countHolesForComponent(binary, width, height, comps[i]);
        totalHoles += comps[i].holeCount;

        string shapeType = classifyShape(comps[i]);
        if (shapeType == "rectangle") {
            rectangleCount++;
        } else {
            circleCount++;
        }
    }

    cout << "Total number of holes: " << totalHoles << endl;
    cout << "Total number of white objects: " << totalObjects << endl;
    cout << "Total number of white rectangle objects: " << rectangleCount << endl;
    cout << "Total number of white circle objects: " << circleCount << endl;

    cout << "\nPer-object details:\n";
    for (size_t i = 0; i < comps.size(); i++) {
        const int boxH = comps[i].maxRow - comps[i].minRow + 1;
        const int boxW = comps[i].maxCol - comps[i].minCol + 1;
        const double fillRatio = static_cast<double>(comps[i].area) /
                                 (static_cast<double>(boxH) * boxW);

        cout << "Object " << (i + 1)
             << ": area=" << comps[i].area
             << ", bbox=(" << boxW << "x" << boxH << ")"
             << ", holes=" << comps[i].holeCount
             << ", fillRatio=" << fillRatio
             << ", class=" << classifyShape(comps[i])
             << endl;
    }

    return 0;
}