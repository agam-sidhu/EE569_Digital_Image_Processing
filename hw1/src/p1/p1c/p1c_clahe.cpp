/*
Name:
USC ID:
USC Email:
Submission Date:

*/

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

static constexpr int width  = 1620;
static constexpr int height = 1080;

// Clamp function
static inline uint8_t clamp(double x) {
    if (x < 0.0) return 0;
    if (x > 255.0) return 255;
    return static_cast<uint8_t>(x + 0.5);
}
// RGB to YUV convert function
static inline void toYUV(uint8_t R, uint8_t G, uint8_t B,
                         uint8_t& Y, uint8_t& U, uint8_t& V) {
    const double y = (0.257 * R) + (0.504 * G) + (0.098 * B) + 16.0;
    const double u = -(0.148 * R) - (0.291 * G) + (0.439 * B) + 128.0;
    const double v = (0.439 * R) - (0.368 * G) - (0.071 * B) + 128.0;
    Y = clamp(y);
    U = clamp(u);
    V = clamp(v);
}
// YUV to RGB convert function
static inline void toRGB(uint8_t Y, uint8_t U, uint8_t V,
                         uint8_t& R, uint8_t& G, uint8_t& B) {
    const double y = static_cast<double>(Y);
    const double u = static_cast<double>(U);
    const double v = static_cast<double>(V);

    const double r = 1.164 * (y - 16.0) + 1.596 * (v - 128.0);
    const double g = 1.164 * (y - 16.0) - 0.813 * (v - 128.0) - 0.391 * (u - 128.0);
    const double b = 1.164 * (y - 16.0) + 2.018 * (u - 128.0);

    R = clamp(r);
    G = clamp(g);
    B = clamp(b);
}
// Main function
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " input_towers.raw output_clahe.raw [clipLimit] [tilesX] [tilesY]\n";
        return 1;
    }

    const std::string input  = argv[1];
    const std::string output = argv[2];
    double clipLim = 2.0;
    int tile_X = 8;
    int tile_Y = 8;

    if (argc >= 4) clipLim = std::stod(argv[3]);
    if (argc >= 5) tile_X = std::stoi(argv[4]);
    if (argc >= 6) tile_Y = std::stoi(argv[5]);

    if (tile_X < 1) tile_X = 1;
    if (tile_Y < 1) tile_Y = 1;
    if (clipLim <= 0.0) clipLim = 0.1;

    const size_t dim = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t in_bytes = dim * 3;

    std::vector<uint8_t> rgb(in_bytes);
    {
        std::ifstream ifs(input, std::ios::binary);
        if (!ifs) { std::cerr << "Cannot open input.\n"; return 1; }
        ifs.read(reinterpret_cast<char*>(rgb.data()),
                 static_cast<std::streamsize>(rgb.size()));
        if (!ifs) { std::cerr << "Input size mismatch.\n"; return 1; }
    }

    std::vector<uint8_t> Y(dim), U(dim), V(dim);
    for (size_t idx = 0; idx < dim; ++idx) {
        const size_t base = 3 * idx;
        uint8_t y, u, v;
        toYUV(rgb[base + 0], rgb[base + 1], rgb[base + 2], y, u, v);
        Y[idx] = y;
        U[idx] = u;
        V[idx] = v;
    }

    cv::Mat yMat(height, width, CV_8UC1, Y.data());
    cv::Mat yOut(height, width, CV_8UC1);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(clipLim);
    clahe->setTilesGridSize(cv::Size(tile_X, tile_Y));
    clahe->apply(yMat, yOut);
    std::vector<uint8_t> Yp(dim);
    std::memcpy(Yp.data(), yOut.data, dim);

    std::vector<uint8_t> rgbOut(in_bytes);
    for (size_t idx = 0; idx < dim; ++idx) {
        const size_t base = 3 * idx;
        uint8_t r, g, b;
        toRGB(Yp[idx], U[idx], V[idx], r, g, b);
        rgbOut[base + 0] = r;
        rgbOut[base + 1] = g;
        rgbOut[base + 2] = b;
    }

    {
        std::ofstream ofs(output, std::ios::binary);
        if (!ofs) { std::cerr << "Cannot open output.\n"; return 1; }
        ofs.write(reinterpret_cast<const char*>(rgbOut.data()),
                  static_cast<std::streamsize>(rgbOut.size()));
        if (!ofs) { std::cerr << "Write failed.\n"; return 1; }
    }

    std::cout << "p1c_clahe done.\n";
    std::cout << "Wrote: " << output
              << " (clipLimit=" << clipLim
              << ", tiles=" << tile_X << "x" << tile_Y << ")\n";
    return 0;
}
