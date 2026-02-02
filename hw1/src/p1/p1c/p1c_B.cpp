/*
Name:
USC ID:
USC Email:
Submission Date:

*/

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static constexpr int width  = 1620;
static constexpr int height = 1080;

// Clamp function
static inline uint8_t clamp(double x) {
    if (x < 0.0)  return 0;
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
    if (argc != 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " input_towers.raw output_globalB.raw\n";
        return 1;
    }

    const std::string input  = argv[1];
    const std::string output = argv[2];
    const int level = 256;
    const size_t dim = static_cast<size_t>(width) * height;
    const size_t in_bytes = dim * 3;

    std::vector<uint8_t> rgb(in_bytes);
    std::vector<uint8_t> rgbOut(in_bytes);
    std::vector<uint32_t> hist(level, 0);
    std::vector<uint32_t> cdf(level, 0);
    std::vector<uint32_t> cdfLow(level, 0);
    std::vector<uint32_t> check(level, 0);
    std::vector<uint8_t> Yp(dim);
    std::vector<uint8_t> Y(dim), U(dim), V(dim);
    {
        std::ifstream ifs(input, std::ios::binary);
        if (!ifs) { std::cerr << "Cannot open input: " << input << "\n"; return 1; }
        ifs.read(reinterpret_cast<char*>(rgb.data()),
                 static_cast<std::streamsize>(rgb.size()));
        if (!ifs) { std::cerr << "Input size mismatch.\n"; return 1; }
    }

    for (size_t idx = 0; idx < dim; ++idx) {
        const size_t base = 3 * idx;
        uint8_t y, u, v;
        toYUV(rgb[base], rgb[base + 1], rgb[base + 2], y, u, v);
        Y[idx] = y;
        U[idx] = u;
        V[idx] = v;
    }
    for (size_t idx = 0; idx < dim; ++idx) {
        hist[Y[idx]]++;
    }

    cdf[0] = hist[0];
    for (int k = 1; k < level; ++k) {
        cdf[k] = cdf[k - 1] + hist[k];
    }

    for (int k = 0; k < level; ++k) {
        cdfLow[k] = cdf[k] - hist[k];
    }

    for (size_t idx = 0; idx < dim; ++idx) {
        const int k = static_cast<int>(Y[idx]);
        const uint32_t rank = cdfLow[k] + check[k];
        check[k]++;

        int mapped = (int)((256ULL * rank) / dim);
        if (mapped > 255) mapped = 255;

        Yp[idx] = static_cast<uint8_t>(mapped);
    }
    for (size_t idx = 0; idx < dim; ++idx) {
        const size_t base = 3 * idx;
        uint8_t r, g, b;
        toRGB(Yp[idx], U[idx], V[idx], r, g, b);
        rgbOut[base]     = r;
        rgbOut[base + 1] = g;
        rgbOut[base + 2] = b;
    }
    {
        std::ofstream ofs(output, std::ios::binary);
        if (!ofs) { std::cerr << "Cannot open output: " << output << "\n"; return 1; }
        ofs.write(reinterpret_cast<const char*>(rgbOut.data()),
                  static_cast<std::streamsize>(rgbOut.size()));
        if (!ofs) { std::cerr << "Write failed.\n"; return 1; }
    }
    std::cout << "p1c_globalB done.\n";
    std::cout << "Wrote: " << output << "\n";
    return 0;
}
