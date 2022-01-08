#pragma once

#ifndef IMAGE_OPS_H
#define IMAGE_OPS_H

#include "image.h"

#include <functional>

#include <cmath>
#include <cstdint>
#include <cassert>

extern const float eps;

float clamp(float f, float min, float max);

int clampi(int i, int min, int max);
int clampi(int i, int max);

/**
 * References
 *
 * Kernel, image processing: https://en.wikipedia.org/wiki/Kernel_(image_processing)
 * Sobel Operator: https://en.wikipedia.org/wiki/Sobel_operator
 */
template<int N> 
struct Kernel {

    constexpr Kernel(const std::initializer_list<std::initializer_list<float>>& il) {
        // static_assert(il.size() == N); does not work bc the expression that created il is not a constexpr.
        int row = 0, col = 0;
        for (const std::initializer_list<float>& rowList: il) {
            col = 0;
            for (float f: rowList) {
                data[row][col] = f;
                ++col;
            }
            ++row;
        }
    }

    float data[N][N];
};

/// Vertical 1D kernel
template<int N>
struct KernelV {
    KernelV(const std::initializer_list<float>& il) {
        int i = 0;
        for (float elem: il) { data[i] = elem; ++i; }
    }
    float data[N];
};

/// Horizontal 1D kernel
template<int N>
struct KernelH {
    KernelH(const std::initializer_list<float>& il) {
        int i = 0;
        for (float elem: il) { data[i] = elem; ++i; }
    }
    float data[N];
};

/// Creates a 3x3 gaussian kernel
Kernel<3> create3gauss();

/// Creates a 5x5 gaussian kernel with sigma = 1.
Kernel<5> create5gauss();

KernelV<5> create5gaussV();

/// Note that this is the same because of the symmetry of the Gaussian.
KernelH<5> create5gaussH();

/// Creates a 5x5 gaussian kernel with sigma = 1.4.
Kernel<5> create5gauss1_4();

KernelV<5> create5gauss14V();

KernelH<5> create5gauss14H();

KernelV<5> create5gauss2V();

KernelH<5> create5gauss2H();

/// Creates a flipped Sobel kernel in x direction.
Kernel<3> createSobelX();

/// Creates a flipped Sobel kernel in y direction.
Kernel<3> createSobelY();

KernelV<3> createSobelXV();
KernelH<3> createSobelXH();

KernelV<3> createSobelYV();
KernelH<3> createSobelYH();

float clamp(float f, float min, float max);

/// Convolves image with a kernel, kernel must be flipped.
template<class ImageT, int N>
FloatImage conv(const ImageT& image, const Kernel<N>& kernel);

/// Same as above but with separated kernels.
template<class ImageT, int N>
FloatImage conv(const ImageT& image, const KernelV<N>& kv, const KernelH<N>& kh);

/// Return the channel value which has the maximum absolute value.
float maxAbsChannel(float r, float g, float b);

float meanChannel(float r, float g, float b);

/// Convolves and reduces the channels to 1. Kernel must be flipped.
template<class ImageT, int N>
FloatImage convReduce(const ImageT& image, const Kernel<N>& kernel,
                      std::function<float(float, float, float)> reduce=maxAbsChannel);

/// Same as above but with separated kernels.
template<class ImageT, int N>
FloatImage convReduce(const ImageT& image, const KernelV<N>& kv, const KernelH<N>& kh,
                      std::function<float(float, float, float)> reduce=maxAbsChannel);

Image invert(const Image& image);

struct MinMaxf { float min; float max;};

MinMaxf findMinMaxf(const FloatImage& image);

/// Returns a new image with all values in the range [0, 1]
FloatImage normalizeImage(const FloatImage& image);

/// Removes the alpha channel under the assumption that the background is white.
Image removeAlpha(const Image& image);

/// Finds a threshold for the top q quartiles pixels.
float findThreshold(const FloatImage& image, float q = 0.1);

#endif
