#include "image_ops.h"

#include <vector>
#include <algorithm>
#include <cfloat>

using std::vector;
using std::function;

using std::max;
using std::min;
using std::sort;

const float eps = 1e-9;

float clamp(float f, float min, float max) {
    if (f < min) { return min; }
    if (f > max) { return max; }
    return f;
}

int clampi(int i, int max) {
    return clampi(i, 0, max);
}

int clampi(int i, int min, int max) {
    if (i < min) { return min; }
    if (i > max) { return max; }
    return i;
}

Kernel<3> create3gauss() {
    return {
        {0.0625, 0.125 , 0.0625},
        {0.125 , 0.25  , 0.125 },
        {0.0625, 0.125 , 0.0625}
    };
}

Kernel<5> create5gauss() {
    return {
        {0.03389831, 0.06779661, 0.08474576, 0.06779661, 0.03389831},
        {0.06779661, 0.15254237, 0.20338983, 0.15254237, 0.06779661},
        {0.08474576, 0.20338983, 0.25423729, 0.20338983, 0.08474576},
        {0.06779661, 0.15254237, 0.20338983, 0.15254237, 0.06779661},
        {0.03389831, 0.06779661, 0.08474576, 0.06779661, 0.03389831}};
}

KernelV<5> create5gaussV() {
    return {0.10165561, 0.45558886, 0.75113904, 0.45558886, 0.10165561};
}

/// Note that this is the same because of the symmetry of the Gaussian.
KernelH<5> create5gaussH() {
    return {0.10165561, 0.45558886, 0.75113904, 0.45558886, 0.10165561};
}


Kernel<5> create5gauss1_4() {
    return {
        {0.01214612, 0.02610994, 0.03369732, 0.02610994, 0.01214612},
        {0.02610994, 0.0561273 , 0.07243752, 0.0561273 , 0.02610994},
        {0.03369732, 0.07243752, 0.09348738, 0.07243752, 0.03369732},
        {0.02610994, 0.0561273 , 0.07243752, 0.0561273 , 0.02610994},
        {0.01214612, 0.02610994, 0.03369732, 0.02610994, 0.01214612}};
}

KernelV<5> create5gauss14V() {
    return {0.2297855 , 0.49395894, 0.6375001 , 0.49395894, 0.2297855};
}

KernelH<5> create5gauss14H() {
    return {0.2297855 , 0.49395894, 0.6375001 , 0.49395894, 0.2297855};
}


KernelV<5> create5gauss2V() {
    return {0.33422053, 0.486288  , 0.5510365 , 0.486288  , 0.33422053};
}

KernelH<5> create5gauss2H() {
    return {0.33422053, 0.486288  , 0.5510365 , 0.486288  , 0.33422053};
}

Kernel<3> createSobelX() {
    return {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };
}

Kernel<3> createSobelY() {
    return {
        {1, 0, -1},
        {2, 0, -2},
        {1, 0, -1}
    };
}

KernelV<3> createSobelXV() { return {1, 2,  1}; }
KernelH<3> createSobelXH() { return {1, 0, -1}; }

KernelV<3> createSobelYV() { return {1, 0, -1}; }
KernelH<3> createSobelYH() { return {1, 2,  1}; }

template<class ImageT, int N>
FloatImage conv(const ImageT& image, const Kernel<N>& kernel) {

    assert(image.getHeight() > N && image.getWidth() > N);
    static_assert(N % 2 == 1);
    
    int w = image.getWidth(), h = image.getHeight(), c = image.getNumChannels();
    FloatImage result(w, h, c);
    const auto * __restrict imgPtr = image.data();
    float * __restrict resultPtr = result.data();
    
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            for (int ci = 0; ci < c; ++ci) {
                float sum = 0;
                for (int kr = 0; kr < N; ++kr) {
                    for (int kc = 0; kc < N; ++kc) {
                        int icol = clampi(col + kc - N/2, w - 1);
                        int irow = clampi(row + kr - N/2, h - 1);
                        // Replaced this for speed with a restricted ptr,
                        // in a small test I almost got a 4x speedup in a
                        // release build.
                        // float r = image.get(icol, irow, ci) * kernel.get(kc, kr);
                        auto val = imgPtr[icol*c + irow*w*c + ci];
                        float r = val * kernel.data[kc][kr];
                        sum += r;
                    }
                }
                resultPtr[row*w*c + col*c + ci] = sum;
            }
        }
    }
    
    return result;
}

template<class ImageT, int N>
FloatImage conv(const ImageT& image, const KernelV<N>& kv, const KernelH<N>& kh) {
    
    static_assert(N % 2 == 1);
    assert(image.getHeight() > N && image.getWidth() > N);
    
    const int w = image.getWidth(), h = image.getHeight(), c = image.getNumChannels();
    
    FloatImage tmp(w, h, c);
    const auto * __restrict ptr = image.data();
    float * __restrict tmpPtr = tmp.data();
    
    // vertical conv
    for (int col = 0; col < w; ++col) {
        for (int ci = 0; ci < c; ++ci) {
            for (int row = 0; row < h; ++row) {
                int pixi = row*w*c + col*c + ci; // index of the pixel in the image
                float sum = 0;
                for (int ki = 0; ki < N; ++ki) {
                    int rowWithOff = clampi(row + ki - N/2, h - 1);
                    int i = rowWithOff*w*c + col*c + ci;
                    sum += kv.data[ki] * ptr[i];
                }
                tmpPtr[pixi] = sum;
            }
        }
    }

    // horizontal conv
    FloatImage result(w, h, c);
    float * __restrict resultPtr = result.data();
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            for (int ci = 0; ci < c; ++ci) {
                int pixi = row*w*c + col*c + ci;
                float sum = 0;
                for (int ki = 0; ki < N; ++ki) {
                    int colWithOffset = clampi(col + ki - N/2, w - 1);
                    int i = row*w*c + colWithOffset*c + ci;
                    sum += kh.data[ki] * tmpPtr[i];
                }
                resultPtr[pixi] = sum;
            }
        }
    }
    
    return result;
}

float maxAbsChannel(float r, float g, float b) {
    float ar = fabs(r), ag = fabs(g), ab = fabs(b);
    if (ar > ag) {
        if (ar > ab) { return r; }
        else { return b; }
    } else {
        if (ag > ab) { return g; }
        else { return b; }
    }
}

float meanChannel(float r, float g, float b){
    return (r+g+b) / 3;
}

template<class ImageT, int N>
FloatImage convReduce(const ImageT& image, const Kernel<N>& kernel,
                      function<float(float, float, float)> reduce) {
    static_assert(N % 2 == 1);

    assert(image.getHeight() > N && image.getWidth() > N);
    int w = image.getWidth(), h = image.getHeight(), c = image.getNumChannels();
    assert(c == 3);
    
    FloatImage result(w, h, 1);
    float * __restrict resultPtr = result.data();
    const auto * __restrict imgPtr = image.data();
    float rgb[3];
    
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            for (int ci = 0; ci < c; ++ci) {
                float sum = 0;
                for (int kr = 0; kr < N; ++kr) {
                    for (int kc = 0; kc < N; ++kc) {
                        int icol = clampi(col + kc - N/2, w - 1);
                        int irow = clampi(row + kr - N/2, h - 1);
                        // float r = image.get(icol, irow, ci) * kernel.get(kc, kr);
                        auto val = imgPtr[icol*c + irow*w*c + ci];
                        float r = val * kernel.data[kc][kr];
                        sum += r;
                    }
                }
                rgb[ci] = sum;
            }
            resultPtr[row*w + col] = reduce(rgb[0], rgb[1], rgb[2]);
        }
    }
    
    return result;
}

template<class ImageT, int N>
FloatImage convReduce(const ImageT& image, const KernelV<N>& kv, const KernelH<N>& kh,
                      function<float(float, float, float)> reduce) {
    static_assert(N % 2 == 1);
    assert(image.getHeight() > N && image.getWidth() > N);
    
    const int w = image.getWidth(), h = image.getHeight(), c = image.getNumChannels();
    assert(c == 3);
    
    FloatImage tmp(w, h, c);
    const uint8_t * __restrict ptr = image.data();
    float * __restrict tmpPtr = tmp.data();
    
    // vertical conv
    for (int col = 0; col < w; ++col) {
        for (int ci = 0; ci < c; ++ci) {
            for (int row = 0; row < h; ++row) {
                int pixi = row*w*c + col*c + ci; // index of the pixel in the image
                float sum = 0;
                for (int ki = 0; ki < N; ++ki) {
                    int rowWithOff = clampi(row + ki - N/2, h - 1);
                    int i = rowWithOff*w*c + col*c + ci;
                    sum += kv.data[ki] * ptr[i];
                }
                tmpPtr[pixi] = sum;
            }
        }
    }
    
    // horizontal conv
    FloatImage result(w, h, 1);
    float * __restrict finalResultPtr = result.data();
    float rgb[3];
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            for (int ci = 0; ci < c; ++ci) {
                float sum = 0;
                for (int ki = 0; ki < N; ++ki) {
                    int colWithOffset = clampi(col + ki - N/2, w - 1);
                    int i = row*w*c + colWithOffset*c + ci;
                    sum += kh.data[ki] * tmpPtr[i];
                }
                rgb[ci] = sum;
            }
            finalResultPtr[row*w + col] = reduce(rgb[0], rgb[1], rgb[2]);
        }
    }
    return result;
}

Image invert(const Image& image) {
    Image inverted(image.getWidth(), image.getHeight(), image.getNumChannels());
    for (int row = 0; row < image.getHeight(); ++row) {
        for (int col = 0; col < image.getWidth(); ++col) {
            for (int c = 0; c < image.getNumChannels(); ++c) {
                uint8_t val = image.get(col, row, c);
                inverted.set(col, row, c, 255 - val);
            }
        }
    }
    return inverted;
}

MinMaxf findMinMaxf(const FloatImage& image) {
    const float * __restrict ptr = image.data();
    int w = image.getWidth(), h = image.getHeight(), c = image.getNumChannels();
    float maxVal = FLT_MIN, minVal = FLT_MAX;
    for (int i = 0; i < w*h*c; ++i) {
        maxVal = fmax(ptr[i], maxVal);
        minVal = fmin(ptr[i], minVal);
    }
    return {minVal, maxVal};
}

FloatImage normalizeImage(const FloatImage& image) {
    auto [min, max] = findMinMaxf(image);
    const float * __restrict ptr = image.data();
    FloatImage normalizedImage(image.getWidth(), image.getHeight(), image.getNumChannels());
    float * __restrict nptr = normalizedImage.data();
    for (int i = 0; i < image.length(); ++i) {
        nptr[i] = (ptr[i] - min) / (max - min);
    }
    return normalizedImage;
}

Image removeAlpha(const Image& image) {
    assert(image.getNumChannels() == 4);
    int w = image.getWidth(), h = image.getHeight();
    const uint8_t * __restrict d = image.data();
    Image image3(w, h, 3);
    uint8_t * __restrict d3 = image3.data();
    
    // Recall the linear interpolation formula:
    // C = alpha*A + (1 - alpha) * B
    // For the image C which results form laying image A over B.
    // All of B's colors are 255 because we assume it is white.
    
    for (int i = 0; i < w*h; ++i) {
        float a = d[i*4 + 3]/255.0f;
        float white = (1 - a) * 255.0f;
        d3[i*3]     = a * d[i*4]     + white;
        d3[i*3 + 1] = a * d[i*4 + 1] + white;
        d3[i*3 + 2] = a * d[i*4 + 2] + white;
    }
    return image3;
}

float findThreshold(const FloatImage& image, float q) {
    assert(0 <= q && q <= 1 && image.getNumChannels() == 1);
    int sz = image.length();
    vector<float> a(sz);
    for (int i = 0; i < sz; ++i) {
        a[i] = image[i];
    }
    sort(a.begin(), a.end());
    return a[int((sz - 1) * (1 - q))];
}

// Explicit template instantiation:
template FloatImage conv(const Image& image, const Kernel<3>& kernel);
template FloatImage conv(const Image& image, const Kernel<5>& kernel);

template FloatImage conv(const FloatImage& image, const Kernel<3>& kernel);
template FloatImage conv(const FloatImage& image, const Kernel<5>& kernel);

template FloatImage conv(const Image& image,      const KernelV<5>& kv, const KernelH<5>& kh);
template FloatImage conv(const FloatImage& image, const KernelV<5>& kv, const KernelH<5>& kh);

template FloatImage convReduce(const Image& image, const Kernel<3>& kernel,
                               function<float(float, float, float)> reduce);
template FloatImage convReduce(const Image& image, const Kernel<5>& kernel,
                               function<float(float, float, float)> reduce);

template FloatImage convReduce(const FloatImage& image, const Kernel<3>& kernel,
                               function<float(float, float, float)> reduce);
template FloatImage convReduce(const FloatImage& image, const Kernel<5>& kernel,
                               function<float(float, float, float)> reduce);


template FloatImage convReduce(const Image& image, const KernelV<3>& kv, const KernelH<3>& kh,
                               std::function<float(float, float, float)> reduce);
