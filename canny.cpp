#include "canny.h"

#include <cfloat>
#include <cmath>

using std::function;
using std::string;
using std::unordered_map;

/// Calcualte gradients using the sobel filter. Blur the images before if desired.
/// The gradient of a color image is reduced by the given reduction function.
void calcGrads(const Image& image, FloatImage *ixPtr, FloatImage *iyPtr,
               FloatImage *magPtr, function<float(float, float, float)> reduction,
               BlurSetting blur, unordered_map<BlurSetting, FloatImage> *cachePtr) {
    
    assert(image.getNumChannels() == 3);
    const int w = image.getWidth(), h = image.getHeight();
    
    FloatImage& ix = *ixPtr; ix.clear(w, h, 1);
    FloatImage& iy = *iyPtr; iy.clear(w, h, 1);
    FloatImage& mag = *magPtr; mag.clear(w, h, 1);
    
    FloatImage floatImage = image.toFloat();
    
    if (blur != NO_BLUR) {
        bool cached = false;
        if (cachePtr) {
            if (auto it = cachePtr->find(blur); it != cachePtr->end()) {
                floatImage = it->second;
                cached = true;
            }
        }
        if (!cached) {
            switch(blur) {
                case NO_BLUR: break;
                case BLUR1:
                    floatImage = conv(floatImage, create5gaussV(), create5gaussH());
                    break;
                case BLUR1_4:
                    floatImage = conv(floatImage, create5gauss14V(), create5gauss14H());
                    break;
                case BLUR2:
                    floatImage = conv(floatImage, create5gauss2V(), create5gauss2H());
                    break;
            }
        }
        if (cachePtr && !cached) { cachePtr->insert({blur, floatImage}); }
    }
    
    ix = convReduce(floatImage, createSobelX(), reduction);
    iy = convReduce(floatImage, createSobelY(), reduction);
    mag = FloatImage(w, h, 1);
    
    float * __restrict magp = mag.data();
    float * __restrict ixp = ix.data();
    float * __restrict iyp = iy.data();
    
    for (int i = 0; i < w*h; ++i) {
        *magp = sqrtf((*ixp)*(*ixp) + (*iyp)*(*iyp));
        ++magp; ++ixp; ++iyp;
    }
}

FloatImage colorizeGrads(const FloatImage& ix, const FloatImage& iy, const FloatImage* magImage) {
    
    FloatImage normalizedMagImage;
    if (magImage) { normalizedMagImage = normalizeImage(*magImage); }
    
    int w = ix.getWidth(), h = ix.getHeight();
    FloatImage angleColorImage(w, h, 3);
    
    const float pi = M_PI;
    
    for (int i = 0; i < ix.length(); ++i) {
        float angle = atan2(ix[i], iy[i]); // in [-PI, PI]
        angle = (angle + pi) / 2;
        float mag = magImage ? normalizedMagImage[i] : 1;
        float r, g, b;
        const float step = pi / 6.0f;
        if (0 <= angle && angle < step) {
            r = mag;
            g = mag * angle / (step);
            b = 0;
        } else if (step <= angle && angle < 2*step) {
            r = mag * (1 - (angle - step) / step);
            g = mag;
            b = 0;
        } else if (2*step <= angle && angle < 3*step) {
            r = 0;
            g = mag;
            b = mag * (angle - 2*step) / step;
        } else if (3*step <= angle && angle < 4*step) {
            r = 0;
            g = mag * (1 - (angle - 3*step) / step);
            b = mag;
        } else if (4*step <= angle && angle < 5*step) {
            r = mag * (angle - 4*step) / step;
            g = 0;
            b = mag;
        } else { // 5*pi/6 <= angle && angle <= pi
            r = mag;
            g = 0;
            b = mag * (1 - (angle - 5*step)/step);
        }
        angleColorImage[i*3]     = r;
        angleColorImage[i*3 + 1] = g;
        angleColorImage[i*3 + 2] = b;
    }
    
    return angleColorImage;
}

/// non-max suppression
float maxOrZero(float val, float nb1, float nb2) {
    return (val > nb1 && val >= nb2) ? val : 0;
}

/// Add 'weak' pixels to the lines image if they are within the 8 neighborhood
/// of a strong pixel.
void connectWeak(FloatImage *linesPtr, const FloatImage& weak, bool keepGrayscale) {
    FloatImage& lines = *linesPtr;
    int w = lines.getWidth(), h = lines.getHeight();
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            if (!weak.get(col, row)) { continue; }
            for (int r = -1; r < 2; ++r) { // -1, 0, 1
                for (int c = -1; c < 2; ++c) {
                    int rowi = clampi(row + r, h - 1);
                    int coli = clampi(col + c, w - 1);
                    if (lines.get(coli, rowi)) {
                        float newVal = keepGrayscale ? weak.get(col, row) : 1;
                        lines.set(col, row, 0, newVal);
                    }
                }
            }
        }
    }
}

bool EdgeFinder::readImage(const string& path) {
    _cache.clear();
    if (!_image.read(path, 0)) {
        return false;
    }
    int c = _image.getNumChannels();
    if (c == 4) {
        _image = removeAlpha(_image);
    }
    return true;
}

bool EdgeFinder::readImage(const uint8_t *ptr, int size) {
    _cache.clear();
    if (!_image.read(ptr, size, 0)) {
        return false;
    }
    int c = _image.getNumChannels();
    if (c == 4) {
        _image = removeAlpha(_image);
    }
    return true;
}

void EdgeFinder::calcGrads(function<float(float, float, float)> reduction, BlurSetting blur) {
    ::calcGrads(_image, &_ix, &_iy, &_mag, reduction, blur, &_cache);
}

void EdgeFinder::calcNonMaxSuppression(float th, float tl, bool keepGrayscale) {
    
    float high = findThreshold(_mag, th), low = findThreshold(_mag, tl);
    
    /*
     Non-max suppression along the gradient.
    
     Finding the discrete gradient direction is simple!
     There are only 4 possibilities: vertical, horizontal, diagonal and anti-
     diagnoal. If ix >= 2*iy then it's horizontal, the other way round it is
     vertical. If the sign of ix an iy is the same it is diagonal otherwise
     anti diagnoal.
     */
    
    int w = _image.getWidth(), h = _image.getHeight();
    
    _lines.clear(w, h, 1);
    _weak.clear(w, h, 1);
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            
            float magVal = _mag.get(col, row);
            float gx = _ix.get(col, row);
            float gy = _iy.get(col, row);
            float val;
            if (fabs(gx) >= fabs(2*gy)) { // horizontal
                float left  = _mag.get(clampi(col - 1, w - 1), row);
                float right = _mag.get(clampi(col + 1, w - 1), row);
                val = maxOrZero(magVal, left, right);
            } else if (fabs(gy) >= fabs(2*gx)) { // vertical
                float above = _mag.get(col, row > 0     ? row - 1 : row);
                float below = _mag.get(col, row < h - 1 ? row + 1 : row);
                val = maxOrZero(magVal, above, below);
            } else if ((gx > 0 && gy > 0) || (gx < 0 && gy < 0)) { // diagnoal
                float leftAbove  = _mag.get(clampi(col - 1, w - 1), clampi(row - 1, h - 1));
                float rightBelow = _mag.get(clampi(col + 1, w - 1), clampi(row + 1, h - 1));
                val = maxOrZero(magVal, leftAbove, rightBelow);
            } else { // anti-diagnoal
                float leftBelow  = _mag.get(clampi(col - 1, w - 1), clampi(row + 1, h - 1));
                float rightAbove = _mag.get(clampi(col + 1, w - 1), clampi(row - 1, h - 1));
                val = maxOrZero(magVal, leftBelow, rightAbove);
            }

            float newVal = 0, lowVal = 0;

            if (val >= high)     { newVal = (keepGrayscale ? val : 1); }
            else if (val >= low) { lowVal = (keepGrayscale ? val : 1); }

            _lines.set(col, row, 0, newVal);
            _weak.set( col, row, 0, lowVal);
        }
    }
    
    connectWeak(&_lines, _weak, keepGrayscale);
}

const FloatImage& EdgeFinder::getLines() const {
    return _lines;
}

const FloatImage& EdgeFinder::getGradient() const {
    return _mag;
}
