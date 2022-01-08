#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
// TODO Use a proper zlib with #define STBIW_ZLIB_COMPRESS
#include "stb_image_write.h"

#include <iostream>
#include <cstring>

#include <cmath>
#include <cfloat>

using std::cerr;
using std::cout;
using std::endl;

using std::memmove;

namespace fs = std::filesystem;

template<class T>
ImageTemplate<T>::ImageTemplate()
: ImageTemplate(0, 0, 1) {}

template<class T>
ImageTemplate<T>::ImageTemplate(int w, int h, int c)
: _w(w), _h(h), _c(c) {
    allocateIfNeeded();
}

template<class T>
ImageTemplate<T>::ImageTemplate(const ImageTemplate& other)
: ImageTemplate(other._w, other._h, other._c) {
    allocateIfNeeded();
    // Note that memmove actually copies.
    memmove(_d, other._d, _w * _h * _c * sizeof(T));
}

template<class T>
ImageTemplate<T>& ImageTemplate<T>::operator=(const ImageTemplate& rhs) {
    delete[] _d;
    _w = rhs._w, _h = rhs._h, _c = rhs._c;
    allocateIfNeeded();
    memmove(_d, rhs._d, _w * _h * _c * sizeof(T));
    return *this;
}

template<class T>
ImageTemplate<T>::~ImageTemplate() {
    delete[] _d;
}

template<class T>
void ImageTemplate<T>::clear(int w, int h, int c) {
    delete[] _d;
    _w = w, _h = h, _c = c;    
    allocateIfNeeded();
}

template<class T>
T ImageTemplate<T>::get(int x, int y, int c) const {
    // NOTE _c and c
    return *(_d + x*_c + y*_w*_c + c);
}

template<class T>
void ImageTemplate<T>::set(int x, int y, int c, T value) {
    *(_d + x*_c + y*_w*_c + c) = value;
}

template<class T>
T ImageTemplate<T>::operator[](int i) const {
    return _d[i];
}

template<class T>
T& ImageTemplate<T>::operator[](int i) {
    return _d[i];
}

template<class T>
T* ImageTemplate<T>::data() {
    return _d;
}

template<class T>
const T* ImageTemplate<T>::data() const {
    return _d;
}

template<class T>
int ImageTemplate<T>::bytesSize() const {
    return _w *_h *_c * sizeof(T);
}

template<class T>
int ImageTemplate<T>::length() const {
    return _w *_h * _c;
}


template<class T>
int ImageTemplate<T>::getWidth() const { return _w; }

template<class T>
int ImageTemplate<T>::getHeight() const { return _h; }

template<class T>
int ImageTemplate<T>::getNumChannels() const { return _c; }

template<class T>
bool ImageTemplate<T>::isEmpty() const {
    return _w == 0 && _h == 0; 
}

template<class T>
void ImageTemplate<T>::allocateIfNeeded() {
    assert(_c != 0 && "Does not make any sense!");
    if (_w == 0 && _h == 0) { _d = nullptr; }
    else { _d = new T[_w * _h * _c]; }
}

bool Image::read(const fs::path& p, int c) {
    delete[] _d;

    int w, h, imgc;
    // Last parameter can be used to force the number of components per pixel
    // if it is not 0.
    _d = stbi_load(p.c_str(), &w, &h, &imgc, (c > 0) ? c : 0);

    if (!_d) {
        clear();
        cerr << "Failed to load image at" << p << endl;
        return false;
    }

    _w = w, _h = h, _c = (c > 0) ? c : imgc;
    
    return true;
}

bool Image::read(const uint8_t *ptr, int size, int c) {
    delete[] _d;
    
    int w, h, imgc;
    _d = stbi_load_from_memory(ptr, size, &w, &h, &imgc, (c > 0) ? c : 0);
    
    if (!_d) {
        clear();
        cerr << "Failed to load from memory!" << endl;
        return false;
    }
    
    _w = w, _h = h, _c = (c > 0) ? c : imgc;

    return true;
}

bool Image::writePng(const fs::path& p) {
    int stride = getWidth() * getNumChannels();
    int ret = stbi_write_png(p.c_str(), getWidth(), getHeight(),
        getNumChannels(), data(), stride);
    if (!ret) {
        cerr << "Failed to write image at " << p << endl;
        return false;
    }
    return true;
}

bool Image::writeBmp(const std::filesystem::path& p) {
    int ret = stbi_write_bmp(p.c_str(), getWidth(), getHeight(),
        getNumChannels(), data());
    return ret;
}

FloatImage Image::toFloat() const {
    FloatImage result(_w, _h, _c);
    float * __restrict resultPtr = result.data();
    uint8_t * __restrict intPtr = _d;
    int i = 0;
    for (; i < _w*_h*_c; ++i, ++resultPtr, ++intPtr) {
        *resultPtr = static_cast<float>(*intPtr) / 255.0f;
    }
    return result;
}

Image Image::makeSingleChannelImage(int c) const {
    int w = getWidth(), h = getHeight(), csrc = getNumChannels();
    Image sImage(w, h, 1);
    const uint8_t * __restrict ptr = data();
    uint8_t * __restrict wptr = sImage.data();
    for (int i = c; i < w*h; ++i) { wptr[i] = ptr[i*csrc];}
    return sImage;
}

Image FloatImage::toUint8() const {
    Image result(_w, _h, _c);
    uint8_t * __restrict resultPtr = result.data();
    float * __restrict floatPtr = _d;
    float maxVal = FLT_MIN, minVal = FLT_MAX;
    for (int i = 0; i < _w*_h*_c; ++i, ++floatPtr) {
        maxVal = fmax(*floatPtr, maxVal);
        minVal = fmin(*floatPtr, minVal);
    }
    floatPtr = _d;
    for (int i = 0; i < _w*_h*_c; ++i, ++resultPtr, ++floatPtr) {
        float val = (*floatPtr - minVal)/(maxVal - minVal);
        *resultPtr = static_cast<uint8_t>(val * 255);
    }
    return result;
}

template class ImageTemplate<uint8_t>;
template class ImageTemplate<float>;
