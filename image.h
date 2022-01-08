#pragma once

#ifndef IMAGE_H
#define IMAGE_H

#include <filesystem>

#include <cstdint>

template<class T>
class ImageTemplate {

public:

    /// Creates an empty image.
    ImageTemplate();

    /// Creates an uninitialized image.
    ImageTemplate(int w, int h, int c=3);

    /// Copy constructor.
    ImageTemplate(const ImageTemplate& other);

    /// Copy assignment.
    ImageTemplate& operator=(const ImageTemplate& rhs);

    ~ImageTemplate();

    /// Clears but does not initialize image.
    void clear(int w=0, int h=0, int c=3);

    T get(int x, int y, int c=0) const;

    void set(int x, int y, int c, T value);

    /// Access the raw pixel values, the index of a pixel can be calculated as
    /// follows: x*c + y*w*c + c where x and y are the coordinates, c the
    /// number of channels and w the width.
    T operator[](int i) const;
    
    T& operator[](int i);
    
    T* data();
    
    const T* data() const;
    
    /// Returns the size of the image in bytes.
    int bytesSize() const;
    
    /// Returns the number of values used in the image.
    int length() const;

    int getWidth() const;

    int getHeight() const;

    int getNumChannels() const;

    bool isEmpty() const;
    
protected:

    void allocateIfNeeded();

    int _w;
    int _h;
    int _c;

    T *_d;
};

class FloatImage;

class Image: public ImageTemplate<uint8_t> {
    
public:
    
    // inherit ctors
    using ImageTemplate::ImageTemplate;
    
    bool read(const std::filesystem::path& p, int c=0);
    
    bool read(const uint8_t *ptr, int size, int c=0);

    bool writePng(const std::filesystem::path& p);
    
    bool writeBmp(const std::filesystem::path& p);
    
    FloatImage toFloat() const;
    
    /// Creates a single channel image.
    Image makeSingleChannelImage(int c) const;
};


class FloatImage: public ImageTemplate<float> {
    
public:
    
    using ImageTemplate::ImageTemplate;
    
    Image toUint8() const;
};


#endif
