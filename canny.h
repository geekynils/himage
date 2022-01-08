#include "image.h"

#include "image_ops.h"

#include <string>
#include <functional>
#include <unordered_map>

enum BlurSetting: int {
    NO_BLUR,
    BLUR1,   // gaussian kernel with a sigma of 1
    BLUR1_4, // ~ sigma of 1.4
    BLUR2    // ~ sigma of 2
};

void calcGrads(const Image& image, FloatImage *ixPtr, FloatImage *iyPtr,
               FloatImage *magPtr,
               std::function<float(float, float, float)> reduction,
               BlurSetting blur=BLUR1,
               std::unordered_map<BlurSetting, FloatImage> *cachePtr = nullptr);

/// Creates a color image where the color corresponds to grad direction and
/// brightness to magnitude (if provided).
FloatImage colorizeGrads(const FloatImage& ix, const FloatImage& iy, const FloatImage* magImage = nullptr);

float maxOrZero(float val, float nb1, float nb2);
        
void connectWeak(Image *linesPtr, const Image& weak);



/// Find edges using Canny's edge detection method.
class EdgeFinder {
    
public:
    
    bool readImage(const std::string& path);
    
    bool readImage(const uint8_t *ptr, int size);
    
    void calcGrads(std::function<float(float, float, float)> reduction=maxAbsChannel,
                   BlurSetting blur = BLUR1_4);
    
    void calcNonMaxSuppression(float th, float tl, bool keepGrayscale = false);
    
    const FloatImage& getLines() const;
    
    const FloatImage& getGradient() const;
    
    const Image& getImage() const { return _image; }
    
    
private:
    
    Image _image;
    std::unordered_map<BlurSetting, FloatImage> _cache;
    
    FloatImage _ix;
    FloatImage _iy;
    FloatImage _mag;
    
    FloatImage _lines;
    FloatImage _weak;
};
