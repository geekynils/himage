#include "image.h"
#include "image_ops.h"
#include "canny.h"

#include <iostream>
#include <map>
#include <filesystem>

#include <cstdio>

#include <ctime>

#include <unistd.h> // TODO Windows

using namespace std;
namespace fs = filesystem;

enum TestResult {
    SUCCESS,
    FAILURE,
    SKIPPED
};

TestResult rwPng(const fs::path& testImagePath, fs::path *outputPath, double* time) {
    
    *time = -1;

    Image image;
    if (!image.read(testImagePath)) {
        return FAILURE;
    }

    char tempPath[] = "/tmp/test_rw_XXXXXX.png";
    if (!mkstemps(tempPath, 4)) {
        cerr << "Failed to create temp file!" << endl;
        return FAILURE;
    } 

    if (!image.writePng(tempPath)) {
        return FAILURE;
    }

    *outputPath = fs::path(tempPath);

    Image image2;
    if (!image2.read(tempPath)) {
        return FAILURE;
    }

    if (!(image.getWidth() == image2.getWidth() 
       && image.getHeight() == image2.getHeight() 
       && image.getNumChannels() == image2.getNumChannels())) {
        cerr << "Dims are not the same after writing and reading again!" << endl;
        return FAILURE;
    }

    for (int row = 0; row < image.getHeight(); ++row) {
        for (int col = 0; col < image.getWidth(); ++col) {
            for (int i = 0; i < image.getNumChannels(); ++i) {
                if (image.get(col, row, i) != image2.get(col, row, i)) {
                    cerr << "Pixels are not equal after writing and reading again!" 
                         << endl;
                    return FAILURE;
                }
            }
        }
    }

    return SUCCESS;
}

TestResult removeAlpha(const fs::path& testImagePath, fs::path *outputPath, double* time) {
    
    Image image;
    if (!image.read(testImagePath)) {
        return FAILURE;
    }
    
    if (image.getNumChannels() != 4) {
        *time = -1;
        return SKIPPED;
    }

    clock_t start, end;
    start = clock();
    Image result = removeAlpha(image);
    end = clock();
    *time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    char tempPath[] = "/tmp/test_rm_alpha_XXXXXX.png";
    if (!mkstemps(tempPath, 4)) {
        cerr << "Failed to create temp file!" << endl;
        return FAILURE;
    }

    if (!result.writePng(tempPath)) {
        return FAILURE;
    }

    *outputPath = fs::path(tempPath);

    return SUCCESS;
    
}

TestResult makeRed(const fs::path& testDataPath, fs::path *outputPath, double* time) {
    
    *time = -1;

    const int w = 800, h = 600, c = 3;

    Image image(w, h, c);
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            image.set(col, row, 0, 255);
            image.set(col, row, 1, 0);
            image.set(col, row, 2, 0);
        }
    }

    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            if (image.get(col, row, 0) != 255) {
                return FAILURE;
            }
            if (image.get(col, row, 1) != 0) {
                return FAILURE;
            }
            if (image.get(col, row, 1) != 0) {
                return FAILURE;
            }
        }
    }

    return SUCCESS;
}

TestResult blur(const fs::path& testImagePath, fs::path *outputPath, double* time) {

    Image image;
    if (!image.read(testImagePath)) {
        return FAILURE;
    }

    Kernel<5> k = create5gauss();

    clock_t start, end;
    start = clock();
    FloatImage result = conv(image, k);
    end = clock();
    *time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    char tempPath[] = "/tmp/test_gauss_XXXXXX.png";
    if (!mkstemps(tempPath, 4)) {
        cerr << "Failed to create temp file!" << endl;
        return FAILURE;
    }

    if (!result.toUint8().writePng(tempPath)) {
        return FAILURE;
    }

    *outputPath = fs::path(tempPath);

    return SUCCESS;
}

TestResult blurSep(const fs::path& testImagePath, fs::path *outputPath, double* time) {

    Image image;
    if (!image.read(testImagePath)) {
        return FAILURE;
    }

    KernelV<5> gaussV = create5gaussV();
    KernelH<5> gaussH = create5gaussH();

    clock_t start, end;
    start = clock();
    FloatImage result = conv(image, gaussV, gaussH);
    end = clock();
    *time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    char tempPath[] = "/tmp/test_gausssep_XXXXXX.png";
    if (!mkstemps(tempPath, 4)) {
        cerr << "Failed to create temp file!" << endl;
        return FAILURE;
    }

    if (!result.toUint8().writePng(tempPath)) {
        return FAILURE;
    }

    *outputPath = fs::path(tempPath);

    return SUCCESS;
}

TestResult sobelX(const fs::path& testImagePath, fs::path *outputPath, double *time) {
    
    Image image;
    if (!image.read(testImagePath)) {
        return FAILURE;
    }
    
    if (image.getNumChannels() == 4) { image = removeAlpha(image); }
    
    Kernel<3> k = createSobelX();
    
    clock_t start, end;
    start = clock();
    FloatImage result = convReduce(image, k);
    end = clock();
    *time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    char tempPath[] = "/tmp/test_sobelx_XXXXXX.png";
    if (!mkstemps(tempPath, 4)) {
        cerr << "Failed to create temp file!" << endl;
        return FAILURE;
    }

    if (!result.toUint8().writePng(tempPath)) {
        return FAILURE;
    }
    
    *outputPath = fs::path(tempPath);
    
    return SUCCESS;
}

TestResult sobelY(const fs::path& testImagePath, fs::path *outputPath, double *time) {
    
    Image image;
    if (!image.read(testImagePath)) {
        return FAILURE;
    }
    
    if (image.getNumChannels() == 4) { image = removeAlpha(image); }
    
    Kernel<3> k = createSobelY();
    
    clock_t start, end;
    start = clock();
    FloatImage result = convReduce(image, k);
    end = clock();
    *time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    char tempPath[] = "/tmp/test_sobely_XXXXXX.png";
    if (!mkstemps(tempPath, 4)) {
        cerr << "Failed to create temp file!" << endl;
        return FAILURE;
    }

    if (!result.toUint8().writePng(tempPath)) {
        return FAILURE;
    }
    
    *outputPath = fs::path(tempPath);
    
    return SUCCESS;
}

TestResult sobelXSep(const fs::path& testImagePath, fs::path *outputPath, double *time) {
    
    Image image;
    if (!image.read(testImagePath)) {
        return FAILURE;
    }
    
    if (image.getNumChannels() == 4) { image = removeAlpha(image); }
    
    KernelH<3> kh = createSobelXH();
    KernelV<3> kv = createSobelXV();
    
    clock_t start, end;
    start = clock();
    FloatImage result = convReduce(image, kv, kh);
    end = clock();
    *time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    char tempPath[] = "/tmp/test_sobelxsep_XXXXXX.png";
    if (!mkstemps(tempPath, 4)) {
        cerr << "Failed to create temp file!" << endl;
        return FAILURE;
    }

    if (!result.toUint8().writePng(tempPath)) {
        return FAILURE;
    }
    
    *outputPath = fs::path(tempPath);
    
    return SUCCESS;
}

TestResult sobelYSep(const fs::path& testImagePath, fs::path *outputPath, double *time) {
    
    Image image;
    if (!image.read(testImagePath)) {
        return FAILURE;
    }
    
    if (image.getNumChannels() == 4) { image = removeAlpha(image); }
    
    KernelH<3> kh = createSobelYH();
    KernelV<3> kv = createSobelYV();
    
    clock_t start, end;
    start = clock();
    FloatImage result = convReduce(image, kv, kh);
    end = clock();
    *time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    char tempPath[] = "/tmp/test_sobelysep_XXXXXX.png";
    if (!mkstemps(tempPath, 4)) {
        cerr << "Failed to create temp file!" << endl;
        return FAILURE;
    }

    if (!result.toUint8().writePng(tempPath)) {
        return FAILURE;
    }
    
    *outputPath = fs::path(tempPath);
    
    return SUCCESS;
}

TestResult gradMag(const fs::path& testImagePath, fs::path *outputPath, double *time) {
    Image image;
    if (!image.read(testImagePath)) {
        return FAILURE;
    }
    
    if (image.getNumChannels() == 4) { image = removeAlpha(image); }
    
    clock_t start, end;
    start = clock();
    FloatImage ix, iy, mag;
    calcGrads(image, &ix, &iy, &mag, meanChannel, NO_BLUR);
    end = clock();
    *time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    char tempPath[] = "/tmp/test_gradMag_XXXXXX.png";
    if (!mkstemps(tempPath, 4)) {
        cerr << "Failed to create temp file!" << endl;
        return FAILURE;
    }

    if (!mag.toUint8().writePng(tempPath)) {
        return FAILURE;
    }
    
    *outputPath = fs::path(tempPath);
    return SUCCESS;
}

TestResult nonMaxSuppresion(const fs::path& testImagePath, fs::path *outputPath, double *time) {
    EdgeFinder edgeFinder;
    if (!edgeFinder.readImage(testImagePath)) {
        return FAILURE;
    }
    
    edgeFinder.calcGrads();
    
    clock_t start, end;
    start = clock();
    edgeFinder.calcNonMaxSuppression(0.3, 0.6);
    end = clock();
    *time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    char tempPath[] = "/tmp/test_nms_XXXXXX.png";
    if (!mkstemps(tempPath, 4)) {
        cerr << "Failed to create temp file!" << endl;
        return FAILURE;
    }

    if (!edgeFinder.getLines().toUint8().writePng(tempPath)) {
        return FAILURE;
    }
    
    *outputPath = fs::path(tempPath);
    return SUCCESS;
}

TestResult drawGradDirection(const fs::path& testImagePath, fs::path *outputPath, double *time) {
    *time = -1;
    
    Image image;
    if (!image.read(testImagePath)) {
        return FAILURE;
    }
    
    int w = image.getWidth(), h = image.getHeight();
    
    if (image.getNumChannels() == 4) { image = removeAlpha(image);}
    
    FloatImage ix, iy, mag;
    calcGrads(image, &ix, &iy, &mag, maxAbsChannel, NO_BLUR);
    FloatImage angleColorImage = colorizeGrads(ix, iy, &mag);

    char tempPath[] = "/tmp/test_gradDir_XXXXXX.png";
    if (!mkstemps(tempPath, 4)) {
        cerr << "Failed to create temp file!" << endl;
        return FAILURE;
    }

    if (!angleColorImage.toUint8().writePng(tempPath)) {
        return FAILURE;
    }
    
    *outputPath = fs::path(tempPath);
    return SUCCESS;
}

typedef TestResult ((*TestFunc)(const fs::path& testImagePath, fs::path *outputPathPtr, double *time));

void runTest(TestFunc testFunc, const string& name, const fs::path& testImagePath) {

    fs::path outputPath;
    double time;
    TestResult success = testFunc(testImagePath, &outputPath, &time);

    cout << name << "\t";
    switch(success) {
        case SUCCESS:
            cout << "[OK]";
            break;
        case FAILURE:
            cout << "[FAILED]";
            break;
        case SKIPPED:
            cout << "[SKIPPED]";
            break;
    }

    if (!outputPath.empty()) {
        cout << " out: " << outputPath.filename();
    }
    
    if (time != -1) { cout << "\t" << time << " secs"; }

    cout << endl;
}

void usage(const string& progName) {
    cout << "Usage: " << progName << " test_image [test]" << endl;
    cout << "If test is not specified it runs all tests. ";
    cout << "Use " << progName << " -l to list all tests" << endl;
}

int main(int argc, char* argv[]) { 

    if (argc < 2) {
        cerr << "Need at least one argument!" << endl;
        usage(argv[0]);
        return 1;
    }

    map<string, TestFunc> testsByName = {
        {"makeRed",     makeRed},
        {"pngRW",       rwPng},
        {"rmAlpha",     removeAlpha},
        {"blur",        blur},
        {"blurSep",     blurSep},
        {"sobelX",      sobelX},
        {"sobelY",      sobelY},
        {"sobelXSep",   sobelXSep},
        {"sobelYSep",   sobelYSep},
        {"gradMag",     gradMag},
        {"gradDir",     drawGradDirection},
        {"nms",         nonMaxSuppresion}
    };
    
    if (argv[1] == string("-l")) {
        for (const auto &[testName, func]: testsByName) {
            cout << testName << endl;
        }
        return 0;
    }
    
    fs::path testImagePath(argv[1]);
    
    if (argc >= 3) {
        for (int i = 2; i < argc; ++i) {
            string testName = argv[i];
            auto it = testsByName.find(testName);
            if (it == testsByName.end()) {
                cerr << "Could not find test '" << testName << "'!" << endl;
                continue;
            }
            runTest(it->second, it->first, testImagePath);
        }
    } else {
        for (const auto &[testName, func]: testsByName) {
            runTest(func, testName, testImagePath);
        }
    }

    return 0;

}
