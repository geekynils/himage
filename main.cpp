#include "image.h"
#include "image_ops.h"
#include "vec.h"
#include "canny.h"

#include <iostream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <functional>

#include <cmath>
#include <cfloat>

using namespace std;

namespace fs = filesystem;

struct Args {
    fs::path inputPath;
    fs::path outputPath;
    fs::path debugOutputPath;
    float th = 0.1;
    float tl = 0.3;
};

void usage(const string& appName, ostream& os) {
    os << "Usage: " << appName << " -o outputFile intputFile "
       << "[-d debugOutputDir -th lowThreshold -th highThreshold]" << endl;
}

bool parseArgs(int argc, char* argv[], Args* pathsPtr) {
    
    Args& args = *pathsPtr;
    
    for (int i = 1; i < argc;) {
        if (argv[i] == string("-o") && i + 1 < argc) {
            args.outputPath = fs::path(argv[i + 1]);
            i += 2;
        } else if (argv[i] == string("-d") && i + 1 < argc) {
            args.debugOutputPath = fs::path(argv[i + 1]);
            i += 2;
        } else if (argv[i] == string("-tl") && i + 1 < argc) {
            try {
                float tl = stof(argv[i + 1]);
                args.tl = tl;
            } catch (std::invalid_argument const& ex) {
                cerr << ex.what() << endl;
                return false;
            }
            i += 2;
        } else if (argv[i] == string("-th") && i + 1 < argc) {
            try {
                float th = stof(argv[i + 1]);
                args.th = th;
            } catch (std::invalid_argument const& ex) {
                cerr << ex.what() << endl;
                return false;
            }
            i += 2;
        } else {
            if (!args.inputPath.empty()) {
                return false;
            }
            args.inputPath = fs::path(argv[i]);
            ++i;
        }
    }

    return !(args.inputPath.empty() || args.outputPath.empty());
}

int main(int argc, char* argv[]) {

    if (argc < 4) {
        usage(argv[0], cerr);
        return 1;
    }

    Args args;
    if (!parseArgs(argc, argv, &args)) {
        usage(argv[0], cerr);
        return 1;
    }

    
    EdgeFinder edgeFinder;
    
    if (!edgeFinder.readImage(args.inputPath)) {
        cerr << "Failed to read: " << args.inputPath << endl;
        return 1;
    }
    
    edgeFinder.calcGrads();
    edgeFinder.calcNonMaxSuppression(args.th, args.tl);
    
    Image edgeImage = invert(edgeFinder.getLines().toUint8());
    
    if (!edgeImage.writePng(args.outputPath)) {
        cerr << "Failed to write: " << args.outputPath << endl;
        return 1;
    }
    
    return 0;
}
