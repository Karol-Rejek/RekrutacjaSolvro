#pragma once
#include <filesystem>
#include <vector>
#include "ImageTypes.h"

namespace fs = std::filesystem;

struct ImageDatasetConfig
{
    fs::path rootDir;
    int64_t imgHeight = 64;
    int64_t imgWidth = 64;
    int64_t numChannels = 1;
    bool augment = false;
    float normMean = 0.5f;
    float normStd = 0.5f;

    std::vector<SourceType> allowedSources = {
        SourceType::HandDrawnPhoto,
        SourceType::HandDrawnScan,
        SourceType::StampPhoto
    };
};
