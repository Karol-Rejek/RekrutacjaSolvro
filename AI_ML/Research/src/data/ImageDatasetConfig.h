#pragma once
#include <filesystem>
#include <vector>
#include "ImageTypes.h"

struct ImageDatasetConfig
{
	std::filesystem::path path;
	int64_t height;
	int64_t width;
	int64_t numChannels;
	std::vector<SourceType> allowedSourceTypes =
	{
		SourceType::HandDrawnPhoto, SourceType::HandDrawnScan, SourceType::StampPhoto
	};
};