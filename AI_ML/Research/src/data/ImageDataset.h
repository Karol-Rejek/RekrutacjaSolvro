#pragma once
#include <torch/torch.h>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include "ImageDatasetConfig.h"
#include "ImageTypes.h"

namespace fs = std::filesystem;

struct ImageSample 
{
    torch::Tensor image;

    DrawingType   drawingType;

    CaptureMethod captureMethod;
    int64_t       contentClass;
    std::string   contentName;
    std::string   filePath;
    uint32_t      sampleIndex;

    torch::Tensor labelTensor() const 
    {
        return torch::tensor(static_cast<int64_t>(drawingType), torch::dtype(torch::kLong));
    }
};

struct SourceMeta 
{ 
    DrawingType drawingType; 
    CaptureMethod captureMethod; 
};

class ImageDataset : public torch::data::datasets::Dataset<ImageDataset, ImageSample>
{
	//----------VARIABLES----------
public:
    using SplitPair = std::pair<ImageDataset, ImageDataset>;

private:
    static const std::unordered_map<std::string, SourceMeta> kSourceDirMap;

    ImageDatasetConfig cfg_;
    std::vector<ImageSample> samples_;
    std::vector<std::string> classNames_;

	//----------CONSTRUCTORS----------
public:
    explicit ImageDataset(ImageDatasetConfig cfg);
    ImageDataset(ImageDatasetConfig cfg, std::vector<ImageSample> samples);

    //----------METHODS----------
public:
    ImageSample get(size_t index) override;
    torch::optional<size_t> size() const override;

public:
    int64_t NumClasses()  const;
    const std::vector<std::string>& ClassNames()  const;
    void PrintStats()  const;
    SplitPair SplitTrainVal(float valRatio = 0.2f, uint32_t seed = 42)   const;

private:
    void BuildIndex();
    torch::Tensor LoadImage(const fs::path& path) const;
    torch::Tensor Normalize(torch::Tensor img) const;
    torch::Tensor Augment(torch::Tensor img, DrawingType dt) const;

};
