#include "ImageDataset.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "../../vendor/stb_image.h"

const std::unordered_map<std::string, SourceMeta> ImageDataset::kSourceDirMap = {
    {"hand_photo",  {DrawingType::HandDrawn, CaptureMethod::Photo}},
    {"hand_scan",   {DrawingType::HandDrawn, CaptureMethod::Scan }},
    {"stamp_photo", {DrawingType::Digital,   CaptureMethod::Photo}}
};

ImageDataset::ImageDataset(ImageDatasetConfig cfg) : cfg_(std::move(cfg)) 
{
    BuildIndex();
}

ImageDataset::ImageDataset(ImageDatasetConfig cfg, std::vector<ImageSample> samples): cfg_(std::move(cfg)), samples_(std::move(samples))
{}

void ImageDataset::BuildIndex()
{
    std::vector<fs::path> classDirs;
    for (auto& e : fs::directory_iterator(cfg_.rootDir))
    {
        if (e.is_directory())
        {
            classDirs.push_back(e.path());
        }
    }
    std::sort(classDirs.begin(), classDirs.end());

    int64_t  classId = 0;
    uint32_t idx = 0;

    for (auto& classDir : classDirs)
    {
        const std::string className = classDir.filename().string();
        classNames_.push_back(className);

        for (auto& [dirName, meta] : kSourceDirMap)
        {
            bool allowed = std::any_of(cfg_.allowedSources.begin(), cfg_.allowedSources.end(),[&](SourceType st) { return toString(st) == dirName; });

            if (!allowed)
                continue;

            fs::path srcDir = classDir / dirName;

            if (!fs::exists(srcDir))
                continue;

            for (auto& imgEntry : fs::directory_iterator(srcDir))
            {
                if (!imgEntry.is_regular_file()) 
                    continue;

                auto ext = imgEntry.path().extension().string();

                if (ext != ".jpg" && ext != ".png" && ext != ".jpeg") 
                    continue;

                samples_.push_back({
                    .image = {},
                    .drawingType = meta.drawingType,
                    .captureMethod = meta.captureMethod,
                    .contentClass = classId,
                    .contentName = className,
                    .filePath = imgEntry.path().string(),
                    .sampleIndex = idx++
                    });
            }
        }
        ++classId;
    }
}

ImageSample ImageDataset::get(size_t index)
{
    auto sample = samples_.at(index);
    sample.image = LoadImage(sample.filePath);
    sample.image = Normalize(sample.image);
    if (cfg_.augment)
        sample.image = Augment(sample.image, sample.drawingType);
    return sample;
}

torch::Tensor ImageDataset::LoadImage(const fs::path& path) const
{
    int w, h, c;
    int desired = static_cast<int>(cfg_.numChannels);
    uint8_t* raw = stbi_load(path.string().c_str(), &w, &h, &c, desired);
    if (!raw)
        throw std::runtime_error("Nie można wczytać: " + path.string());

    auto t = torch::from_blob(raw, { h, w, cfg_.numChannels }, torch::kUInt8)
        .clone().to(torch::kFloat32).div_(255.0f);
    stbi_image_free(raw);

    t = t.permute({ 2, 0, 1 }).unsqueeze(0);

    t = torch::nn::functional::interpolate(t,
        torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<int64_t>{cfg_.imgHeight, cfg_.imgWidth})
        .mode(torch::kBilinear)
        .align_corners(false));

    return t.squeeze(0);
}

torch::Tensor ImageDataset::Normalize(torch::Tensor img) const
{
    return (img - cfg_.normMean) / cfg_.normStd;
}

torch::Tensor ImageDataset::Augment(torch::Tensor img, DrawingType dt) const
{
    if (torch::rand(1).item<float>() > 0.5f)
        img = torch::flip(img, { 2 });

    if (dt == DrawingType::HandDrawn)
    {
        auto padded = torch::nn::functional::pad(img, torch::nn::functional::PadFuncOptions({ 4, 4, 4, 4 }).mode(torch::kReflect));
        int64_t H = img.size(1), W = img.size(2);
        int64_t top = (torch::rand(1) * 8).to(torch::kLong).item<int64_t>();
        int64_t left = (torch::rand(1) * 8).to(torch::kLong).item<int64_t>();
        img = padded.slice(1, top, top + H).slice(2, left, left + W);

        img = (img + 0.02f * torch::randn_like(img)).clamp(-1.0f, 1.0f);
    }
    else 
    {
        float   scale = 0.9f + torch::rand(1).item<float>() * 0.2f;
        int64_t nH = static_cast<int64_t>(img.size(1) * scale);
        int64_t nW = static_cast<int64_t>(img.size(2) * scale);
        auto opts = torch::nn::functional::InterpolateFuncOptions().mode(torch::kBilinear).align_corners(false);
        img = torch::nn::functional::interpolate(img.unsqueeze(0), opts.size(std::vector<int64_t>{nH, nW})).squeeze(0);
        img = torch::nn::functional::interpolate(img.unsqueeze(0), opts.size(std::vector<int64_t>{cfg_.imgHeight, cfg_.imgWidth})).squeeze(0);
    }
    return img;
}

torch::optional<size_t> ImageDataset::size() const
{
    return samples_.size();
}

int64_t ImageDataset::NumClasses() const
{
    return static_cast<int64_t>(classNames_.size());
}

const std::vector<std::string>& ImageDataset::ClassNames() const
{
    return classNames_;
}

ImageDataset::SplitPair ImageDataset::SplitTrainVal(float valRatio, uint32_t seed) const
{
    std::unordered_map<int64_t, std::vector<size_t>> buckets;
    for (size_t i = 0; i < samples_.size(); ++i)
    {
        int64_t key = static_cast<int64_t>(samples_[i].drawingType) * 10 + samples_[i].contentClass;
        buckets[key].push_back(i);
    }

    std::mt19937 rng(seed);
    std::vector<ImageSample> trainSamples, valSamples;

    for (auto& [key, indices] : buckets)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        size_t valN = static_cast<size_t>(indices.size() * valRatio);
        for (size_t i = 0; i < indices.size(); ++i)
        {
            (i < valN ? valSamples : trainSamples).push_back(samples_[indices[i]]);
        }
    }

    auto trainCfg = cfg_; trainCfg.augment = true;
    auto valCfg = cfg_; valCfg.augment = false;

    return { ImageDataset(trainCfg, std::move(trainSamples)), ImageDataset(valCfg,   std::move(valSamples)) };
}

void ImageDataset::PrintStats() const
{
    size_t hand = 0, dig = 0;
    for (auto& s : samples_)
    {
        s.drawingType == DrawingType::HandDrawn ? ++hand : ++dig;
    }

    std::cout << "=== ImageDataset stats ===\n" << "  Łącznie próbek : " << samples_.size() << "\n" << "  Klas treści: " << classNames_.size() << "\n" << "  HandDrawn: " << hand << "\n" << "  Digital: " << dig << "\n";
}
