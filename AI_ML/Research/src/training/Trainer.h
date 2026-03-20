#pragma once
#include <torch/torch.h>
#include <filesystem>
#include <string>
#include <vector>
#include "metrics.h"
#include "../models/Classifier.h"
#include "../models/ModelLoader.h"
#include "../data/ImageDataset.h"

namespace fs = std::filesystem;

struct TrainerConfig 
{
    int64_t epochs = 30;
    int64_t batchSize = 32;
    float learningRate = 1e-4f;
    float weightDecay = 1e-4f;
    int64_t numWorkers = 0;
    fs::path checkpointDir = "checkpoints";
    uint32_t seed = 42;
};

class Trainer
{
    //----------VARIABLES----------
private:
    Classifier classifier_;
    TrainerConfig cfg_;
    torch::Device device_;
    torch::optim::AdamW optimizer_;
    std::vector<double> trainLossHistory_;
    std::vector<double> valLossHistory_;

	//----------CONSTRUCTORS----------
public:
    Trainer(Classifier classifier, TrainerConfig cfg, torch::Device device);

	//----------METHODS----------
public:
    ClassificationMetrics Train(ImageDataset& trainDataset, ImageDataset& valDataset);
    ClassificationMetrics Evaluate(ImageDataset& testDataset);

public:
    void SaveCheckpoint(const std::string& tag)  const;
    void LoadCheckpoint(const std::string& path);

    const std::vector<double>& TrainLossHistory() const;
    const std::vector<double>& ValLossHistory()   const;

private:
    double TrainOneEpoch(torch::data::StatelessDataLoader<ImageDataset,torch::data::samplers::RandomSampler>& loader);
    ClassificationMetrics EvalSplit(ImageDataset& dataset);
};
