// Research.h: plik dołączany dla standardowych systemowych plików dołączanych,
// lub pliki dołączane specyficzne dla projektu.

#pragma once
#include <torch/torch.h>
#include <filesystem>
#include "src/data/ImageDataset.h"
#include "src/data/ImageDatasetConfig.h"
#include "src/models/ModelLoader.h"
#include "src/models/Classifier.h"
#include "src/training/Trainer.h"
#include "src/training/metrics.h"

namespace fs = std::filesystem;

struct ExperimentConfig
{
    fs::path    datasetRoot = "data/raw";
    fs::path    modelsDir = "models";
    fs::path    resultsDir = "results";
    std::string modelName = "resnet18";   
    int64_t     imgSize = 64;
    uint32_t    seed = 42;
    TrainerConfig trainerCfg;
};

void runExperiment(const ExperimentConfig& cfg);
void compareModels(const ExperimentConfig& cfg);
torch::Device selectDevice();

// TODO: W tym miejscu przywołaj dodatkowe nagłówki wymagane przez program.
