// Research.cpp: definiuje punkt wejścia dla aplikacji.
//

#include "Research.h"
#include <iostream>
#include <fstream>
#include "vendor/json.hpp"

using json = nlohmann::json;

torch::Device selectDevice()
{
    torch::Device device = torch::cuda::is_available()
        ? torch::kCUDA : torch::kCPU;
    std::cout << "Urządzenie: "
        << (device == torch::kCUDA ? "CUDA" : "CPU") << "\n";
    return device;
}

void runExperiment(const ExperimentConfig& expCfg)
{
    torch::manual_seed(expCfg.seed);
    auto device = selectDevice();

    ImageDatasetConfig dataCfg{.rootDir = expCfg.datasetRoot, .imgHeight = expCfg.imgSize, .imgWidth = expCfg.imgSize, .numChannels = 1};
    ImageDataset fullDataset(dataCfg);
    fullDataset.PrintStats();

    auto [trainVal, test] = fullDataset.SplitTrainVal(0.15f, expCfg.seed);
    auto [train, val] = trainVal.SplitTrainVal(0.176f, expCfg.seed);

    std::cout << "\n=== EKSPERYMENT 1: Frozen backbone ===\n";
    auto frozenModule = ModelLoader::load(expCfg.modelsDir / (expCfg.modelName + ".pt"), device, true);
    Classifier frozenCls(frozenModule, FineTuningMode::Frozen, device);

    auto frozenCfg = expCfg.trainerCfg;
    frozenCfg.checkpointDir = "checkpoints/frozen";
    Trainer frozenTrainer(frozenCls, frozenCfg, device);
    frozenTrainer.Train(train, val);
    auto frozenTest = frozenTrainer.Evaluate(test);
    std::cout << "Frozen test:\n" << frozenTest.toString();

    std::cout << "\n=== EKSPERYMENT 2: Full fine-tuning ===\n";
    auto fullModule = ModelLoader::load(expCfg.modelsDir / (expCfg.modelName + ".pt"), device, false);
    Classifier fullCls(fullModule, FineTuningMode::Full, device);

    auto fullCfg = expCfg.trainerCfg;
    fullCfg.learningRate = 1e-5f;
    fullCfg.checkpointDir = "checkpoints/full_finetune";
    Trainer fullTrainer(fullCls, fullCfg, device);
    fullTrainer.Train(train, val);
    auto fullTest = fullTrainer.Evaluate(test);
    std::cout << "Full FT test:\n" << fullTest.toString();

    json comparison;
    comparison["model"] = expCfg.modelName;
    comparison["frozen"] = { {"f1_score", frozenTest.f1Score}, {"roc_auc",  frozenTest.rocAuc}, {"accuracy", frozenTest.accuracy} };
    comparison["full_ft"] = { {"f1_score", fullTest.f1Score}, {"roc_auc",  fullTest.rocAuc}, {"accuracy", fullTest.accuracy} };
    comparison["winner"] = frozenTest.f1Score > fullTest.f1Score ? "frozen" : "full_finetune";

    fs::create_directories(expCfg.resultsDir);
    std::ofstream(expCfg.resultsDir / "comparison.json") << comparison.dump(2);
    std::cout << "\nWyniki zapisano w: " << expCfg.resultsDir / "comparison.json" << "\n";
}

int main()
{
    ExperimentConfig cfg;
    cfg.trainerCfg.epochs = 30;
    cfg.trainerCfg.batchSize = 32;

    runExperiment(cfg);
    return 0;
}
