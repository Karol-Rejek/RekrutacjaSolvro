#include "Trainer.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include "../../vendor/json.hpp"

using json = nlohmann::json;

Trainer::Trainer(Classifier classifier, TrainerConfig cfg, torch::Device device): classifier_(std::move(classifier)), cfg_(std::move(cfg)), device_(device),
                 optimizer_(ModelLoader::trainableParams(classifier_.module()),torch::optim::AdamWOptions(cfg_.learningRate).weight_decay(cfg_.weightDecay))
{
    torch::manual_seed(cfg_.seed);
    fs::create_directories(cfg_.checkpointDir);
}

ClassificationMetrics Trainer::Train(ImageDataset& trainDataset, ImageDataset& valDataset)
{
    auto trainLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(trainDataset,torch::data::DataLoaderOptions().batch_size(cfg_.batchSize).workers(cfg_.numWorkers));

    double bestValF1 = 0.0;
    ClassificationMetrics bestMetrics;

    for (int64_t epoch = 1; epoch <= cfg_.epochs; ++epoch)
    {
        classifier_.module().train();
        double trainLoss = TrainOneEpoch(*trainLoader);
        trainLossHistory_.push_back(trainLoss);

        auto valMetrics = EvalSplit(valDataset);
        valLossHistory_.push_back(1.0 - valMetrics.f1Score);

        std::cout << "Epoch [" << epoch << "/" << cfg_.epochs << "]" << "  train_loss=" << trainLoss << "  val_F1=" << valMetrics.f1Score << "  val_AUC=" << valMetrics.rocAuc << "\n";

        if (valMetrics.f1Score > bestValF1) {
            bestValF1 = valMetrics.f1Score;
            bestMetrics = valMetrics;
            SaveCheckpoint("best_" + toString(classifier_.mode()));
        }
    }

    json history;
    history["mode"] = toString(classifier_.mode());
    history["train_loss"] = trainLossHistory_;
    history["val_proxy"] = valLossHistory_;

    auto histPath = cfg_.checkpointDir
        / ("history_" + toString(classifier_.mode()) + ".json");
    std::ofstream(histPath) << history.dump(2);
    std::cout << "Historia zapisana: " << histPath << "\n";

    return bestMetrics;
}

double Trainer::TrainOneEpoch(torch::data::StatelessDataLoader<ImageDataset, torch::data::samplers::RandomSampler>& loader)
{
    double  totalLoss = 0.0;
    int64_t batches = 0;
    auto    criterion = torch::nn::CrossEntropyLoss();

    for (auto& batch : loader) 
    {
        optimizer_.zero_grad();

        std::vector<torch::Tensor> imgs, lbls;
        for (auto& sample : batch) {
            imgs.push_back(sample.image);
            lbls.push_back(sample.labelTensor());
        }
        auto images = torch::stack(imgs).to(device_);
        auto targets = torch::stack(lbls).to(device_);

        auto logits = classifier_.forward(images);
        auto loss = criterion(logits, targets);

        loss.backward();
        optimizer_.step();

        totalLoss += loss.item<double>();
        ++batches;
    }

    return batches > 0 ? totalLoss / batches : 0.0;
}

ClassificationMetrics Trainer::EvalSplit(ImageDataset& dataset)
{
    classifier_.module().eval();
    torch::NoGradGuard noGrad;

    auto loader = torch::data::make_data_loader(
        dataset,
        torch::data::DataLoaderOptions()
        .batch_size(cfg_.batchSize)
        .workers(0));

    std::vector<torch::Tensor> allPreds, allTargets, allProbs;

    for (auto& batch : *loader) {
        std::vector<torch::Tensor> imgs, lbls;
        for (auto& sample : batch) {
            imgs.push_back(sample.image);
            lbls.push_back(sample.labelTensor());
        }

        auto images = torch::stack(imgs).to(device_);
        auto targets = torch::stack(lbls);

        auto logits = classifier_.forward(images).cpu();
        auto probs = torch::softmax(logits, 1).select(1, 1);
        auto preds = logits.argmax(1);

        allPreds.push_back(preds);
        allTargets.push_back(targets);
        allProbs.push_back(probs);
    }

    return ClassificationMetrics::compute(torch::cat(allPreds), torch::cat(allTargets), torch::cat(allProbs));
}

ClassificationMetrics Trainer::Evaluate(ImageDataset& testDataset)
{
    auto metrics = EvalSplit(testDataset);

    json results;
    results["mode"] = toString(classifier_.mode());
    results["accuracy"] = metrics.accuracy;
    results["precision"] = metrics.precision;
    results["recall"] = metrics.recall;
    results["f1_score"] = metrics.f1Score;
    results["roc_auc"] = metrics.rocAuc;

    fs::path outDir = "results";
    fs::create_directories(outDir);
    auto outPath = outDir / (toString(classifier_.mode()) + "_metrics.json");
    std::ofstream(outPath) << results.dump(2);
    std::cout << "Wyniki testowe zapisano: " << outPath << "\n";

    return metrics;
}

void Trainer::SaveCheckpoint(const std::string& tag) const
{
    auto path = cfg_.checkpointDir / (tag + ".pt");
    classifier_.module().save(path.string());
    std::cout << "Checkpoint zapisany: " << path << "\n";
}

void Trainer::LoadCheckpoint(const std::string& path)
{
    if (!fs::exists(path))
        throw std::runtime_error("Checkpoint nie istnieje: " + path);

    classifier_.module() = torch::jit::load(path, device_);
    std::cout << "Checkpoint wczytany: " << path << "\n";
}

const std::vector<double>& Trainer::TrainLossHistory() const
{
    return trainLossHistory_;
}

const std::vector<double>& Trainer::ValLossHistory() const
{
    return valLossHistory_;
}
