#pragma once
#include <torch/torch.h>
#include <string>

struct ClassificationMetrics 
{
    double accuracy = 0.0;
    double precision = 0.0;   
    double recall = 0.0;  
    double f1Score = 0.0;  
    double rocAuc = 0.0;

    torch::Tensor confusionMatrix;

    static ClassificationMetrics compute(const torch::Tensor& preds, const torch::Tensor& targets, const torch::Tensor& probs, int64_t numClasses = 2);

    std::string toString() const;
};