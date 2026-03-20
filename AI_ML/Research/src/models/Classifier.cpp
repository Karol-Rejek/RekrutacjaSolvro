#include "Classifier.h"
#include <stdexcept>

Classifier::Classifier(torch::jit::script::Module module, FineTuningMode mode, torch::Device device) : module_(std::move(module)), mode_(mode), device_(device)
{
    module_.to(device_);
}

torch::Tensor Classifier::forward(const torch::Tensor& input)
{
    std::vector<torch::jit::IValue> inputs{ input.to(device_) };
    return module_.forward(inputs).toTensor();
}