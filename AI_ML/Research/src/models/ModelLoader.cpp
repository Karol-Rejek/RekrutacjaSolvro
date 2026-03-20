#include "ModelLoader.h"
#include <stdexcept>
#include <iostream>

torch::jit::script::Module ModelLoader::load(const fs::path& modelPath, torch::Device device, bool frozen)
{
    if (!fs::exists(modelPath))
        throw std::runtime_error("Model nie istnieje: " + modelPath.string());

    auto module = torch::jit::load(modelPath.string(), device);
    module.train();

    if (frozen) 
    {
        for (auto param : module.parameters())
        {
            param.requires_grad_(false);
        }

        auto params = module.parameters();
        auto all = std::vector<torch::Tensor>(params.begin(), params.end());
        if (all.size() >= 2)
        {
            all[all.size() - 1].requires_grad_(true); 
            all[all.size() - 2].requires_grad_(true); 
        }
        std::cout << "[ModelLoader] Tryb: frozen backbone\n";
    }
    else 
    {
        for (auto param : module.parameters())
        {
            param.requires_grad_(true);
        }
        std::cout << "[ModelLoader] Tryb: full fine-tuning\n";
    }

    return module;
}

std::vector<torch::Tensor> ModelLoader::trainableParams(torch::jit::script::Module& module)
{
    std::vector<torch::Tensor> params;
    for (auto param : module.parameters())
    {
        if (param.requires_grad())
        {
            params.push_back(param);
        }
    }
    return params;
}