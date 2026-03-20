#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

struct ModelLoader 
{
    static torch::jit::script::Module load(const fs::path& modelPath, torch::Device device,  bool frozen = false);

    static std::vector<torch::Tensor> trainableParams(torch::jit::script::Module& module);
};

