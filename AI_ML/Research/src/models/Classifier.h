#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <string>

enum class FineTuningMode { Frozen, Full };

inline std::string toString(FineTuningMode m)
{
    return m == FineTuningMode::Frozen ? "frozen" : "full_finetune";
}

class Classifier
{
	//----------VARIABLES----------
private:
    torch::jit::script::Module module_;
    FineTuningMode mode_;
    torch::Device device_;

	//----------CONSTRUCTORS----------
public:
    Classifier(torch::jit::script::Module module, FineTuningMode mode, torch::Device device);


    //----------METHODS----------
public:
    torch::Tensor forward(const torch::Tensor& input);

	//----------GETTERS----------
public:
    FineTuningMode mode()   const { return mode_; }
    torch::Device device() const { return device_; }
    torch::jit::script::Module& module() { return module_; }
    const torch::jit::script::Module& module() const { return module_; }
};
