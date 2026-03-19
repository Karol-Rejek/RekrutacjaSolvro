#pragma once
#include <torch/torch.h>

#include "ImageDatasetConfig.h"
#include "ImageTypes.h"


struct ImageSample
{
	torch::Tensor image;
	DrawingType drawingType;
	CaptureMetod captureMetod;
	uint32_t sampleIndex;

	std::string filePath;
	
	torch::Tensor label() const
	{
		return torch::tensor(static_cast<int64_t>(drawingType), torch::dtype(torch::kLong));
	}
};

struct SourceData
{
	DrawingType drawingType;
	CaptureMetod captureMetod;
};

class ImageDataset : public torch::data::Dataset<ImageDataset, ImageSample>
{
	//----------Variables----------
private:
	SourceData sourceData;

	ImageDatasetConfig config;
	std::vector<ImageSample> samples;

	//----------Constructors----------
public:
	explicit ImageDataset(ImageDatasetConfig config);

	//----------Destructors----------
	~ImageDataset();

	//----------Metods----------
public:
	ImageSample get(size_t index) override;
	torch::optional<size_t> size() const override;

private:
	void BuildIndex();
};