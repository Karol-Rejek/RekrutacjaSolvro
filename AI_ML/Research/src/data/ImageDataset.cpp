#include "ImageDataset.h"

ImageDataset::ImageDataset(ImageDatasetConfig config) : config(std::move(config))
{
	BuildIndex();
}

void ImageDataset::BuildIndex()
{
	
}