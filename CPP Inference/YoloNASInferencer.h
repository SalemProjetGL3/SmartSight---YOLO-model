#ifndef YOLONAS_INFERENCER_H
#define YOLONAS_INFERENCER_H

#include "OrtModel.h"
#include "ImagePreprocessor.h"
#include <vector>
#include <string>

class YoloNASInferencer {
public:
    explicit YoloNASInferencer(const std::wstring& model_path);
    std::vector<std::vector<float>> runInference(const std::string& image_path);

private:
    OrtModel model;
};

#endif
