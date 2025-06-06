#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class ImagePreprocessor {
public:
    static std::vector<float> preprocess(const std::string& image_path, int& width, int& height);
};
