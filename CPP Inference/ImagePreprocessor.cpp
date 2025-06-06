#include "ImagePreprocessor.h"
#include <stdexcept>

std::vector<float> ImagePreprocessor::preprocess(const std::string& image_path, int& width, int& height) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    width = image.cols;
    height = image.rows;

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(640, 640));
    image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

    std::vector<float> tensor_data(3 * 640 * 640);
    int idx = 0;

    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < 640; ++y) {
            for (int x = 0; x < 640; ++x) {
                tensor_data[idx++] = image.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    return tensor_data;
}
