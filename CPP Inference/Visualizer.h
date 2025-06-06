#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class Visualizer {
public:
    static void drawDetections(cv::Mat& image, const std::vector<std::vector<float>>& detections, float conf_threshold = 0.5f);
};
