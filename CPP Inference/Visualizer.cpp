#include "Visualizer.h"

void Visualizer::drawDetections(cv::Mat& image, const std::vector<std::vector<float>>& detections, float conf_threshold) {
    int img_width = image.cols;
    int img_height = image.rows;

    for (const auto& det : detections) {
        if (det.size() < 6 || det[4] < conf_threshold) continue;

        // Clip coordinates to image boundaries
        float x1 = std::max(0.0f, std::min(det[0], static_cast<float>(img_width - 1)));
        float y1 = std::max(0.0f, std::min(det[1], static_cast<float>(img_height - 1)));
        float x2 = std::max(0.0f, std::min(det[2], static_cast<float>(img_width - 1)));
        float y2 = std::max(0.0f, std::min(det[3], static_cast<float>(img_height - 1)));

        cv::Point top_left(static_cast<int>(x1), static_cast<int>(y1));
        cv::Point bottom_right(static_cast<int>(x2), static_cast<int>(y2));

        cv::rectangle(image, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);
        std::string label = "Class: " + std::to_string(static_cast<int>(det[5])) +
                          " Conf: " + std::to_string(det[4]).substr(0, 4);
        cv::putText(image, label, cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
    for (const auto& det : detections) {
        std::cout << "Detection: [" << det[0] << ", " << det[1] << ", "
                  << det[2] << ", " << det[3] << "]" << std::endl;
    }
}
