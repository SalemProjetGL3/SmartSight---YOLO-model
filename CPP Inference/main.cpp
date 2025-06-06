#include "YoloNASInferencer.h"
#include "Visualizer.h"
#include <iostream>
#include <filesystem>
#include <random>

namespace fs = std::filesystem;

// C++ equivalent of your Python get_random_image_from_dataset()
std::string getRandomImageFromDataset(const std::string& dataset_path = "C:/Users/21654/OneDrive/Desktop/PPP/DATA/OOD.v1i.yolov8/test/images") {
    std::vector<std::string> image_files;

    for (const auto& entry : fs::directory_iterator(dataset_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                image_files.push_back(entry.path().string());
            }
        }
    }

    if (image_files.empty()) {
        throw std::runtime_error("No image files found in dataset path.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, image_files.size() - 1);
    return image_files[dist(gen)];
}

int main() {
    try {
        YoloNASInferencer inferencer(L"yolo_nas_s.onnx");

        std::string image_path = getRandomImageFromDataset();
        std::cout << "Using image: " << image_path << std::endl;

        std::vector<std::vector<float>> detections = inferencer.runInference(image_path);

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) throw std::runtime_error("Failed to read selected image.");

        Visualizer::drawDetections(image, detections);

        cv::imshow("Detections", image);
        cv::waitKey(0);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
