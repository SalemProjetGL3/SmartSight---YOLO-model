#include "YoloNASInferencer.h"
#include <stdexcept>
#include <iostream>

YoloNASInferencer::YoloNASInferencer(const std::wstring& model_path)
    : model(model_path) {}

std::vector<std::vector<float>> YoloNASInferencer::runInference(const std::string& image_path) {
    int orig_w, orig_h;
    std::vector<float> input_tensor_values = ImagePreprocessor::preprocess(image_path, orig_w, orig_h);

    std::vector<int64_t> input_dims = {1, 3, 640, 640};
    size_t tensor_size = 1 * 3 * 640 * 640;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), tensor_size,
        input_dims.data(), input_dims.size()
    );

    const std::vector<std::string>& input_names = model.getInputNames();
    const std::vector<std::string>& output_names = model.getOutputNames();

    std::vector<const char*> input_name_ptrs = {input_names[0].c_str()};
    std::vector<const char*> output_name_ptrs;
    for (const std::string& name : output_names) {
        output_name_ptrs.push_back(name.c_str());
    }

    auto output_tensors = model.getSession().Run(
        Ort::RunOptions{nullptr},
        input_name_ptrs.data(), &input_tensor, 1,
        output_name_ptrs.data(), output_name_ptrs.size()
    );

    if (output_tensors.empty()) {
        throw std::runtime_error("Model returned no outputs.");
    }

    std::cout << "Number of output tensors: " << output_tensors.size() << std::endl;
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        Ort::TensorTypeAndShapeInfo info = output_tensors[i].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = info.GetShape();
        std::cout << "Output " << i << " shape: ";
        for (auto dim : shape) std::cout << dim << " ";
        std::cout << std::endl;
    }

    if (output_tensors.size() < 2) {
        throw std::runtime_error("Expected at least 2 output tensors (boxes and scores).");
    }

    Ort::Value& box_tensor = output_tensors[0];
    float* boxes_data = box_tensor.GetTensorMutableData<float>();
    auto box_shape = box_tensor.GetTensorTypeAndShapeInfo().GetShape();  // [1, N, 4]
    int64_t num_boxes = box_shape[1];

    Ort::Value& score_tensor = output_tensors[1];
    float* scores_data = score_tensor.GetTensorMutableData<float>();
    auto score_shape = score_tensor.GetTensorTypeAndShapeInfo().GetShape();  // [1, N, num_classes]
    int64_t num_classes = score_shape[2];

    std::vector<std::vector<float>> detections;
    std::vector<std::vector<float>> raw_detections;
    int detected_count = 0;

    std::unordered_map<int64_t, std::tuple<float, int, float>> box_detections;

    // First collect all potential detections
    for (int64_t i = 0; i < num_boxes; ++i) {
        float* box_ptr = boxes_data + i * 4;
        float* class_ptr = scores_data + i * num_classes;

        float max_conf = 0.0f;
        int max_class = -1;

        for (int c = 0; c < num_classes; ++c) {
            if (class_ptr[c] > max_conf) {
                max_conf = class_ptr[c];
                max_class = c;
            }
        }

        if (max_conf > 0.5f) {
            float x1 = box_ptr[0];
            float y1 = box_ptr[1];
            float x2 = box_ptr[2];
            float y2 = box_ptr[3];

            // Scale to original image dimensions
            float scale_x = static_cast<float>(orig_w) / 640.0f;
            float scale_y = static_cast<float>(orig_h) / 640.0f;

            x1 = std::max(0.0f, std::min(x1 * scale_x, static_cast<float>(orig_w)));
            y1 = std::max(0.0f, std::min(y1 * scale_y, static_cast<float>(orig_h)));
            x2 = std::max(0.0f, std::min(x2 * scale_x, static_cast<float>(orig_w)));
            y2 = std::max(0.0f, std::min(y2 * scale_y, static_cast<float>(orig_h)));

            raw_detections.push_back({x1, y1, x2, y2, max_conf, static_cast<float>(max_class)});
        }
    }

    // Sort detections by confidence (highest first)
    std::sort(raw_detections.begin(), raw_detections.end(),
        [](const std::vector<float>& a, const std::vector<float>& b) {
            return a[4] > b[4]; // Sort by confidence (index 4)
        });

    // Apply Non-Maximum Suppression (NMS) to eliminate overlapping boxes
    std::vector<bool> keep(raw_detections.size(), true);
    const float nms_threshold = 0.5f; // IoU threshold for NMS

    for (size_t i = 0; i < raw_detections.size(); ++i) {
        if (!keep[i]) continue;

        for (size_t j = i + 1; j < raw_detections.size(); ++j) {
            if (!keep[j]) continue;

            // Calculate IoU (Intersection over Union)
            float x1_i = raw_detections[i][0];
            float y1_i = raw_detections[i][1];
            float x2_i = raw_detections[i][2];
            float y2_i = raw_detections[i][3];

            float x1_j = raw_detections[j][0];
            float y1_j = raw_detections[j][1];
            float x2_j = raw_detections[j][2];
            float y2_j = raw_detections[j][3];

            float xi1 = std::max(x1_i, x1_j);
            float yi1 = std::max(y1_i, y1_j);
            float xi2 = std::min(x2_i, x2_j);
            float yi2 = std::min(y2_i, y2_j);

            float inter_area = std::max(0.0f, xi2 - xi1) * std::max(0.0f, yi2 - yi1);
            float box1_area = (x2_i - x1_i) * (y2_i - y1_i);
            float box2_area = (x2_j - x1_j) * (y2_j - y1_j);
            float union_area = box1_area + box2_area - inter_area;
            float iou = inter_area / union_area;

            if (iou > nms_threshold) {
                keep[j] = false; // Suppress this detection
            }
        }
    }

    // Keep only the non-suppressed detections
    for (size_t i = 0; i < raw_detections.size(); ++i) {
        if (keep[i]) {
            detections.push_back(raw_detections[i]);
            detected_count++;

            // Debug output
            const auto& det = raw_detections[i];
            float cx = (det[0] + det[2]) / 2.0f;
            float cy = (det[1] + det[3]) / 2.0f;
            float w = det[2] - det[0];
            float h = det[3] - det[1];

            std::cout << "Selected detection " << detected_count << ":\n";
            std::cout << "  Class ID: " << static_cast<int>(det[5]) << "\n";
            std::cout << "  Confidence: " << det[4] << "\n";
            std::cout << "  Box: [" << det[0] << ", " << det[1] << ", "
                      << det[2] << ", " << det[3] << "]\n";
            std::cout << "  Center: (" << cx << ", " << cy << ")\n";
            std::cout << "  Size: " << w << "x" << h << "\n";
            std::cout << "  Image Size: " << orig_w << "x" << orig_h << "\n\n";
        }
    }

    if (detected_count == 0) {
        std::cout << "⚠️ No detections above the confidence threshold.\n";
    } else {
        std::cout << "✅ Final detections after NMS: " << detected_count << std::endl;
    }

    return detections;
}
