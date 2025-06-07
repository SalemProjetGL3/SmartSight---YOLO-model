#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

struct Label {
    int class_id;
    float x_center;
    float y_center;
    float width;
    float height;
};

// Read label file into a vector of Label structs
std::vector<Label> read_labels(const std::string& filepath) {
    std::vector<Label> labels;
    std::ifstream file(filepath);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        Label lbl;
        if (iss >> lbl.class_id >> lbl.x_center >> lbl.y_center >> lbl.width >> lbl.height) {
            labels.push_back(lbl);
        }
    }
    return labels;
}

// Calculate if a side is clear based on object area
bool is_side_clear(const std::vector<Label>& labels, const std::string& side, float threshold = 0.05) {
    float total_area = 0.0f;
    for (const auto& lbl : labels) {
        if ((side == "left" && lbl.x_center < 0.4f) || (side == "right" && lbl.x_center > 0.6f)) {
            total_area += lbl.width * lbl.height;
        }
    }
    return total_area < threshold;
}

// Decide direction based on labels
std::string decide_direction(const std::vector<Label>& labels) {
    int left = 0, right = 0, middle = 0;
    for (const auto& lbl : labels) {
        if (lbl.x_center >= 0.4f && lbl.x_center <= 0.6f) {
            middle++;
        }
        else if (lbl.x_center < 0.4f) {
            left++;
        }
        else {
            right++;
        }
    }

    if (middle > 0) {
        bool left_clear = is_side_clear(labels, "left");
        bool right_clear = is_side_clear(labels, "right");

        if (left < right && left_clear) return "left";
        else if (right_clear) return "right";
        else if (left_clear) return "left";
        else return "stop";
    }

    return "straight";
}

int main() {
    std::string label_dir = "C:/Users/mahdi/OneDrive/Bureau/TP/PPP/Decision Tree/labels"; // Change to your label directory

    for (const auto& entry : fs::directory_iterator(label_dir)) {
        if (entry.path().extension() == ".txt") {
            std::vector<Label> labels = read_labels(entry.path().string());
            std::string decision = decide_direction(labels);
            std::cout << entry.path().filename() << ": " << decision << std::endl;
        }
    }

    return 0;
}
