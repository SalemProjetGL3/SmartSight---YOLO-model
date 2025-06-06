#include "OrtModel.h"
#include <stdexcept>
#include <iostream>

OrtModel::OrtModel(const std::wstring& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "YOLO-NAS"),
      session_options(),
      session(env, model_path.c_str(), session_options) {

    // Just call the methods directly
    input_names = session.GetInputNames();
    output_names = session.GetOutputNames();

    if (input_names.empty() || output_names.empty()) {
        throw std::runtime_error("Input or output names could not be loaded.");
    }
}

Ort::Session& OrtModel::getSession() {
    return session;
}

const std::vector<std::string>& OrtModel::getInputNames() const {
    return input_names;
}

const std::vector<std::string>& OrtModel::getOutputNames() const {
    return output_names;
}
