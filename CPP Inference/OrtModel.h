#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

class OrtModel {
public:
    OrtModel(const std::wstring& model_path);
    Ort::Session& getSession();
    const std::vector<std::string>& getInputNames() const;
    const std::vector<std::string>& getOutputNames() const;

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};
