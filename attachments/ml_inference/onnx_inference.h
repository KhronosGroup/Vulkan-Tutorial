#pragma once

#include <string>
#include <vector>
#include <memory>
#include <array>
#include "common/preprocessing/vulkan_preprocessing.h"

#ifdef HAS_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

class ONNXClassifier {
public:
    struct ClassificationResult {
        std::vector<std::pair<int, float>> topK;  // (class_id, confidence)
        float inferenceTimeMs;
    };

    ONNXClassifier(const std::string& modelPath);
    ~ONNXClassifier() = default;

    ClassificationResult classify(const PreprocessedImage& image, int topK = 5);

    struct GenericResult {
        std::vector<float> data;
        std::vector<int64_t> shape;
        float inferenceTimeMs;
    };
    GenericResult runGeneric(const PreprocessedImage& image);

private:
    std::vector<float> softmax(const float* logits, size_t size);
    std::vector<std::pair<int, float>> getTopK(const std::vector<float>& probabilities, int k);

#ifdef HAS_ONNX_RUNTIME
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;

    size_t numClasses = 0;
#endif
};
