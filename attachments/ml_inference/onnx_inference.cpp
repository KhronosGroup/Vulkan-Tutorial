#include "onnx_inference.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cmath>

#ifdef HAS_ONNX_RUNTIME

ONNXClassifier::ONNXClassifier(const std::string& modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "ONNXClassifier") {

    // Configure session options
    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Try to use a GPU execution provider if available
    bool providerFound = false;
    
    // 1. WebGPU - Modern cross-platform GPU API (Leave in for future support)
    try {
        sessionOptions.AppendExecutionProvider("WebGPU");
        std::cout << "Using WebGPU execution provider\n";
        providerFound = true;
    } catch (...) {
        // Not available in this build
    }

    if (!providerFound) {
#ifdef _WIN32
        // 2. DirectML - Standard high-performance GPU API for Windows
        try {
            sessionOptions.AppendExecutionProvider("DML");
            std::cout << "Using DirectML execution provider\n";
            providerFound = true;
        } catch (...) {
            // DirectML not available
        }
#else
        // 2. CUDA - Standard high-performance GPU API for Linux
        try {
            sessionOptions.AppendExecutionProvider("CUDA");
            std::cout << "Using CUDA execution provider\n";
            providerFound = true;
        } catch (...) {
            // CUDA not available
        }
#endif
    }

    if (!providerFound) {
        std::cout << "No high-performance GPU execution provider available (WebGPU, DirectML, or CUDA).\n";
        std::cout << "Using default CPU provider.\n";
    }

    // Create session
    #ifdef _WIN32
    std::wstring wModelPath(modelPath.begin(), modelPath.end());
    session = std::make_unique<Ort::Session>(env, wModelPath.c_str(), sessionOptions);
    #else
    session = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);
    #endif

    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;

    auto inputName = session->GetInputNameAllocated(0, allocator);
    inputNames.push_back(inputName.get());
    inputName.release();

    auto outputName = session->GetOutputNameAllocated(0, allocator);
    outputNames.push_back(outputName.get());
    outputName.release();

    // Get output shape to determine number of classes
    auto outputTypeInfo = session->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    auto outputShape = outputTensorInfo.GetShape();
    numClasses = outputShape.back();

    std::cout << "ONNX model loaded successfully\n";
    std::cout << "  Input: " << inputNames[0] << "\n";
    std::cout << "  Output: " << outputNames[0] << " (classes: " << numClasses << ")\n";
}

ONNXClassifier::ClassificationResult ONNXClassifier::classify(
    const PreprocessedImage& image, int topK) {

    auto startTime = std::chrono::high_resolution_clock::now();

    // Create input tensor
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        const_cast<float*>(image.data.data()),
        image.data.size(),
        image.shape.data(),
        image.shape.size());

    // Run inference
    auto outputTensors = session->Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        &inputTensor,
        1,
        outputNames.data(),
        1);

    auto endTime = std::chrono::high_resolution_clock::now();
    float inferenceTime = std::chrono::duration<float, std::milli>(
        endTime - startTime).count();

    // Get output
    float* outputData = outputTensors[0].GetTensorMutableData<float>();

    // Apply softmax and get top-K
    std::vector<float> probabilities = softmax(outputData, numClasses);
    auto topKResults = getTopK(probabilities, topK);

    return {topKResults, inferenceTime};
}

ONNXClassifier::GenericResult ONNXClassifier::runGeneric(const PreprocessedImage& image) {
    auto startTime = std::chrono::high_resolution_clock::now();

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float*>(image.data.data()), image.data.size(),
        image.shape.data(), image.shape.size());

    auto outputTensors = session->Run(
        Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), 1);

    auto endTime = std::chrono::high_resolution_clock::now();
    float inferenceTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();

    float* outputData = outputTensors[0].GetTensorMutableData<float>();
    auto outputInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
    auto outputShape = outputInfo.GetShape();
    
    size_t totalSize = 1;
    for (auto dim : outputShape) totalSize *= dim;

    std::vector<float> data(outputData, outputData + totalSize);
    return {data, outputShape, inferenceTime};
}

std::vector<float> ONNXClassifier::softmax(const float* logits, size_t size) {
    std::vector<float> result(size);

    // Find max for numerical stability
    float maxLogit = *std::max_element(logits, logits + size);

    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        result[i] = std::exp(logits[i] - maxLogit);
        sum += result[i];
    }

    // Normalize
    for (size_t i = 0; i < size; ++i) {
        result[i] /= sum;
    }

    return result;
}

std::vector<std::pair<int, float>> ONNXClassifier::getTopK(
    const std::vector<float>& probabilities, int k) {

    std::vector<std::pair<int, float>> indexed;
    indexed.reserve(probabilities.size());

    for (size_t i = 0; i < probabilities.size(); ++i) {
        indexed.emplace_back(static_cast<int>(i), probabilities[i]);
    }

    std::partial_sort(indexed.begin(),
                     indexed.begin() + k,
                     indexed.end(),
                     [](const auto& a, const auto& b) {
                         return a.second > b.second;
                     });

    indexed.resize(k);
    return indexed;
}

#else // HAS_ONNX_RUNTIME

ONNXClassifier::ONNXClassifier(const std::string& modelPath) {
    std::cerr << "ONNX Runtime not available. Model not loaded: " << modelPath << "\n";
}

ONNXClassifier::ClassificationResult ONNXClassifier::classify(const PreprocessedImage&, int topK) {
    return {{}, 0.0f};
}

ONNXClassifier::GenericResult ONNXClassifier::runGeneric(const PreprocessedImage&) {
    return {{}, {}, 0.0f};
}

std::vector<float> ONNXClassifier::softmax(const float*, size_t size) {
    return std::vector<float>(size, 0.0f);
}

std::vector<std::pair<int, float>> ONNXClassifier::getTopK(const std::vector<float>& probabilities, int k) {
    return {};
}

#endif // HAS_ONNX_RUNTIME
