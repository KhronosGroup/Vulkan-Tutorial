#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <fstream>
#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include "common/renderer/renderer.h"
#include "common/preprocessing/vulkan_preprocessing.h"

// Common interface for all ML backends
class ClassifierInterface {
public:
    virtual ~ClassifierInterface() = default;
    struct Result {
        int classId;
        float confidence;
        float inferenceTimeMs;
    };
    virtual Result classify(const PreprocessedImage& img) = 0;
};

#ifdef USE_NNEF
#include "common/vulkan_nnef_inference.h"
class NNEFClassifier : public ClassifierInterface, public VulkanNNEFInference {
public:
    using VulkanNNEFInference::VulkanNNEFInference;
    Result classify(const PreprocessedImage& img) override {
        auto start = std::chrono::high_resolution_clock::now();
        auto probs = infer(img.data);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto maxIt = std::max_element(probs.begin(), probs.end());
        int classId = (int)std::distance(probs.begin(), maxIt);
        float confidence = *maxIt;
        float timeMs = std::chrono::duration<float, std::milli>(end - start).count();
        
        return { classId, confidence, timeMs };
    }
};
#endif

#ifdef USE_LITERT
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_options.h"

class LiteRTClassifier : public ClassifierInterface {
public:
    LiteRTClassifier(const std::string& modelPath) {
        if (LiteRtCreateEnvironment(0, nullptr, &env) != kLiteRtStatusOk) {
            std::cerr << "LiteRT: Failed to create environment" << std::endl;
            exit(1);
        }

        if (LiteRtCreateModelFromFile(modelPath.c_str(), &model) != kLiteRtStatusOk) {
            std::cerr << "LiteRT: Failed to load model " << modelPath << std::endl;
            exit(1);
        }

        LiteRtOptions options = nullptr;
        LiteRtCreateOptions(&options);
        LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorCpu);

        LiteRtStatus status = LiteRtCreateCompiledModel(env, model, options, &compiledModel);
        if (status != kLiteRtStatusOk) {
            std::cerr << "LiteRT: Failed to compile model: " << LiteRtGetStatusString(status) << std::endl;
            exit(1);
        }
        LiteRtDestroyOptions(options);

        // Prepare input/output buffers
        LiteRtTensorBufferRequirements inputReqs;
        if (LiteRtGetCompiledModelInputBufferRequirements(compiledModel, 0, 0, &inputReqs) != kLiteRtStatusOk) {
             std::cerr << "LiteRT: Failed to get input requirements" << std::endl;
             exit(1);
        }

        LiteRtRankedTensorType inputType;
        LiteRtGetCompiledModelInputTensorLayout(compiledModel, 0, 0, &inputType.layout);
        inputType.element_type = kLiteRtElementTypeFloat32; // Assuming float32 for MobileNetV2

        if (LiteRtCreateManagedTensorBufferFromRequirements(env, &inputType, inputReqs, &inputBuffer) != kLiteRtStatusOk) {
            std::cerr << "LiteRT: Failed to create input buffer" << std::endl;
            exit(1);
        }

        LiteRtTensorBufferRequirements outputReqs;
        if (LiteRtGetCompiledModelOutputBufferRequirements(compiledModel, 0, 0, &outputReqs) != kLiteRtStatusOk) {
             std::cerr << "LiteRT: Failed to get output requirements" << std::endl;
             exit(1);
        }

        LiteRtLayout outputLayout;
        LiteRtGetCompiledModelOutputTensorLayouts(compiledModel, 0, 1, &outputLayout, true);
        LiteRtRankedTensorType outputType;
        outputType.layout = outputLayout;
        outputType.element_type = kLiteRtElementTypeFloat32;

        if (LiteRtCreateManagedTensorBufferFromRequirements(env, &outputType, outputReqs, &outputBuffer) != kLiteRtStatusOk) {
            std::cerr << "LiteRT: Failed to create output buffer" << std::endl;
            exit(1);
        }
    }

    ~LiteRTClassifier() {
        if (inputBuffer) LiteRtDestroyTensorBuffer(inputBuffer);
        if (outputBuffer) LiteRtDestroyTensorBuffer(outputBuffer);
        if (compiledModel) LiteRtDestroyCompiledModel(compiledModel);
        if (model) LiteRtDestroyModel(model);
        if (env) LiteRtDestroyEnvironment(env);
    }

    ClassifierInterface::Result classify(const PreprocessedImage& img) override {
        auto start = std::chrono::high_resolution_clock::now();

        void* inputData;
        if (LiteRtLockTensorBuffer(inputBuffer, &inputData, kLiteRtTensorBufferLockModeWrite) != kLiteRtStatusOk) {
            return { -1, 0.0f, 0.0f };
        }
        
        // Convert NCHW to NHWC
        float* dst = static_cast<float*>(inputData);
        const float* src = img.data.data();
        const int H = 224;
        const int W = 224;
        const int C = 3;
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    dst[(h * W + w) * C + c] = src[c * (H * W) + h * W + w];
                }
            }
        }
        LiteRtUnlockTensorBuffer(inputBuffer);

        if (LiteRtRunCompiledModel(compiledModel, 0, 1, &inputBuffer, 1, &outputBuffer) != kLiteRtStatusOk) {
            std::cerr << "LiteRT: Failed to run model" << std::endl;
            return { -1, 0.0f, 0.0f };
        }

        void* outputData;
        if (LiteRtLockTensorBuffer(outputBuffer, &outputData, kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk) {
            return { -1, 0.0f, 0.0f };
        }

        size_t outputSize;
        LiteRtGetTensorBufferPackedSize(outputBuffer, &outputSize);
        int numElements = (int)(outputSize / sizeof(float));
        const float* scores = static_cast<const float*>(outputData);

        // Softmax
        std::vector<float> probs(numElements);
        float maxLogit = scores[0];
        for (int i = 1; i < numElements; ++i) if (scores[i] > maxLogit) maxLogit = scores[i];
        
        float sum = 0.0f;
        for (int i = 0; i < numElements; ++i) {
            probs[i] = std::exp(scores[i] - maxLogit);
            sum += probs[i];
        }
        
        int maxId = 0;
        float maxVal = 0.0f;
        for (int i = 0; i < numElements; ++i) {
            probs[i] /= sum;
            if (probs[i] > maxVal) {
                maxVal = probs[i];
                maxId = i;
            }
        }
        LiteRtUnlockTensorBuffer(outputBuffer);

        auto end = std::chrono::high_resolution_clock::now();
        float timeMs = std::chrono::duration<float, std::milli>(end - start).count();

        return { maxId, maxVal, timeMs };
    }

private:
    LiteRtEnvironment env = nullptr;
    LiteRtModel model = nullptr;
    LiteRtCompiledModel compiledModel = nullptr;
    LiteRtTensorBuffer inputBuffer = nullptr;
    LiteRtTensorBuffer outputBuffer = nullptr;
};
#endif

#ifdef USE_IREE
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/api.h"
#include "iree/hal/drivers/vulkan/registration/driver_module.h"
#include <algorithm>
#include <cmath>

#define CHECK_OK(status, msg) \
  if (!iree_status_is_ok(status)) { \
    std::cerr << msg << " failed" << std::endl; \
    iree_status_fprint(stderr, status); \
    iree_status_free(status); \
    exit(1); \
  }

class IREEClassifier : public ClassifierInterface {
public:
    IREEClassifier(const std::string& modelPath, vk::raii::Device& device) {
        iree_runtime_instance_options_t instance_options;
        iree_runtime_instance_options_initialize(&instance_options);
        iree_runtime_instance_options_use_all_available_drivers(&instance_options);
        
        CHECK_OK(iree_runtime_instance_create(&instance_options, iree_allocator_system(), &instance),
                 "iree_runtime_instance_create");

        // We wrap the existing Vulkan device into IREE
        // Note: In a real app we might want IREE to create its own device or share better.
        // For this sample, we'll let IREE create its own Vulkan device to keep it simple and decoupled
        // as per iree_vulkan_inference.cpp pattern.
        CHECK_OK(iree_hal_create_device(iree_runtime_instance_driver_registry(instance),
                                          iree_make_cstring_view("vulkan"),
                                          iree_runtime_instance_host_allocator(instance),
                                          &iree_device),
                   "iree_hal_create_device");

        iree_runtime_session_options_t session_options;
        iree_runtime_session_options_initialize(&session_options);
        
        CHECK_OK(iree_runtime_session_create_with_device(instance, &session_options, iree_device, 
                                                           iree_runtime_instance_host_allocator(instance), 
                                                           &session),
                   "iree_runtime_session_create_with_device");

        CHECK_OK(iree_runtime_session_append_bytecode_module_from_file(session, modelPath.c_str()),
                   "iree_runtime_session_append_bytecode_module_from_file");

        // Find entry point
        const char* candidates[] = { "module.main_graph", "module.main", "main_graph", "main" };
        for (const char* candidate : candidates) {
            iree_runtime_call_t call;
            iree_status_t status = iree_runtime_call_initialize_by_name(session, iree_make_cstring_view(candidate), &call);
            if (iree_status_is_ok(status)) {
                function_name = candidate;
                iree_runtime_call_deinitialize(&call);
                break;
            }
            iree_status_free(status);
        }
        
        if (function_name.empty()) {
            std::cerr << "IREE: Could not find entry point in " << modelPath << std::endl;
            exit(1);
        }
    }

    ~IREEClassifier() {
        if (session) iree_runtime_session_release(session);
        if (iree_device) iree_hal_device_release(iree_device);
        if (instance) iree_runtime_instance_release(instance);
    }

    ClassifierInterface::Result classify(const PreprocessedImage& img) override {
        auto start = std::chrono::high_resolution_clock::now();
        
        iree_runtime_call_t call;
        CHECK_OK(iree_runtime_call_initialize_by_name(session, iree_make_cstring_view(function_name.c_str()), &call),
                 "iree_runtime_call_initialize_by_name");

        iree_hal_buffer_view_t* input_view = nullptr;
        std::vector<iree_hal_dim_t> iree_shape;
        for (auto d : img.shape) iree_shape.push_back((iree_hal_dim_t)d);
        
        iree_hal_buffer_params_t buffer_params;
        memset(&buffer_params, 0, sizeof(buffer_params));
        buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
        buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
        iree_hal_buffer_params_canonicalize(&buffer_params);

        CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(iree_device,
                                                                 iree_hal_device_allocator(iree_device),
                                                                 iree_shape.size(), iree_shape.data(),
                                                                 IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                                                 IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                                                 buffer_params,
                                                                 iree_make_const_byte_span(img.data.data(), img.data.size() * sizeof(float)),
                                                                 &input_view),
                        "iree_hal_buffer_view_allocate_buffer_copy");

        CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, input_view),
                        "iree_runtime_call_inputs_push_back_buffer_view");
        iree_hal_buffer_view_release(input_view);

        CHECK_OK(iree_runtime_call_invoke(&call, 0), "iree_runtime_call_invoke");

        iree_hal_buffer_view_t* output_view = nullptr;
        CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &output_view),
                        "iree_runtime_call_outputs_pop_front_buffer_view");

        iree_host_size_t num_elements = 1;
        for (size_t i = 0; i < iree_hal_buffer_view_shape_rank(output_view); ++i) {
            num_elements *= iree_hal_buffer_view_shape_dim(output_view, i);
        }
        
        std::vector<float> logits(num_elements);
        CHECK_OK(iree_hal_device_transfer_d2h(iree_device,
                                                     iree_hal_buffer_view_buffer(output_view), 0,
                                                     logits.data(),
                                                     logits.size() * sizeof(float),
                                                     IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                                     iree_infinite_timeout()),
                        "iree_hal_device_transfer_d2h");

        iree_hal_buffer_view_release(output_view);
        iree_runtime_call_deinitialize(&call);

        // Softmax & Find Max
        float maxLogit = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        int maxIdx = 0;
        float maxProb = 0.0f;
        
        std::vector<float> probs(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            probs[i] = std::exp(logits[i] - maxLogit);
            sum += probs[i];
        }
        for (size_t i = 0; i < probs.size(); ++i) {
            probs[i] /= sum;
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIdx = i;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::milli>(end - start).count();
        
        return {maxIdx, maxProb, time};
    }

private:
    iree_runtime_instance_t* instance = nullptr;
    iree_hal_device_t* iree_device = nullptr;
    iree_runtime_session_t* session = nullptr;
    std::string function_name;
};
#endif

#ifdef USE_TFLITE
class TFLiteClassifier : public ClassifierInterface {
public:
    TFLiteClassifier(const std::string& modelPath) {
        model = TfLiteModelCreateFromFile(modelPath.c_str());
        if (!model) {
            std::cerr << "TFLite: Failed to load model " << modelPath << std::endl;
            exit(1);
        }

        options = TfLiteInterpreterOptionsCreate();
        TfLiteInterpreterOptionsSetNumThreads(options, 4);

        interpreter = TfLiteInterpreterCreate(model, options);
        if (!interpreter) {
            std::cerr << "TFLite: Failed to create interpreter" << std::endl;
            exit(1);
        }

        if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
            std::cerr << "TFLite: Failed to allocate tensors" << std::endl;
            exit(1);
        }

        input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
        output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    }

    ~TFLiteClassifier() {
        if (interpreter) TfLiteInterpreterDelete(interpreter);
        if (options) TfLiteInterpreterOptionsDelete(options);
        if (model) TfLiteModelDelete(model);
    }

    ClassifierInterface::Result classify(const PreprocessedImage& img) override {
        auto start = std::chrono::high_resolution_clock::now();

        float* input_data = static_cast<float*>(TfLiteTensorData(input_tensor));
        std::memcpy(input_data, img.data.data(), img.data.size() * sizeof(float));

        // Invoke
        if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
            std::cerr << "TFLite: Failed to invoke interpreter" << std::endl;
            return { -1, 0.0f, 0.0f };
        }

        // Get output
        const float* output_data = static_cast<const float*>(TfLiteTensorData(output_tensor));
        int output_size = 1;
        for (int i = 0; i < TfLiteTensorNumDims(output_tensor); ++i) {
            output_size *= TfLiteTensorDim(output_tensor, i);
        }

        auto end = std::chrono::high_resolution_clock::now();
        float timeMs = std::chrono::duration<float, std::milli>(end - start).count();

        int maxId = 0;
        float maxVal = output_data[0];
        for (int i = 1; i < output_size; ++i) {
            if (output_data[i] > maxVal) {
                maxVal = output_data[i];
                maxId = i;
            }
        }

        // MobileNetV2 TFLite often outputs logits or probabilities depending on version
        // We assume probabilities or we'd need softmax
        return { maxId, maxVal, timeMs };
    }

private:
    TfLiteModel* model = nullptr;
    TfLiteInterpreterOptions* options = nullptr;
    TfLiteInterpreter* interpreter = nullptr;
    TfLiteTensor* input_tensor = nullptr;
    const TfLiteTensor* output_tensor = nullptr;
};
#endif

#ifdef HAVE_OPENCV

int main(int argc, char** argv) {
    std::string modelPath = "models/mobilenetv2_nnef_optimized";
    std::string labelPath = "data/imagenet_classes.txt";
    std::string testImagePath = "";
    int cameraIndex = 0;
    int maxFrames = -1;
    bool forceBlank = false;
    bool saveDebug = false;
    float threshold = 0.7f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) modelPath = argv[++i];
        else if (arg == "--labels" && i + 1 < argc) labelPath = argv[++i];
        else if (arg == "--cam" && i + 1 < argc) cameraIndex = std::stoi(argv[++i]);
        else if (arg == "--frames" && i + 1 < argc) maxFrames = std::stoi(argv[++i]);
        else if (arg == "--image" && i + 1 < argc) testImagePath = argv[++i];
        else if (arg == "--blank") forceBlank = true;
        else if (arg == "--debug-save") saveDebug = true;
        else if (arg == "--threshold" && i + 1 < argc) threshold = std::stof(argv[++i]);
    }

    std::cout << "--- Embedded Wildlife Cam App (rPi) ---" << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    
    // 1. Initialize Headless Renderer
    std::cout << "Initializing Vulkan..." << std::endl;
    auto renderer = std::make_unique<Renderer>(nullptr, 640, 480);
    
    // 2. Initialize Preprocessor
    auto preprocessor = std::make_unique<VulkanPreprocessor>(
        renderer->GetRaiiDevice(),
        renderer->GetGraphicsQueue(),
        renderer->GetCommandPool(),
        renderer->GetPhysicalDevice()
    );

    // 3. Initialize ML Backend
    std::unique_ptr<ClassifierInterface> classifier;
    
    auto endsWith = [](const std::string& str, const std::string& suffix) {
        return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    if (endsWith(modelPath, ".vmfb")) {
#ifdef USE_IREE
        std::cout << "Using IREE Backend for " << modelPath << std::endl;
        classifier = std::make_unique<IREEClassifier>(modelPath, renderer->GetRaiiDevice());
#else
        std::cerr << "Error: IREE backend not enabled at compile time. Cannot load " << modelPath << std::endl;
        return 1;
#endif
    } else if (endsWith(modelPath, ".tflite")) {
#ifdef USE_LITERT
        std::cout << "Using LiteRT Backend for " << modelPath << std::endl;
        classifier = std::make_unique<LiteRTClassifier>(modelPath);
#elif defined(USE_TFLITE)
        std::cout << "Using TFLite Backend for " << modelPath << std::endl;
        classifier = std::make_unique<TFLiteClassifier>(modelPath);
#else
        std::cerr << "Error: TFLite/LiteRT backend not enabled at compile time. Cannot load " << modelPath << std::endl;
        return 1;
#endif
    } else {
#ifdef USE_NNEF
        std::cout << "Using NNEF (Custom Vulkan) Backend for " << modelPath << std::endl;
        auto nnef = std::make_unique<NNEFClassifier>(*renderer);
        if (!nnef->loadModel(modelPath)) {
            std::cerr << "Error: Could not load NNEF model from " << modelPath << std::endl;
            return 1;
        }
        classifier = std::move(nnef);
#else
        std::cerr << "Error: NNEF backend not enabled at compile time. Cannot load " << modelPath << std::endl;
        return 1;
#endif
    }

    // 4. Load Labels
    std::vector<std::string> labels;
    std::ifstream labelFile(labelPath);
    std::string line;
    while (std::getline(labelFile, line)) labels.push_back(line);
    if (labels.empty()) {
        std::cerr << "Error: No labels loaded from " << labelPath << std::endl;
        return 1;
    }
    if (labels.size() < 1000) {
        std::cout << "Warning: Label file has only " << labels.size() << " entries (expected 1000 for MobileNetV2)." << std::endl;
    }

    // 5. Initialize Camera
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera " << cameraIndex << std::endl;
        std::cout << "Falling back to simulation mode..." << std::endl;
    }

    std::cout << "Starting wildlife monitoring loop..." << std::endl;
    cv::Mat frame;
    int frameCount = 0;
    auto lastCheck = std::chrono::high_resolution_clock::now();

    while (true) {
        if (forceBlank) {
            frame = cv::Mat::zeros(480, 640, CV_8UC3);
        } else if (!testImagePath.empty()) {
            frame = cv::imread(testImagePath);
            if (frame.empty()) {
                std::cerr << "Error: Could not load test image: " << testImagePath << std::endl;
                break;
            }
        } else if (cap.isOpened()) {
            cap >> frame;
        } else {
            // Simulation: Read a sample image if it exists, or just a black frame
            frame = cv::Mat::zeros(480, 640, CV_8UC3);
            cv::putText(frame, "SIMULATION", cv::Point(200, 240), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 3);
        }

        if (frame.empty()) break;

        frameCount++;
        if (maxFrames > 0 && frameCount > maxFrames) break;

        // Vulkan Preprocessing
        auto preprocessed = preprocessor->preprocess(frame.data, frame.cols, frame.rows, frame.step, true);

        // Optional: save a visualization of what the model sees (after normalization)
        auto savePreprocIfNeeded = [&](const std::string& path){
            // Convert NCHW normalized float back to HWC uint8 [0,255] for visualization
            const int W = 224, H = 224;
            cv::Mat vis(H, W, CV_8UC3);
            auto idx = [&](int c, int y, int x){ return c*H*W + y*W + x; };
            
            // Standard ImageNet normalization: (val - mean) / std
            float mean[3] = {0.485f, 0.456f, 0.406f};
            float std_dev[3] = {0.229f, 0.224f, 0.225f};

            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    float r = preprocessed.data[idx(0,y,x)] * std_dev[0] + mean[0];
                    float g = preprocessed.data[idx(1,y,x)] * std_dev[1] + mean[1];
                    float b = preprocessed.data[idx(2,y,x)] * std_dev[2] + mean[2];
                    // to [0,255]
                    int R = std::clamp(int(r * 255.0f), 0, 255);
                    int G = std::clamp(int(g * 255.0f), 0, 255);
                    int B = std::clamp(int(b * 255.0f), 0, 255);
                    vis.at<cv::Vec3b>(y,x) = cv::Vec3b(B,G,R); // BGR for OpenCV
                }
            }
            cv::imwrite(path, vis);
        };
        
        // Inference
        auto result = classifier->classify(preprocessed);
        int classId = result.classId;
        float confidence = result.confidence;
        float infTime = result.inferenceTimeMs;

        std::vector<std::pair<float, int>> topK;
        topK.push_back({confidence, classId});

        // Logging & Overlay
        std::string label = (classId < (int)labels.size()) ? labels[classId] : "Unknown";
        
        if (confidence > threshold) {
            std::cout << "[DETECTED] " << label << " (" << (confidence * 100.0f) << "%) in " << infTime << "ms" << std::endl;
            
            // Print top-5
            std::cout << "Top candidates:" << std::endl;
            for (int k = 0; k < std::min(5, (int)topK.size()); ++k) {
                std::string kLabel = (topK[k].second < (int)labels.size()) ? labels[topK[k].second] : "Unknown";
                std::cout << "  - " << kLabel << " (" << (topK[k].first * 100.0f) << "%)" << std::endl;
            }

            // Add overlay to frame
            cv::putText(frame, label + " " + std::to_string((int)(confidence * 100)) + "%", 
                        cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

            if (saveDebug) {
                std::string rawName = "debug_detection_" + std::to_string(frameCount) + ".jpg";
                cv::imwrite(rawName, frame);
                char absPath[4096];
                if (realpath(rawName.c_str(), absPath)) {
                    std::cout << "Saved debug image to: " << absPath << std::endl;
                } else {
                    std::cout << "Saved debug image to " << rawName << std::endl;
                }
                
                std::string preName = "debug_preproc_" + std::to_string(frameCount) + ".png";
                savePreprocIfNeeded(preName);
                if (realpath(preName.c_str(), absPath)) {
                    std::cout << "Saved preprocessed visualization to: " << absPath << std::endl;
                } else {
                    std::cout << "Saved preprocessed visualization to " << preName << std::endl;
                }
            }
        } else if (saveDebug && maxFrames > 0) {
            // If running bounded frames with debug enabled, still drop preprocessed frame for inspection
            std::string preName = "debug_preproc_" + std::to_string(frameCount) + ".png";
            savePreprocIfNeeded(preName);
            char absPath[4096];
            if (realpath(preName.c_str(), absPath)) {
                std::cout << "Saved debug preprocessed view to: " << absPath << std::endl;
            }
        }

        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastCheck).count();
        if (elapsed >= 5) {
            float fps = (float)frameCount / elapsed;
            std::cout << "\n[STATS] Steady-state FPS: " << fps << std::endl;
            frameCount = 0;
            lastCheck = now;
        }

        // Optional: Break if user presses 'q' (if running in a shell that supports it)
        // Since we are headless, we'll just run indefinitely for this sample
        // or add a limit. Let's add a 100 frame limit if in simulation mode for CI.
        if (!cap.isOpened() && frameCount > 100) {
            std::cout << "\nSimulation finished." << std::endl;
            break;
        }
    }

    return 0;
}

#else // HAVE_OPENCV

int main(int, char**) {
    std::cerr << "embedded_wildlife_cam: built without OpenCV. Camera input is not available.\n";
    std::cerr << "Install OpenCV and rebuild with -DHAVE_OPENCV to enable camera support.\n";
    return 1;
}

#endif // HAVE_OPENCV
