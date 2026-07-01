#include <opencv2/opencv.hpp>
#include "onnx_inference.h"
#include "renderer.h"
#include "imgui_system.h"
#include "vulkan_preprocessing.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>
#include <GLFW/glfw3.h>

// This example demonstrates real-time camera integration with ML inference.

class RealTimeClassifier {
public:
    RealTimeClassifier(const std::string& modelPath, const std::string& labelPath)
        : classifier(modelPath) {
        loadLabels(labelPath);

        // 1. Initialize GLFW & Window
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(1280, 720, "Vulkan Real-Time ML Classification", nullptr, nullptr);
        if (!window) {
            throw std::runtime_error("Failed to create window");
        }

        // 2. Initialize Vulkan Renderer and ImGui
        renderer = std::make_unique<Renderer>(window, 1280, 720);
        imguiSystem = std::make_unique<ImGuiSystem>(renderer.get(), window, 1280, 720);

        // 3. Initialize Vulkan Preprocessor
        preprocessor = std::make_unique<VulkanPreprocessor>(
            renderer->GetRaiiDevice(),
            renderer->GetGraphicsQueue(),
            renderer->GetCommandPool(),
            renderer->GetPhysicalDevice()
        );

        // 4. Setup display texture
        createDisplayResources();
        
        std::cout << "Vulkan Real-Time Classifier ready." << std::endl;
    }

    ~RealTimeClassifier() {
        if (renderer) renderer->WaitIdle();
        
        // Destroy all Vulkan resources before the window is destroyed.
        // These members must be cleared while renderer (and its device/surface) is still alive.
        displaySampler = nullptr;
        displayView = nullptr;
        displayMemory = nullptr;
        displayImage = nullptr;
        
        preprocessor.reset();
        imguiSystem.reset();
        renderer.reset();

        if (window) {
            glfwDestroyWindow(window);
            glfwTerminate();
        }
    }

    void run() {
        cv::VideoCapture cap(0);
        bool cameraAvailable = cap.isOpened();
        if (!cameraAvailable) {
            std::cout << "Camera not found. Using simulation mode." << std::endl;
        }

        cv::Mat frame;
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            if (cameraAvailable) {
                cap >> frame;
            } else {
                // Dummy frame for simulation
                frame = cv::Mat::zeros(480, 640, CV_8UC3);
                cv::putText(frame, "No camera found. Simulation mode.", cv::Point(50, 240), 
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            }

            if (frame.empty()) break;

            // 1. Update display and process ML in one go
            if (renderer->BeginFrame()) {
                processFrame(frame);
                
                imguiSystem->NewFrame();
                drawUI();
                
                imguiSystem->Render(renderer->GetCurrentCommandBuffer(), renderer->GetCurrentFrame());
                renderer->EndFrame();
            }
        }
    }

private:
    void createDisplayResources() {
        // Max resolution for display texture (1080p)
        uint32_t texW = 1920;
        uint32_t texH = 1080;

        vk::ImageCreateInfo imageInfo{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .extent = {texW, texH, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst
        };

        displayImage = vk::raii::Image(renderer->GetRaiiDevice(), imageInfo);
        
        auto memReqs = displayImage.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memReqs.size,
            .memoryTypeIndex = renderer->FindMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
        };
        displayMemory = vk::raii::DeviceMemory(renderer->GetRaiiDevice(), allocInfo);
        displayImage.bindMemory(*displayMemory, 0);

        vk::ImageViewCreateInfo viewInfo{
            .image = *displayImage,
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
        };
        displayView = vk::raii::ImageView(renderer->GetRaiiDevice(), viewInfo);

        vk::SamplerCreateInfo samplerInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge
        };
        displaySampler = vk::raii::Sampler(renderer->GetRaiiDevice(), samplerInfo);

        displayTexID = imguiSystem->AddTexture(displayView, displaySampler);
    }

    void updateDisplayTexture(const cv::Mat& frame) {
        // Convert BGR to RGBA for display
        cv::Mat rgba;
        cv::cvtColor(frame, rgba, cv::COLOR_BGR2RGBA);

        size_t size = rgba.total() * rgba.elemSize();
        
        // Use a temporary staging buffer for upload
        vk::BufferCreateInfo stagingInfo{
            .size = size, 
            .usage = vk::BufferUsageFlagBits::eTransferSrc
        };
        vk::raii::Buffer stagingBuffer(renderer->GetRaiiDevice(), stagingInfo);
        
        auto memReqs = stagingBuffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memReqs.size,
            .memoryTypeIndex = renderer->FindMemoryType(memReqs.memoryTypeBits, 
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
        };
        vk::raii::DeviceMemory stagingMem(renderer->GetRaiiDevice(), allocInfo);
        stagingBuffer.bindMemory(*stagingMem, 0);

        void* data = stagingMem.mapMemory(0, size);
        std::memcpy(data, rgba.data, size);
        stagingMem.unmapMemory();

        renderer->TransitionImageLayout(*displayImage, vk::Format::eR8G8B8A8Unorm, 
            vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        renderer->CopyBufferToImage(*stagingBuffer, *displayImage, rgba.cols, rgba.rows);
        renderer->TransitionImageLayout(*displayImage, vk::Format::eR8G8B8A8Unorm, 
            vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    void processFrame(cv::Mat& frame) {
        // 1. Update display texture (one OpenCV call here for RGBA conversion)
        updateDisplayTexture(frame);

        // 2. Preprocess with Vulkan Compute
        // Passing raw BGR data directly - no redundant OpenCV conversion!
        auto img = preprocessor->preprocess(frame.data, frame.cols, frame.rows, true);

        // 3. Inference - Now requesting Top-5 results
        lastResult = classifier.classify(img, 5);
    }

    void drawUI() {
        // Fullscreen camera background
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("CameraBackground", nullptr, 
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | 
            ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoInputs);

        if (displayTexID != 0) {
            ImGui::GetWindowDrawList()->AddImage(displayTexID, ImVec2(0, 0), ImGui::GetIO().DisplaySize);
        }
        ImGui::End();

        // Control / Results Overlay - Positioned at top-right with semi-transparency
        float overlayWidth = 400.0f;
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - overlayWidth - 20, 20), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(overlayWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.7f); // Semi-transparent
        
        ImGui::Begin("Classification Results", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
        
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "MobileNetV2 Predictions");
        ImGui::Separator();

        if (!lastResult.topK.empty()) {
            for (size_t i = 0; i < lastResult.topK.size(); ++i) {
                int classId = lastResult.topK[i].first;
                float confidence = lastResult.topK[i].second;
                std::string label = (static_cast<size_t>(classId) < labels.size()) ? labels[classId] : "Unknown";

                ImGui::PushID(static_cast<int>(i));
                
                // Show rank and label (wrapped to prevent cropping)
                ImGui::Text("%zu. %s", i + 1, label.c_str());
                
                // Confidence bar
                char buf[32];
                snprintf(buf, sizeof(buf), "%.1f%%", confidence * 100.0f);
                ImGui::ProgressBar(confidence, ImVec2(-1, 0), buf);
                
                ImGui::PopID();
            }
            
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Performance:");
            ImGui::BulletText("Inference: %.1f ms", lastResult.inferenceTimeMs);
            ImGui::BulletText("Preprocessing: GPU-Accelerated");
        } else {
            ImGui::Text("Initializing camera...");
        }
        
        ImGui::Spacing();
        ImGui::Separator();
        if (ImGui::Button("Close Application", ImVec2(-1, 0))) {
            glfwSetWindowShouldClose(window, true);
        }
        
        ImGui::End();
    }

    void loadLabels(const std::string& path) {
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            labels.push_back(line);
        }
    }

    GLFWwindow* window = nullptr;
    ONNXClassifier classifier;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<ImGuiSystem> imguiSystem;
    std::unique_ptr<VulkanPreprocessor> preprocessor;
    std::vector<std::string> labels;
    
    // Display resources
    vk::raii::Image displayImage = nullptr;
    vk::raii::DeviceMemory displayMemory = nullptr;
    vk::raii::ImageView displayView = nullptr;
    vk::raii::Sampler displaySampler = nullptr;
    ImTextureID displayTexID = 0;

    ONNXClassifier::ClassificationResult lastResult;
};

int main(int argc, char** argv) {
    std::string modelPath = "models/mobilenetv2.onnx";
    std::string labelPath = "data/imagenet_classes.txt";

    if (argc > 1) modelPath = argv[1];
    if (argc > 2) labelPath = argv[2];

    try {
        RealTimeClassifier app(modelPath, labelPath);
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
