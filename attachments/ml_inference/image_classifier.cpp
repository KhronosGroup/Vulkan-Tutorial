#include "renderer.h"
#include "imgui_system.h"
#include "vulkan_preprocessing.h"
#include "onnx_inference.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "imgui.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>

#include <GLFW/glfw3.h>

class ImageClassifierApp {
public:
    ~ImageClassifierApp() {
        if (renderer) {
            renderer->WaitIdle();
        }
        // Destroy in reverse order of creation
        classifier.reset();
        vulkanPreprocessor.reset();
        imguiSystem.reset();
        renderer.reset();

        if (window) {
            glfwDestroyWindow(window);
            glfwTerminate();
        }
    }

    void run() {
        initialize();
        mainLoop();
    }

private:
    void initialize() {
        // Initialize GLFW window
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(1280, 720, "Image Classifier - MobileNetV2", nullptr, nullptr);

        // Initialize Vulkan renderer
        renderer = std::make_unique<Renderer>(window, 1280, 720);
        imguiSystem = std::make_unique<ImGuiSystem>(renderer.get(), window, 1280, 720);

        std::cout << "Initializing ML components...\n";

        // Initialize Vulkan preprocessor
        vulkanPreprocessor = std::make_unique<VulkanPreprocessor>(
            renderer->GetRaiiDevice(),
            renderer->GetGraphicsQueue(),
            renderer->GetCommandPool(),
            renderer->GetPhysicalDevice());

        // Initialize ONNX classifier
        classifier = std::make_unique<ONNXClassifier>("models/mobilenetv2.onnx");

        // Load ImageNet class names
        loadClassNames();

        std::cout << "Initialization complete!\n";
        std::cout << "Ready to classify images.\n";
    }

    void loadClassNames() {
        std::ifstream file("data/imagenet_classes.txt");
        if (!file.is_open()) {
            std::cerr << "Warning: Could not load class names from data/imagenet_classes.txt\n";
            std::cerr << "Using generic class labels.\n";
            for (int i = 0; i < 1000; ++i) {
                classNames.push_back("Class " + std::to_string(i));
            }
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            classNames.push_back(line);
        }

        std::cout << "Loaded " << classNames.size() << " class names\n";
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            // Begin frame
            if (!renderer->BeginFrame()) {
                continue;
            }
            imguiSystem->NewFrame();

            // Draw UI
            drawUI();

            // End frame
            imguiSystem->Render(renderer->GetCurrentCommandBuffer(), renderer->GetCurrentFrame());
            renderer->EndFrame();
        }
    }

    void drawUI() {
        auto& io = ImGui::GetIO();

        ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Always);
        ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_Always);

        ImGui::Begin("Image Classifier", nullptr,
                    ImGuiWindowFlags_NoTitleBar |
                    ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoCollapse);

        // Title
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "MobileNetV2 Image Classification");
        ImGui::Separator();

        // Layout: Left panel for controls, right panel for results
        float leftWidth = 400.0f;
        float contentHeight = ImGui::GetContentRegionAvail().y;

        ImGui::BeginChild("LeftPanel", ImVec2(leftWidth, contentHeight), true);
        {
            ImGui::TextWrapped("Load an image to classify it using MobileNetV2 (ImageNet 1000 classes).");
            ImGui::Spacing();

            // Simple inline text input for image path
            static char pathBuffer[512] = "";
            ImGui::Text("Image Path:");
            ImGui::InputText("##imagepath", pathBuffer, sizeof(pathBuffer));

            if (ImGui::Button("Load Image", ImVec2(-1, 0))) {
                if (strlen(pathBuffer) > 0) {
                    std::cout << "Loading image: " << pathBuffer << "\n";
                    currentImagePath = pathBuffer;
                    runInference();
                } else {
                    std::cout << "Please enter an image path\n";
                }
            }

            if (ImGui::Button("Load Sample Image", ImVec2(-1, 0))) {
                const char* samplePath = "../sample_images/golden_retriever.jpg";
                strncpy(pathBuffer, samplePath, sizeof(pathBuffer) - 1);
                pathBuffer[sizeof(pathBuffer) - 1] = '\0';
                std::cout << "Loading sample image: " << samplePath << "\n";
                currentImagePath = samplePath;
                runInference();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (!currentImagePath.empty()) {
                ImGui::Text("Current Image:");
                ImGui::TextWrapped("%s", currentImagePath.c_str());

                if (imageWidth > 0 && imageHeight > 0) {
                    ImGui::Text("Size: %d x %d", imageWidth, imageHeight);
                }

                ImGui::Spacing();

                if (ImGui::Button("Classify Again", ImVec2(-1, 0))) {
                    runInference();
                }
            } else {
                ImGui::TextWrapped("No image loaded. Click 'Load Image...' to begin.");
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Performance metrics
            if (lastPreprocessTimeMs > 0) {
                ImGui::Text("Performance:");
                ImGui::Spacing();
                ImGui::Text("  Vulkan Preprocessing:  %.2f ms", lastPreprocessTimeMs);
                ImGui::Text("  ONNX Inference:        %.2f ms", lastResult.inferenceTimeMs);
                ImGui::Text("  Total:                 %.2f ms", lastPreprocessTimeMs + lastResult.inferenceTimeMs);
                ImGui::Spacing();
                float fps = 1000.0f / (lastPreprocessTimeMs + lastResult.inferenceTimeMs);
                ImGui::Text("  Throughput: ~%.0f FPS", fps);
            }
        }
        ImGui::EndChild();

        ImGui::SameLine();

        // Right panel: Results
        ImGui::BeginChild("RightPanel", ImVec2(0, contentHeight), true);
        {
            if (!lastResult.topK.empty()) {
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Top-5 Predictions:");
                ImGui::Spacing();

                for (size_t i = 0; i < lastResult.topK.size(); ++i) {
                    int classId = lastResult.topK[i].first;
                    float confidence = lastResult.topK[i].second;

                    // Ensure class ID is in valid range
                    if (classId < 0 || classId >= static_cast<int>(classNames.size())) {
                        continue;
                    }

                    ImGui::PushID(static_cast<int>(i));

                    // Rank and class name
                    ImGui::Text("%zu.", i + 1);
                    ImGui::SameLine();
                    ImGui::TextWrapped("%s", classNames[classId].c_str());

                    // Confidence percentage
                    ImGui::Text("   %.1f%%", confidence * 100.0f);

                    // Progress bar
                    char label[32];
                    snprintf(label, sizeof(label), "##bar%zu", i);
                    ImGui::ProgressBar(confidence, ImVec2(-1, 0), label);

                    ImGui::Spacing();

                    ImGui::PopID();
                }
            } else {
                ImGui::TextWrapped("Load and classify an image to see predictions here.");
            }
        }
        ImGui::EndChild();

        ImGui::End();
    }

    void runInference() {
        if (currentImagePath.empty()) {
            std::cerr << "No image loaded\n";
            return;
        }

        std::cout << "Loading image: " << currentImagePath << "\n";

        // Load image
        int channels;
        unsigned char* image = stbi_load(currentImagePath.c_str(),
                                        &imageWidth, &imageHeight, &channels,
                                        STBI_rgb);

        if (!image) {
            std::cerr << "Failed to load image: " << currentImagePath << "\n";
            std::cerr << "STB Error: " << stbi_failure_reason() << "\n";
            return;
        }

        std::cout << "Image loaded: " << imageWidth << "x" << imageHeight << "\n";

        auto startTime = std::chrono::high_resolution_clock::now();

        // Preprocess with Vulkan Compute
        auto preprocessed = vulkanPreprocessor->preprocess(image, imageWidth, imageHeight, imageWidth * 3);

        auto preprocessEnd = std::chrono::high_resolution_clock::now();
        lastPreprocessTimeMs = std::chrono::duration<float, std::milli>(
            preprocessEnd - startTime).count();

        // Run inference
        lastResult = classifier->classify(preprocessed, 5);

        std::cout << "Inference complete!\n";
        std::cout << "Top prediction: " << classNames[lastResult.topK[0].first]
                  << " (" << (lastResult.topK[0].second * 100.0f) << "%)\n";

        stbi_image_free(image);
    }

    // Members
    GLFWwindow* window = nullptr;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<ImGuiSystem> imguiSystem;
    std::unique_ptr<VulkanPreprocessor> vulkanPreprocessor;
    std::unique_ptr<ONNXClassifier> classifier;

    std::string currentImagePath;
    int imageWidth = 0;
    int imageHeight = 0;
    ONNXClassifier::ClassificationResult lastResult;
    float lastPreprocessTimeMs = 0.0f;

    std::vector<std::string> classNames;
};

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    try {
        ImageClassifierApp app;
        app.run();

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
