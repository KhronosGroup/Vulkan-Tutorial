/**
 * MNIST Digit Recognition with Vulkan and ImGui
 *
 * Professional implementation following Simple Game Engine patterns.
 * - Pure Vulkan rendering (no OpenGL)
 * - Custom ImGui integration
 * - Real ML inference with Vulkan compute shaders
 */

#include "renderer.h"
#include "imgui_system.h"
#include "vulkan_mnist_inference.h"
#ifdef HAS_ONNX_RUNTIME
#include "onnx_inference.h"
#endif
#ifdef HAS_IREE
#include "iree_mnist_inference.h"
#endif
#include "common/mnist_ui.h"
#include "imgui.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <cmath>

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    GLFWwindow* window = glfwCreateWindow(800, 600, "MNIST Digit Recognition - Vulkan", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return 1;
    }

    // Set minimum window size to prevent layout issues
    glfwSetWindowSizeLimits(window, 800, 500, GLFW_DONT_CARE, GLFW_DONT_CARE);

    // Application state
    MNISTUIState uiState;
    int currentWidth = 800, currentHeight = 600;

    // Scope to ensure imgui is destroyed before renderer
    {
        // Create renderer and ImGui system
        Renderer renderer(window, 800, 600);
        ImGuiSystem imgui(&renderer, window, 800, 600);

        // Create Vulkan compute inference engine
        VulkanMNISTInference engine(renderer);
        uiState.engineReady = engine.loadWeights("mnist_weights.bin");
        if (!uiState.engineReady) {
            uiState.engineReady = engine.loadWeights("../mnist_weights.bin");
        }

#ifdef HAS_ONNX_RUNTIME
        std::unique_ptr<ONNXClassifier> onnxEngine;
        try {
            onnxEngine = std::make_unique<ONNXClassifier>("mnist_model.onnx");
            uiState.onnxReady = true;
        } catch (...) {
            try {
                onnxEngine = std::make_unique<ONNXClassifier>("../mnist_model.onnx");
                uiState.onnxReady = true;
            } catch (...) {
                std::cerr << "Warning: Could not load ONNX MNIST model\n";
            }
        }
#endif

#ifdef HAS_IREE
        std::unique_ptr<IREEMNISTInference> ireeEngine;
        try {
            ireeEngine = std::make_unique<IREEMNISTInference>("mnist_model.vmfb");
            if (ireeEngine->isReady()) uiState.ireeReady = true;
        } catch (...) {
            try {
                ireeEngine = std::make_unique<IREEMNISTInference>("../mnist_model.vmfb");
                if (ireeEngine->isReady()) uiState.ireeReady = true;
            } catch (...) {
                std::cerr << "Warning: Could not load IREE MNIST model\n";
            }
        }
#endif

        // Main loop
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            // Check for window resize
            int newWidth, newHeight;
            glfwGetFramebufferSize(window, &newWidth, &newHeight);

            // Skip rendering if minimized
            if (newWidth == 0 || newHeight == 0) {
                continue;
            }

            // Update sizes if changed
            if (newWidth != currentWidth || newHeight != currentHeight) {
                currentWidth = newWidth;
                currentHeight = newHeight;
                imgui.UpdateDisplaySize(currentWidth, currentHeight);
            }

            if (!renderer.BeginFrame()) continue;

            imgui.NewFrame();

            auto events = RenderMNISTUI(uiState, currentWidth, currentHeight);

            if (events.clearClicked) {
                uiState.canvas.clear();
                std::fill(uiState.probabilities.begin(), uiState.probabilities.end(), 0.0f);
                uiState.predictedDigit = -1;
                uiState.statusMessage = "";
            }

            if (events.recognizeClicked) {
                auto startTime = std::chrono::high_resolution_clock::now();
                if (uiState.inferenceMode == 0) {
                    // Vulkan Inference
                    uiState.probabilities = engine.infer(uiState.canvas.getPixels());
                } else if (uiState.inferenceMode == 1) {
                    // ONNX Inference
#ifdef HAS_ONNX_RUNTIME
                    if (onnxEngine) {
                        // Apply SAME preprocessing as Vulkan
                        std::vector<float> preprocessed = VulkanMNISTInference::preprocess(uiState.canvas.getPixels());
                        
                        // Apply normalization (Mean: 0.1307, Std: 0.3081)
                        std::vector<float> normalized(784);
                        for (size_t i = 0; i < 784; ++i) {
                            normalized[i] = (preprocessed[i] - 0.1307f) / 0.3081f;
                        }

                        PreprocessedImage img;
                        img.data = normalized;
                        // MNIST ONNX expects [batch, channels, height, width] = [1, 1, 28, 28]
                        img.shape = {1, 1, 28, 28};
                        auto result = onnxEngine->classify(img, 10);
                        
                        // Convert top-K back to flat probabilities vector
                        std::fill(uiState.probabilities.begin(), uiState.probabilities.end(), 0.0f);
                        for (auto& p : result.topK) {
                            if (p.first >= 0 && p.first < 10) {
                                uiState.probabilities[p.first] = p.second;
                            }
                        }
                    }
#endif
                } else if (uiState.inferenceMode == 2) {
                    // IREE Inference
#ifdef HAS_IREE
                    if (ireeEngine) {
                        uiState.probabilities = ireeEngine->infer(uiState.canvas.getPixels());
                    }
#endif
                }
                auto endTime = std::chrono::high_resolution_clock::now();
                uiState.lastInferenceTimeMs = std::chrono::duration<float, std::milli>(endTime - startTime).count();

                auto maxIt = std::max_element(uiState.probabilities.begin(), uiState.probabilities.end());
                if (*maxIt > 0.01f) {
                    uiState.predictedDigit = std::distance(uiState.probabilities.begin(), maxIt);
                    uiState.statusMessage = "";
                } else {
                    uiState.statusMessage = "Canvas is empty - draw something first!";
                    uiState.predictedDigit = -1;
                }
            }

            // Render
            imgui.Render(renderer.GetCurrentCommandBuffer(), renderer.GetCurrentFrame());
            renderer.EndFrame();
        }

        // Wait for device to finish before cleanup
        renderer.WaitIdle();

        // ImGui and Renderer will be destroyed here in correct order (imgui first, then renderer)
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
