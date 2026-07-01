#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "common/renderer/renderer.h"
#include "common/imgui_system.h"
#include "common/mnist_ui.h"
#include "imgui.h"

/**
 * RL Automated Exploration (RND) Demonstration with GTest
 * 
 * This sample demonstrates Curiosity-Driven Exploration using 
 * Random Network Distillation (RND) on the actual MNIST application UI.
 * 
 * Concept:
 * 1. An RL agent "explores" the Vulkan application.
 * 2. It uses a 'Target Network' (fixed) and a 'Predictor Network' (trainable).
 * 3. Prediction Error = Curiosity Reward.
 * 4. This allows the agent to autonomously find "novel" states or bugs 
 *    without manual test scripts.
 */

class RNDModule {
public:
    RNDModule(size_t inputSize, size_t featureSize)
        : inputSize(inputSize), featureSize(featureSize) {

        // Initialize Random (Target) Network weights - FIXED
        targetWeights.resize(inputSize * featureSize);
        std::mt19937 targetGen(42); // Fixed seed for target
        std::uniform_real_distribution<float> dist{-1.0f, 1.0f};
        for (auto& w : targetWeights) w = dist(targetGen);

        // Initialize Predictor Network weights
        predictorWeights.resize(inputSize * featureSize);
        std::mt19937 predGen(123);
        for (auto& w : predictorWeights) w = dist(predGen);
    }

    float computeCuriosityReward(const std::vector<float>& state) {
        auto targetFeatures = forward(targetWeights, state);
        auto predictedFeatures = forward(predictorWeights, state);

        float mse = 0;
        for (size_t i = 0; i < featureSize; ++i) {
            float diff = targetFeatures[i] - predictedFeatures[i];
            mse += diff * diff;
        }
        mse /= static_cast<float>(featureSize);

        // Train predictor toward target for this state (Gradient Descent step)
        trainPredictor(state, targetFeatures);

        return mse; // Higher error = higher curiosity reward
    }

private:
    std::vector<float> forward(const std::vector<float>& weights, const std::vector<float>& input) {
        std::vector<float> output(featureSize, 0.0f);
        for (size_t i = 0; i < featureSize; ++i) {
            for (size_t j = 0; j < inputSize; ++j) {
                output[i] += weights[i * inputSize + j] * input[j];
            }
            output[i] = std::tanh(output[i]); // Simple non-linearity
        }
        return output;
    }

    void trainPredictor(const std::vector<float>& input, const std::vector<float>& target) {
        float learningRate = 0.05f; // Increased learning rate
        for (int step = 0; step < 10; ++step) { // Increased steps
            auto current = forward(predictorWeights, input);
            for (size_t i = 0; i < featureSize; ++i) {
                float error = target[i] - current[i];
                for (size_t j = 0; j < inputSize; ++j) {
                    float grad = error * (1.0f - current[i] * current[i]);
                    predictorWeights[i * inputSize + j] += learningRate * grad * input[j];
                }
            }
        }
    }

    size_t inputSize, featureSize;
    std::vector<float> targetWeights;
    std::vector<float> predictorWeights;
};

class VulkanExplorationTest : public ::testing::Test {
protected:
    void SetUp() override {
        width = 800;
        height = 600;
        // Initialize Headless Renderer
        renderer = std::make_unique<Renderer>(nullptr, width, height);
        imgui = std::make_unique<ImGuiSystem>(renderer.get(), nullptr, width, height);
    }

    // Capture the frame and downsample for the RND module using averaging
    std::vector<float> CaptureAndDownsample(int targetW, int targetH) {
        size_t bufferSize = width * height * 4;
        vk::BufferCreateInfo bufferInfo{.size = bufferSize, .usage = vk::BufferUsageFlagBits::eTransferDst};
        vk::raii::Buffer stagingBuffer(renderer->GetRaiiDevice(), bufferInfo);
        auto memReqs = stagingBuffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memReqs.size,
            .memoryTypeIndex = renderer->FindMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
        };
        vk::raii::DeviceMemory stagingMemory(renderer->GetRaiiDevice(), allocInfo);
        stagingBuffer.bindMemory(*stagingMemory, 0);

        renderer->TransitionImageLayout(renderer->GetSwapchainImages()[0], vk::Format::eB8G8R8A8Unorm,
                                     vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal);
        renderer->CopyImageToBuffer(renderer->GetSwapchainImages()[0], *stagingBuffer, width, height);

        uint8_t* mapped = static_cast<uint8_t*>(stagingMemory.mapMemory(0, bufferSize));
        
        std::vector<float> downsampled(targetW * targetH * 3, 0.0f);
        int blockW = width / targetW;
        int blockH = height / targetH;
        float invBlockArea = 1.0f / static_cast<float>(blockW * blockH);

        for (int y = 0; y < targetH; ++y) {
            for (int x = 0; x < targetW; ++x) {
                float sumR = 0, sumG = 0, sumB = 0;
                for (int sy = 0; sy < blockH; ++sy) {
                    for (int sx = 0; sx < blockW; ++sx) {
                        int srcIdx = ((y * blockH + sy) * width + (x * blockW + sx)) * 4;
                        sumB += mapped[srcIdx + 0]; // B8G8R8A8
                        sumG += mapped[srcIdx + 1];
                        sumR += mapped[srcIdx + 2];
                    }
                }
                downsampled[(y * targetW + x) * 3 + 0] = (sumR * invBlockArea) / 255.0f;
                downsampled[(y * targetW + x) * 3 + 1] = (sumG * invBlockArea) / 255.0f;
                downsampled[(y * targetW + x) * 3 + 2] = (sumB * invBlockArea) / 255.0f;
            }
        }
        
        stagingMemory.unmapMemory();
        return downsampled;
    }

    uint32_t width, height;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<ImGuiSystem> imgui;
};

TEST_F(VulkanExplorationTest, MNIST_Novelty_Detection) {
    // 1. Initialize RND Module
    // Input: 80x60 RGB = 14400 values, Latent: 128 features
    const int DS_W = 80;
    const int DS_H = 60;
    RNDModule rnd(DS_W * DS_H * 3, 128);

    MNISTUIState uiState;
    uiState.engineReady = true;
    uiState.onnxReady = true;

    std::cout << "[RL Exploration] Starting Automated Exploration Simulation..." << std::endl;

    // 2. EXPLORATION PHASE: Agent sees the "Normal" UI (empty canvas) multiple times.
    // The predictor learns this state, and curiosity rewards should drop.
    float initialReward = 0;
    float lastStaticReward = 0;

    for (int i = 0; i < 20; ++i) {
        if (renderer->BeginFrame()) {
            imgui->NewFrame();
            RenderMNISTUI(uiState, width, height);
            imgui->Render(renderer->GetCurrentCommandBuffer(), renderer->GetCurrentFrame());
            renderer->EndFrame();
        }

        auto observation = CaptureAndDownsample(DS_W, DS_H);
        float reward = rnd.computeCuriosityReward(observation);
        
        if (i == 0) initialReward = reward;
        lastStaticReward = reward;
        
        if (i % 5 == 0) {
            std::cout << "  Frame " << std::setw(2) << i << " (Familiar UI) - Curiosity: " 
                      << std::fixed << std::setprecision(6) << reward << std::endl;
        }
    }

    EXPECT_LT(lastStaticReward, initialReward) << "Curiosity should drop as the agent becomes 'bored' with the static UI";

    // 3. NOVELTY EVENT: Something unexpected happens (e.g. User draws a digit or an error occurs)
    std::cout << "\n[RL Exploration] SUDDEN NOVELTY: Drawing a large digit on the canvas..." << std::endl;
    
    // Draw a thick '7'
    for(int x=5; x<23; ++x) {
        uiState.canvas.setPixel(x, 5, 1.0f);
        uiState.canvas.setPixel(x, 6, 1.0f);
    }
    for(int i=0; i<15; ++i) {
        uiState.canvas.setPixel(22-i, 5+i, 1.0f);
        uiState.canvas.setPixel(21-i, 5+i, 1.0f);
    }

    if (renderer->BeginFrame()) {
        imgui->NewFrame();
        RenderMNISTUI(uiState, width, height);
        imgui->Render(renderer->GetCurrentCommandBuffer(), renderer->GetCurrentFrame());
        renderer->EndFrame();
    }

    auto novelObservation = CaptureAndDownsample(DS_W, DS_H);
    float novelReward = rnd.computeCuriosityReward(novelObservation);

    std::cout << "  Frame 20 (Novel UI)    - Curiosity: " << novelReward << " (Spike!)" << std::endl;

    // 4. VERIFICATION: Curiosity reward for novel state should be higher
    EXPECT_GT(novelReward, lastStaticReward * 1.2f) << "Novel UI state should trigger a curiosity spike!";
    
    if (novelReward > lastStaticReward * 1.1f) {
        std::cout << "\n✓ SUCCESS: The RL agent successfully identified the UI change as a highly novel state!" << std::endl;
        std::cout << "This demonstrates how autonomous agents can find edge cases or crashes by seeking surprise." << std::endl;
    }
}
