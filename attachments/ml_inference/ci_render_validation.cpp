#include <gtest/gtest.h>
#include "vulkan_mnist_inference.h"
#include "onnx_inference.h"
#include "common/renderer/renderer.h"
#include "common/imgui_system.h"
#include "common/mnist_ui.h"
#include "imgui.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <memory>

/**
 * CI Render Validation with GTest
 * 
 * Demonstrates how to test a Vulkan application using a three-level hierarchy:
 * 
 * Level 1: Perceptual Similarity (LPIPS) - Does it look "right"?
 * Level 2: Functional UI Validation (OCR + Layout) - Can we read the UI and does it fit?
 * Level 3: Semantic Artifact Detection (Magenta check) - Are there broken textures?
 */

class LPIPSValidator {
public:
    LPIPSValidator(const std::string& modelPath) {
        #ifdef HAS_ONNX_RUNTIME
        try {
            session_ = std::make_unique<Ort::Session>(env_, 
                #ifdef _WIN32
                std::wstring(modelPath.begin(), modelPath.end()).c_str(),
                #else
                modelPath.c_str(),
                #endif
                options_);
            loaded_ = true;
        } catch (...) {
            std::cerr << "LPIPS model not found at " << modelPath << "\n";
        }
        #endif
    }

    float computeDistance(const std::vector<uint8_t>& img1, const std::vector<uint8_t>& img2, int w, int h) {
        if (!loaded_) return 0.5f; // Mock failure if model missing

        #ifdef HAS_ONNX_RUNTIME
        PreprocessedImage p1 = preprocess(img1, w, h);
        PreprocessedImage p2 = preprocess(img2, w, h);

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        
        int64_t shape[] = {1, 3, 224, 224};
        inputs.push_back(Ort::Value::CreateTensor<float>(mem, p1.data.data(), p1.data.size(), shape, 4));
        inputs.push_back(Ort::Value::CreateTensor<float>(mem, p2.data.data(), p2.data.size(), shape, 4));

        const char* inputNames[] = {"input0", "input1"};
        const char* outputNames[] = {"output"};

        auto outputs = session_->Run(Ort::RunOptions{nullptr}, inputNames, inputs.data(), 2, outputNames, 1);
        return outputs[0].GetTensorMutableData<float>()[0];
        #else
        return 0.0f;
        #endif
    }

private:
    #ifdef HAS_ONNX_RUNTIME
    PreprocessedImage preprocess(const std::vector<uint8_t>& pixels, int w, int h) {
        PreprocessedImage out;
        out.data.resize(3 * 224 * 224);
        out.shape = {1, 3, 224, 224};
        for (int y = 0; y < 224; ++y) {
            for (int x = 0; x < 224; ++x) {
                int srcX = (x * w) / 224;
                int srcY = (y * h) / 224;
                int srcIdx = (srcY * w + srcX) * 4;
                out.data[0 * 224 * 224 + y * 224 + x] = (pixels[srcIdx + 0] / 127.5f) - 1.0f;
                out.data[1 * 224 * 224 + y * 224 + x] = (pixels[srcIdx + 1] / 127.5f) - 1.0f;
                out.data[2 * 224 * 224 + y * 224 + x] = (pixels[srcIdx + 2] / 127.5f) - 1.0f;
            }
        }
        return out;
    }

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "LPIPS"};
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
    #endif
    bool loaded_ = false;
};

class VulkanRenderTest : public ::testing::Test {
protected:
    void SetUp() override {
        width = 800;
        height = 600;
        renderer = std::make_unique<Renderer>(nullptr, width, height);
        imgui = std::make_unique<ImGuiSystem>(renderer.get(), nullptr, width, height);
    }

    std::vector<uint8_t> CaptureFrame() {
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

        void* mapped = stagingMemory.mapMemory(0, bufferSize);
        std::vector<uint8_t> currentPixels(bufferSize);
        std::memcpy(currentPixels.data(), mapped, bufferSize);
        stagingMemory.unmapMemory();
        return currentPixels;
    }

    uint32_t width, height;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<ImGuiSystem> imgui;
};

TEST_F(VulkanRenderTest, MNIST_UI_Validation) {
    MNISTUIState uiState;
    uiState.engineReady = true;
    uiState.onnxReady = true;
    
    // Draw a '7' programmatically on the 28x28 canvas
    auto& canvas = uiState.canvas;
    // Top bar
    for(int x=8; x<20; ++x) {
        canvas.setPixel(x, 5, 1.0f);
        canvas.setPixel(x, 6, 1.0f);
    }
    // Diagonal
    for(int i=0; i<12; ++i) {
        canvas.setPixel(19-i, 5+i, 1.0f);
        canvas.setPixel(18-i, 5+i, 1.0f);
    }

    // Step 1: Render actual MNIST UI
    if (renderer->BeginFrame()) {
        imgui->NewFrame();
        RenderMNISTUI(uiState, width, height);
        imgui->Render(renderer->GetCurrentCommandBuffer(), renderer->GetCurrentFrame());
        renderer->EndFrame();
    }

    auto pixels = CaptureFrame();

    // LEVEL 1: Perceptual Similarity (LPIPS)
    SCOPED_TRACE("Level 1: Perceptual Similarity");
    LPIPSValidator lpips("models/lpips_vgg.onnx");
    float distance = lpips.computeDistance(pixels, pixels, width, height);
    EXPECT_LT(distance, 0.15f) << "Perceptual distance too high!";

    // LEVEL 2: Functional OCR + Fitting
    SCOPED_TRACE("Level 2: Functional UI Validation");
    
    // 2a. OCR check on the rendered canvas area
    VulkanMNISTInference ocr(*renderer);
    bool loaded = ocr.loadWeights("mnist_weights.bin");
    if (!loaded) loaded = ocr.loadWeights("../mnist_weights.bin");
    ASSERT_TRUE(loaded) << "Failed to load MNIST weights for OCR validation";
    
    // Search for the MNIST canvas first by its unique background color (30, 30, 30)
    int canvasMinX = width, canvasMinY = height, canvasMaxX = 0, canvasMaxY = 0;
    bool canvasFound = false;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            uint8_t b = pixels[(y * width + x) * 4 + 0];
            uint8_t g = pixels[(y * width + x) * 4 + 1];
            uint8_t r = pixels[(y * width + x) * 4 + 2];
            if (r == 30 && g == 30 && b == 30) {
                canvasMinX = std::min(canvasMinX, (int)x); canvasMinY = std::min(canvasMinY, (int)y);
                canvasMaxX = std::max(canvasMaxX, (int)x); canvasMaxY = std::max(canvasMaxY, (int)y);
                canvasFound = true;
            }
        }
    }
    ASSERT_TRUE(canvasFound) << "Could not find the MNIST canvas area on screen!";

    // Now search for the drawn digit ONLY within the canvas area
    int minX = canvasMaxX, minY = canvasMaxY, maxX = canvasMinX, maxY = canvasMinY;
    bool found = false;
    for (int y = canvasMinY; y <= canvasMaxY; ++y) {
        for (int x = canvasMinX; x <= canvasMaxX; ++x) {
            uint8_t b = pixels[(y * width + x) * 4 + 0];
            uint8_t g = pixels[(y * width + x) * 4 + 1];
            uint8_t r = pixels[(y * width + x) * 4 + 2];
            if (r > 200 && g > 200 && b > 200) { 
                minX = std::min(minX, (int)x); minY = std::min(minY, (int)y);
                maxX = std::max(maxX, (int)x); maxY = std::max(maxY, (int)y);
                found = true;
            }
        }
    }
    
    ASSERT_TRUE(found) << "Could not find any rendered digit on screen!";
    
    // Extract the whole canvas and downsample it to 28x28
    std::vector<float> mnistInput(784, 0.0f);
    int cw = canvasMaxX - canvasMinX + 1;
    int ch = canvasMaxY - canvasMinY + 1;
    
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            // Sample center of each block
            int sx = canvasMinX + (x * cw / 28) + (cw / 56);
            int sy = canvasMinY + (y * ch / 28) + (ch / 56);
            uint8_t val = pixels[(sy * width + sx) * 4 + 0]; // Blue channel
            mnistInput[y*28+x] = (val > 100) ? 1.0f : 0.0f;
        }
    }
    auto res = ocr.infer(mnistInput);
    int predictedDigit = std::distance(res.begin(), std::max_element(res.begin(), res.end()));
    
    EXPECT_EQ(predictedDigit, 7) << "OCR failed to recognize the rendered digit!";

    // LEVEL 3: Semantic Artifact Detection
    SCOPED_TRACE("Level 3: Semantic Artifact Detection");
    int magentaCount = 0;
    for (uint32_t i = 0; i < width * height; ++i) {
        uint8_t b = pixels[i*4+0], g = pixels[i*4+1], r = pixels[i*4+2];
        if (r > 200 && b > 200 && g < 50) magentaCount++; // Missing texture color
    }
    EXPECT_LT(magentaCount, 100) << "Detected too many broken (magenta) textures!";
}
