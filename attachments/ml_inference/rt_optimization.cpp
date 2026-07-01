#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <memory>
#include <GLFW/glfw3.h>
#include "onnx_inference.h"
#include "common/renderer/renderer.h"
#include "common/imgui_system.h"
#include "imgui.h"

/**
 * RT (Real-Time) Ray Tracing Optimization Demonstration
 * 
 * ACTUAL demonstration of AI-driven Ray Tracing optimization techniques:
 * 1. AI-Guided Adaptive Sampling (Error Predictor -> Heatmap)
 * 2. AI Denoising (U-Net Reconstruction)
 * 
 * Pipeline:
 * [Pass 1] Probe (1spp) -> [Pass 2] Error Predictor (Heatmap) -> [Pass 3] Targeted Trace -> [Pass 4] AI Denoiser
 */

class RayTracingOptimizer {
public:
    RayTracingOptimizer(Renderer& renderer, GLFWwindow* window) 
        : renderer_(renderer), window_(window) {
        
        imgui_ = std::make_unique<ImGuiSystem>(&renderer_, window_, 1280, 720);

        #ifdef HAS_ONNX_RUNTIME
        try {
            errorPredictor_ = std::make_unique<ONNXClassifier>("models/error_predictor.onnx");
            denoiser_ = std::make_unique<ONNXClassifier>("models/denoiser.onnx");
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not load AI models: " << e.what() << "\n";
        }
        #endif
        
        createResources();
        createPipeline();
        createDisplayTextures();
    }

    ~RayTracingOptimizer() {
        renderer_.WaitIdle();
    }

    void run() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();

            auto startFrame = std::chrono::high_resolution_clock::now();

            if (renderer_.BeginFrame()) {
                // RESET TIMES
                probeTime_ = 0;
                heatmapTime_ = 0;
                targetedTime_ = 0;
                denoiseTime_ = 0;

                // 1. PROBE PASS (1 spp)
                auto startProbe = std::chrono::high_resolution_clock::now();
                dispatchRayTrace(1);
                auto endProbe = std::chrono::high_resolution_clock::now();
                probeTime_ = std::chrono::duration<float, std::milli>(endProbe - startProbe).count();

                // 2. DOWNLOAD PROBE FOR AI (Needed if Adaptive or Heatmap view is on)
                bool needsHeatmap = adaptiveEnabled_ || viewMode_ == 1;
                std::vector<float> probePixels;
                
                if (needsHeatmap || viewMode_ == 2) {
                    probePixels = downloadImage();
                    lastProbePixels_ = probePixels;
                }

                // 3. AI VARIANCE PREDICTION
                if (needsHeatmap && !probePixels.empty()) {
                    #ifdef HAS_ONNX_RUNTIME
                    if (errorPredictor_) {
                        PreprocessedImage img;
                        img.shape = {1, 3, 256, 256};
                        img.data.resize(3 * 256 * 256);
                        
                        // Downsample probePixels (800x600) to 256x256 for AI
                        for (int y = 0; y < 256; ++y) {
                            for (int x = 0; x < 256; ++x) {
                                int sx = x * 800 / 256;
                                int sy = y * 600 / 256;
                                int sidx = (sy * 800 + sx) * 3;
                                int didx = (y * 256 + x);
                                img.data[0 * 65536 + didx] = probePixels[sidx + 0];
                                img.data[1 * 65536 + didx] = probePixels[sidx + 1];
                                img.data[2 * 65536 + didx] = probePixels[sidx + 2];
                            }
                        }

                        auto res = errorPredictor_->runGeneric(img);
                        heatmapTime_ = res.inferenceTimeMs;
                        lastHeatmapData_ = res.data;

                        // UPLOAD TO GPU FOR GUIDANCE
                        if (adaptiveEnabled_) {
                            void* hData = heatmapMemoryBuffer_.mapMemory(0, 256 * 256 * sizeof(float));
                            std::memcpy(hData, lastHeatmapData_.data(), 256 * 256 * sizeof(float));
                            heatmapMemoryBuffer_.unmapMemory();
                        }
                    }
                    #endif
                }

                // 4. TARGETED PASS (Guided by Heatmap)
                if (adaptiveEnabled_) {
                    auto startTargeted = std::chrono::high_resolution_clock::now();
                    dispatchRayTrace(31); // Adaptive samples
                    auto endTargeted = std::chrono::high_resolution_clock::now();
                    targetedTime_ = std::chrono::duration<float, std::milli>(endTargeted - startTargeted).count();
                }

                // 5. DOWNLOAD FINAL RESULT (From Pass 1 or Pass 3)
                auto currentResultPixels = downloadImage();

                // 6. AI DENOISING
                if (denoiseEnabled_) {
                    #ifdef HAS_ONNX_RUNTIME
                    if (denoiser_ && !currentResultPixels.empty()) {
                        PreprocessedImage multiChannel;
                        multiChannel.shape = {1, 10, 256, 256};
                        multiChannel.data.assign(10 * 256 * 256, 0.0f);
                        
                        // Fill first 3 channels with current result data
                        for (int y = 0; y < 256; ++y) {
                            for (int x = 0; x < 256; ++x) {
                                int sx = x * 800 / 256;
                                int sy = y * 600 / 256;
                                int sidx = (sy * 800 + sx) * 3;
                                int didx = (y * 256 + x);
                                multiChannel.data[0 * 65536 + didx] = currentResultPixels[sidx + 0];
                                multiChannel.data[1 * 65536 + didx] = currentResultPixels[sidx + 1];
                                multiChannel.data[2 * 65536 + didx] = currentResultPixels[sidx + 2];
                            }
                        }

                        auto res = denoiser_->runGeneric(multiChannel);
                        denoiseTime_ = res.inferenceTimeMs;
                        lastDenoisedPixels_ = res.data;
                    }
                    #endif
                }

                // Update display texture from buffers
                updateDisplayTexture(currentResultPixels);

                imgui_->NewFrame();
                drawUI();
                imgui_->Render(renderer_.GetCurrentCommandBuffer(), renderer_.GetCurrentFrame());

                renderer_.EndFrame();
            }

            auto endFrame = std::chrono::high_resolution_clock::now();
            totalTime_ = std::chrono::duration<float, std::milli>(endFrame - startFrame).count();
            frameCount_++;
        }
    }

private:
    void createDisplayTextures() {
        // Main output texture
        displayTexID_ = createTexture(800, 600, displayImage_, displayMemory_, displayView_, displaySampler_);
        // Heatmap texture
        heatmapTexID_ = createTexture(800, 600, heatmapImage_, heatmapMemory_, heatmapView_, heatmapSampler_);
        // Probe texture
        probeTexID_ = createTexture(800, 600, probeImage_, probeMemory_, probeView_, probeSampler_);
    }

    ImTextureID createTexture(uint32_t w, uint32_t h, vk::raii::Image& img, vk::raii::DeviceMemory& mem, vk::raii::ImageView& view, vk::raii::Sampler& sampler) {
        auto& device = renderer_.GetRaiiDevice();
        
        vk::ImageCreateInfo imageInfo{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .extent = {w, h, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst
        };

        img = vk::raii::Image(device, imageInfo);
        auto memReqs = img.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memReqs.size,
            .memoryTypeIndex = renderer_.FindMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
        };
        mem = vk::raii::DeviceMemory(device, allocInfo);
        img.bindMemory(*mem, 0);

        vk::ImageViewCreateInfo viewInfo{
            .image = *img,
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
        };
        view = vk::raii::ImageView(device, viewInfo);

        vk::SamplerCreateInfo samplerInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge
        };
        sampler = vk::raii::Sampler(device, samplerInfo);

        return imgui_->AddTexture(view, sampler);
    }

    void updateDisplayTexture(const std::vector<float>& floatPixels) {
        // If denoised result is available and enabled, we use it for display
        bool useDenoised = denoiseEnabled_ && !lastDenoisedPixels_.empty();

        // Convert to RGBA8 for display
        std::vector<uint8_t> rgbaPixels(800 * 600 * 4);
        std::vector<uint8_t> heatPixels(800 * 600 * 4);
        std::vector<uint8_t> probePixels(800 * 600 * 4);

        for (uint32_t i = 0; i < 800 * 600; ++i) {
            int px = i % 800;
            int py = i / 800;

            if (useDenoised) {
                // Bilinear upsample denoised result (256x256) back to 800x600
                float fx = (static_cast<float>(px) + 0.5f) * 256.0f / 800.0f - 0.5f;
                float fy = (static_cast<float>(py) + 0.5f) * 256.0f / 600.0f - 0.5f;
                fx = std::clamp(fx, 0.0f, 255.0f);
                fy = std::clamp(fy, 0.0f, 255.0f);
                int x0 = static_cast<int>(fx);
                int y0 = static_cast<int>(fy);
                int x1 = std::min(x0 + 1, 255);
                int y1 = std::min(y0 + 1, 255);
                float wx = fx - static_cast<float>(x0);
                float wy = fy - static_cast<float>(y0);
                for (int c = 0; c < 3; ++c) {
                    int off = c * 65536;
                    float v00 = lastDenoisedPixels_[off + y0 * 256 + x0];
                    float v01 = lastDenoisedPixels_[off + y0 * 256 + x1];
                    float v10 = lastDenoisedPixels_[off + y1 * 256 + x0];
                    float v11 = lastDenoisedPixels_[off + y1 * 256 + x1];
                    float val = v00 * (1.0f - wx) * (1.0f - wy)
                              + v01 * wx * (1.0f - wy)
                              + v10 * (1.0f - wx) * wy
                              + v11 * wx * wy;
                    rgbaPixels[i * 4 + c] = static_cast<uint8_t>(std::clamp(val * 255.0f, 0.0f, 255.0f));
                }
            } else {
                // Main image (Final from Ray Tracer)
                rgbaPixels[i * 4 + 0] = static_cast<uint8_t>(std::clamp(floatPixels[i * 3 + 0] * 255.0f, 0.0f, 255.0f));
                rgbaPixels[i * 4 + 1] = static_cast<uint8_t>(std::clamp(floatPixels[i * 3 + 1] * 255.0f, 0.0f, 255.0f));
                rgbaPixels[i * 4 + 2] = static_cast<uint8_t>(std::clamp(floatPixels[i * 3 + 2] * 255.0f, 0.0f, 255.0f));
            }
            rgbaPixels[i * 4 + 3] = 255;

            // Probe image (1spp)
            if (!lastProbePixels_.empty()) {
                probePixels[i * 4 + 0] = static_cast<uint8_t>(std::clamp(lastProbePixels_[i * 3 + 0] * 255.0f, 0.0f, 255.0f));
                probePixels[i * 4 + 1] = static_cast<uint8_t>(std::clamp(lastProbePixels_[i * 3 + 1] * 255.0f, 0.0f, 255.0f));
                probePixels[i * 4 + 2] = static_cast<uint8_t>(std::clamp(lastProbePixels_[i * 3 + 2] * 255.0f, 0.0f, 255.0f));
                probePixels[i * 4 + 3] = 255;
            }

            // Generate/Display heatmap using a Blue-Green-Red colormap
            float h = 0.0f;
            if (!lastHeatmapData_.empty()) {
                int hx = (i % 800) * 256 / 800;
                int hy = (i / 800) * 256 / 600;
                h = std::clamp(lastHeatmapData_[hy * 256 + hx], 0.0f, 1.0f);
            } else {
                // Fallback: brightness-based heatmap
                h = std::clamp(floatPixels[i * 3 + 0] * 0.8f + 0.1f, 0.0f, 1.0f);
            }
            
            if (h < 0.5f) {
                float t = h * 2.0f;
                heatPixels[i * 4 + 0] = 0;
                heatPixels[i * 4 + 1] = static_cast<uint8_t>(t * 255);
                heatPixels[i * 4 + 2] = static_cast<uint8_t>((1.0f - t) * 255);
            } else {
                float t = (h - 0.5f) * 2.0f;
                heatPixels[i * 4 + 0] = static_cast<uint8_t>(t * 255);
                heatPixels[i * 4 + 1] = static_cast<uint8_t>((1.0f - t) * 255);
                heatPixels[i * 4 + 2] = 0;
            }
            heatPixels[i * 4 + 3] = 255;
        }

        uploadToTexture(displayImage_, rgbaPixels);
        uploadToTexture(heatmapImage_, heatPixels);
        uploadToTexture(probeImage_, probePixels);
    }

    void uploadToTexture(vk::raii::Image& img, const std::vector<uint8_t>& pixels) {
        size_t size = pixels.size();
        vk::BufferCreateInfo stagingInfo{.size = size, .usage = vk::BufferUsageFlagBits::eTransferSrc};
        vk::raii::Buffer stagingBuffer(renderer_.GetRaiiDevice(), stagingInfo);
        auto memReqs = stagingBuffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memReqs.size,
            .memoryTypeIndex = renderer_.FindMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
        };
        vk::raii::DeviceMemory stagingMem(renderer_.GetRaiiDevice(), allocInfo);
        stagingBuffer.bindMemory(*stagingMem, 0);

        void* data = stagingMem.mapMemory(0, size);
        std::memcpy(data, pixels.data(), size);
        stagingMem.unmapMemory();

        renderer_.TransitionImageLayout(*img, vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        renderer_.CopyBufferToImage(*stagingBuffer, *img, 800, 600);
        renderer_.TransitionImageLayout(*img, vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    void drawUI() {
        // Single fullscreen window for robust layout
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize, ImGuiCond_Always);
        ImGui::Begin("RT Optimization Demo", nullptr, 
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | 
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus);

        if (ImGui::BeginTable("MainTable", 2, ImGuiTableFlags_Resizable)) {
            ImGui::TableSetupColumn("Controls", ImGuiTableColumnFlags_WidthFixed, 360.0f);
            ImGui::TableSetupColumn("Viewport", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableNextRow();

            // 1. Controls Column
            ImGui::TableNextColumn();
            drawControls();

            // 2. Viewport Column
            ImGui::TableNextColumn();
            drawViewport();

            ImGui::EndTable();
        }
        ImGui::End();
    }

    void drawControls() {
        ImGui::BeginChild("ControlsChild", ImVec2(0, 0), true, ImGuiWindowFlags_AlwaysVerticalScrollbar);
        
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "AI-Driven Ray Tracing");
        ImGui::Separator();

        ImGui::Checkbox("Enable Adaptive Sampling", &adaptiveEnabled_);
        ImGui::Checkbox("Enable AI Denoising", &denoiseEnabled_);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text("Performance Metrics:");
        
        float currentTotal = probeTime_ + heatmapTime_ + targetedTime_ + denoiseTime_;

        ImGui::BulletText("Pass 1: Probe (1spp): %.2f ms", probeTime_);
        if (heatmapTime_ > 0) {
            ImGui::BulletText("Pass 2: AI Heatmap: %.2f ms", heatmapTime_);
        }
        if (adaptiveEnabled_) {
            ImGui::BulletText("Pass 3: Adaptive Trace: %.2f ms", targetedTime_);
        }
        if (denoiseEnabled_) {
            ImGui::BulletText("Pass 4: AI Denoising: %.2f ms", denoiseTime_);
        }

        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Total Frame: %.2f ms", currentTotal);
        
        float bruteForceTime = probeTime_ * 1000.0f;
        if (currentTotal > 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Speedup vs 1000spp: %.1fx", bruteForceTime / currentTotal);
            ImGui::BulletText("Inference Savings: %.1f%%", (1.0f - currentTotal / bruteForceTime) * 100.0f);
        }

        ImGui::Separator();
        ImGui::Text("Visualization:");
        ImGui::RadioButton("Final Result", &viewMode_, 0);
        ImGui::RadioButton("AI Heatmap", &viewMode_, 1);
        ImGui::RadioButton("Noisy Probe (1spp)", &viewMode_, 2);
        
        ImGui::Spacing();
        ImGui::Dummy(ImVec2(0, 20)); // Extra padding at bottom to prevent cropping
        
        ImGui::EndChild();
    }

    void drawViewport() {
        ImGui::BeginChild("ViewportChild", ImVec2(0, 0), true);
        
        ImTextureID texToShow = displayTexID_;
        if (viewMode_ == 1) texToShow = heatmapTexID_;
        if (viewMode_ == 2) texToShow = probeTexID_;
        
        ImVec2 avail = ImGui::GetContentRegionAvail();
        // Maintain 4:3 aspect ratio
        float scale = std::min(avail.x / 800.0f, avail.y / 600.0f);
        ImVec2 size(800.0f * scale, 600.0f * scale);
        
        // Center image
        ImGui::SetCursorPos(ImVec2((avail.x - size.x) * 0.5f, (avail.y - size.y) * 0.5f));
        ImGui::Image(texToShow, size);
        
        ImGui::EndChild();
    }

    void createResources() {
        auto& device = renderer_.GetRaiiDevice();
        vk::BufferCreateInfo bufferInfo;
        bufferInfo.size = 800 * 600 * 3 * sizeof(float);
        bufferInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc;
        outputBuffer_ = vk::raii::Buffer(device, bufferInfo);
        
        auto memReqs = outputBuffer_.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = renderer_.FindMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        outputMemory_ = vk::raii::DeviceMemory(device, allocInfo);
        outputBuffer_.bindMemory(*outputMemory_, 0);

        // Heatmap buffer (256x256 floats)
        vk::BufferCreateInfo hBufferInfo;
        hBufferInfo.size = 256 * 256 * sizeof(float);
        hBufferInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer;
        heatmapBuffer_ = vk::raii::Buffer(device, hBufferInfo);
        
        auto hMemReqs = heatmapBuffer_.getMemoryRequirements();
        vk::MemoryAllocateInfo hAllocInfo;
        hAllocInfo.allocationSize = hMemReqs.size;
        hAllocInfo.memoryTypeIndex = renderer_.FindMemoryType(hMemReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        heatmapMemoryBuffer_ = vk::raii::DeviceMemory(device, hAllocInfo);
        heatmapBuffer_.bindMemory(*heatmapMemoryBuffer_, 0);
        
        // Zero out heatmap initially
        void* hData = heatmapMemoryBuffer_.mapMemory(0, 256 * 256 * sizeof(float));
        std::memset(hData, 0, 256 * 256 * sizeof(float));
        heatmapMemoryBuffer_.unmapMemory();
    }

    void createPipeline() {
        auto& device = renderer_.GetRaiiDevice();
        shaderModule_ = renderer_.CreateShaderModule("shaders/raytrace.comp.spv");

        std::array<vk::DescriptorSetLayoutBinding, 2> bindings;
        bindings[0].binding = 0;
        bindings[0].descriptorType = vk::DescriptorType::eStorageBuffer;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = vk::ShaderStageFlagBits::eCompute;

        bindings[1].binding = 1;
        bindings[1].descriptorType = vk::DescriptorType::eStorageBuffer;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = vk::ShaderStageFlagBits::eCompute;

        vk::DescriptorSetLayoutCreateInfo layoutInfo;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        descriptorSetLayout_ = vk::raii::DescriptorSetLayout(device, layoutInfo);

        vk::PushConstantRange pcRange;
        pcRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
        pcRange.offset = 0;
        pcRange.size = 16;

        vk::PipelineLayoutCreateInfo pipeLayoutInfo;
        pipeLayoutInfo.setLayoutCount = 1;
        pipeLayoutInfo.pSetLayouts = &*descriptorSetLayout_;
        pipeLayoutInfo.pushConstantRangeCount = 1;
        pipeLayoutInfo.pPushConstantRanges = &pcRange;
        pipelineLayout_ = vk::raii::PipelineLayout(device, pipeLayoutInfo);

        vk::ComputePipelineCreateInfo pipeInfo;
        pipeInfo.stage.stage = vk::ShaderStageFlagBits::eCompute;
        pipeInfo.stage.module = *shaderModule_;
        pipeInfo.stage.pName = "main";
        pipeInfo.layout = *pipelineLayout_;
        pipeline_ = vk::raii::Pipeline(device, nullptr, pipeInfo);

        vk::DescriptorPoolSize poolSize;
        poolSize.type = vk::DescriptorType::eStorageBuffer;
        poolSize.descriptorCount = 2;

        vk::DescriptorPoolCreateInfo poolInfo;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        descriptorPool_ = vk::raii::DescriptorPool(device, poolInfo);

        vk::DescriptorSetAllocateInfo allocInfo;
        allocInfo.descriptorPool = *descriptorPool_;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &*descriptorSetLayout_;
        descriptorSet_ = std::move(vk::raii::DescriptorSets(device, allocInfo)[0]);

        vk::DescriptorBufferInfo buffInfo;
        buffInfo.buffer = *outputBuffer_;
        buffInfo.offset = 0;
        buffInfo.range = VK_WHOLE_SIZE;

        vk::DescriptorBufferInfo hBuffInfo;
        hBuffInfo.buffer = *heatmapBuffer_;
        hBuffInfo.offset = 0;
        hBuffInfo.range = VK_WHOLE_SIZE;

        std::array<vk::WriteDescriptorSet, 2> writes;
        writes[0].dstSet = *descriptorSet_;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = vk::DescriptorType::eStorageBuffer;
        writes[0].pBufferInfo = &buffInfo;

        writes[1].dstSet = *descriptorSet_;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = vk::DescriptorType::eStorageBuffer;
        writes[1].pBufferInfo = &hBuffInfo;

        device.updateDescriptorSets(writes, nullptr);
    }

    void dispatchRayTrace(uint32_t spp) {
        // Use a temporary command buffer for immediate execution to allow same-frame readback
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *renderer_.GetCommandPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };
        auto cmdBuffers = renderer_.GetRaiiDevice().allocateCommandBuffers(allocInfo);
        auto& cmd = cmdBuffers[0];

        cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline_);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout_, 0, {*descriptorSet_}, nullptr);
        uint32_t params[4] = {800, 600, frameCount_, spp};
        cmd.pushConstants<uint32_t>(*pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, params);
        cmd.dispatch((800 + 15) / 16, (600 + 15) / 16, 1);
        
        // Barrier to ensure compute writes are visible to host read
        vk::MemoryBarrier barrier{
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
            .dstAccessMask = vk::AccessFlagBits::eHostRead
        };
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, {barrier}, nullptr, nullptr);
        cmd.end();

        vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*cmd};
        renderer_.GetGraphicsQueue().submit(submitInfo);
        renderer_.GetGraphicsQueue().waitIdle();
    }

    std::vector<float> downloadImage() {
        void* data = outputMemory_.mapMemory(0, 800 * 600 * 3 * sizeof(float));
        std::vector<float> pixels(800 * 600 * 3);
        std::memcpy(pixels.data(), data, pixels.size() * sizeof(float));
        outputMemory_.unmapMemory();
        return pixels;
    }

    Renderer& renderer_;
    GLFWwindow* window_;
    std::unique_ptr<ImGuiSystem> imgui_;
    
    #ifdef HAS_ONNX_RUNTIME
    std::unique_ptr<ONNXClassifier> errorPredictor_;
    std::unique_ptr<ONNXClassifier> denoiser_;
    #endif

    vk::raii::Buffer outputBuffer_{nullptr};
    vk::raii::DeviceMemory outputMemory_{nullptr};
    vk::raii::Buffer heatmapBuffer_{nullptr};
    vk::raii::DeviceMemory heatmapMemoryBuffer_{nullptr};
    vk::raii::ShaderModule shaderModule_{nullptr};
    vk::raii::DescriptorSetLayout descriptorSetLayout_{nullptr};
    vk::raii::PipelineLayout pipelineLayout_{nullptr};
    vk::raii::Pipeline pipeline_{nullptr};
    vk::raii::DescriptorPool descriptorPool_{nullptr};
    vk::raii::DescriptorSet descriptorSet_{nullptr};

    // Display resources
    vk::raii::Image displayImage_ = nullptr;
    vk::raii::DeviceMemory displayMemory_ = nullptr;
    vk::raii::ImageView displayView_ = nullptr;
    vk::raii::Sampler displaySampler_ = nullptr;
    ImTextureID displayTexID_ = 0;

    vk::raii::Image heatmapImage_ = nullptr;
    vk::raii::DeviceMemory heatmapMemory_ = nullptr;
    vk::raii::ImageView heatmapView_ = nullptr;
    vk::raii::Sampler heatmapSampler_ = nullptr;
    ImTextureID heatmapTexID_ = 0;

    vk::raii::Image probeImage_ = nullptr;
    vk::raii::DeviceMemory probeMemory_ = nullptr;
    vk::raii::ImageView probeView_ = nullptr;
    vk::raii::Sampler probeSampler_ = nullptr;
    ImTextureID probeTexID_ = 0;

    std::vector<float> lastHeatmapData_;
    std::vector<float> lastProbePixels_;
    std::vector<float> lastDenoisedPixels_;

    bool adaptiveEnabled_ = true;
    bool denoiseEnabled_ = true;
    int viewMode_ = 0; // 0: Final, 1: Heatmap
    uint32_t frameCount_ = 0;

    float probeTime_ = 0;
    float heatmapTime_ = 0;
    float targetedTime_ = 0;
    float denoiseTime_ = 0;
    float totalTime_ = 0;
};

int main() {
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Vulkan RT Optimization Demo", nullptr, nullptr);
    if (!window) return 1;

    try {
        {
            Renderer renderer(window, 1280, 720);
            RayTracingOptimizer optimizer(renderer, window);
            optimizer.run();
        } // optimizer and renderer destroyed here, while window is still alive
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
