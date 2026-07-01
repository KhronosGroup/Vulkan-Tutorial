/**
 * Minimal ImGui System implementation for MNIST
 */
#include "imgui_system.h"
#include "renderer.h"
#include "imgui.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <cstring>

ImGuiSystem::ImGuiSystem(Renderer* renderer, GLFWwindow* window, uint32_t width, uint32_t height)
    : renderer_(renderer), window_(window), width_(width), height_(height) {
    if (!initialize()) {
        throw std::runtime_error("Failed to initialize ImGui system");
    }
}

ImGuiSystem::~ImGuiSystem() {
    if (context_) {
        ImGui::DestroyContext(context_);
    }
}

bool ImGuiSystem::initialize() {
    // Create ImGui context
    context_ = ImGui::CreateContext();
    ImGui::SetCurrentContext(context_);

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(width_), static_cast<float>(height_));
    io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

    // Setup style
    ImGui::StyleColorsDark();

    // Initialize buffers for each frame in flight
    uint32_t maxFrames = renderer_->GetMaxFramesInFlight();
    vertexBuffers_.reserve(maxFrames);
    vertexMemories_.reserve(maxFrames);
    indexBuffers_.reserve(maxFrames);
    indexMemories_.reserve(maxFrames);

    for (uint32_t i = 0; i < maxFrames; i++) {
        vertexBuffers_.emplace_back(nullptr);
        vertexMemories_.emplace_back(nullptr);
        indexBuffers_.emplace_back(nullptr);
        indexMemories_.emplace_back(nullptr);
    }

    // Create Vulkan resources
    if (!createFontTexture()) return false;
    if (!createDescriptorSetLayout()) return false;
    if (!createDescriptorPool()) return false;
    if (!createDescriptorSet()) return false;
    if (!createPipelineLayout()) return false;
    if (!createPipeline()) return false;

    return true;
}

bool ImGuiSystem::createFontTexture() {
    ImGuiIO& io = ImGui::GetIO();
    unsigned char* pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    VkDeviceSize uploadSize = width * height * 4 * sizeof(char);

    // Create font image
    vk::ImageCreateInfo imageInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst
    };

    fontImage_ = vk::raii::Image(renderer_->GetRaiiDevice(), imageInfo);

    auto memRequirements = fontImage_.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = renderer_->FindMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
    };

    fontMemory_ = vk::raii::DeviceMemory(renderer_->GetRaiiDevice(), allocInfo);
    fontImage_.bindMemory(*fontMemory_, 0);

    // Create staging buffer
    vk::BufferCreateInfo bufferInfo{
        .size = uploadSize,
        .usage = vk::BufferUsageFlagBits::eTransferSrc
    };
    vk::raii::Buffer stagingBuffer(renderer_->GetRaiiDevice(), bufferInfo);

    auto bufferMemReq = stagingBuffer.getMemoryRequirements();
    vk::MemoryAllocateInfo bufferAllocInfo{
        .allocationSize = bufferMemReq.size,
        .memoryTypeIndex = renderer_->FindMemoryType(bufferMemReq.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
    };

    vk::raii::DeviceMemory stagingMemory(renderer_->GetRaiiDevice(), bufferAllocInfo);
    stagingBuffer.bindMemory(*stagingMemory, 0);

    void* data = stagingMemory.mapMemory(0, uploadSize);
    std::memcpy(data, pixels, uploadSize);
    stagingMemory.unmapMemory();

    // Transition and copy
    renderer_->TransitionImageLayout(*fontImage_, vk::Format::eR8G8B8A8Unorm,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    renderer_->CopyBufferToImage(*stagingBuffer, *fontImage_, width, height);
    renderer_->TransitionImageLayout(*fontImage_, vk::Format::eR8G8B8A8Unorm,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    // Create image view
    vk::ImageViewCreateInfo viewInfo{
        .image = *fontImage_,
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
    };
    fontView_ = vk::raii::ImageView(renderer_->GetRaiiDevice(), viewInfo);

    // Create sampler
    vk::SamplerCreateInfo samplerInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eRepeat,
        .addressModeV = vk::SamplerAddressMode::eRepeat,
        .addressModeW = vk::SamplerAddressMode::eRepeat
    };
    fontSampler_ = vk::raii::Sampler(renderer_->GetRaiiDevice(), samplerInfo);

    VkImage vkImage = *fontImage_;
    io.Fonts->TexID = (ImTextureID)(uintptr_t)vkImage;

    return true;
}

bool ImGuiSystem::createDescriptorSetLayout() {
    vk::DescriptorSetLayoutBinding binding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .bindingCount = 1,
        .pBindings = &binding
    };
    descriptorSetLayout_ = vk::raii::DescriptorSetLayout(renderer_->GetRaiiDevice(), layoutInfo);

    return true;
}

bool ImGuiSystem::createDescriptorPool() {
    vk::DescriptorPoolSize poolSize{
        .type = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 32
    };
    vk::DescriptorPoolCreateInfo poolInfo{
        .maxSets = 32,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize
    };
    descriptorPool_ = vk::raii::DescriptorPool(renderer_->GetRaiiDevice(), poolInfo);

    return true;
}

bool ImGuiSystem::createDescriptorSet() {
    vk::DescriptorSetLayout layouts[] = {*descriptorSetLayout_};
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *descriptorPool_,
        .descriptorSetCount = 1,
        .pSetLayouts = layouts
    };
    auto sets = vk::raii::DescriptorSets(renderer_->GetRaiiDevice(), allocInfo);
    descriptorSet_ = std::move(sets[0]);

    vk::DescriptorImageInfo imageInfo{
        .sampler = *fontSampler_,
        .imageView = *fontView_,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
    };
    vk::WriteDescriptorSet write{
        .dstSet = *descriptorSet_,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &imageInfo
    };
    renderer_->GetRaiiDevice().updateDescriptorSets(write, nullptr);

    return true;
}

bool ImGuiSystem::createPipelineLayout() {
    vk::PushConstantRange pushConstant{
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
        .offset = 0,
        .size = sizeof(float) * 4
    };
    vk::DescriptorSetLayout layouts[] = {*descriptorSetLayout_};
    vk::PipelineLayoutCreateInfo layoutInfo{
        .setLayoutCount = 1,
        .pSetLayouts = layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstant
    };
    pipelineLayout_ = vk::raii::PipelineLayout(renderer_->GetRaiiDevice(), layoutInfo);

    return true;
}

bool ImGuiSystem::createPipeline() {
    // Load shaders (compiled from Slang)
    auto vertShader = renderer_->CreateShaderModule("shaders/imgui.vert.spv");
    auto fragShader = renderer_->CreateShaderModule("shaders/imgui.frag.spv");

    vk::PipelineShaderStageCreateInfo shaderStages[] = {
        {.stage = vk::ShaderStageFlagBits::eVertex, .module = *vertShader, .pName = "main"},
        {.stage = vk::ShaderStageFlagBits::eFragment, .module = *fragShader, .pName = "main"}
    };

    // Vertex input
    vk::VertexInputBindingDescription binding{
        .binding = 0,
        .stride = sizeof(ImDrawVert),
        .inputRate = vk::VertexInputRate::eVertex
    };
    vk::VertexInputAttributeDescription attributes[] = {
        {.location = 0, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(ImDrawVert, pos)},
        {.location = 1, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(ImDrawVert, uv)},
        {.location = 2, .binding = 0, .format = vk::Format::eA8B8G8R8UnormPack32, .offset = offsetof(ImDrawVert, col)}
    };

    vk::PipelineVertexInputStateCreateInfo vertexInput{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 3,
        .pVertexAttributeDescriptions = attributes
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList
    };

    vk::Viewport viewport{.x = 0, .y = 0, .width = static_cast<float>(width_), .height = static_cast<float>(height_), .minDepth = 0.0f, .maxDepth = 1.0f};
    vk::Rect2D scissor{.offset = {0, 0}, .extent = {width_, height_}};
    vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor
    };

    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = false,
        .lineWidth = 1.0f
    };

    vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = vk::SampleCountFlagBits::e1
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = true,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                         vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = false,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment
    };

    vk::DynamicState dynamicStates[] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = 2,
        .pDynamicStates = dynamicStates
    };

    vk::GraphicsPipelineCreateInfo pipelineInfo{
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = *pipelineLayout_,
        .renderPass = renderer_->GetRenderPass(),
        .subpass = 0
    };

    pipeline_ = vk::raii::Pipeline(renderer_->GetRaiiDevice(), nullptr, pipelineInfo);

    return true;
}

void ImGuiSystem::NewFrame() {
    ImGui::SetCurrentContext(context_);
    ImGuiIO& io = ImGui::GetIO();

    // Update display size
    io.DisplaySize = ImVec2(static_cast<float>(width_), static_cast<float>(height_));
    io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
    io.DeltaTime = 1.0f / 60.0f;

    // Update mouse position
    if (window_) {
        double mouse_x, mouse_y;
        glfwGetCursorPos(window_, &mouse_x, &mouse_y);
        io.MousePos = ImVec2(static_cast<float>(mouse_x), static_cast<float>(mouse_y));

        // Update mouse buttons
        io.MouseDown[0] = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        io.MouseDown[1] = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        io.MouseDown[2] = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    } else {
        io.MousePos = ImVec2(-FLT_MAX, -FLT_MAX);
        io.MouseDown[0] = io.MouseDown[1] = io.MouseDown[2] = false;
    }

    ImGui::NewFrame();
}

ImTextureID ImGuiSystem::AddTexture(vk::raii::ImageView const& imageView, vk::raii::Sampler const& sampler) {
    vk::DescriptorSetLayout layouts[] = {*descriptorSetLayout_};
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *descriptorPool_,
        .descriptorSetCount = 1,
        .pSetLayouts = layouts
    };
    auto sets = vk::raii::DescriptorSets(renderer_->GetRaiiDevice(), allocInfo);
    vk::raii::DescriptorSet set = std::move(sets[0]);

    vk::DescriptorImageInfo imageInfo{
        .sampler = *sampler,
        .imageView = *imageView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
    };

    vk::WriteDescriptorSet write{
        .dstSet = *set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &imageInfo
    };

    renderer_->GetRaiiDevice().updateDescriptorSets(write, nullptr);

    ImTextureID id = reinterpret_cast<ImTextureID>(static_cast<uintptr_t>(textureSets_.size() + 1)); // Avoid 0
    textureSets_.push_back(std::move(set));
    textureIds_.push_back(id);

    return id;
}

void ImGuiSystem::UpdateDisplaySize(uint32_t width, uint32_t height) {
    width_ = width;
    height_ = height;
    if (context_) {
        ImGui::SetCurrentContext(context_);
        ImGui::GetIO().DisplaySize = ImVec2(static_cast<float>(width), static_cast<float>(height));
    }
}

void ImGuiSystem::Render(vk::raii::CommandBuffer& cmd, uint32_t frameIndex) {
    ImGui::SetCurrentContext(context_);
    ImGui::Render();

    ImDrawData* drawData = ImGui::GetDrawData();
    if (!drawData || drawData->TotalVtxCount == 0) return;

    updateBuffers(frameIndex);

    // Render
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline_);
    
    ImTextureID currentTexture = 0;
    // Bind font texture by default
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout_, 0, *descriptorSet_, nullptr);
    currentTexture = ImGui::GetIO().Fonts->TexID;

    vk::Viewport viewport(0, 0, static_cast<float>(width_), static_cast<float>(height_), 0.0f, 1.0f);
    cmd.setViewport(0, viewport);

    ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    ImVec2 displayPos(0, 0);
    float scale[2] = {2.0f / displaySize.x, 2.0f / displaySize.y};
    float translate[2] = {-1.0f - displayPos.x * scale[0], -1.0f - displayPos.y * scale[1]};
    cmd.pushConstants<float>(*pipelineLayout_, vk::ShaderStageFlagBits::eVertex, 0, {scale[0], scale[1], translate[0], translate[1]});

    vk::DeviceSize offset = 0;
    cmd.bindVertexBuffers(0, *vertexBuffers_[frameIndex], offset);
    cmd.bindIndexBuffer(*indexBuffers_[frameIndex], 0, sizeof(ImDrawIdx) == 2 ? vk::IndexType::eUint16 : vk::IndexType::eUint32);

    uint32_t vtxOffset = 0, idxOffset = 0;
    for (int n = 0; n < drawData->CmdListsCount; n++) {
        const ImDrawList* cmdList = drawData->CmdLists[n];
        for (int cmdI = 0; cmdI < cmdList->CmdBuffer.Size; cmdI++) {
            const ImDrawCmd* pcmd = &cmdList->CmdBuffer[cmdI];

            // Texture switching
            ImTextureID texID = pcmd->GetTexID();
            if (texID != currentTexture) {
                currentTexture = texID;
                if (currentTexture == ImGui::GetIO().Fonts->TexID) {
                    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout_, 0, *descriptorSet_, nullptr);
                } else {
                    // Find our registered texture set
                    for (size_t i = 0; i < textureIds_.size(); i++) {
                        if (textureIds_[i] == currentTexture) {
                            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout_, 0, *textureSets_[i], nullptr);
                            break;
                        }
                    }
                }
            }

            vk::Rect2D scissor(
                {std::max(0, static_cast<int32_t>(pcmd->ClipRect.x)), std::max(0, static_cast<int32_t>(pcmd->ClipRect.y))},
                {static_cast<uint32_t>(pcmd->ClipRect.z - pcmd->ClipRect.x), static_cast<uint32_t>(pcmd->ClipRect.w - pcmd->ClipRect.y)}
            );
            cmd.setScissor(0, scissor);
            cmd.drawIndexed(pcmd->ElemCount, 1, idxOffset, vtxOffset, 0);
        }
        idxOffset += cmdList->IdxBuffer.Size;
        vtxOffset += cmdList->VtxBuffer.Size;
    }
}

void ImGuiSystem::updateBuffers(uint32_t frameIndex) {
    ImDrawData* drawData = ImGui::GetDrawData();

    VkDeviceSize vtxSize = drawData->TotalVtxCount * sizeof(ImDrawVert);
    VkDeviceSize idxSize = drawData->TotalIdxCount * sizeof(ImDrawIdx);

    // Create or resize vertex buffer
    bool needNewVertexBuffer = !*vertexBuffers_[frameIndex];
    if (!needNewVertexBuffer) {
        needNewVertexBuffer = vertexBuffers_[frameIndex].getMemoryRequirements().size < vtxSize;
    }
    if (needNewVertexBuffer) {
        vk::BufferCreateInfo bufferInfo{
            .size = vtxSize,
            .usage = vk::BufferUsageFlagBits::eVertexBuffer
        };
        vertexBuffers_[frameIndex] = vk::raii::Buffer(renderer_->GetRaiiDevice(), bufferInfo);

        auto memReq = vertexBuffers_[frameIndex].getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memReq.size,
            .memoryTypeIndex = renderer_->FindMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
        };
        vertexMemories_[frameIndex] = vk::raii::DeviceMemory(renderer_->GetRaiiDevice(), allocInfo);
        vertexBuffers_[frameIndex].bindMemory(*vertexMemories_[frameIndex], 0);
    }

    // Create or resize index buffer
    bool needNewIndexBuffer = !*indexBuffers_[frameIndex];
    if (!needNewIndexBuffer) {
        needNewIndexBuffer = indexBuffers_[frameIndex].getMemoryRequirements().size < idxSize;
    }
    if (needNewIndexBuffer) {
        vk::BufferCreateInfo bufferInfo{
            .size = idxSize,
            .usage = vk::BufferUsageFlagBits::eIndexBuffer
        };
        indexBuffers_[frameIndex] = vk::raii::Buffer(renderer_->GetRaiiDevice(), bufferInfo);

        auto memReq = indexBuffers_[frameIndex].getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memReq.size,
            .memoryTypeIndex = renderer_->FindMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
        };
        indexMemories_[frameIndex] = vk::raii::DeviceMemory(renderer_->GetRaiiDevice(), allocInfo);
        indexBuffers_[frameIndex].bindMemory(*indexMemories_[frameIndex], 0);
    }

    // Upload data
    ImDrawVert* vtxDst = static_cast<ImDrawVert*>(vertexMemories_[frameIndex].mapMemory(0, vtxSize));
    ImDrawIdx* idxDst = static_cast<ImDrawIdx*>(indexMemories_[frameIndex].mapMemory(0, idxSize));

    for (int n = 0; n < drawData->CmdListsCount; n++) {
        const ImDrawList* cmdList = drawData->CmdLists[n];
        std::memcpy(vtxDst, cmdList->VtxBuffer.Data, cmdList->VtxBuffer.Size * sizeof(ImDrawVert));
        std::memcpy(idxDst, cmdList->IdxBuffer.Data, cmdList->IdxBuffer.Size * sizeof(ImDrawIdx));
        vtxDst += cmdList->VtxBuffer.Size;
        idxDst += cmdList->IdxBuffer.Size;
    }

    vertexMemories_[frameIndex].unmapMemory();
    indexMemories_[frameIndex].unmapMemory();
}
