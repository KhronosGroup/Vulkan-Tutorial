#include "renderer.h"
#include <fstream>
#include <array>
#include <iostream>
#include "mesh_component.h"

// This file contains pipeline-related methods from the Renderer class

// Create a descriptor set layout
bool Renderer::createDescriptorSetLayout() {
    try {
        // Create binding for a uniform buffer
        vk::DescriptorSetLayoutBinding uboLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            .pImmutableSamplers = nullptr
        };

        // Create binding for texture sampler
        vk::DescriptorSetLayoutBinding samplerLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .pImmutableSamplers = nullptr
        };

        // Create a descriptor set layout
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()
        };

        descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

// Create PBR descriptor set layout
bool Renderer::createPBRDescriptorSetLayout() {
    try {
        // Create descriptor set layout bindings for PBR shader
        std::array bindings = {
            // Binding 0: Uniform buffer (UBO)
            vk::DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                .pImmutableSamplers = nullptr
            },
            // Binding 1: Base color map and sampler
            vk::DescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
                .pImmutableSamplers = nullptr
            },
            // Binding 2: Metallic roughness map and sampler
            vk::DescriptorSetLayoutBinding{
                .binding = 2,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
                .pImmutableSamplers = nullptr
            },
            // Binding 3: Normal map and sampler
            vk::DescriptorSetLayoutBinding{
                .binding = 3,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
                .pImmutableSamplers = nullptr
            },
            // Binding 4: Occlusion map and sampler
            vk::DescriptorSetLayoutBinding{
                .binding = 4,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
                .pImmutableSamplers = nullptr
            },
            // Binding 5: Emissive map and sampler
            vk::DescriptorSetLayoutBinding{
                .binding = 5,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
                .pImmutableSamplers = nullptr
            },
            // Binding 6: Light storage buffer (shadows removed)
            vk::DescriptorSetLayoutBinding{
                .binding = 6,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
                .pImmutableSamplers = nullptr
            }
        };

        // Create a descriptor set layout
        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()
        };

        pbrDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create PBR descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

// Create a graphics pipeline
bool Renderer::createGraphicsPipeline() {
    try {
        // Read shader code
        auto shaderCode = readFile("shaders/texturedMesh.spv");

        // Create shader modules
        vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

        // Create shader stage info
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = *shaderModule,
            .pName = "VSMain"
        };

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = *shaderModule,
            .pName = "PSMain"
        };

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        // Create vertex input info with instancing support
        auto vertexBindingDescription = Vertex::getBindingDescription();
        auto instanceBindingDescription = InstanceData::getBindingDescription();
        std::array<vk::VertexInputBindingDescription, 2> bindingDescriptions = {
            vertexBindingDescription,
            instanceBindingDescription
        };

        auto vertexAttributeDescriptions = Vertex::getAttributeDescriptions();
        auto instanceAttributeDescriptions = InstanceData::getAttributeDescriptions();

        // Combine all attribute descriptions (no duplicates)
        std::vector<vk::VertexInputAttributeDescription> allAttributeDescriptions;
        allAttributeDescriptions.insert(allAttributeDescriptions.end(), vertexAttributeDescriptions.begin(), vertexAttributeDescriptions.end());
        allAttributeDescriptions.insert(allAttributeDescriptions.end(), instanceAttributeDescriptions.begin(), instanceAttributeDescriptions.end());

        // Note: materialIndex attribute (Location 11) is not used by current shaders
        // Removed to fix validation layer error - shaders don't expect input at location 11

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size()),
            .pVertexBindingDescriptions = bindingDescriptions.data(),
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(allAttributeDescriptions.size()),
            .pVertexAttributeDescriptions = allAttributeDescriptions.data()
        };

        // Create input assembly info
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = VK_FALSE
        };

        // Create viewport state info
        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .scissorCount = 1
        };

        // Create rasterization state info
        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f
        };

        // Create multisample state info
        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = VK_FALSE
        };

        // Create depth stencil state info
        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE
        };

        // Create a color blend attachment state
        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = VK_FALSE,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        // Create color blend state info
        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = VK_FALSE,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment
        };

        // Create dynamic state info
        std::vector dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };

        vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };

        // Create pipeline layout
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*descriptorSetLayout,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr
        };

        pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        // Create pipeline rendering info
        vk::Format depthFormat = findDepthFormat();
        std::cout << "Creating main graphics pipeline with depth format: " << static_cast<int>(depthFormat) << std::endl;

        // Initialize member variable for proper lifetime management
        mainPipelineRenderingCreateInfo = vk::PipelineRenderingCreateInfo{
            .sType = vk::StructureType::ePipelineRenderingCreateInfo,
            .pNext = nullptr,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &swapChainImageFormat,
            .depthAttachmentFormat = depthFormat,
            .stencilAttachmentFormat = vk::Format::eUndefined
        };

        // Create the graphics pipeline
        vk::GraphicsPipelineCreateInfo pipelineInfo{
            .sType = vk::StructureType::eGraphicsPipelineCreateInfo,
            .pNext = &mainPipelineRenderingCreateInfo,
            .flags = vk::PipelineCreateFlags{},
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = *pipelineLayout,
            .renderPass = nullptr,
            .subpass = 0,
            .basePipelineHandle = nullptr,
            .basePipelineIndex = -1
        };

        graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create graphics pipeline: " << e.what() << std::endl;
        return false;
    }
}

// Create PBR pipeline
bool Renderer::createPBRPipeline() {
    try {
        // Create PBR descriptor set layout
        if (!createPBRDescriptorSetLayout()) {
            return false;
        }

        // Read shader code
        auto shaderCode = readFile("shaders/pbr.spv");

        // Create shader modules
        vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

        // Create shader stage info
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = *shaderModule,
            .pName = "VSMain"
        };

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = *shaderModule,
            .pName = "PSMain"
        };

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        // Define vertex and instance binding descriptions
        auto vertexBindingDescription = Vertex::getBindingDescription();
        auto instanceBindingDescription = InstanceData::getBindingDescription();
        std::array<vk::VertexInputBindingDescription, 2> bindingDescriptions = {
            vertexBindingDescription,
            instanceBindingDescription
        };

        // Define vertex and instance attribute descriptions
        auto vertexAttributeDescriptions = Vertex::getAttributeDescriptions();
        auto instanceModelMatrixAttributes = InstanceData::getModelMatrixAttributeDescriptions();
        auto instanceNormalMatrixAttributes = InstanceData::getNormalMatrixAttributeDescriptions();

        // Combine all attribute descriptions
        std::vector<vk::VertexInputAttributeDescription> allAttributeDescriptions;
        allAttributeDescriptions.insert(allAttributeDescriptions.end(), vertexAttributeDescriptions.begin(), vertexAttributeDescriptions.end());
        allAttributeDescriptions.insert(allAttributeDescriptions.end(), instanceModelMatrixAttributes.begin(), instanceModelMatrixAttributes.end());
        allAttributeDescriptions.insert(allAttributeDescriptions.end(), instanceNormalMatrixAttributes.begin(), instanceNormalMatrixAttributes.end());

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size()),
            .pVertexBindingDescriptions = bindingDescriptions.data(),
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(allAttributeDescriptions.size()),
            .pVertexAttributeDescriptions = allAttributeDescriptions.data()
        };

        // Create input assembly info
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = VK_FALSE
        };

        // Create viewport state info
        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .scissorCount = 1
        };

        // Create rasterization state info
        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f
        };

        // Create multisample state info
        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = VK_FALSE
        };

        // Create depth stencil state info
        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE
        };

        // Create a color blend attachment state
        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = VK_FALSE,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        // Create color blend state info
        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = VK_FALSE,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment
        };

        // Create dynamic state info
        std::vector dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };

        vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };

        // Create push constant range for material properties
        vk::PushConstantRange pushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .offset = 0,
            .size = sizeof(MaterialProperties)
        };

        // Create a pipeline layout using the PBR descriptor set layout
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*pbrDescriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange
        };

        pbrPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        // Create pipeline rendering info
        vk::Format depthFormat = findDepthFormat();

        // Initialize member variable for proper lifetime management
        pbrPipelineRenderingCreateInfo = vk::PipelineRenderingCreateInfo{
            .sType = vk::StructureType::ePipelineRenderingCreateInfo,
            .pNext = nullptr,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &swapChainImageFormat,
            .depthAttachmentFormat = depthFormat,
            .stencilAttachmentFormat = vk::Format::eUndefined
        };

        // 1) Opaque PBR pipeline (no blending, depth writes enabled)
        vk::PipelineColorBlendAttachmentState opaqueBlendAttachment = colorBlendAttachment;
        opaqueBlendAttachment.blendEnable = VK_FALSE;
        vk::PipelineColorBlendStateCreateInfo colorBlendingOpaque{
            .logicOpEnable = VK_FALSE,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &opaqueBlendAttachment
        };
        vk::PipelineDepthStencilStateCreateInfo depthStencilOpaque = depthStencil;
        depthStencilOpaque.depthWriteEnable = VK_TRUE;

        vk::GraphicsPipelineCreateInfo opaquePipelineInfo{
            .sType = vk::StructureType::eGraphicsPipelineCreateInfo,
            .pNext = &pbrPipelineRenderingCreateInfo,
            .flags = vk::PipelineCreateFlags{},
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencilOpaque,
            .pColorBlendState = &colorBlendingOpaque,
            .pDynamicState = &dynamicState,
            .layout = *pbrPipelineLayout,
            .renderPass = nullptr,
            .subpass = 0,
            .basePipelineHandle = nullptr,
            .basePipelineIndex = -1
        };
        pbrGraphicsPipeline = vk::raii::Pipeline(device, nullptr, opaquePipelineInfo);

        // 2) Blended PBR pipeline (alpha blending, depth writes disabled for translucency)
        vk::PipelineColorBlendAttachmentState blendedAttachment = colorBlendAttachment;
        blendedAttachment.blendEnable = VK_TRUE;
        blendedAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
        blendedAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        blendedAttachment.colorBlendOp = vk::BlendOp::eAdd;
        blendedAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
        blendedAttachment.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        blendedAttachment.alphaBlendOp = vk::BlendOp::eAdd;
        vk::PipelineColorBlendStateCreateInfo colorBlendingBlended{
            .logicOpEnable = VK_FALSE,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &blendedAttachment
        };
        vk::PipelineDepthStencilStateCreateInfo depthStencilBlended = depthStencil;
        depthStencilBlended.depthWriteEnable = VK_FALSE;

        vk::GraphicsPipelineCreateInfo blendedPipelineInfo{
            .sType = vk::StructureType::eGraphicsPipelineCreateInfo,
            .pNext = &pbrPipelineRenderingCreateInfo,
            .flags = vk::PipelineCreateFlags{},
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencilBlended,
            .pColorBlendState = &colorBlendingBlended,
            .pDynamicState = &dynamicState,
            .layout = *pbrPipelineLayout,
            .renderPass = nullptr,
            .subpass = 0,
            .basePipelineHandle = nullptr,
            .basePipelineIndex = -1
        };
        pbrBlendGraphicsPipeline = vk::raii::Pipeline(device, nullptr, blendedPipelineInfo);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create PBR pipeline: " << e.what() << std::endl;
        return false;
    }
}

// Create a lighting pipeline
bool Renderer::createLightingPipeline() {
    try {
        // Read shader code
        auto shaderCode = readFile("shaders/lighting.spv");

        // Create shader modules
        vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

        // Create shader stage info
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = *shaderModule,
            .pName = "VSMain"
        };

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = *shaderModule,
            .pName = "PSMain"
        };

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        // Create vertex input info
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data()
        };

        // Create input assembly info
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = VK_FALSE
        };

        // Create viewport state info
        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .scissorCount = 1
        };

        // Create rasterization state info
        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f
        };

        // Create multisample state info
        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = VK_FALSE
        };

        // Create depth stencil state info
        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE
        };

        // Create a color blend attachment state
        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = VK_TRUE,
            .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
            .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        // Create color blend state info
        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = VK_FALSE,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment
        };

        // Create dynamic state info
        std::vector dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };

        vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };

        // Create push constant range for material properties
        vk::PushConstantRange pushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .offset = 0,
            .size = sizeof(MaterialProperties)
        };

        // Create pipeline layout
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*descriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange
        };

        lightingPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        // Create pipeline rendering info
        vk::Format depthFormat = findDepthFormat();

        // Initialize member variable for proper lifetime management
        lightingPipelineRenderingCreateInfo = vk::PipelineRenderingCreateInfo{
            .sType = vk::StructureType::ePipelineRenderingCreateInfo,
            .pNext = nullptr,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &swapChainImageFormat,
            .depthAttachmentFormat = depthFormat,
            .stencilAttachmentFormat = vk::Format::eUndefined
        };

        // Create a graphics pipeline
        vk::GraphicsPipelineCreateInfo pipelineInfo{
            .sType = vk::StructureType::eGraphicsPipelineCreateInfo,
            .pNext = &lightingPipelineRenderingCreateInfo,
            .flags = vk::PipelineCreateFlags{},
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = *lightingPipelineLayout,
            .renderPass = nullptr,
            .subpass = 0,
            .basePipelineHandle = nullptr,
            .basePipelineIndex = -1
        };

        lightingPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create lighting pipeline: " << e.what() << std::endl;
        return false;
    }
}

// Push material properties to the pipeline
void Renderer::pushMaterialProperties(vk::CommandBuffer commandBuffer, const MaterialProperties& material) const {
    commandBuffer.pushConstants(*pbrPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(MaterialProperties), &material);
}
