#pragma once

#include <string>
#include <vector>
#include <memory>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vk_platform.h>

// Forward declarations
class Renderer;
class AudioSystem;
class AudioSource;
struct ImGuiContext;

/**
 * @brief Class for managing ImGui integration with Vulkan.
 *
 * This class implements the ImGui integration as described in the GUI chapter:
 * @see en/Building_a_Simple_Engine/GUI/02_imgui_setup.adoc
 */
class ImGuiSystem {
public:
    /**
     * @brief Default constructor.
     */
    ImGuiSystem();

    /**
     * @brief Destructor for proper cleanup.
     */
    ~ImGuiSystem();

    /**
     * @brief Initialize the ImGui system.
     * @param renderer Pointer to the renderer.
     * @param width The width of the window.
     * @param height The height of the window.
     * @return True if initialization was successful, false otherwise.
     */
    bool Initialize(Renderer* renderer, uint32_t width, uint32_t height);

    /**
     * @brief Clean up ImGui resources.
     */
    void Cleanup();

    /**
     * @brief Start a new ImGui frame.
     */
    void NewFrame();

    /**
     * @brief Render the ImGui frame.
     * @param commandBuffer The command buffer to record rendering commands to.
     */
    void Render(vk::raii::CommandBuffer & commandBuffer);

    /**
     * @brief Handle mouse input.
     * @param x The x-coordinate of the mouse.
     * @param y The y-coordinate of the mouse.
     * @param buttons The state of the mouse buttons.
     */
    void HandleMouse(float x, float y, uint32_t buttons);

    /**
     * @brief Handle keyboard input.
     * @param key The key code.
     * @param pressed Whether the key was pressed or released.
     */
    void HandleKeyboard(uint32_t key, bool pressed);

    /**
     * @brief Handle character input.
     * @param c The character.
     */
    void HandleChar(uint32_t c);

    /**
     * @brief Handle window resize.
     * @param width The new width of the window.
     * @param height The new height of the window.
     */
    void HandleResize(uint32_t width, uint32_t height);

    /**
     * @brief Check if ImGui wants to capture keyboard input.
     * @return True if ImGui wants to capture keyboard input, false otherwise.
     */
    bool WantCaptureKeyboard() const;

    /**
     * @brief Check if ImGui wants to capture mouse input.
     * @return True if ImGui wants to capture mouse input, false otherwise.
     */
    bool WantCaptureMouse() const;

    /**
     * @brief Set the audio system reference for audio controls.
     * @param audioSystem Pointer to the audio system.
     */
    void SetAudioSystem(AudioSystem* audioSystem);

    /**
     * @brief Get the current PBR rendering state.
     * @return True if PBR rendering is enabled, false otherwise.
     */
    bool IsPBREnabled() const { return pbrEnabled; }

private:
    // ImGui context
    ImGuiContext* context = nullptr;

    // Renderer reference
    Renderer* renderer = nullptr;

    // Audio system reference
    AudioSystem* audioSystem = nullptr;
    AudioSource* audioSource = nullptr;
    AudioSource* debugPingSource = nullptr;

    // Audio position tracking
    float audioSourceX = 1.0f;
    float audioSourceY = 0.0f;
    float audioSourceZ = 0.0f;

    // Vulkan resources
    vk::raii::DescriptorPool descriptorPool = nullptr;
    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::DescriptorSet descriptorSet = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline pipeline = nullptr;
    vk::raii::Sampler fontSampler = nullptr;
    vk::raii::Image fontImage = nullptr;
    vk::raii::DeviceMemory fontMemory = nullptr;
    vk::raii::ImageView fontView = nullptr;
    vk::raii::Buffer vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;
    vk::raii::Buffer indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;

    // Window dimensions
    uint32_t width = 0;
    uint32_t height = 0;

    // Mouse state
    float mouseX = 0.0f;
    float mouseY = 0.0f;
    uint32_t mouseButtons = 0;

    // Initialization flag
    bool initialized = false;

    // PBR rendering state
    bool pbrEnabled = false;

    /**
     * @brief Create Vulkan resources for ImGui.
     * @return True if creation was successful, false otherwise.
     */
    bool createResources();

    /**
     * @brief Create font texture.
     * @return True if creation was successful, false otherwise.
     */
    bool createFontTexture();

    /**
     * @brief Create descriptor set layout.
     * @return True if creation was successful, false otherwise.
     */
    bool createDescriptorSetLayout();

    /**
     * @brief Create descriptor pool.
     * @return True if creation was successful, false otherwise.
     */
    bool createDescriptorPool();

    /**
     * @brief Create descriptor set.
     * @return True if creation was successful, false otherwise.
     */
    bool createDescriptorSet();

    /**
     * @brief Create pipeline layout.
     * @return True if creation was successful, false otherwise.
     */
    bool createPipelineLayout();

    /**
     * @brief Create pipeline.
     * @return True if creation was successful, false otherwise.
     */
    bool createPipeline();

    /**
     * @brief Update vertex and index buffers.
     */
    void updateBuffers();
};
