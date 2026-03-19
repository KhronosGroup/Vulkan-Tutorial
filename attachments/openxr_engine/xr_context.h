#pragma once

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#define XR_USE_PLATFORM_XLIB
#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>
#include <string>
#include <array>
#include <map>

// Helper structure for spatial meshes (Chapter 16)
struct XrSpatialMesh {
    XrUuidMSFT meshGuid;
    std::vector<glm::vec3> vertices;
    std::vector<uint32_t> indices;
    glm::mat4 transform;
};

class XrContext {
public:
    XrContext();
    ~XrContext();

    bool createInstance(const std::string& appName);
    void setVulkanInstance(vk::Instance instance) { vkInstance = instance; }
    bool createSession(vk::PhysicalDevice physicalDevice, vk::Device device, uint32_t queueFamilyIndex, uint32_t queueIndex);
    void cleanup();

#if defined(PLATFORM_ANDROID)
    void setAndroidApp(struct android_app* app) { androidApp = app; }
#endif

    // Core Handshake (Chapter 2)
    std::vector<const char*> getVulkanInstanceExtensions();
    std::vector<const char*> getVulkanDeviceExtensions(vk::PhysicalDevice physicalDevice);
    const uint8_t* getRequiredLUID();

    // Swapchain Management (Chapter 3 & 8)
    vk::Extent2D getRecommendedExtent() const;
    void createSwapchains(vk::Device device, vk::Format format, vk::Extent2D extent);
    std::vector<vk::Image> enumerateSwapchainImages(); // Returns images with 2 layers for multiview
    vk::Extent2D getSwapchainExtent() const { return extent; }
    vk::Format getSwapchainFormat() const { return format; }

    void waitSwapchainImage();
    uint32_t acquireSwapchainImage();
    void releaseSwapchainImage();

    // Frame Lifecycle (Chapter 5)
    XrFrameState waitFrame();
    void beginFrame();
    void endFrame(const std::array<std::vector<vk::ImageView>, 2>& eyeViews);

    // View & Projection (Chapter 4 & 11)
    void locateViews(XrTime predictedTime);
    std::vector<XrView> getLatestViews() const { return views; }
    std::array<XrPosef, 2> getLatestViewPoses() const { return {views[0].pose, views[1].pose}; }
    vk::Viewport getViewport(uint32_t eye) const;
    vk::Rect2D getScissor(uint32_t eye) const;
    glm::mat4 getProjectionMatrix(uint32_t eye) const;
    glm::mat4 getViewMatrix(uint32_t eye) const;
    glm::vec3 getEyePosition(uint32_t eye) const;

    // Input Actions (Chapter 7)
    void pollActions();
    bool isActionActive(const std::string& name) const;
    XrPosef getActionPose(const std::string& name) const;

    // Scene Understanding (Chapter 16)
    std::vector<XrSpatialMesh> getLatestSpatialMeshes();

    // ML & Occlusion (Chapter 17 & 18)
    glm::vec2 getGazeNDC() const;
    XrReferenceSpaceType getReferenceSpace() const { return referenceSpaceType; }

    bool isExtensionEnabled(const char* extName) const;
    static bool checkRuntimeAvailable();

private:
    PFN_xrGetVulkanInstanceExtensionsKHR pfnGetVulkanInstanceExtensionsKHR = nullptr;
    PFN_xrGetVulkanDeviceExtensionsKHR pfnGetVulkanDeviceExtensionsKHR = nullptr;
    PFN_xrGetVulkanGraphicsRequirementsKHR pfnGetVulkanGraphicsRequirementsKHR = nullptr;
    PFN_xrGetVulkanGraphicsRequirements2KHR pfnGetVulkanGraphicsRequirements2KHR = nullptr;
    PFN_xrGetVulkanGraphicsDeviceKHR pfnGetVulkanGraphicsDeviceKHR = nullptr;
    PFN_xrGetVulkanGraphicsDevice2KHR pfnGetVulkanGraphicsDevice2KHR = nullptr;

    XrInstance instance;
    vk::Instance vkInstance;
    XrSystemId systemId;
    XrSession session;
    XrSpace appSpace;
    XrReferenceSpaceType referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;

    uint8_t requiredLuid[VK_LUID_SIZE] = {0};
    bool luidValid = false;

#if defined(PLATFORM_ANDROID)
    struct android_app* androidApp = nullptr;
#endif

    vk::Format format;
    vk::Extent2D extent;

    struct SwapchainData {
        XrSwapchain handle;
        std::vector<XrSwapchainImageVulkanKHR> images;
    };
    std::vector<SwapchainData> swapchains;

    XrFrameState frameState;
    std::vector<XrView> views;

    // Action system members
    XrActionSet actionSet;
    std::map<std::string, XrAction> actions;
    std::map<std::string, XrActionType> actionTypes;
    std::map<std::string, XrSpace> actionSpaces;

    // Gaze interaction member
    XrSpace gazeSpace;

    // Scene understanding member
    // XrSceneObserverMSFT sceneObserver;

    std::vector<std::string> enabledExtensions;
};

// Common Helper: Convert XrPosef to glm::mat4
inline glm::mat4 xrPoseToMatrix(const XrPosef& pose) {
    glm::quat q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    glm::mat4 m = glm::mat4_cast(q);
    m[3] = glm::vec4(pose.position.x, pose.position.y, pose.position.z, 1.0f);
    return m;
}
