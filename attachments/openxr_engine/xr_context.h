#pragma once

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#if defined(_WIN32)
#define XR_USE_PLATFORM_WIN32
// openxr_platform.h with XR_USE_PLATFORM_WIN32 needs LARGE_INTEGER, IUnknown, etc.
// <windows.h> provides the base types; <unknwn.h> provides IUnknown which is
// excluded by WIN32_LEAN_AND_MEAN (defined globally in the build).
#include <windows.h>
#include <unknwn.h>
#elif defined(__linux__)
#define XR_USE_PLATFORM_XLIB
#endif
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
    XrInstance getXrInstance() const { return instance; }
    XrSystemId getSystemId() const { return systemId; }
    bool createSession(vk::PhysicalDevice physicalDevice, vk::Device device, uint32_t queueFamilyIndex, uint32_t queueIndex);
    void updateReferenceSpacePose(const XrPosef& pose);
    void cleanup();

#if defined(PLATFORM_ANDROID)
    void setAndroidApp(struct android_app* app) { androidApp = app; }
#endif

    // Core Handshake (Chapter 2)
    std::vector<const char*> getVulkanInstanceExtensions();
    std::vector<const char*> getVulkanDeviceExtensions(vk::PhysicalDevice physicalDevice);
    // The runtime hands back the exact VkPhysicalDevice it wants us to use — no LUID
    // matching required. Returns VK_NULL_HANDLE if the query fails.
    vk::PhysicalDevice getRequiredPhysicalDevice();

    // Swapchain Management (Chapter 3 & 8)
    vk::Extent2D getRecommendedExtent() const;
    void createSwapchains(vk::Device device, vk::Format format, vk::Extent2D extent);
    std::vector<vk::Image> enumerateSwapchainImages(uint32_t swapchainIdx = 0);
    vk::Extent2D getSwapchainExtent() const { return extent; }
    vk::Format getSwapchainFormat() const { return format; }

    void waitSwapchainImage();
    uint32_t acquireSwapchainImage();
    void releaseSwapchainImage();

    // Session lifecycle
    void pollEvents();
    // True when the session is running and frames should be rendered.
    bool isSessionRunning() const {
        return sessionState == XR_SESSION_STATE_SYNCHRONIZED ||
               sessionState == XR_SESSION_STATE_VISIBLE ||
               sessionState == XR_SESSION_STATE_FOCUSED;
    }
    // True when xrWaitFrame/xrBeginFrame/xrEndFrame should be called.
    // Includes READY state because some runtimes (Monado) require xrWaitFrame
    // to be called to advance from READY to SYNCHRONIZED.
    bool isSessionFraming() const {
        return sessionBegun &&
               (sessionState == XR_SESSION_STATE_READY ||
                sessionState == XR_SESSION_STATE_SYNCHRONIZED ||
                sessionState == XR_SESSION_STATE_VISIBLE ||
                sessionState == XR_SESSION_STATE_FOCUSED);
    }
    XrSessionState getSessionState() const { return sessionState; }

    // Frame Lifecycle (Chapter 5)
    XrFrameState waitFrame();
    void beginFrame();
    void endFrame(const std::array<std::vector<vk::ImageView>, 2>& eyeViews);

    // View & Projection (Chapter 4 & 11)
    void locateViews(XrTime predictedTime);
    std::vector<XrView> getLatestViews() const { return views; }
    std::array<XrPosef, 2> getLatestViewPoses() const {
        static const XrPosef identity = {{0,0,0,1},{0,0,0}};
        return {views.size() > 0 ? views[0].pose : identity,
                views.size() > 1 ? views[1].pose : identity};
    }
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
    PFN_xrGetVulkanGraphicsRequirements2KHR pfnGetVulkanGraphicsRequirements2KHR = nullptr;
    PFN_xrGetVulkanGraphicsDevice2KHR pfnGetVulkanGraphicsDevice2KHR = nullptr;

    XrInstance instance;
    vk::Instance vkInstance;
    XrSystemId systemId;
    XrSession session;
    XrSpace appSpace;
    bool sessionBegun = false;
    XrReferenceSpaceType referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;

    VkPhysicalDevice requiredPhysicalDevice = VK_NULL_HANDLE;
    bool requiredPhysicalDeviceQueried = false;

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
    XrSessionState sessionState = XR_SESSION_STATE_UNKNOWN;

    // Action system members
    XrActionSet actionSet;
    std::map<std::string, XrAction> actions;
    std::map<std::string, XrActionType> actionTypes;
    std::map<std::string, XrSpace> actionSpaces;

    // Gaze interaction member
    XrSpace gazeSpace;
    bool systemSupportsEyeGaze = false;

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
