#include "xr_context.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <sstream>
#include <glm/gtc/matrix_transform.hpp>

#if defined(PLATFORM_ANDROID)
#include <openxr/openxr_platform.h>
#include <vulkan/vulkan_android.h>
#include <game-activity/native_app_glue/android_native_app_glue.h>
#endif

XrContext::XrContext() : 
    instance(XR_NULL_HANDLE), 
    systemId(XR_NULL_SYSTEM_ID), 
    session(XR_NULL_HANDLE), 
    appSpace(XR_NULL_HANDLE),
    actionSet(XR_NULL_HANDLE),
    gazeSpace(XR_NULL_HANDLE)
{}
XrContext::~XrContext() { cleanup(); }

bool XrContext::checkRuntimeAvailable() {
    uint32_t extCount = 0;
    if (xrEnumerateInstanceExtensionProperties(nullptr, 0, &extCount, nullptr) != XR_SUCCESS) {
        return false;
    }
    std::vector<XrExtensionProperties> extensions(extCount, {XR_TYPE_EXTENSION_PROPERTIES});
    if (xrEnumerateInstanceExtensionProperties(nullptr, extCount, &extCount, extensions.data()) != XR_SUCCESS) {
        return false;
    }
    for (const auto& ext : extensions) {
        if (std::strcmp(ext.extensionName, XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME) == 0) {
            return true;
        }
    }
    return false;
}

bool XrContext::isExtensionEnabled(const char* extName) const {
    for (const auto& ext : enabledExtensions) {
        if (ext == extName) return true;
    }
    return false;
}

bool XrContext::createInstance(const std::string& appName) {
    std::cout << "XrContext: Creating OpenXR instance for " << appName << std::endl;

    uint32_t extCount = 0;
    xrEnumerateInstanceExtensionProperties(nullptr, 0, &extCount, nullptr);
    std::vector<XrExtensionProperties> availableExtensions(extCount, {XR_TYPE_EXTENSION_PROPERTIES});
    xrEnumerateInstanceExtensionProperties(nullptr, extCount, &extCount, availableExtensions.data());

    auto checkExt = [&](const char* name) {
        for (const auto& ext : availableExtensions) {
            if (std::strcmp(ext.extensionName, name) == 0) return true;
        }
        return false;
    };

    std::vector<const char*> extensions = { XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME };
    const char* optionalExtensions[] = {
        XR_MSFT_SCENE_UNDERSTANDING_EXTENSION_NAME,
        XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME,
        XR_MSFT_HAND_INTERACTION_EXTENSION_NAME
    };

    for (auto ext : optionalExtensions) {
        if (checkExt(ext)) {
            extensions.push_back(ext);
            enabledExtensions.push_back(ext);
        }
    }

#if defined(PLATFORM_ANDROID)
    if (checkExt(XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME)) {
        extensions.push_back(XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME);
        enabledExtensions.push_back(XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME);
    }
#endif

    XrInstanceCreateInfo instanceCreateInfo{XR_TYPE_INSTANCE_CREATE_INFO};
    std::strncpy(instanceCreateInfo.applicationInfo.applicationName, appName.c_str(), XR_MAX_APPLICATION_NAME_SIZE);
    instanceCreateInfo.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
    instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    instanceCreateInfo.enabledExtensionNames = extensions.data();

#if defined(PLATFORM_ANDROID)
    XrInstanceCreateInfoAndroidKHR androidCreateInfo{XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR};
    if (isExtensionEnabled(XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME)) {
        if (!androidApp) {
            std::cerr << "XrContext: androidApp not set" << std::endl;
            return false;
        }
        androidCreateInfo.applicationVM = androidApp->activity->vm;
        androidCreateInfo.applicationActivity = androidApp->activity->clazz;
        instanceCreateInfo.next = &androidCreateInfo;
    }
#endif

    if (xrCreateInstance(&instanceCreateInfo, &this->instance) != XR_SUCCESS) {
        return false;
    }

    // Load Vulkan extension functions
    xrGetInstanceProcAddr(instance, "xrGetVulkanInstanceExtensionsKHR", (PFN_xrVoidFunction*)&pfnGetVulkanInstanceExtensionsKHR);
    xrGetInstanceProcAddr(instance, "xrGetVulkanDeviceExtensionsKHR", (PFN_xrVoidFunction*)&pfnGetVulkanDeviceExtensionsKHR);
    xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsRequirementsKHR", (PFN_xrVoidFunction*)&pfnGetVulkanGraphicsRequirementsKHR);
    xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsRequirements2KHR", (PFN_xrVoidFunction*)&pfnGetVulkanGraphicsRequirements2KHR);
    xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsDeviceKHR", (PFN_xrVoidFunction*)&pfnGetVulkanGraphicsDeviceKHR);
    xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsDevice2KHR", (PFN_xrVoidFunction*)&pfnGetVulkanGraphicsDevice2KHR);

    XrSystemGetInfo systemGetInfo{XR_TYPE_SYSTEM_GET_INFO};
    systemGetInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    if (xrGetSystem(this->instance, &systemGetInfo, &this->systemId) != XR_SUCCESS) {
        return false;
    }

    return true;
}

bool XrContext::createSession(vk::PhysicalDevice physicalDevice, vk::Device device, uint32_t queueFamilyIndex, uint32_t queueIndex) {
    std::cout << "XrContext: Creating session" << std::endl;

    XrGraphicsBindingVulkanKHR graphicsBinding{XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR};
    graphicsBinding.instance = (VkInstance)this->vkInstance;
    graphicsBinding.physicalDevice = (VkPhysicalDevice)physicalDevice;
    graphicsBinding.device = (VkDevice)device;
    graphicsBinding.queueFamilyIndex = queueFamilyIndex;
    graphicsBinding.queueIndex = queueIndex;

    XrSessionCreateInfo sessionCreateInfo{XR_TYPE_SESSION_CREATE_INFO};
    sessionCreateInfo.next = &graphicsBinding;
    sessionCreateInfo.systemId = this->systemId;
    if (xrCreateSession(this->instance, &sessionCreateInfo, &this->session) != XR_SUCCESS) {
        return false;
    }

    XrReferenceSpaceCreateInfo spaceCreateInfo{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    spaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
    spaceCreateInfo.poseInReferenceSpace = {{0,0,0,1}, {0,0,0}};
    if (xrCreateReferenceSpace(this->session, &spaceCreateInfo, &this->appSpace) != XR_SUCCESS) {
        return false;
    }

    this->views.resize(2, {XR_TYPE_VIEW});
    for (uint32_t i = 0; i < 2; ++i) {
        this->views[i].pose = {{0,0,0,1}, {0,0,0}};
        this->views[i].fov = {-1, 1, 1, -1};
    }

    XrActionSetCreateInfo actionSetInfo{XR_TYPE_ACTION_SET_CREATE_INFO};
    std::strncpy(actionSetInfo.actionSetName, "main", XR_MAX_ACTION_SET_NAME_SIZE);
    std::strncpy(actionSetInfo.localizedActionSetName, "Main Actions", XR_MAX_LOCALIZED_ACTION_SET_NAME_SIZE);
    xrCreateActionSet(this->instance, &actionSetInfo, &this->actionSet);

    auto createAction = [&](const std::string& name, const std::string& localizedName, XrActionType type) {
        XrActionCreateInfo actionInfo{XR_TYPE_ACTION_CREATE_INFO};
        actionInfo.actionType = type;
        std::strncpy(actionInfo.actionName, name.c_str(), XR_MAX_ACTION_NAME_SIZE);
        std::strncpy(actionInfo.localizedActionName, localizedName.c_str(), XR_MAX_LOCALIZED_ACTION_NAME_SIZE);
        XrAction action;
        xrCreateAction(this->actionSet, &actionInfo, &action);
        this->actions[name] = action;
        this->actionTypes[name] = type;
    };

    createAction("trigger_left", "Left Trigger", XR_ACTION_TYPE_BOOLEAN_INPUT);
    createAction("trigger_right", "Right Trigger", XR_ACTION_TYPE_BOOLEAN_INPUT);
    createAction("pose_left", "Left Hand Pose", XR_ACTION_TYPE_POSE_INPUT);
    createAction("pose_right", "Right Hand Pose", XR_ACTION_TYPE_POSE_INPUT);
    createAction("grab_left", "Left Grab", XR_ACTION_TYPE_FLOAT_INPUT);
    createAction("grab_right", "Right Grab", XR_ACTION_TYPE_FLOAT_INPUT);
    createAction("Grab", "Grab", XR_ACTION_TYPE_BOOLEAN_INPUT);
    createAction("GrabPose", "Grab Pose", XR_ACTION_TYPE_POSE_INPUT);
    createAction("menu", "Menu Button", XR_ACTION_TYPE_BOOLEAN_INPUT);

    XrPath khrSimplePath;
    xrStringToPath(this->instance, "/interaction_profiles/khr/simple_controller", &khrSimplePath);
    std::vector<XrActionSuggestedBinding> bindings;
    auto addBinding = [&](const std::string& act, const char* path) {
        XrPath p;
        xrStringToPath(this->instance, path, &p);
        bindings.push_back({actions[act], p});
    };
    addBinding("trigger_left", "/user/hand/left/input/select/click");
    addBinding("trigger_right", "/user/hand/right/input/select/click");
    addBinding("pose_left", "/user/hand/left/input/grip/pose");
    addBinding("pose_right", "/user/hand/right/input/grip/pose");
    addBinding("Grab", "/user/hand/right/input/select/click");
    addBinding("GrabPose", "/user/hand/right/input/grip/pose");
    addBinding("menu", "/user/hand/left/input/menu/click");

    XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
    suggestedBindings.interactionProfile = khrSimplePath;
    suggestedBindings.suggestedBindings = bindings.data();
    suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
    xrSuggestInteractionProfileBindings(this->instance, &suggestedBindings);

    XrSessionActionSetsAttachInfo attachInfo{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
    attachInfo.countActionSets = 1;
    attachInfo.actionSets = &this->actionSet;
    xrAttachSessionActionSets(this->session, &attachInfo);

    for (const auto& actionName : {"pose_left", "pose_right", "GrabPose"}) {
        XrActionSpaceCreateInfo actionSpaceInfo{XR_TYPE_ACTION_SPACE_CREATE_INFO};
        actionSpaceInfo.action = actions[actionName];
        actionSpaceInfo.poseInActionSpace = {{0,0,0,1}, {0,0,0}};
        XrSpace space;
        xrCreateActionSpace(this->session, &actionSpaceInfo, &space);
        this->actionSpaces[actionName] = space;
    }

    if (isExtensionEnabled(XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME)) {
        XrReferenceSpaceCreateInfo gazeSpaceInfo{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
        gazeSpaceInfo.referenceSpaceType = (XrReferenceSpaceType)1000031008; // XR_REFERENCE_SPACE_TYPE_EYE_GAZE_EXT
        gazeSpaceInfo.poseInReferenceSpace = {{0,0,0,1}, {0,0,0}};
        xrCreateReferenceSpace(this->session, &gazeSpaceInfo, &this->gazeSpace);
    }

    return true;
}

void XrContext::cleanup() {
    for (auto& swapchain : swapchains) {
        xrDestroySwapchain(swapchain.handle);
    }
    swapchains.clear();

    if (gazeSpace != XR_NULL_HANDLE) {
        xrDestroySpace(gazeSpace);
        gazeSpace = XR_NULL_HANDLE;
    }

    for (auto& pair : actionSpaces) {
        xrDestroySpace(pair.second);
    }
    actionSpaces.clear();

    for (auto& pair : actions) {
        xrDestroyAction(pair.second);
    }
    actions.clear();

    if (actionSet != XR_NULL_HANDLE) {
        xrDestroyActionSet(actionSet);
        actionSet = XR_NULL_HANDLE;
    }

    if (appSpace != XR_NULL_HANDLE) {
        xrDestroySpace(appSpace);
        appSpace = XR_NULL_HANDLE;
    }

    if (session != XR_NULL_HANDLE) {
        xrDestroySession(session);
        session = XR_NULL_HANDLE;
    }

    if (instance != XR_NULL_HANDLE) {
        xrDestroyInstance(instance);
        instance = XR_NULL_HANDLE;
    }
}

std::vector<const char*> XrContext::getVulkanInstanceExtensions() {
    if (instance == XR_NULL_HANDLE) return { "XR_KHR_vulkan_enable2" };
    uint32_t size = 0;
    if (!pfnGetVulkanInstanceExtensionsKHR) return {};
    pfnGetVulkanInstanceExtensionsKHR(instance, systemId, 0, &size, nullptr);
    std::vector<char> buffer(size);
    pfnGetVulkanInstanceExtensionsKHR(instance, systemId, size, &size, buffer.data());
    
    static std::vector<std::string> extStrings;
    extStrings.clear();
    std::string extensions(buffer.data());
    std::istringstream iss(extensions);
    std::string ext;
    while (iss >> ext) extStrings.push_back(ext);

    static std::vector<const char*> extPtrs;
    extPtrs.clear();
    for (const auto& s : extStrings) extPtrs.push_back(s.c_str());
    return extPtrs;
}

std::vector<const char*> XrContext::getVulkanDeviceExtensions(vk::PhysicalDevice physicalDevice) {
    if (instance == XR_NULL_HANDLE) return { "VK_KHR_external_memory", "VK_KHR_external_semaphore" };
    uint32_t size = 0;
    if (!pfnGetVulkanDeviceExtensionsKHR) return {};
    pfnGetVulkanDeviceExtensionsKHR(instance, systemId, 0, &size, nullptr);
    std::vector<char> buffer(size);
    pfnGetVulkanDeviceExtensionsKHR(instance, systemId, size, &size, buffer.data());

    static std::vector<std::string> devExtStrings;
    devExtStrings.clear();
    std::string extensions(buffer.data());
    std::istringstream iss(extensions);
    std::string ext;
    while (iss >> ext) devExtStrings.push_back(ext);

    static std::vector<const char*> devExtPtrs;
    devExtPtrs.clear();
    for (const auto& s : devExtStrings) devExtPtrs.push_back(s.c_str());
    return devExtPtrs;
}

const uint8_t* XrContext::getRequiredLUID() {
    if (!luidValid && vkInstance && instance != XR_NULL_HANDLE && systemId != XR_NULL_SYSTEM_ID) {
        // Step 1: Call graphics requirements as mandated by spec before getting graphics device
        XrGraphicsRequirementsVulkanKHR graphicsRequirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
        if (pfnGetVulkanGraphicsRequirements2KHR) {
            pfnGetVulkanGraphicsRequirements2KHR(instance, systemId, &graphicsRequirements);
        } else if (pfnGetVulkanGraphicsRequirementsKHR) {
            pfnGetVulkanGraphicsRequirementsKHR(instance, systemId, &graphicsRequirements);
        }

        // Step 2: Get the physical device from OpenXR
        VkPhysicalDevice vkPhysicalDevice = VK_NULL_HANDLE;
        XrResult result = XR_ERROR_FUNCTION_UNSUPPORTED;

        if (pfnGetVulkanGraphicsDevice2KHR) {
            XrVulkanGraphicsDeviceGetInfoKHR getInfo{XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
            getInfo.systemId = systemId;
            getInfo.vulkanInstance = (VkInstance)vkInstance;
            result = pfnGetVulkanGraphicsDevice2KHR(instance, &getInfo, &vkPhysicalDevice);
        } else if (pfnGetVulkanGraphicsDeviceKHR) {
            result = pfnGetVulkanGraphicsDeviceKHR(instance, systemId, (VkInstance)vkInstance, &vkPhysicalDevice);
        }

        if (result == XR_SUCCESS && vkPhysicalDevice != VK_NULL_HANDLE) {
            // Step 3: Extract LUID from the physical device
            VkPhysicalDeviceIDProperties idProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            idProps.pNext = nullptr;
            VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            props2.pNext = &idProps;

            auto pfnGetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr((VkInstance)vkInstance, "vkGetPhysicalDeviceProperties2");
            if (pfnGetPhysicalDeviceProperties2) {
                pfnGetPhysicalDeviceProperties2(vkPhysicalDevice, &props2);
                if (idProps.deviceLUIDValid) {
                    std::memcpy(requiredLuid, idProps.deviceLUID, VK_LUID_SIZE);
                    luidValid = true;
                    std::cout << "XrContext: Required LUID found and stored." << std::endl;
                } else {
                    std::cout << "XrContext: Physical device LUID is not valid." << std::endl;
                }
            } else {
                std::cerr << "XrContext: Failed to load vkGetPhysicalDeviceProperties2" << std::endl;
            }
        } else {
            std::cerr << "XrContext: Failed to get Vulkan graphics device from OpenXR (XrResult=" << result << ")" << std::endl;
        }
    }
    return luidValid ? requiredLuid : nullptr;
}

vk::Extent2D XrContext::getRecommendedExtent() const {
    uint32_t count = 0;
    xrEnumerateViewConfigurationViews(instance, systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &count, nullptr);
    std::vector<XrViewConfigurationView> vcv(count, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
    xrEnumerateViewConfigurationViews(instance, systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, count, &count, vcv.data());
    if (vcv.empty()) return {1024, 1024};
    vk::Extent2D ext{vcv[0].recommendedImageRectWidth, vcv[0].recommendedImageRectHeight};
    return ext;
}

void XrContext::createSwapchains(vk::Device device, vk::Format format, vk::Extent2D extent) {
    this->format = format;
    this->extent = extent;

    XrSwapchainCreateInfo ci{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    ci.arraySize = 2;
    ci.format = (int64_t)format;
    ci.width = extent.width;
    ci.height = extent.height;
    ci.mipCount = 1;
    ci.faceCount = 1;
    ci.sampleCount = 1;
    ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;

    XrSwapchain handle;
    xrCreateSwapchain(this->session, &ci, &handle);

    uint32_t count = 0;
    xrEnumerateSwapchainImages(handle, 0, &count, nullptr);
    std::vector<XrSwapchainImageVulkanKHR> images(count, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
    xrEnumerateSwapchainImages(handle, count, &count, (XrSwapchainImageBaseHeader*)images.data());

    SwapchainData data;
    data.handle = handle;
    data.images = std::move(images);
    this->swapchains.push_back(std::move(data));
}

std::vector<vk::Image> XrContext::enumerateSwapchainImages() {
    std::vector<vk::Image> vkImages;
    if (!swapchains.empty()) {
        for (const auto& img : swapchains[0].images) {
            vkImages.push_back(vk::Image(img.image));
        }
    }
    return vkImages;
}

void XrContext::waitSwapchainImage() {
    if (swapchains.empty()) return;
    XrSwapchainImageWaitInfo wi{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wi.timeout = XR_INFINITE_DURATION;
    xrWaitSwapchainImage(swapchains[0].handle, &wi);
}

uint32_t XrContext::acquireSwapchainImage() {
    if (swapchains.empty()) return 0;
    XrSwapchainImageAcquireInfo ai{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    uint32_t index = 0;
    xrAcquireSwapchainImage(swapchains[0].handle, &ai, &index);
    return index;
}

void XrContext::releaseSwapchainImage() {
    if (swapchains.empty()) return;
    XrSwapchainImageReleaseInfo ri{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    xrReleaseSwapchainImage(swapchains[0].handle, &ri);
}

XrFrameState XrContext::waitFrame() {
    XrFrameWaitInfo wi{XR_TYPE_FRAME_WAIT_INFO};
    this->frameState = {XR_TYPE_FRAME_STATE};
    xrWaitFrame(this->session, &wi, &this->frameState);
    return this->frameState;
}

void XrContext::beginFrame() {
    XrFrameBeginInfo bi{XR_TYPE_FRAME_BEGIN_INFO};
    xrBeginFrame(this->session, &bi);
}

void XrContext::endFrame(const std::array<std::vector<vk::ImageView>, 2>& eyeViews) {
    XrFrameEndInfo ei{XR_TYPE_FRAME_END_INFO};
    ei.displayTime = this->frameState.predictedDisplayTime;
    ei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;

    XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    layer.space = this->appSpace;

    static XrCompositionLayerProjectionView projectionViews[2];
    for (uint32_t i = 0; i < 2; ++i) {
        projectionViews[i] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
        projectionViews[i].pose = this->views[i].pose;
        projectionViews[i].fov = this->views[i].fov;
        projectionViews[i].subImage.swapchain = this->swapchains[0].handle;
        projectionViews[i].subImage.imageRect = {{0, 0}, {(int32_t)extent.width, (int32_t)extent.height}};
        projectionViews[i].subImage.imageArrayIndex = i;
    }

    layer.viewCount = 2;
    layer.views = projectionViews;

    static const XrCompositionLayerBaseHeader* layers[1];
    uint32_t layerCount = 0;
    if (this->frameState.shouldRender) {
        layers[layerCount++] = (XrCompositionLayerBaseHeader*)&layer;
    }

    ei.layerCount = layerCount;
    ei.layers = layers;

    xrEndFrame(this->session, &ei);
}

void XrContext::locateViews(XrTime predictedTime) {
    XrViewLocateInfo li{XR_TYPE_VIEW_LOCATE_INFO};
    li.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    li.displayTime = predictedTime;
    li.space = this->appSpace;

    XrViewState vs{XR_TYPE_VIEW_STATE};
    uint32_t count = 0;
    xrLocateViews(this->session, &li, &vs, (uint32_t)this->views.size(), &count, this->views.data());
}

vk::Viewport XrContext::getViewport(uint32_t eye) const {
    return vk::Viewport(0, 0, (float)extent.width, (float)extent.height, 0.0f, 1.0f);
}

vk::Rect2D XrContext::getScissor(uint32_t eye) const {
    return vk::Rect2D({0, 0}, extent);
}

glm::mat4 XrContext::getProjectionMatrix(uint32_t eye) const {
    if (eye >= views.size()) return glm::mat4(1.0f);
    const auto& fov = views[eye].fov;
    float nearZ = 0.1f, farZ = 100.0f;
    float tanLeft = std::tan(fov.angleLeft), tanRight = std::tan(fov.angleRight);
    float tanDown = std::tan(fov.angleDown), tanUp = std::tan(fov.angleUp);
    float tanWidth = tanRight - tanLeft, tanHeight = tanUp - tanDown;
    glm::mat4 projection = glm::mat4(0.0f);
    projection[0][0] = 2.0f / tanWidth;
    projection[1][1] = 2.0f / tanHeight;
    projection[2][0] = (tanRight + tanLeft) / tanWidth;
    projection[2][1] = (tanUp + tanDown) / tanHeight;
    projection[2][2] = -farZ / (farZ - nearZ);
    projection[2][3] = -1.0f;
    projection[3][2] = -(farZ * nearZ) / (farZ - nearZ);
    return projection;
}

glm::mat4 XrContext::getViewMatrix(uint32_t eye) const {
    if (eye >= views.size()) return glm::mat4(1.0f);
    return glm::inverse(xrPoseToMatrix(views[eye].pose));
}

glm::vec3 XrContext::getEyePosition(uint32_t eye) const {
    if (eye >= views.size()) return glm::vec3(0.0f);
    return glm::vec3(views[eye].pose.position.x, views[eye].pose.position.y, views[eye].pose.position.z);
}

void XrContext::pollActions() {
    if (session == XR_NULL_HANDLE || actionSet == XR_NULL_HANDLE) return;
    XrActionsSyncInfo si{XR_TYPE_ACTIONS_SYNC_INFO};
    static XrActiveActionSet as;
    as.actionSet = actionSet;
    as.subactionPath = XR_NULL_PATH;
    si.activeActionSets = &as;
    si.countActiveActionSets = 1;
    xrSyncActions(session, &si);
}

bool XrContext::isActionActive(const std::string& name) const {
    if (session == XR_NULL_HANDLE || actions.find(name) == actions.end()) return false;
    XrAction action = actions.at(name);
    XrActionType type = actionTypes.at(name);
    XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
    gi.action = action;
    if (type == XR_ACTION_TYPE_BOOLEAN_INPUT) {
        XrActionStateBoolean st{XR_TYPE_ACTION_STATE_BOOLEAN};
        xrGetActionStateBoolean(session, &gi, &st);
        if (st.isActive) return st.currentState;
    } else if (type == XR_ACTION_TYPE_FLOAT_INPUT) {
        XrActionStateFloat st{XR_TYPE_ACTION_STATE_FLOAT};
        xrGetActionStateFloat(session, &gi, &st);
        if (st.isActive) return st.currentState > 0.1f;
    }
    return false;
}

XrPosef XrContext::getActionPose(const std::string& name) const {
    if (session == XR_NULL_HANDLE || actionSpaces.find(name) == actionSpaces.end()) {
        return {{0,0,0,1}, {0,0,0}};
    }
    XrAction action = actions.at(name);
    XrSpace space = actionSpaces.at(name);
    XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
    xrLocateSpace(space, appSpace, frameState.predictedDisplayTime, &loc);
    if ((loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) && (loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT)) {
        return loc.pose;
    }
    return {{0,0,0,1}, {0,0,0}};
}

std::vector<XrSpatialMesh> XrContext::getLatestSpatialMeshes() {
    return {};
}

glm::vec2 XrContext::getGazeNDC() const {
    if (gazeSpace == XR_NULL_HANDLE || views.empty()) return glm::vec2(0.5f, 0.5f);
    XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
    xrLocateSpace(gazeSpace, appSpace, frameState.predictedDisplayTime, &loc);
    if (!(loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)) return glm::vec2(0.5f, 0.5f);
    glm::mat4 gazeMat = xrPoseToMatrix(loc.pose);
    glm::vec3 gazeOrigin = glm::vec3(gazeMat[3]);
    glm::vec3 gazeDir = -glm::vec3(gazeMat[2]);
    glm::mat4 viewProj = getProjectionMatrix(0) * getViewMatrix(0);
    glm::vec4 projected = viewProj * glm::vec4(gazeOrigin + gazeDir, 1.0f);
    if (projected.w == 0.0f) return glm::vec2(0.5f, 0.5f);
    glm::vec3 ndc = glm::vec3(projected) / projected.w;
    return glm::vec2(ndc.x * 0.5f + 0.5f, ndc.y * 0.5f + 0.5f);
}
