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

XrContext::XrContext() {}
XrContext::~XrContext() { cleanup(); }

bool XrContext::checkRuntimeAvailable() {
    try {
        std::vector<xr::ExtensionProperties> extensions = xr::enumerateInstanceExtensionProperties(nullptr);
        for (const auto& ext : extensions) {
            if (std::strcmp(ext.extensionName, XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME) == 0) {
                return true;
            }
        }
    } catch (...) {
        return false;
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

    try {
        // Query available extensions
        std::vector<xr::ExtensionProperties> availableExtensions = xr::enumerateInstanceExtensionProperties(nullptr);
        auto checkExt = [&](const char* name) {
            for (const auto& ext : availableExtensions) {
                if (std::strcmp(ext.extensionName, name) == 0) return true;
            }
            return false;
        };

        // 1. Create OpenXR Instance
        std::vector<const char*> extensions = { XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME };

        // Add optional extensions if available
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

        xr::InstanceCreateInfo instanceCreateInfo;
        std::strncpy(instanceCreateInfo.applicationInfo.applicationName, appName.c_str(), XR_MAX_APPLICATION_NAME_SIZE);
        instanceCreateInfo.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
        instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        instanceCreateInfo.enabledExtensionNames = extensions.data();

#if defined(PLATFORM_ANDROID)
        if (isExtensionEnabled(XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME)) {
            if (!androidApp) {
                std::cerr << "XrContext: androidApp not set" << std::endl;
                return false;
            }

            xr::InstanceCreateInfoAndroidKHR androidCreateInfo{XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR};
            androidCreateInfo.applicationVM = androidApp->activity->vm;
            androidCreateInfo.applicationActivity = androidApp->activity->clazz;
            instanceCreateInfo.next = &androidCreateInfo;
        }
#endif

        this->instance = xr::createInstance(instanceCreateInfo);

        // 2. Get System ID
        xr::SystemGetInfo systemGetInfo;
        systemGetInfo.formFactor = xr::FormFactor::HeadMountedDisplay;
        this->systemId = this->instance.getSystem(systemGetInfo);

    } catch (const std::exception& e) {
        std::cerr << "XrContext: Instance initialization failed: " << e.what() << std::endl;
        return false;
    }

    return true;
}

bool XrContext::createSession(vk::PhysicalDevice physicalDevice, vk::Device device, uint32_t queueFamilyIndex, uint32_t queueIndex) {
    std::cout << "XrContext: Creating session" << std::endl;

    try {
        // 4. Create Session
        XrGraphicsBindingVulkanKHR graphicsBinding{XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR};
        graphicsBinding.instance = (VkInstance)this->vkInstance;
        graphicsBinding.physicalDevice = (VkPhysicalDevice)physicalDevice;
        graphicsBinding.device = (VkDevice)device;
        graphicsBinding.queueFamilyIndex = queueFamilyIndex;
        graphicsBinding.queueIndex = queueIndex;

        xr::SessionCreateInfo sessionCreateInfo;
        sessionCreateInfo.next = &graphicsBinding;
        sessionCreateInfo.systemId = this->systemId;
        this->session = this->instance.createSession(sessionCreateInfo);

        // 5. Create Reference Space
        xr::ReferenceSpaceCreateInfo spaceCreateInfo;
        spaceCreateInfo.referenceSpaceType = xr::ReferenceSpaceType::Stage;
        spaceCreateInfo.poseInReferenceSpace = {{0,0,0,1}, {0,0,0}};
        this->appSpace = this->session.createReferenceSpace(spaceCreateInfo);

        // Initialize views with identity/default values to avoid out-of-bounds access
        this->views.resize(2);
        for (uint32_t i = 0; i < 2; ++i) {
            this->views[i].pose = {{0,0,0,1}, {0,0,0}};
            this->views[i].fov = {-1, 1, 1, -1};
        }

        // 6. Initialize Action System (Chapter 7)
        xr::ActionSetCreateInfo actionSetInfo;
        std::strncpy(actionSetInfo.actionSetName, "main", XR_MAX_ACTION_SET_NAME_SIZE);
        std::strncpy(actionSetInfo.localizedActionSetName, "Main Actions", XR_MAX_LOCALIZED_ACTION_SET_NAME_SIZE);
        this->actionSet = this->instance.createActionSet(actionSetInfo);

        auto createAction = [&](const std::string& name, const std::string& localizedName, xr::ActionType type) {
            xr::ActionCreateInfo actionInfo;
            actionInfo.actionType = type;
            std::strncpy(actionInfo.actionName, name.c_str(), XR_MAX_ACTION_NAME_SIZE);
            std::strncpy(actionInfo.localizedActionName, localizedName.c_str(), XR_MAX_LOCALIZED_ACTION_NAME_SIZE);
            this->actions[name] = this->actionSet.createAction(actionInfo);
            this->actionTypes[name] = type;
        };

        createAction("trigger_left", "Left Trigger", xr::ActionType::BooleanInput);
        createAction("trigger_right", "Right Trigger", xr::ActionType::BooleanInput);
        createAction("pose_left", "Left Hand Pose", xr::ActionType::PoseInput);
        createAction("pose_right", "Right Hand Pose", xr::ActionType::PoseInput);
        createAction("grab_left", "Left Grab", xr::ActionType::FloatInput);
        createAction("grab_right", "Right Grab", xr::ActionType::FloatInput);
        createAction("Grab", "Grab", xr::ActionType::BooleanInput);
        createAction("GrabPose", "Grab Pose", xr::ActionType::PoseInput);
        createAction("menu", "Menu Button", xr::ActionType::BooleanInput);

        // Suggest bindings for simple controller
        xr::Path khrSimplePath = this->instance.stringToPath("/interaction_profiles/khr/simple_controller");
        std::vector<xr::ActionSuggestedBinding> bindings = {
            {actions["trigger_left"], this->instance.stringToPath("/user/hand/left/input/select/click")},
            {actions["trigger_right"], this->instance.stringToPath("/user/hand/right/input/select/click")},
            {actions["pose_left"], this->instance.stringToPath("/user/hand/left/input/grip/pose")},
            {actions["pose_right"], this->instance.stringToPath("/user/hand/right/input/grip/pose")},
            {actions["Grab"], this->instance.stringToPath("/user/hand/right/input/select/click")},
            {actions["GrabPose"], this->instance.stringToPath("/user/hand/right/input/grip/pose")},
            {actions["menu"], this->instance.stringToPath("/user/hand/left/input/menu/click")}
        };
        xr::InteractionProfileSuggestedBinding suggestedBindings;
        suggestedBindings.interactionProfile = khrSimplePath;
        suggestedBindings.suggestedBindings = bindings.data();
        suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
        this->instance.suggestInteractionProfileBindings(suggestedBindings);

        xr::SessionActionSetsAttachInfo attachInfo;
        attachInfo.countActionSets = 1;
        attachInfo.actionSets = &this->actionSet;
        this->session.attachSessionActionSets(attachInfo);

        // Create spaces for pose actions
        for (const auto& actionName : {"pose_left", "pose_right", "GrabPose"}) {
            xr::ActionSpaceCreateInfo actionSpaceInfo;
            actionSpaceInfo.action = actions[actionName];
            actionSpaceInfo.poseInActionSpace = {{0,0,0,1}, {0,0,0}};
            this->actionSpaces[actionName] = this->session.createActionSpace(actionSpaceInfo);
        }

        // 7. Initialize Eye Gaze (Chapter 17)
        if (isExtensionEnabled(XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME)) {
            xr::ReferenceSpaceCreateInfo gazeSpaceInfo;
            gazeSpaceInfo.referenceSpaceType = (xr::ReferenceSpaceType)XR_REFERENCE_SPACE_TYPE_EYE_GAZE_EXT;
            gazeSpaceInfo.poseInReferenceSpace = {{0,0,0,1}, {0,0,0}};
            try {
                this->gazeSpace = this->session.createReferenceSpace(gazeSpaceInfo);
            } catch (const std::exception& e) {
                std::cout << "XrContext: Eye gaze space creation failed: " << e.what() << std::endl;
            }
        }

        // 8. Initialize Scene Understanding (Chapter 16)
        if (isExtensionEnabled(XR_MSFT_SCENE_UNDERSTANDING_EXTENSION_NAME)) {
            PFN_xrCreateSceneObserverMSFT xrCreateSceneObserverMSFT_ptr;
            this->instance.getInstanceProcAddr("xrCreateSceneObserverMSFT", (PFN_xrVoidFunction*)&xrCreateSceneObserverMSFT_ptr);
            if (xrCreateSceneObserverMSFT_ptr) {
                XrSceneObserverCreateInfoMSFT createInfo{XR_TYPE_SCENE_OBSERVER_CREATE_INFO_MSFT};
                XrSceneObserverMSFT observer;
                if (xrCreateSceneObserverMSFT_ptr((XrSession)this->session, &createInfo, &observer) == XR_SUCCESS) {
                    this->sceneObserver = xr::SceneObserverMSFT(observer);
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "XrContext: Session creation failed: " << e.what() << std::endl;
        return false;
    }

    return true;
}

void XrContext::cleanup() {
    for (auto& swapchain : swapchains) {
        swapchain.handle.destroy();
    }
    swapchains.clear();

    if (sceneObserver) {
        PFN_xrDestroySceneObserverMSFT xrDestroySceneObserverMSFT_ptr;
        this->instance.getInstanceProcAddr("xrDestroySceneObserverMSFT", (PFN_xrVoidFunction*)&xrDestroySceneObserverMSFT_ptr);
        if (xrDestroySceneObserverMSFT_ptr) {
            xrDestroySceneObserverMSFT_ptr((XrSceneObserverMSFT)sceneObserver);
        }
        sceneObserver = nullptr;
    }

    if (gazeSpace) {
        gazeSpace.destroy();
        gazeSpace = nullptr;
    }

    for (auto& pair : actionSpaces) {
        pair.second.destroy();
    }
    actionSpaces.clear();

    for (auto& pair : actions) {
        pair.second.destroy();
    }
    actions.clear();

    if (actionSet) {
        actionSet.destroy();
        actionSet = nullptr;
    }

    if (appSpace) {
        appSpace.destroy();
        appSpace = nullptr;
    }

    if (session) {
        session.destroy();
        session = nullptr;
    }

    if (instance) {
        instance.destroy();
        instance = nullptr;
    }
}

std::vector<const char*> XrContext::getVulkanInstanceExtensions() {
    if (!instance) {
        // Fallback for when instance isn't created yet (though it should be)
        return { "XR_KHR_vulkan_enable2" };
    }

    // Use the real API to query required extensions
    std::string extensions = this->instance.getVulkanInstanceExtensionsKHR(this->systemId);

    // Parse the space-separated string into a vector of const char*
    static std::vector<std::string> extList;
    extList.clear();
    std::string ext;
    std::istringstream iss(extensions);
    while (iss >> ext) {
        extList.push_back(ext);
    }

    static std::vector<const char*> extPtrs;
    extPtrs.clear();
    for (const auto& s : extList) {
        extPtrs.push_back(s.c_str());
    }

    return extPtrs;
}

std::vector<const char*> XrContext::getVulkanDeviceExtensions(vk::PhysicalDevice physicalDevice) {
    if (!instance) {
        return { "VK_KHR_external_memory", "VK_KHR_external_semaphore" };
    }

    std::string extensions = this->instance.getVulkanDeviceExtensionsKHR(this->systemId);

    static std::vector<std::string> devExtList;
    devExtList.clear();
    std::string ext;
    std::istringstream iss(extensions);
    while (iss >> ext) {
        devExtList.push_back(ext);
    }

    static std::vector<const char*> devExtPtrs;
    devExtPtrs.clear();
    for (const auto& s : devExtList) {
        devExtPtrs.push_back(s.c_str());
    }

    return devExtPtrs;
}

const uint8_t* XrContext::getRequiredLUID() {
    static uint8_t luid[8] = {0};
    xr::GraphicsRequirementsVulkanKHR graphicsRequirements;
    this->instance.getVulkanGraphicsRequirements2KHR(this->systemId, &graphicsRequirements);
    std::memcpy(luid, &graphicsRequirements.graphicsDeviceLuid, 8);
    return luid;
}

vk::Extent2D XrContext::getRecommendedExtent() const {
    uint32_t viewCount = 0;
    std::vector<xr::ViewConfigurationView> views = this->instance.enumerateViewConfigurationViewsToVector(this->systemId, xr::ViewConfigurationType::PrimaryStereo);
    if (views.empty()) return {1024, 1024};
    return {views[0].recommendedImageRectWidth, views[0].recommendedImageRectHeight};
}

void XrContext::createSwapchains(vk::Device device, vk::Format format, vk::Extent2D extent) {
    this->format = format;
    this->extent = extent;

    xr::SwapchainCreateInfo swapchainCreateInfo;
    swapchainCreateInfo.arrayCount = 2; // Multiview (Layered)
    swapchainCreateInfo.format = (int64_t)format;
    swapchainCreateInfo.width = extent.width;
    swapchainCreateInfo.height = extent.height;
    swapchainCreateInfo.mipCount = 1;
    swapchainCreateInfo.faceCount = 1;
    swapchainCreateInfo.sampleCount = 1;
    swapchainCreateInfo.usageFlags = xr::SwapchainUsageFlagBits::ColorAttachment;

    xr::Swapchain swapchainHandle = this->session.createSwapchain(swapchainCreateInfo);

    std::vector<xr::SwapchainImageVulkanKHR> images = swapchainHandle.enumerateSwapchainImagesToVector<xr::SwapchainImageVulkanKHR>();

    SwapchainData data;
    data.handle = swapchainHandle;
    data.images = std::move(images);
    this->swapchains.push_back(std::move(data));
}

std::vector<vk::Image> XrContext::enumerateSwapchainImages() {
    std::vector<vk::Image> vkImages;
    if (!swapchains.empty()) {
        for (const auto& img : swapchains[0].images) {
            vkImages.push_back((VkImage)img.image);
        }
    }
    return vkImages;
}

void XrContext::waitSwapchainImage() {
    if (swapchains.empty()) return;
    xr::SwapchainWaitInfo waitInfo;
    waitInfo.timeout = xr::Duration::infinite();
    swapchains[0].handle.waitSwapchainImage(waitInfo);
}

uint32_t XrContext::acquireSwapchainImage() {
    if (swapchains.empty()) return 0;
    xr::SwapchainAcquireInfo acquireInfo;
    return swapchains[0].handle.acquireSwapchainImage(acquireInfo);
}

void XrContext::releaseSwapchainImage() {
    if (swapchains.empty()) return;
    xr::SwapchainReleaseInfo releaseInfo;
    swapchains[0].handle.releaseSwapchainImage(releaseInfo);
}

XrFrameState XrContext::waitFrame() {
    xr::FrameWaitInfo waitInfo;
    this->frameState = this->session.waitFrame(waitInfo);
    return (XrFrameState)this->frameState;
}

void XrContext::beginFrame() {
    xr::FrameBeginInfo beginInfo;
    this->session.beginFrame(beginInfo);
}

void XrContext::endFrame(const std::array<std::vector<vk::raii::ImageView>, 2>& eyeViews) {
    xr::FrameEndInfo endInfo;
    endInfo.displayTime = this->frameState.predictedDisplayTime;
    endInfo.environmentBlendMode = xr::EnvironmentBlendMode::Opaque;

    xr::CompositionLayerProjection layer;
    layer.space = this->appSpace;

    std::vector<xr::CompositionLayerProjectionView> projectionViews(2);
    for (uint32_t i = 0; i < 2; ++i) {
        projectionViews[i].pose = this->views[i].pose;
        projectionViews[i].fov = this->views[i].fov;
        projectionViews[i].subImage.swapchain = this->swapchains[0].handle;
        projectionViews[i].subImage.imageRect = {{0, 0}, {(int32_t)extent.width, (int32_t)extent.height}};
        projectionViews[i].subImage.imageArrayIndex = i;
    }

    layer.viewCount = (uint32_t)projectionViews.size();
    layer.views = projectionViews.data();

    std::vector<xr::CompositionLayerBaseHeader*> layers;
    if (this->frameState.shouldRender) {
        layers.push_back(reinterpret_cast<xr::CompositionLayerBaseHeader*>(&layer));
    }

    endInfo.layerCount = (uint32_t)layers.size();
    endInfo.layers = layers.data();

    this->session.endFrame(endInfo);
}

void XrContext::locateViews(XrTime predictedTime) {
    xr::ViewLocateInfo locateInfo;
    locateInfo.viewConfigurationType = xr::ViewConfigurationType::PrimaryStereo;
    locateInfo.displayTime = predictedTime;
    locateInfo.space = this->appSpace;

    auto [result, viewState, locatedViews] = this->session.locateViewsToVector(locateInfo);
    this->views = std::move(locatedViews);
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
    float nearZ = 0.1f;
    float farZ = 100.0f;

    float tanLeft = std::tan(fov.angleLeft);
    float tanRight = std::tan(fov.angleRight);
    float tanDown = std::tan(fov.angleDown);
    float tanUp = std::tan(fov.angleUp);

    float tanWidth = tanRight - tanLeft;
    float tanHeight = tanUp - tanDown;

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
    if (!session || !actionSet) return;

    xr::ActionsSyncInfo syncInfo;
    xr::ActiveActionSet activeSet{actionSet, xr::Space(nullptr)};
    syncInfo.activeActionSets = &activeSet;
    syncInfo.countActiveActionSets = 1;

    if (session.syncActions(syncInfo) != xr::Result::Success) {
        return;
    }
}

bool XrContext::isActionActive(const std::string& name) const {
    if (!session || actions.find(name) == actions.end()) return false;

    xr::Action action = actions.at(name);
    xr::ActionType type = actionTypes.at(name);
    xr::ActionStateGetInfo getInfo{action};

    if (type == xr::ActionType::BooleanInput) {
        auto state = session.getActionStateBoolean(getInfo);
        if (state.isActive) return state.currentState;
    } else if (type == xr::ActionType::FloatInput) {
        auto state = session.getActionStateFloat(getInfo);
        if (state.isActive) return state.currentState > 0.1f;
    }

    return false;
}

XrPosef XrContext::getActionPose(const std::string& name) const {
    if (!session || actionSpaces.find(name) == actionSpaces.end()) {
        XrPosef pose;
        pose.orientation = {0,0,0,1};
        pose.position = {0,0,0};
        return pose;
    }

    xr::Space space = actionSpaces.at(name);
    auto location = space.locateSpace(appSpace, frameState.predictedDisplayTime);

    if (location.locationFlags & xr::SpaceLocationFlagBits::OrientationValid &&
        location.locationFlags & xr::SpaceLocationFlagBits::PositionValid) {
        return (XrPosef)location.pose;
    }

    XrPosef pose;
    pose.orientation = {0,0,0,1};
    pose.position = {0,0,0};
    return pose;
}

std::vector<XrSpatialMesh> XrContext::getLatestSpatialMeshes() {
    if (!sceneObserver) return {};

    std::vector<XrSpatialMesh> meshes;

    // Use extension function pointers
    PFN_xrComputeNewSceneMSFT xrComputeNewSceneMSFT_ptr;
    PFN_xrGetSceneComputeStateMSFT xrGetSceneComputeStateMSFT_ptr;
    PFN_xrCreateSceneMSFT xrCreateSceneMSFT_ptr;
    PFN_xrGetSceneComponentsMSFT xrGetSceneComponentsMSFT_ptr;
    PFN_xrGetSceneMeshBuffersMSFT xrGetSceneMeshBuffersMSFT_ptr;
    PFN_xrGetSceneComponentLocationsMSFT xrGetSceneComponentLocationsMSFT_ptr;
    PFN_xrDestroySceneMSFT xrDestroySceneMSFT_ptr;

    this->instance.getInstanceProcAddr("xrComputeNewSceneMSFT", (PFN_xrVoidFunction*)&xrComputeNewSceneMSFT_ptr);
    this->instance.getInstanceProcAddr("xrGetSceneComputeStateMSFT", (PFN_xrVoidFunction*)&xrGetSceneComputeStateMSFT_ptr);
    this->instance.getInstanceProcAddr("xrCreateSceneMSFT", (PFN_xrVoidFunction*)&xrCreateSceneMSFT_ptr);
    this->instance.getInstanceProcAddr("xrGetSceneComponentsMSFT", (PFN_xrVoidFunction*)&xrGetSceneComponentsMSFT_ptr);
    this->instance.getInstanceProcAddr("xrGetSceneMeshBuffersMSFT", (PFN_xrVoidFunction*)&xrGetSceneMeshBuffersMSFT_ptr);
    this->instance.getInstanceProcAddr("xrGetSceneComponentLocationsMSFT", (PFN_xrVoidFunction*)&xrGetSceneComponentLocationsMSFT_ptr);
    this->instance.getInstanceProcAddr("xrDestroySceneMSFT", (PFN_xrVoidFunction*)&xrDestroySceneMSFT_ptr);

    if (!xrComputeNewSceneMSFT_ptr || !xrGetSceneComputeStateMSFT_ptr || !xrCreateSceneMSFT_ptr ||
        !xrGetSceneComponentsMSFT_ptr || !xrGetSceneMeshBuffersMSFT_ptr || !xrGetSceneComponentLocationsMSFT_ptr || !xrDestroySceneMSFT_ptr) {
        return {};
    }

    // 1. Check if we need to start a new computation or if one is ready
    XrSceneComputeStateMSFT state;
    xrGetSceneComputeStateMSFT_ptr((XrSceneObserverMSFT)sceneObserver, &state);

    if (state == XR_SCENE_COMPUTE_STATE_NONE_MSFT) {
        XrSceneComputeInfoMSFT computeInfo{XR_TYPE_SCENE_COMPUTE_INFO_MSFT};
        computeInfo.consistency = XR_SCENE_COMPUTE_CONSISTENCY_SNAPSHOT_MSFT;
        XrSceneSphereBoundMSFT sphereBound;
        sphereBound.center = {0,0,0};
        sphereBound.radius = 10.0f;
        computeInfo.bounds.sphereCount = 1;
        computeInfo.bounds.spheres = &sphereBound;
        xrComputeNewSceneMSFT_ptr((XrSceneObserverMSFT)sceneObserver, &computeInfo);
        return {};
    }

    if (state != XR_SCENE_COMPUTE_STATE_COMPLETED_MSFT) return {};

    // 2. Create the scene
    XrSceneCreateInfoMSFT createInfo{XR_TYPE_SCENE_CREATE_INFO_MSFT};
    XrSceneMSFT scene;
    if (xrCreateSceneMSFT_ptr((XrSceneObserverMSFT)sceneObserver, &createInfo, &scene) != XR_SUCCESS) return {};

    // 3. Get mesh components
    XrSceneComponentsGetInfoMSFT getInfo{XR_TYPE_SCENE_COMPONENTS_GET_INFO_MSFT};
    getInfo.componentType = XR_SCENE_COMPONENT_TYPE_MESH_MSFT;

    XrSceneComponentsMSFT components{XR_TYPE_SCENE_COMPONENTS_MSFT};
    xrGetSceneComponentsMSFT_ptr(scene, &getInfo, &components);

    std::vector<XrSceneComponentMSFT> componentBuffer(components.componentCountOutput);
    components.componentBuffer = componentBuffer.data();
    components.componentCapacityInput = components.componentCountOutput;
    xrGetSceneComponentsMSFT_ptr(scene, &getInfo, &components);

    // 4. Get locations for transforms
    XrSceneComponentLocationsGetInfoMSFT locGetInfo{XR_TYPE_SCENE_COMPONENT_LOCATIONS_GET_INFO_MSFT};
    locGetInfo.baseSpace = (XrSpace)appSpace;
    locGetInfo.time = frameState.predictedDisplayTime;

    std::vector<XrSceneComponentLocationMSFT> locationBuffer(components.componentCountOutput);
    XrSceneComponentLocationsMSFT locations{XR_TYPE_SCENE_COMPONENT_LOCATIONS_MSFT};
    locations.componentCount = components.componentCountOutput;
    locations.componentLocations = locationBuffer.data();
    xrGetSceneComponentLocationsMSFT_ptr(scene, &locGetInfo, &locations);

    // 5. Get mesh data for each component
    for (uint32_t i = 0; i < components.componentCountOutput; ++i) {
        XrSceneMeshBuffersGetInfoMSFT meshGetInfo{XR_TYPE_SCENE_MESH_BUFFERS_GET_INFO_MSFT};
        meshGetInfo.meshComponentId = componentBuffer[i].id;

        XrSceneMeshBuffersMSFT meshBuffers{XR_TYPE_SCENE_MESH_BUFFERS_MSFT};
        xrGetSceneMeshBuffersMSFT_ptr(scene, &meshGetInfo, &meshBuffers);

        XrSpatialMesh mesh;
        mesh.meshGuid = componentBuffer[i].id;
        mesh.vertices.resize(meshBuffers.vertexCountOutput);
        mesh.indices.resize(meshBuffers.indexCountOutput);

        XrSceneMeshVertexBufferMSFT vBuffer{XR_TYPE_SCENE_MESH_VERTEX_BUFFER_MSFT};
        vBuffer.vertexCapacityInput = (uint32_t)mesh.vertices.size();
        vBuffer.vertices = (XrVector3f*)mesh.vertices.data();

        XrSceneMeshIndexBufferMSFT iBuffer{XR_TYPE_SCENE_MESH_INDEX_BUFFER_MSFT};
        iBuffer.indexCapacityInput = (uint32_t)mesh.indices.size();
        iBuffer.indices = mesh.indices.data();

        meshBuffers.vertexBuffer = &vBuffer;
        meshBuffers.indexBuffer = &iBuffer;

        xrGetSceneMeshBuffersMSFT_ptr(scene, &meshGetInfo, &meshBuffers);

        if (locationBuffer[i].flags & (XR_SPACE_LOCATION_ORIENTATION_VALID_BIT | XR_SPACE_LOCATION_POSITION_VALID_BIT)) {
            mesh.transform = xrPoseToMatrix(locationBuffer[i].pose);
        } else {
            mesh.transform = glm::mat4(1.0f);
        }

        meshes.push_back(std::move(mesh));
    }

    // 6. Cleanup scene handle
    xrDestroySceneMSFT_ptr(scene);

    // Start a new computation for the next call
    XrSceneComputeInfoMSFT computeInfo{XR_TYPE_SCENE_COMPUTE_INFO_MSFT};
    computeInfo.consistency = XR_SCENE_COMPUTE_CONSISTENCY_SNAPSHOT_MSFT;
    XrSceneSphereBoundMSFT sphereBound;
    sphereBound.center = {0,0,0};
    sphereBound.radius = 10.0f;
    computeInfo.bounds.sphereCount = 1;
    computeInfo.bounds.spheres = &sphereBound;
    xrComputeNewSceneMSFT_ptr((XrSceneObserverMSFT)sceneObserver, &computeInfo);

    return meshes;
}

glm::vec2 XrContext::getGazeNDC() const {
    if (!gazeSpace || views.empty()) return glm::vec2(0.5f, 0.5f);

    auto location = gazeSpace.locateSpace(appSpace, frameState.predictedDisplayTime);
    if (!(location.locationFlags & xr::SpaceLocationFlagBits::OrientationValid)) {
        return glm::vec2(0.5f, 0.5f);
    }

    // Project gaze vector onto the near plane
    // Gaze is along the -Z axis of the gaze space
    glm::mat4 gazeMat = xrPoseToMatrix((XrPosef)location.pose);
    glm::vec3 gazeOrigin = glm::vec3(gazeMat[3]);
    glm::vec3 gazeDir = -glm::vec3(gazeMat[2]); // Forward is -Z

    // Use the first eye's view/projection for NDC calculation
    glm::mat4 viewProj = getProjectionMatrix(0) * getViewMatrix(0);
    glm::vec4 projected = viewProj * glm::vec4(gazeOrigin + gazeDir, 1.0f);

    if (projected.w == 0.0f) return glm::vec2(0.5f, 0.5f);

    glm::vec3 ndc = glm::vec3(projected) / projected.w;
    return glm::vec2(ndc.x * 0.5f + 0.5f, ndc.y * 0.5f + 0.5f);
}
