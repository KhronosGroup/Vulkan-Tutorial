/* Copyright (c) 2025 Holochip Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// OpenXR-engine local shadow of simple_engine/platform.h.
// Adds PLATFORM_HEADSET_ONLY support for standalone headsets (PICO, embedded
// Linux XR devices) that do not have a display/window system and therefore
// cannot link against GLFW.  All other paths are identical to the upstream file.
#pragma once

#include <functional>
#include <memory>
#include <string>

#if defined(PLATFORM_ANDROID)
#	include <vulkan/vulkan.h>
#	include <android/asset_manager.h>
#	include <android/asset_manager_jni.h>
#	include <android/log.h>
#	include <android/native_activity.h>
#	include <game-activity/native_app_glue/android_native_app_glue.h>
#	define LOGI(...) ((void) __android_log_print(ANDROID_LOG_INFO, "SimpleEngine", __VA_ARGS__))
#	define LOGW(...) ((void) __android_log_print(ANDROID_LOG_WARN, "SimpleEngine", __VA_ARGS__))
#	define LOGE(...) ((void) __android_log_print(ANDROID_LOG_ERROR, "SimpleEngine", __VA_ARGS__))
#elif defined(PLATFORM_HEADSET_ONLY)
#	include <vulkan/vulkan.h>
#	define LOGI(...)        \
		printf(__VA_ARGS__); \
		printf("\n")
#	define LOGW(...)        \
		printf(__VA_ARGS__); \
		printf("\n")
#	define LOGE(...)                 \
		fprintf(stderr, __VA_ARGS__); \
		fprintf(stderr, "\n")
#else
#	define GLFW_INCLUDE_VULKAN
#	include <GLFW/glfw3.h>
#	define LOGI(...)        \
		printf(__VA_ARGS__); \
		printf("\n")
#	define LOGW(...)        \
		printf(__VA_ARGS__); \
		printf("\n")
#	define LOGE(...)                 \
		fprintf(stderr, __VA_ARGS__); \
		fprintf(stderr, "\n")
#endif

class Platform {
  public:
    Platform() = default;
    virtual ~Platform() = default;

    virtual bool Initialize(const std::string& appName, int width, int height) = 0;
    virtual void Cleanup() = 0;
    virtual bool ProcessEvents() = 0;
    virtual bool HasWindowResized() = 0;
    virtual int GetWindowWidth() const = 0;
    virtual int GetWindowHeight() const = 0;
    virtual void GetWindowSize(int* width, int* height) const {
      *width = GetWindowWidth();
      *height = GetWindowHeight();
    }
    virtual bool CreateVulkanSurface(VkInstance instance, VkSurfaceKHR* surface) = 0;
    virtual void SetResizeCallback(std::function<void(int, int)> callback) = 0;
    virtual void SetMouseCallback(std::function<void(float, float, uint32_t)> callback) = 0;
    virtual void SetKeyboardCallback(std::function<void(uint32_t, bool)> callback) = 0;
    virtual void SetCharCallback(std::function<void(uint32_t)> callback) = 0;
    virtual void SetWindowTitle(const std::string& title) = 0;
};

#if defined(PLATFORM_ANDROID)
// Full AndroidPlatform definition — copy of simple_engine/platform.h section
class AndroidPlatform : public Platform {
  private:
    android_app* app = nullptr;
    int width = 0;
    int height = 0;
    bool windowResized = false;
    std::function<void(int, int)> resizeCallback;
    std::function<void(float, float, uint32_t)> mouseCallback;
    std::function<void(uint32_t, bool)> keyboardCallback;
    std::function<void(uint32_t)> charCallback;

    struct DeviceCapabilities {
      int apiLevel = 0;
      std::string deviceModel;
      std::string deviceManufacturer;
      int cpuCores = 0;
      int64_t totalMemory = 0;
      bool supportsVulkan = false;
      bool supportsVulkan11 = false;
      bool supportsVulkan12 = false;
      std::vector<std::string> supportedVulkanExtensions;
    };
    DeviceCapabilities deviceCapabilities;
    bool powerSavingMode = false;
    bool multiTouchEnabled = true;

    void DetectDeviceCapabilities();
    void SetupPowerSavingMode();
    void InitializeTouchInput();

  public:
    void EnablePowerSavingMode(bool enable);
    bool IsPowerSavingModeEnabled() const { return powerSavingMode; }
    void EnableMultiTouch(bool enable) { multiTouchEnabled = enable; }
    bool IsMultiTouchEnabled() const { return multiTouchEnabled; }
    const DeviceCapabilities& GetDeviceCapabilities() const { return deviceCapabilities; }

    explicit AndroidPlatform(android_app* androidApp);

    bool Initialize(const std::string& appName, int width, int height) override;
    void Cleanup() override;
    bool ProcessEvents() override;
    bool HasWindowResized() override;
    int GetWindowWidth() const override { return width; }
    int GetWindowHeight() const override { return height; }
    bool CreateVulkanSurface(VkInstance instance, VkSurfaceKHR* surface) override;
    void SetResizeCallback(std::function<void(int, int)> callback) override;
    void SetMouseCallback(std::function<void(float, float, uint32_t)> callback) override;
    void SetKeyboardCallback(std::function<void(uint32_t, bool)> callback) override;
    void SetCharCallback(std::function<void(uint32_t)> callback) override;
    void SetWindowTitle(const std::string& title) override;

    android_app* GetApp() const { return app; }
    AAssetManager* GetAssetManager() const { return app ? app->activity->assetManager : nullptr; }
};

#elif defined(PLATFORM_HEADSET_ONLY)

// HeadlessPlatform: stub platform for standalone XR headsets that have no
// display, window system, or GLFW available.  The XR swapchain is used directly.
class HeadlessPlatform final : public Platform {
  private:
    int width = 0;
    int height = 0;
    std::function<void(int, int)> resizeCallback;
    std::function<void(float, float, uint32_t)> mouseCallback;
    std::function<void(uint32_t, bool)> keyboardCallback;
    std::function<void(uint32_t)> charCallback;

  public:
    HeadlessPlatform() = default;

    bool Initialize(const std::string& appName, int w, int h) override {
      width = w; height = h;
      return true;
    }
    void Cleanup() override {}
    bool ProcessEvents() override { return true; }
    bool HasWindowResized() override { return false; }
    int GetWindowWidth() const override { return width; }
    int GetWindowHeight() const override { return height; }

    // No real surface — caller (Renderer) must skip surface-dependent operations.
    bool CreateVulkanSurface(VkInstance, VkSurfaceKHR* surface) override {
      *surface = VK_NULL_HANDLE;
      return false; // signals "no surface" to the renderer
    }

    void SetResizeCallback(std::function<void(int, int)> cb) override { resizeCallback = std::move(cb); }
    void SetMouseCallback(std::function<void(float, float, uint32_t)> cb) override { mouseCallback = std::move(cb); }
    void SetKeyboardCallback(std::function<void(uint32_t, bool)> cb) override { keyboardCallback = std::move(cb); }
    void SetCharCallback(std::function<void(uint32_t)> cb) override { charCallback = std::move(cb); }
    void SetWindowTitle(const std::string&) override {}

    void SetSize(int w, int h) { width = w; height = h; }
};

#else

// DesktopPlatform: full GLFW-based windowed platform for desktop+XR companion window.
class DesktopPlatform final : public Platform {
  private:
    GLFWwindow* window = nullptr;
    int width = 0;
    int height = 0;
    bool windowResized = false;
    std::function<void(int, int)> resizeCallback;
    std::function<void(float, float, uint32_t)> mouseCallback;
    std::function<void(uint32_t, bool)> keyboardCallback;
    std::function<void(uint32_t)> charCallback;

    static void WindowResizeCallback(GLFWwindow* window, int width, int height);
    static void MousePositionCallback(GLFWwindow* window, double xpos, double ypos);
    static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void CharCallback(GLFWwindow* window, unsigned int codepoint);

  public:
    DesktopPlatform() = default;

    bool Initialize(const std::string& appName, int width, int height) override;
    void Cleanup() override;
    bool ProcessEvents() override;
    bool HasWindowResized() override;
    int GetWindowWidth() const override { return width; }
    int GetWindowHeight() const override { return height; }
    bool CreateVulkanSurface(VkInstance instance, VkSurfaceKHR* surface) override;
    void SetResizeCallback(std::function<void(int, int)> callback) override;
    void SetMouseCallback(std::function<void(float, float, uint32_t)> callback) override;
    void SetKeyboardCallback(std::function<void(uint32_t, bool)> callback) override;
    void SetCharCallback(std::function<void(uint32_t)> callback) override;
    void SetWindowTitle(const std::string& title) override;
    GLFWwindow* GetWindow() const { return window; }
};
#endif

template<typename... Args>
std::unique_ptr<Platform> CreatePlatform(Args&&... args) {
#if defined(PLATFORM_ANDROID)
  return std::make_unique<AndroidPlatform>(std::forward<Args>(args)...);
#elif defined(PLATFORM_HEADSET_ONLY)
  return std::make_unique<HeadlessPlatform>();
#else
  return std::make_unique<DesktopPlatform>();
#endif
}
