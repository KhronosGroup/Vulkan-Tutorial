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
#include "platform.h"

#include <stdexcept>

#if defined(PLATFORM_ANDROID)
#	include <cassert>
#	include <vulkan/vulkan_android.h>

// Android platform implementation

AndroidPlatform::AndroidPlatform(android_app* androidApp) : app(androidApp) {
  // Set up the app's user data
  app->userData = this;

  // Set up the command callback
  app->onAppCmd = [](android_app* app, int32_t cmd) {
    auto* platform = static_cast<AndroidPlatform *>(app->userData);

    switch (cmd) {
      case APP_CMD_INIT_WINDOW:
      case APP_CMD_WINDOW_RESIZED:
        if (app->window != nullptr) {
          // Get the window dimensions
          ANativeWindow* window = app->window;
          platform->width = ANativeWindow_getWidth(window);
          platform->height = ANativeWindow_getHeight(window);
          platform->windowResized = true;

          // Call the resize callback if set
          if (platform->resizeCallback) {
            platform->resizeCallback(platform->width, platform->height);
          }
        }
        break;

      case APP_CMD_TERM_WINDOW:
        // Window is being hidden or closed
        break;

      default:
        break;
    }
  };
}

bool AndroidPlatform::Initialize(const std::string& appName, int requestedWidth, int requestedHeight) {
  // On Android, the window dimensions are determined by the device
  if (app->window != nullptr) {
    width = ANativeWindow_getWidth(app->window);
    height = ANativeWindow_getHeight(app->window);

    // Get device information for performance optimizations
    DetectDeviceCapabilities();

    // Set up power-saving mode based on battery level
    SetupPowerSavingMode();

    // Initialize touch input handling
    InitializeTouchInput();

    return true;
  }
  return false;
}

void AndroidPlatform::Cleanup() {
  // Nothing to clean up for Android
}

bool AndroidPlatform::ProcessEvents() {
  // Process Android events
  int events;
  android_poll_source* source;

  // Poll for events with a timeout of 0 (non-blocking)
  while (ALooper_pollOnce(0, nullptr, &events, (void **) &source) >= 0) {
    if (source != nullptr) {
      source->process(app, source);
    }

    // Check if we are exiting
    if (app->destroyRequested != 0) {
      return false;
    }
  }

  return true;
}

bool AndroidPlatform::HasWindowResized() {
  bool resized = windowResized;
  windowResized = false;
  return resized;
}

bool AndroidPlatform::CreateVulkanSurface(VkInstance instance, VkSurfaceKHR* surface) {
  if (app->window == nullptr) {
    return false;
  }

  VkAndroidSurfaceCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
  createInfo.window = app->window;

  if (vkCreateAndroidSurfaceKHR(instance, &createInfo, nullptr, surface) != VK_SUCCESS) {
    return false;
  }

  return true;
}

void AndroidPlatform::SetResizeCallback(std::function<void(int, int)> callback) {
  resizeCallback = std::move(callback);
}

void AndroidPlatform::SetMouseCallback(std::function < void(float, float, uint32_t) > callback) {
  mouseCallback = std::move(callback);
}

void AndroidPlatform::SetKeyboardCallback(std::function < void(uint32_t, bool) > callback) {
  keyboardCallback = std::move(callback);
}

void AndroidPlatform::SetCharCallback(std::function<void(uint32_t)> callback) {
  charCallback = std::move(callback);
}

void AndroidPlatform::SetWindowTitle([[maybe_unused]] const std::string& title) {
  // No-op on Android - mobile apps don't have window titles
}

void AndroidPlatform::DetectDeviceCapabilities() {
  if (!app) {
    return;
  }

  // Get API level via JNI
  JNIEnv* env = nullptr;
  app->activity->vm->AttachCurrentThread(&env, nullptr);
  if (env) {
    jclass versionClass = env->FindClass("android/os/Build$VERSION");
    jfieldID sdkFieldID = env->GetStaticFieldID(versionClass, "SDK_INT", "I");
    deviceCapabilities.apiLevel = env->GetStaticIntField(versionClass, sdkFieldID);

    jclass buildClass = env->FindClass("android/os/Build");
    jfieldID modelFieldID = env->GetStaticFieldID(buildClass, "MODEL", "Ljava/lang/String;");
    jfieldID manufacturerFieldID = env->GetStaticFieldID(buildClass, "MANUFACTURER", "Ljava/lang/String;");

    jstring modelJString = (jstring) env->GetStaticObjectField(buildClass, modelFieldID);
    jstring manufacturerJString = (jstring) env->GetStaticObjectField(buildClass, manufacturerFieldID);

    const char* modelChars = env->GetStringUTFChars(modelJString, nullptr);
    const char* manufacturerChars = env->GetStringUTFChars(manufacturerJString, nullptr);

    deviceCapabilities.deviceModel = modelChars;
    deviceCapabilities.deviceManufacturer = manufacturerChars;

    env->ReleaseStringUTFChars(modelJString, modelChars);
    env->ReleaseStringUTFChars(manufacturerJString, manufacturerChars);

    // Get total memory
    jclass activityManagerClass = env->FindClass("android/app/ActivityManager");
    jclass memoryInfoClass = env->FindClass("android/app/ActivityManager$MemoryInfo");
    jmethodID memoryInfoConstructor = env->GetMethodID(memoryInfoClass, "<init>", "()V");
    jobject memoryInfo = env->NewObject(memoryInfoClass, memoryInfoConstructor);

    jmethodID getSystemService = env->GetMethodID(env->GetObjectClass(app->activity->javaGameActivity),
                                                  "getSystemService",
                                                  "(Ljava/lang/String;)Ljava/lang/Object;");
    jstring serviceStr = env->NewStringUTF("activity");
    jobject activityManager = env->CallObjectMethod(app->activity->javaGameActivity, getSystemService, serviceStr);

    jmethodID getMemoryInfo = env->GetMethodID(activityManagerClass,
                                               "getMemoryInfo",
                                               "(Landroid/app/ActivityManager$MemoryInfo;)V");
    env->CallVoidMethod(activityManager, getMemoryInfo, memoryInfo);

    jfieldID totalMemField = env->GetFieldID(memoryInfoClass, "totalMem", "J");
    deviceCapabilities.totalMemory = env->GetLongField(memoryInfo, totalMemField);

    env->DeleteLocalRef(serviceStr);

    deviceCapabilities.supportsVulkan = true;
    deviceCapabilities.supportsVulkan11 = deviceCapabilities.apiLevel >= 28;
    deviceCapabilities.supportsVulkan12 = deviceCapabilities.apiLevel >= 29;

    app->activity->vm->DetachCurrentThread();
  }

  LOGI("Device detected: %s %s (API %d)", deviceCapabilities.deviceManufacturer.c_str(), deviceCapabilities.deviceModel.c_str(), deviceCapabilities.apiLevel);
}

void AndroidPlatform::SetupPowerSavingMode() {
  if (!app)
    return;

  JNIEnv* env = nullptr;
  app->activity->vm->AttachCurrentThread(&env, nullptr);
  if (env) {
    jclass intentFilterClass = env->FindClass("android/content/IntentFilter");
    jmethodID intentFilterConstructor = env->GetMethodID(intentFilterClass, "<init>", "(Ljava/lang/String;)V");
    jstring actionBatteryChanged = env->NewStringUTF("android.intent.action.BATTERY_CHANGED");
    jobject filter = env->NewObject(intentFilterClass, intentFilterConstructor, actionBatteryChanged);

    jmethodID registerReceiver = env->GetMethodID(env->GetObjectClass(app->activity->javaGameActivity),
                                                  "registerReceiver",
                                                  "(Landroid/content/BroadcastReceiver;Landroid/content/IntentFilter;)Landroid/content/Intent;");
    jobject intent = env->CallObjectMethod(app->activity->javaGameActivity, registerReceiver, nullptr, filter);

    if (intent) {
      jclass intentClass = env->GetObjectClass(intent);
      jmethodID getIntExtra = env->GetMethodID(intentClass, "getIntExtra", "(Ljava/lang/String;I)I");

      jstring levelKey = env->NewStringUTF("level");
      jstring scaleKey = env->NewStringUTF("scale");

      int level = env->CallIntMethod(intent, getIntExtra, levelKey, -1);
      int scale = env->CallIntMethod(intent, getIntExtra, scaleKey, -1);

      env->DeleteLocalRef(levelKey);
      env->DeleteLocalRef(scaleKey);

      if (level != -1 && scale != -1) {
        float batteryPct = (float) level / (float) scale;
        if (batteryPct < 0.2f) {
          EnablePowerSavingMode(true);
        }
      }
    }

    env->DeleteLocalRef(actionBatteryChanged);
    app->activity->vm->DetachCurrentThread();
  }
}

void AndroidPlatform::InitializeTouchInput() {
  if (!app)
    return;

  // GameActivity specific input handling is handled via GameActivity_set*Callback in the glue,
  // but the android_app structure in the new glue doesn't have onInputEvent anymore.
  // Instead, we rely on the activity callbacks or the internal event processing.
}

void AndroidPlatform::EnablePowerSavingMode(bool enable) {
  powerSavingMode = enable;
  LOGI("Power-saving mode %s", enable ? "enabled" : "disabled");
}

#else
// Desktop platform implementation

bool DesktopPlatform::Initialize(const std::string& appName, int requestedWidth, int requestedHeight) {
  if (!glfwInit()) {
    throw std::runtime_error("Failed to initialize GLFW");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  window = glfwCreateWindow(requestedWidth, requestedHeight, appName.c_str(), nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    throw std::runtime_error("Failed to create GLFW window");
  }

  glfwSetWindowUserPointer(window, this);

  glfwSetFramebufferSizeCallback(window, WindowResizeCallback);
  glfwSetCursorPosCallback(window, MousePositionCallback);
  glfwSetMouseButtonCallback(window, MouseButtonCallback);
  glfwSetKeyCallback(window, KeyCallback);
  glfwSetCharCallback(window, CharCallback);

  glfwGetFramebufferSize(window, &width, &height);

  return true;
}

void DesktopPlatform::Cleanup() {
  if (window) {
    glfwDestroyWindow(window);
    window = nullptr;
  }
  glfwTerminate();
}

bool DesktopPlatform::ProcessEvents() {
  glfwPollEvents();
  return !glfwWindowShouldClose(window);
}

bool DesktopPlatform::HasWindowResized() {
  bool resized = windowResized;
  windowResized = false;
  return resized;
}

bool DesktopPlatform::CreateVulkanSurface(VkInstance instance, VkSurfaceKHR* surface) {
  if (glfwCreateWindowSurface(instance, window, nullptr, surface) != VK_SUCCESS) {
    return false;
  }
  return true;
}

void DesktopPlatform::SetResizeCallback(std::function<void(int, int)> callback) {
  resizeCallback = std::move(callback);
}

void DesktopPlatform::SetMouseCallback(std::function < void(float, float, uint32_t) > callback) {
  mouseCallback = std::move(callback);
}

void DesktopPlatform::SetKeyboardCallback(std::function < void(uint32_t, bool) > callback) {
  keyboardCallback = std::move(callback);
}

void DesktopPlatform::SetCharCallback(std::function<void(uint32_t)> callback) {
  charCallback = std::move(callback);
}

void DesktopPlatform::SetWindowTitle(const std::string& title) {
  if (window) {
    glfwSetWindowTitle(window, title.c_str());
  }
}

void DesktopPlatform::WindowResizeCallback(GLFWwindow* window, int width, int height) {
  auto* platform = static_cast<DesktopPlatform *>(glfwGetWindowUserPointer(window));
  platform->width = width;
  platform->height = height;
  platform->windowResized = true;

  if (platform->resizeCallback) {
    platform->resizeCallback(width, height);
  }
}

void DesktopPlatform::MousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  auto* platform = static_cast<DesktopPlatform *>(glfwGetWindowUserPointer(window));
  if (platform->mouseCallback) {
    uint32_t buttons = 0;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
      buttons |= 0x01;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
      buttons |= 0x02;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
      buttons |= 0x04;
    }
    platform->mouseCallback(static_cast<float>(xpos), static_cast<float>(ypos), buttons);
  }
}

void DesktopPlatform::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  auto* platform = static_cast<DesktopPlatform *>(glfwGetWindowUserPointer(window));
  if (platform->mouseCallback) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    uint32_t buttons = 0;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
      buttons |= 0x01;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
      buttons |= 0x02;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
      buttons |= 0x04;
    }
    platform->mouseCallback(static_cast<float>(xpos), static_cast<float>(ypos), buttons);
  }
}

void DesktopPlatform::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto* platform = static_cast<DesktopPlatform *>(glfwGetWindowUserPointer(window));
  if (platform->keyboardCallback) {
    platform->keyboardCallback(key, action != GLFW_RELEASE);
  }
}

void DesktopPlatform::CharCallback(GLFWwindow* window, unsigned int codepoint) {
  auto* platform = static_cast<DesktopPlatform *>(glfwGetWindowUserPointer(window));
  if (platform->charCallback) {
    platform->charCallback(codepoint);
  }
}
#endif