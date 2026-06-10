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

  // Initialize sensors
  // Use the deprecated but widely available getInstance() for compatibility with minSdk 24
  sensorManager = ASensorManager_getInstance();

  if (sensorManager) {
    accelerometerSensor = ASensorManager_getDefaultSensor(sensorManager, ASENSOR_TYPE_ACCELEROMETER);
    if (accelerometerSensor) {
       ALooper* looper = ALooper_forThread();
       if (!looper) {
         looper = ALooper_prepare(ALOOPER_PREPARE_ALLOW_NON_CALLBACKS);
       }
       sensorEventQueue = ASensorManager_createEventQueue(sensorManager, looper, 3 /*IDENT_SENSOR*/, nullptr, nullptr);
    }
  }

  // Set up the command callback
  app->onAppCmd = [](android_app* app, int32_t cmd) {
    auto* platform = static_cast<AndroidPlatform *>(app->userData);

    switch (cmd) {
      case APP_CMD_INIT_WINDOW:
      case APP_CMD_WINDOW_RESIZED:
      case APP_CMD_CONFIG_CHANGED:
        if (app->window != nullptr) {
          // Get the window dimensions
          ANativeWindow* window = app->window;
          int32_t newWidth = ANativeWindow_getWidth(window);
          int32_t newHeight = ANativeWindow_getHeight(window);

          LOGI("AndroidPlatform: Window event %d. Dimensions: %dx%d", cmd, newWidth, newHeight);

          if (newWidth > 0 && newHeight > 0 && (newWidth != platform->width || newHeight != platform->height)) {
            platform->width = newWidth;
            platform->height = newHeight;
            platform->windowResized = true;

            LOGI("AndroidPlatform: Resizing to %dx%d", platform->width, platform->height);

            // Call the resize callback if set
            if (platform->resizeCallback) {
              platform->resizeCallback(platform->width, platform->height);
            }
          }
        }
        break;

      case APP_CMD_TERM_WINDOW:
        LOGI("AndroidPlatform: APP_CMD_TERM_WINDOW");
        // Window is being hidden or closed. Mark as resized with 0 size to stop rendering.
        platform->width = 0;
        platform->height = 0;
        platform->windowResized = true;
        break;

      case APP_CMD_GAINED_FOCUS:
        LOGI("AndroidPlatform: APP_CMD_GAINED_FOCUS");
        break;

      case APP_CMD_LOST_FOCUS:
        LOGI("AndroidPlatform: APP_CMD_LOST_FOCUS");
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

    // Enable accelerometer
    if (sensorEventQueue && accelerometerSensor) {
      ASensorEventQueue_enableSensor(sensorEventQueue, accelerometerSensor);
      // Set sensor rate (e.g., 60Hz)
      ASensorEventQueue_setEventRate(sensorEventQueue, accelerometerSensor, (1000L / 60) * 1000);
    }

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
  if (sensorEventQueue) {
    if (accelerometerSensor) {
      ASensorEventQueue_disableSensor(sensorEventQueue, accelerometerSensor);
    }
    ASensorManager_destroyEventQueue(sensorManager, sensorEventQueue);
    sensorEventQueue = nullptr;
  }
}

bool AndroidPlatform::ProcessEvents() {
  // Process Android events
  int events;
  android_poll_source* source;

  int ident;
  // Poll for events with a timeout of 0 (non-blocking)
  // We check for both LOOPER_ID_MAIN (cmd/input) and IDENT_SENSOR (3)
  while ((ident = ALooper_pollOnce(0, nullptr, &events, (void **) &source)) >= 0) {
    if (source != nullptr) {
      source->process(app, source);
    }

    // Handle sensors if they triggered the looper
    if (ident == 3 /*IDENT_SENSOR*/ && sensorEventQueue) {
      ASensorEvent event;
      while (ASensorEventQueue_getEvents(sensorEventQueue, &event, 1) > 0) {
        if (event.type == ASENSOR_TYPE_ACCELEROMETER) {
          accelX = event.acceleration.x;
          accelY = event.acceleration.y;
          accelZ = event.acceleration.z;
        }
      }
    }

    // Check if we are exiting
    if (app->destroyRequested != 0) {
      return false;
    }
  }

  // Handle GameActivity input events
  android_input_buffer* inputBuffer = android_app_swap_input_buffers(app);
  if (inputBuffer) {
    // Process motion events (touches)
    for (uint64_t i = 0; i < inputBuffer->motionEventsCount; ++i) {
      GameActivityMotionEvent& event = inputBuffer->motionEvents[i];

      int32_t action = event.action & AMOTION_EVENT_ACTION_MASK;

      if (event.pointerCount > 0) {
        // For mouse emulation, always follow the primary finger (index 0).
        // This avoids position "jumps" when multiple fingers are used.
        float x = GameActivityPointerAxes_getX(&event.pointers[0]);
        float y = GameActivityPointerAxes_getY(&event.pointers[0]);

        uint32_t buttons = 0;
        if (action == AMOTION_EVENT_ACTION_DOWN ||
            action == AMOTION_EVENT_ACTION_MOVE ||
            action == AMOTION_EVENT_ACTION_POINTER_DOWN) {
          buttons = 0x01; // Finger(s) down
        } else if (action == AMOTION_EVENT_ACTION_UP ||
                   action == AMOTION_EVENT_ACTION_CANCEL) {
          buttons = 0x00; // All fingers up
        } else if (action == AMOTION_EVENT_ACTION_POINTER_UP) {
          // One finger up, but others might still be down.
          // If the primary finger (0) was the one that left, the next finger
          // will become index 0 in the NEXT event, so we'll get a release
          // only when the LAST finger is lifted.
          buttons = (event.pointerCount > 1) ? 0x01 : 0x00;
        }

        // Diagnostic log for touch events (throttled)
        static int moveLogThrottler = 0;
        if (action != AMOTION_EVENT_ACTION_MOVE || ++moveLogThrottler % 30 == 0) {
           LOGI("Touch: act=%d pos=(%.1f, %.1f) btn=%u count=%d",
                action, x, y, buttons, event.pointerCount);
        }

        if (mouseCallback) {
          mouseCallback(x, y, buttons);
        }
      }
    }
    android_app_clear_motion_events(inputBuffer);

    // Process key events
    for (uint64_t i = 0; i < inputBuffer->keyEventsCount; ++i) {
      GameActivityKeyEvent& event = inputBuffer->keyEvents[i];
      if (keyboardCallback) {
        keyboardCallback(event.keyCode, event.action == AKEY_EVENT_ACTION_DOWN);
      }
    }
    android_app_clear_key_events(inputBuffer);
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

int AndroidPlatform::GetDisplayRotation() const {
  if (!app || !app->activity || !app->activity->javaGameActivity) return 0;

  JNIEnv* env = nullptr;
  app->activity->vm->AttachCurrentThread(&env, nullptr);
  int rotation = 0;
  if (env) {
    // 1. Get WindowManager from Activity via getWindowManager()
    jclass activityClass = env->GetObjectClass(app->activity->javaGameActivity);
    jmethodID getWindowManager = env->GetMethodID(activityClass, "getWindowManager", "()Landroid/view/WindowManager;");
    jobject windowManager = env->CallObjectMethod(app->activity->javaGameActivity, getWindowManager);

    // 2. Get Default Display from WindowManager
    jclass windowManagerClass = env->FindClass("android/view/WindowManager");
    jmethodID getDefaultDisplay = env->GetMethodID(windowManagerClass, "getDefaultDisplay", "()Landroid/view/Display;");
    jobject display = env->CallObjectMethod(windowManager, getDefaultDisplay);

    // 3. Get Rotation from Display
    jclass displayClass = env->FindClass("android/view/Display");
    jmethodID getRotation = env->GetMethodID(displayClass, "getRotation", "()I");
    rotation = env->CallIntMethod(display, getRotation);

    app->activity->vm->DetachCurrentThread();
  }
  return rotation;
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

  // Configure GameActivity to pass all motion and key events to the native side
  // without filtering. This ensures we see all touches, moves, and releases.
  android_app_set_motion_event_filter(app, nullptr);
  android_app_set_key_event_filter(app, nullptr);
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