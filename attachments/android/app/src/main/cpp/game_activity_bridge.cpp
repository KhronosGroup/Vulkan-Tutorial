#include <game-activity/GameActivity.h>
#include <game-activity/native_app_glue/android_native_app_glue.h>
#include <android/log.h>
#include <cstring>

// Define logging macros
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "VulkanTutorial", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "VulkanTutorial", __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "VulkanTutorial", __VA_ARGS__))

// Forward declaration of the main entry point
extern "C" void android_main(android_app* app);

// GameActivity entry point
extern "C" {
// This is the function the GameActivity library will call.
// Ensure its signature matches what GameActivity expects:
// void GameActivity_onCreate(GameActivity* activity, void* savedState, size_t savedStateSize)
void GameActivity_onCreate(GameActivity* activity, void* savedState, size_t savedStateSize) {
    LOGI("GameActivity_onCreate");

    // Create an android_app structure
    android_app* app = new android_app(); // Consider using std::unique_ptr for better memory management
    memset(app, 0, sizeof(android_app));

    // Set up the android_app structure
    app->activity = activity;
    app->window = nullptr; // Window will be provided later through onNativeWindowCreated callback

    // Call the original android_main function
    android_main(app);

    // Clean up
    // IMPORTANT: The lifetime of 'app' needs to be managed carefully.
    // If android_main runs asynchronously or expects 'app' to live longer,
    // deleting it here might be premature.
    // Consider the lifecycle of your native_app_glue integration.
    // For example, 'app' might need to be freed when the activity is destroyed.
    delete app;
}
}