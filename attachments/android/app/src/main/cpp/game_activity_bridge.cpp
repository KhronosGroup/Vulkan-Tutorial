#include <game-activity/GameActivity.h>
#include <game-activity/native_app_glue/android_native_app_glue.h>
#include <android/log.h>

// Define logging macros
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "VulkanTutorial", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "VulkanTutorial", __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "VulkanTutorial", __VA_ARGS__))

// Forward declaration of the main entry point
extern "C" void android_main(android_app* app);

// GameActivity entry point
extern "C" {
    void GameActivity_onCreate(GameActivity* activity) {
        LOGI("GameActivity_onCreate");

        // Create an android_app structure
        android_app* app = new android_app();
        memset(app, 0, sizeof(android_app));

        // Set up the android_app structure
        app->activity = activity;
        app->window = activity->window;

        // Call the original android_main function
        android_main(app);

        // Clean up
        delete app;
    }
}
