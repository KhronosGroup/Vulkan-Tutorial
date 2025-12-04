// Intentionally empty bridge: rely entirely on GameActivity's native_app_glue
// provided by the prefab (libgame-activity). That glue will invoke our
// android_main(android_app*) defined in main.cpp. Defining another
// GameActivity_onCreate here causes duplicate symbol linker errors.
// Keeping a translation unit avoids removing the target from CMake.

#include <android/log.h>

#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "SimpleEngine", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "SimpleEngine", __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "SimpleEngine", __VA_ARGS__))

// Nothing to do here.