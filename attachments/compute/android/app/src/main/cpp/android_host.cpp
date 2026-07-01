// android_host.cpp
// Defines the globals declared extern in glfw_android_shim.h and provides
// the JNI entry points for MainActivity and VulkanChapterActivity.

#include "glfw_android_shim.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/log.h>
#include <jni.h>

// ---------------------------------------------------------------------------
// Global definitions (declared extern in glfw_android_shim.h)
// ---------------------------------------------------------------------------
ANativeWindow*               g_androidWindow  = nullptr;
std::atomic<bool>            g_shouldStop     = false;
void*                        g_userPointer    = nullptr;
double                       g_cursorX        = 0.0;
double                       g_cursorY        = 0.0;
std::mutex                   g_eventMutex;
std::queue<GlfwAndroidEvent> g_eventQueue;
GLFWframebuffersizefun       g_fbSizeCallback = nullptr;
GLFWscrollfun                g_scrollCb       = nullptr;
GLFWmousebuttonfun           g_mouseButtonCb  = nullptr;
GLFWcursorposfun             g_cursorPosCb    = nullptr;
GLFWkeyfun                   g_keyCb          = nullptr;

// Shared asset manager — used by android_host to pass to chapters.
// Named g_hostAssetMgr to avoid clashing with ch12's internal static g_assetMgr.
AAssetManager* g_hostAssetMgr = nullptr;

// ---------------------------------------------------------------------------
// Chapter run-function forward declarations
// ---------------------------------------------------------------------------
extern "C" void chapter02_run();
extern "C" void chapter03_run();
extern "C" void chapter04_run();
extern "C" void chapter05_run();
extern "C" void chapter06_run();
extern "C" void chapter07_run();
extern "C" void chapter08_run();
extern "C" void chapter09_run();
extern "C" void chapter10_run();
// Chapter 12 (12_embedded.cpp) is a headless embedded sample; it is intentionally
// absent from the Android launcher.  Build it for Raspberry Pi / Jetson via the
// desktop CMake target instead.

// ---------------------------------------------------------------------------
// Chapter metadata table (index matches kChapters[] position, not chapter number)
// ---------------------------------------------------------------------------
struct ChapterInfo {
    int         index;
    const char* name;
    const char* desc;
    bool        available;
    void      (*runFn)();
};

static const ChapterInfo kChapters[] = {
    { 2,  "Compute Architecture",
          "Mandelbrot set explorer — compute dispatch + storage image blit",
          true,  chapter02_run },
    { 3,  "Memory Models",
          "Fluid simulation (SPH) — multiple compute passes",
          true,  chapter03_run },
    { 4,  "Subgroup Operations",
          "Hair strands in wind — WaveActiveSum + WavePrefixSum",
          true,  chapter04_run },
    { 5,  "OpenCL on Vulkan",
          "Forest fractal — requires clspv; not available on Android",
          false, chapter05_run },
    { 6,  "Advanced Data Structures",
          "BVH ray tracer — GPU-side tree traversal",
          true,  chapter06_run },
    { 7,  "GPU-Driven Pipelines",
          "Asteroid field — GPU culling + indirect dispatch",
          true,  chapter07_run },
    { 8,  "Async Compute",
          "Cloth simulation — async compute + timeline semaphores",
          true,  chapter08_run },
    { 9,  "Specialized Math",
          "FP16 noise + denoise — cooperative math ops",
          true,  chapter09_run },
    { 10, "Performance Optimization",
          "GPU heatmap — occupancy + roofline analysis",
          true,  chapter10_run },
};
static constexpr int kChapterCount = (int)(sizeof(kChapters) / sizeof(kChapters[0]));

// ---------------------------------------------------------------------------
// JNI — MainActivity
// ---------------------------------------------------------------------------

extern "C" JNIEXPORT jint JNICALL
Java_com_vulkan_compute_MainActivity_nativeChapterCount(JNIEnv*, jclass)
{
    return kChapterCount;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_vulkan_compute_MainActivity_nativeChapterInfo(JNIEnv* env, jclass, jint i)
{
    if (i < 0 || i >= kChapterCount) return nullptr;
    jclass   strClass = env->FindClass("java/lang/String");
    jobjectArray arr  = env->NewObjectArray(3, strClass, nullptr);
    env->SetObjectArrayElement(arr, 0, env->NewStringUTF(kChapters[i].name));
    env->SetObjectArrayElement(arr, 1, env->NewStringUTF(kChapters[i].desc));
    env->SetObjectArrayElement(arr, 2, env->NewStringUTF(kChapters[i].available ? "true" : "false"));
    return arr;
}

// ---------------------------------------------------------------------------
// JNI — VulkanChapterActivity
// ---------------------------------------------------------------------------

extern "C" JNIEXPORT void JNICALL
Java_com_vulkan_compute_VulkanChapterActivity_nativeStart(
    JNIEnv* env, jobject, jobject surface, jobject assetMgr, jint chapterIdx)
{
    if (chapterIdx < 0 || chapterIdx >= kChapterCount) {
        __android_log_print(ANDROID_LOG_ERROR, "VulkanCompute",
                            "Invalid chapter index %d", chapterIdx);
        return;
    }
    if (!kChapters[chapterIdx].available) {
        __android_log_print(ANDROID_LOG_WARN, "VulkanCompute",
                            "Chapter %d not available on Android", chapterIdx);
        return;
    }

    g_androidWindow  = ANativeWindow_fromSurface(env, surface);
    g_hostAssetMgr   = AAssetManager_fromJava(env, assetMgr);
    g_shouldStop.store(false);

    // Reset shared GLFW state for this run
    {
        std::lock_guard<std::mutex> lk(g_eventMutex);
        while (!g_eventQueue.empty()) g_eventQueue.pop();
    }
    g_fbSizeCallback = nullptr;
    g_scrollCb       = nullptr;
    g_mouseButtonCb  = nullptr;
    g_cursorPosCb    = nullptr;
    g_keyCb          = nullptr;
    g_userPointer    = nullptr;

    __android_log_print(ANDROID_LOG_INFO, "VulkanCompute",
                        "Starting chapter %d: %s",
                        kChapters[chapterIdx].index, kChapters[chapterIdx].name);

    kChapters[chapterIdx].runFn();

    ANativeWindow_release(g_androidWindow);
    g_androidWindow = nullptr;
    g_hostAssetMgr  = nullptr;
}

extern "C" JNIEXPORT void JNICALL
Java_com_vulkan_compute_VulkanChapterActivity_nativeStop(JNIEnv*, jobject)
{
    g_shouldStop.store(true);
}

extern "C" JNIEXPORT void JNICALL
Java_com_vulkan_compute_VulkanChapterActivity_nativeTouchCursorPos(
    JNIEnv*, jobject, jdouble x, jdouble y)
{
    g_cursorX = x;
    g_cursorY = y;
    std::lock_guard<std::mutex> lk(g_eventMutex);
    GlfwAndroidEvent ev{};
    ev.type = GlfwAndroidEvent::CURSOR_POS;
    ev.x    = x;
    ev.y    = y;
    g_eventQueue.push(ev);
}

extern "C" JNIEXPORT void JNICALL
Java_com_vulkan_compute_VulkanChapterActivity_nativeMouseButton(
    JNIEnv*, jobject, jint button, jint action, jdouble x, jdouble y)
{
    g_cursorX = x;
    g_cursorY = y;
    std::lock_guard<std::mutex> lk(g_eventMutex);
    GlfwAndroidEvent ev{};
    ev.type   = GlfwAndroidEvent::MOUSE_BUTTON;
    ev.button = button;
    ev.action = action;
    ev.x      = x;
    ev.y      = y;
    g_eventQueue.push(ev);
}

extern "C" JNIEXPORT void JNICALL
Java_com_vulkan_compute_VulkanChapterActivity_nativeScroll(
    JNIEnv*, jobject, jdouble dx, jdouble dy)
{
    std::lock_guard<std::mutex> lk(g_eventMutex);
    GlfwAndroidEvent ev{};
    ev.type = GlfwAndroidEvent::SCROLL;
    ev.x    = dx;
    ev.y    = dy;
    g_eventQueue.push(ev);
}

extern "C" JNIEXPORT void JNICALL
Java_com_vulkan_compute_VulkanChapterActivity_nativeResize(
    JNIEnv*, jobject, jint w, jint h)
{
    std::lock_guard<std::mutex> lk(g_eventMutex);
    GlfwAndroidEvent ev{};
    ev.type  = GlfwAndroidEvent::FB_RESIZE;
    ev.width  = w;
    ev.height = h;
    g_eventQueue.push(ev);
}
