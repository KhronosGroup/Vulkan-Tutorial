#ifndef GLFW_ANDROID_SHIM_H
#define GLFW_ANDROID_SHIM_H

// GLFW compatibility shim for Android.
// Provides enough of the GLFW API surface to compile chapters 02-10 on Android NDK
// using ANativeWindow + touch events instead of a real GLFW window.
//
// All functions are inline; globals are declared extern and defined in android_host.cpp.

#define VK_USE_PLATFORM_ANDROID_KHR
#include <vulkan/vulkan.h>
#include <android/native_window.h>
#include <android/log.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <queue>
#include <string>

// ---------------------------------------------------------------------------
// GLFW types
// ---------------------------------------------------------------------------
typedef struct GLFWwindow  GLFWwindow;
typedef struct GLFWmonitor GLFWmonitor;

typedef void (*GLFWframebuffersizefun)(GLFWwindow*, int, int);
typedef void (*GLFWscrollfun)(GLFWwindow*, double, double);
typedef void (*GLFWmousebuttonfun)(GLFWwindow*, int, int, int);
typedef void (*GLFWcursorposfun)(GLFWwindow*, double, double);
typedef void (*GLFWkeyfun)(GLFWwindow*, int, int, int, int);

// ---------------------------------------------------------------------------
// GLFW constants
// ---------------------------------------------------------------------------
#define GLFW_CLIENT_API        0x22001
#define GLFW_NO_API            0
#define GLFW_RESIZABLE         0x20003
#define GLFW_TRUE              1
#define GLFW_FALSE             0
#define GLFW_PRESS             1
#define GLFW_RELEASE           0
#define GLFW_MOUSE_BUTTON_LEFT 0

#define GLFW_KEY_ESCAPE        256
#define GLFW_KEY_R             82
#define GLFW_KEY_EQUAL         61
#define GLFW_KEY_MINUS         45
#define GLFW_KEY_W             87
#define GLFW_KEY_A             65
#define GLFW_KEY_S             83
#define GLFW_KEY_D             68
#define GLFW_KEY_E             69
#define GLFW_KEY_Q             81
#define GLFW_KEY_LEFT_SHIFT    340

// ---------------------------------------------------------------------------
// Event queue
// ---------------------------------------------------------------------------
struct GlfwAndroidEvent {
    enum { CURSOR_POS, MOUSE_BUTTON, SCROLL, FB_RESIZE } type;
    double x = 0, y = 0;
    int button = 0, action = 0;
    int width = 0, height = 0;
};

// ---------------------------------------------------------------------------
// Extern globals — defined in android_host.cpp
// ---------------------------------------------------------------------------
extern ANativeWindow*          g_androidWindow;
extern std::atomic<bool>       g_shouldStop;
extern void*                   g_userPointer;
extern double                  g_cursorX;
extern double                  g_cursorY;
extern std::mutex              g_eventMutex;
extern std::queue<GlfwAndroidEvent> g_eventQueue;

extern GLFWframebuffersizefun  g_fbSizeCallback;
extern GLFWscrollfun           g_scrollCb;
extern GLFWmousebuttonfun      g_mouseButtonCb;
extern GLFWcursorposfun        g_cursorPosCb;
extern GLFWkeyfun              g_keyCb;

// ---------------------------------------------------------------------------
// GLFW function implementations (inline — compiled once per TU that includes this)
// ---------------------------------------------------------------------------
inline int glfwInit() { return 1; }
inline void glfwTerminate() {}
inline void glfwWindowHint(int, int) {}

inline GLFWwindow* glfwCreateWindow(int, int, const char*, GLFWmonitor*, GLFWwindow*) {
    static int dummy = 0;
    return reinterpret_cast<GLFWwindow*>(&dummy);
}

inline void glfwDestroyWindow(GLFWwindow*) {}

inline int glfwWindowShouldClose(GLFWwindow*) {
    return g_shouldStop.load() ? 1 : 0;
}

inline void glfwSetWindowShouldClose(GLFWwindow*, int v) {
    g_shouldStop.store(v != 0);
}

inline void glfwPollEvents() {
    std::queue<GlfwAndroidEvent> local;
    {
        std::lock_guard<std::mutex> lk(g_eventMutex);
        local.swap(g_eventQueue);
    }
    while (!local.empty()) {
        GlfwAndroidEvent ev = local.front();
        local.pop();
        GLFWwindow* w = glfwCreateWindow(0, 0, nullptr, nullptr, nullptr);
        switch (ev.type) {
            case GlfwAndroidEvent::CURSOR_POS:
                if (g_cursorPosCb) g_cursorPosCb(w, ev.x, ev.y);
                break;
            case GlfwAndroidEvent::MOUSE_BUTTON:
                if (g_mouseButtonCb) g_mouseButtonCb(w, ev.button, ev.action, 0);
                break;
            case GlfwAndroidEvent::SCROLL:
                if (g_scrollCb) g_scrollCb(w, ev.x, ev.y);
                break;
            case GlfwAndroidEvent::FB_RESIZE:
                if (g_fbSizeCallback) g_fbSizeCallback(w, ev.width, ev.height);
                break;
        }
    }
}

inline void glfwWaitEvents() {
    glfwPollEvents();
}

inline void* glfwGetWindowUserPointer(GLFWwindow*) { return g_userPointer; }
inline void  glfwSetWindowUserPointer(GLFWwindow*, void* p) { g_userPointer = p; }

inline void glfwGetFramebufferSize(GLFWwindow*, int* w, int* h) {
    if (g_androidWindow) {
        *w = ANativeWindow_getWidth(g_androidWindow);
        *h = ANativeWindow_getHeight(g_androidWindow);
    } else {
        *w = *h = 0;
    }
}

inline void glfwGetWindowSize(GLFWwindow* win, int* w, int* h) {
    glfwGetFramebufferSize(win, w, h);
}

inline void glfwGetCursorPos(GLFWwindow*, double* x, double* y) {
    *x = g_cursorX;
    *y = g_cursorY;
}

inline int glfwGetKey(GLFWwindow*, int) { return GLFW_RELEASE; }

inline double glfwGetTime() {
    static auto start = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start).count();
}

inline const char** glfwGetRequiredInstanceExtensions(uint32_t* count) {
    static const char* exts[] = { "VK_KHR_surface", "VK_KHR_android_surface" };
    *count = 2;
    return exts;
}

inline VkResult glfwCreateWindowSurface(
    VkInstance inst,
    GLFWwindow*,
    const VkAllocationCallbacks* alloc,
    VkSurfaceKHR* out)
{
    VkAndroidSurfaceCreateInfoKHR ci{};
    ci.sType  = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
    ci.window = g_androidWindow;
    return vkCreateAndroidSurfaceKHR(inst, &ci, alloc, out);
}

// Callback setters — store the function pointer and return the old one.
inline GLFWframebuffersizefun glfwSetFramebufferSizeCallback(GLFWwindow*, GLFWframebuffersizefun fn) {
    GLFWframebuffersizefun old = g_fbSizeCallback;
    g_fbSizeCallback = fn;
    return old;
}
inline GLFWscrollfun glfwSetScrollCallback(GLFWwindow*, GLFWscrollfun fn) {
    GLFWscrollfun old = g_scrollCb;
    g_scrollCb = fn;
    return old;
}
inline GLFWmousebuttonfun glfwSetMouseButtonCallback(GLFWwindow*, GLFWmousebuttonfun fn) {
    GLFWmousebuttonfun old = g_mouseButtonCb;
    g_mouseButtonCb = fn;
    return old;
}
inline GLFWcursorposfun glfwSetCursorPosCallback(GLFWwindow*, GLFWcursorposfun fn) {
    GLFWcursorposfun old = g_cursorPosCb;
    g_cursorPosCb = fn;
    return old;
}
inline GLFWkeyfun glfwSetKeyCallback(GLFWwindow*, GLFWkeyfun fn) {
    GLFWkeyfun old = g_keyCb;
    g_keyCb = fn;
    return old;
}

#endif // GLFW_ANDROID_SHIM_H
