#pragma once

#include "debug_system.h"

// RenderDoc integration is optional and loaded at runtime.
// This header intentionally does NOT include <renderdoc_app.h> to avoid a hard dependency.
// Instead, we declare a minimal interface and dynamically resolve the API if present.

class RenderDocDebugSystem final : public DebugSystem {
public:
    static RenderDocDebugSystem& GetInstance() {
        static RenderDocDebugSystem instance;
        return instance;
    }

    // Attempt to load the RenderDoc API from the current process.
    // Safe to call multiple times.
    bool LoadRenderDocAPI();

    // Returns true if the RenderDoc API has been successfully loaded.
    bool IsAvailable() const { return renderdocAvailable; }

    // Triggers an immediate capture (equivalent to pressing the capture hotkey in the UI).
    void TriggerCapture();

    // Starts a frame capture for the given device/window (can be nullptr to auto-detect on many backends).
    void StartFrameCapture(void* device = nullptr, void* window = nullptr);

    // Ends a frame capture previously started. Returns true on success.
    bool EndFrameCapture(void* device = nullptr, void* window = nullptr);

private:
    RenderDocDebugSystem() = default;
    ~RenderDocDebugSystem() override = default;

    RenderDocDebugSystem(const RenderDocDebugSystem&) = delete;
    RenderDocDebugSystem& operator=(const RenderDocDebugSystem&) = delete;

    // Internal function pointers matching the subset of RenderDoc API we use.
    // We avoid including the official header by declaring minimal signatures.
    using pRENDERDOC_GetAPI = int (*)(int, void**);

    // Subset of API function pointers
    typedef void (*pRENDERDOC_TriggerCapture)();
    typedef void (*pRENDERDOC_StartFrameCapture)(void* device, void* window);
    typedef unsigned int (*pRENDERDOC_EndFrameCapture)(void* device, void* window); // returns bool in C API

    // Storage for resolved API
    pRENDERDOC_TriggerCapture fnTriggerCapture = nullptr;
    pRENDERDOC_StartFrameCapture fnStartFrameCapture = nullptr;
    pRENDERDOC_EndFrameCapture fnEndFrameCapture = nullptr;

    bool renderdocAvailable = false;
};
