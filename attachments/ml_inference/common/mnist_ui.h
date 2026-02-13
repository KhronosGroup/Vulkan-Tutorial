#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include "imgui.h"

// 28x28 drawing canvas
class DrawingCanvas {
public:
    static constexpr int SIZE = 28;

    DrawingCanvas() : pixels_(SIZE * SIZE, 0.0f) {}

    void clear() { std::fill(pixels_.begin(), pixels_.end(), 0.0f); }

    void draw(int x, int y) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int px = x + dx;
                int py = y + dy;
                if (px >= 0 && px < SIZE && py >= 0 && py < SIZE) {
                    pixels_[py * SIZE + px] = std::min(1.0f, pixels_[py * SIZE + px] + 0.3f);
                }
            }
        }
    }

    float getPixel(int x, int y) const { return pixels_[y * SIZE + x]; }
    const std::vector<float>& getPixels() const { return pixels_; }
    
    // For programmatic filling (validation)
    void setPixel(int x, int y, float val) {
        if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
            pixels_[y * SIZE + x] = val;
        }
    }

private:
    std::vector<float> pixels_;
};

struct MNISTUIState {
    DrawingCanvas canvas;
    int inferenceMode = 0; // 0: Vulkan, 1: ONNX, 2: IREE
    std::vector<float> probabilities = std::vector<float>(10, 0.0f);
    int predictedDigit = -1;
    float lastInferenceTimeMs = 0.0f;
    std::string statusMessage = "";
    bool engineReady = false;
    bool onnxReady = false;
    bool ireeReady = false;
    bool isDrawing = false;
};

struct MNISTUIEvents {
    bool recognizeClicked = false;
    bool clearClicked = false;
};

inline MNISTUIEvents RenderMNISTUI(MNISTUIState& state, int width, int height) {
    MNISTUIEvents events;
    
    // Main UI - use current window size, force update every frame
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(static_cast<float>(width), static_cast<float>(height)), ImGuiCond_Always);
    ImGui::Begin("MNIST Recognition", nullptr,
                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus);

    // Header with status
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "MNIST Digit Recognition");

    if (state.engineReady) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), " [Ready]");
    } else {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), " [No Model - Train First]");
    }

    ImGui::Separator();
    ImGui::TextWrapped("Draw a digit (0-9) in the canvas. Click and drag to draw.");
    ImGui::Spacing();
    
    // Use a table for reliable two-column layout
    if (ImGui::BeginTable("MNISTLayout", 2)) {
        ImGui::TableSetupColumn("Controls", ImGuiTableColumnFlags_WidthFixed, 360.0f);
        ImGui::TableSetupColumn("Results", ImGuiTableColumnFlags_WidthStretch);
        
        ImGui::TableNextRow();
        ImGui::TableNextColumn();

        float contentHeight = ImGui::GetContentRegionAvail().y;

        // LEFT SIDE: Engine, Canvas and buttons
        ImGui::BeginChild("LeftPanel", ImVec2(0, contentHeight), false);
        
        ImGui::Text("Engine:");
        ImGui::SameLine();
        ImGui::RadioButton("Vulkan", &state.inferenceMode, 0);
        ImGui::SameLine();
        
        if (state.onnxReady) {
            ImGui::RadioButton("ONNX", &state.inferenceMode, 1);
        } else {
            ImGui::BeginDisabled();
            ImGui::RadioButton("ONNX", &state.inferenceMode, 1);
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1, 0, 0, 1), "(ONNX Missing)");
        }
        ImGui::SameLine();

        if (state.ireeReady) {
            ImGui::RadioButton("IREE", &state.inferenceMode, 2);
        } else {
            ImGui::BeginDisabled();
            ImGui::RadioButton("IREE", &state.inferenceMode, 2);
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1, 0, 0, 1), "(IREE Missing)");
        }
        ImGui::Separator();
        
        ImGui::Text("Canvas:");

        // Canvas
        ImVec2 canvasPos = ImGui::GetCursorScreenPos();
        ImVec2 canvasSize(280, 280);
        ImDrawList* drawList = ImGui::GetWindowDrawList();

        drawList->AddRectFilled(canvasPos,
                               ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
                               IM_COL32(30, 30, 30, 255));

        // Draw pixels
        for (int y = 0; y < DrawingCanvas::SIZE; ++y) {
            for (int x = 0; x < DrawingCanvas::SIZE; ++x) {
                float val = state.canvas.getPixel(x, y);
                if (val > 0.01f) {
                    int color = static_cast<int>(val * 255);
                    drawList->AddRectFilled(
                        ImVec2(canvasPos.x + x * 10, canvasPos.y + y * 10),
                        ImVec2(canvasPos.x + (x + 1) * 10, canvasPos.y + (y + 1) * 10),
                        IM_COL32(color, color, color, 255)
                    );
                }
            }
        }

        drawList->AddRect(canvasPos,
                         ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
                         IM_COL32(100, 100, 100, 255), 0.0f, 0, 2.0f);

        // Mouse input
        ImGui::InvisibleButton("canvas", canvasSize);
        if (ImGui::IsItemActive() && ImGui::IsMouseDown(0)) {
            ImVec2 mousePos = ImGui::GetMousePos();
            int cx = static_cast<int>((mousePos.x - canvasPos.x) / 10);
            int cy = static_cast<int>((mousePos.y - canvasPos.y) / 10);
            if (cx >= 0 && cx < DrawingCanvas::SIZE && cy >= 0 && cy < DrawingCanvas::SIZE) {
                state.canvas.draw(cx, cy);
                state.isDrawing = true;
            }
        } else if (state.isDrawing) {
            state.isDrawing = false;
        }

        ImGui::Spacing();

        // Buttons
        if (ImGui::Button("Clear", ImVec2(130, 35))) {
            events.clearClicked = true;
        }

        ImGui::SameLine();

        if (!state.engineReady) {
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5f);
        }

        if (ImGui::Button("Recognize", ImVec2(130, 35)) && state.engineReady) {
            events.recognizeClicked = true;
        }

        if (!state.engineReady) {
            ImGui::PopStyleVar();
        }

        ImGui::EndChild();

        ImGui::TableNextColumn();

        // RIGHT SIDE: Results and probabilities
        ImGui::BeginChild("RightPanel", ImVec2(0, contentHeight), false);

        ImGui::Text("Results:");
        ImGui::Separator();

        // Status message
        if (!state.statusMessage.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "%s", state.statusMessage.c_str());
            ImGui::Spacing();
        }

        // Prediction result
        if (state.predictedDigit >= 0 && state.engineReady) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.2f, 1.0f, 0.4f, 1.0f));
            ImGui::Text("Prediction: %d", state.predictedDigit);
            ImGui::PopStyleColor();
            ImGui::Text("Confidence: %.1f%%", state.probabilities[state.predictedDigit] * 100.0f);
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Inference: %.2f ms", state.lastInferenceTimeMs);
            ImGui::Spacing();
        } else if (state.statusMessage.empty()) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Draw and click Recognize");
            ImGui::Spacing();
        }

        ImGui::Separator();
        ImGui::Text("All Probabilities:");
        ImGui::Spacing();

        // Probabilities
        for (int i = 0; i < 10; ++i) {
            char label[32];
            snprintf(label, sizeof(label), "%d: %.1f%%", i, state.probabilities[i] * 100.0f);
            ImGui::ProgressBar(state.probabilities[i], ImVec2(-1, 0), label);
        }

        ImGui::EndChild();
        ImGui::EndTable();
    }

    ImGui::End();
    
    return events;
}
