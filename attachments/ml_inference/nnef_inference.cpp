#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include "common/renderer/renderer.h"
#include "common/vulkan_nnef_inference.h"

/**
 * Sample 7: Desktop NNEF Engine
 * 
 * A standalone CLI tool to run NNEF models using the custom Vulkan engine.
 * Demonstrates the Engine Chapter 06: "NNEF Deep Dive".
 * 
 * Usage: nnef_inference --model <path> --input <bin_file> [--output <path>]
 */

int main(int argc, char** argv) {
    std::string modelPath = "models/mobilenetv2_nnef_optimized";
    std::string inputPath = "";
    std::string outputPath = "output.bin";
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) modelPath = argv[++i];
        else if (arg == "--input" && i + 1 < argc) inputPath = argv[++i];
        else if (arg == "--output" && i + 1 < argc) outputPath = argv[++i];
        else if (arg == "--verbose") verbose = true;
    }

    if (inputPath.empty()) {
        std::cout << "Usage: nnef_inference --model <path> --input <bin_file> [--output <path>] [--verbose]" << std::endl;
        std::cout << "  --model:  Path to NNEF model directory" << std::endl;
        std::cout << "  --input:  Path to raw float32 input binary file" << std::endl;
        std::cout << "  --output: Path to save raw float32 output binary file (default: output.bin)" << std::endl;
        return 1;
    }

    std::cout << "--- Desktop NNEF Engine ---" << std::endl;

    // 1. Initialize Headless Renderer
    // We use a dummy window size as we only need the device for compute
    Renderer renderer(nullptr, 224, 224);
    
    // 2. Load NNEF Model
    VulkanNNEFInference engine(renderer);
    if (!engine.loadModel(modelPath)) {
        std::cerr << "Failed to load NNEF model: " << modelPath << std::endl;
        return 1;
    }

    // 3. Load Input Data
    std::ifstream is(inputPath, std::ios::binary | std::ios::ate);
    if (!is.is_open()) {
        std::cerr << "Failed to open input file: " << inputPath << std::endl;
        return 1;
    }
    std::streamsize size = is.tellg();
    is.seekg(0, std::ios::beg);
    
    std::vector<float> input(size / sizeof(float));
    if (!is.read(reinterpret_cast<char*>(input.data()), size)) {
        std::cerr << "Failed to read input data." << std::endl;
        return 1;
    }

    std::cout << "Loaded input: " << input.size() << " floats." << std::endl;

    // 4. Run Inference
    auto start = std::chrono::high_resolution_clock::now();
    auto results = engine.infer(input);
    auto end = std::chrono::high_resolution_clock::now();
    
    float duration = std::chrono::duration<float, std::milli>(end - start).count();
    std::cout << "Inference completed in " << duration << " ms." << std::endl;

    if (results.empty()) {
        std::cerr << "Inference returned no results." << std::endl;
        return 1;
    }

    // 5. Save Results
    std::ofstream os(outputPath, std::ios::binary);
    if (!os.is_open()) {
        std::cerr << "Failed to open output file: " << outputPath << std::endl;
        return 1;
    }
    os.write(reinterpret_cast<const char*>(results.data()), results.size() * sizeof(float));
    
    std::cout << "Results saved to " << outputPath << std::endl;
    
    if (verbose) {
        std::cout << "Top 5 results:" << std::endl;
        std::vector<std::pair<float, int>> topK;
        for (int i = 0; i < (int)results.size(); ++i) topK.push_back({results[i], i});
        std::sort(topK.begin(), topK.end(), std::greater<>());
        for (int i = 0; i < std::min(5, (int)topK.size()); ++i) {
            std::cout << "  Class " << topK[i].second << ": " << topK[i].first << std::endl;
        }
    }

    return 0;
}
