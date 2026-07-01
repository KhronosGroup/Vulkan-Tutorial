#pragma once

#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>

/**
 * Loads model weights from binary file created by train_mnist.py
 */
class WeightLoader {
public:
    struct ModelWeights {
        std::vector<float> conv1_weights;
        std::vector<float> conv1_bias;
        std::vector<float> conv2_weights;
        std::vector<float> conv2_bias;
        std::vector<float> fc1_weights;
        std::vector<float> fc1_bias;
        std::vector<float> fc2_weights;
        std::vector<float> fc2_bias;
        std::vector<float> fc3_weights;
        std::vector<float> fc3_bias;
    };

    static ModelWeights load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open weights file: " + filename);
        }

        // Read and verify magic number
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x4D4E5354) {  // 'MNST'
            throw std::runtime_error("Invalid weights file format");
        }

        // Read version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) {
            throw std::runtime_error("Unsupported weights file version");
        }

        ModelWeights weights;

        // Helper to read tensor
        auto readTensor = [&file]() {
            uint32_t count;
            file.read(reinterpret_cast<char*>(&count), sizeof(count));

            std::vector<float> data(count);
            file.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));

            if (!file.good()) {
                throw std::runtime_error("Error reading tensor data");
            }

            return data;
        };

        // Read weights in same order as exported
        weights.conv1_weights = readTensor();
        weights.conv1_bias = readTensor();
        weights.conv2_weights = readTensor();
        weights.conv2_bias = readTensor();
        weights.fc1_weights = readTensor();
        weights.fc1_bias = readTensor();
        weights.fc2_weights = readTensor();
        weights.fc2_bias = readTensor();
        weights.fc3_weights = readTensor();
        weights.fc3_bias = readTensor();

        return weights;
    }
};
