#pragma once

#include "weight_loader.hpp"
#include "third_party/nnef-parser/include/nnef.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>

class NNEFLoader {
public:
    static WeightLoader::ModelWeights load(const std::string& path) {
        nnef::Graph graph;
        std::string error;
        
        // Load the NNEF graph
        if (!nnef::load_graph(path, graph, error)) {
            throw std::runtime_error("Failed to load NNEF graph: " + error);
        }
        
        // Load the NNEF variables
        if (!nnef::load_variables(path, graph, error)) {
            throw std::runtime_error("Failed to load NNEF variables: " + error);
        }
        
        WeightLoader::ModelWeights weights;
        
        // Map NNEF variables to our weights structure
        // Based on graph.nnef inspection:
        // variable1: bias for linear1 (128) -> fc1_bias
        // variable2: bias for linear2 (10)  -> fc2_bias
        // variable3: weights for linear1 (128, 784) -> fc1_weights
        // variable4: weights for linear2 (10, 128)  -> fc2_weights
        
        weights.fc1_bias = get_tensor_data(graph, "variable1");
        weights.fc2_bias = get_tensor_data(graph, "variable2");
        weights.fc1_weights = get_tensor_data(graph, "variable3");
        weights.fc2_weights = get_tensor_data(graph, "variable4");
        
        // Fill other fields with dummy data if needed (for MNIST sample compatibility)
        weights.conv1_weights = {0.0f};
        weights.conv1_bias = {0.0f};
        weights.conv2_weights = {0.0f};
        weights.conv2_bias = {0.0f};
        weights.fc3_weights = {0.0f};
        weights.fc3_bias = {0.0f};
        
        std::cout << "Successfully loaded NNEF model from: " << path << std::endl;
        
        return weights;
    }

private:
    static std::vector<float> get_tensor_data(const nnef::Graph& graph, const std::string& name) {
        auto it = graph.tensors.find(name);
        if (it != graph.tensors.end()) {
            const nnef::Tensor& tensor = it->second;
            // Check if it's float32/scalar
            if (tensor.dtype != "float32" && tensor.dtype != "scalar") {
                throw std::runtime_error("Tensor " + name + " is not float32 or scalar (found: " + tensor.dtype + ")");
            }
            
            size_t count = 1;
            for (int dim : tensor.shape) {
                count *= dim;
            }
            
            const float* data = reinterpret_cast<const float*>(tensor.data.data());
            return std::vector<float>(data, data + count);
        }
        throw std::runtime_error("Tensor not found: " + name);
    }
};
