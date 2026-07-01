#pragma once

#include <vector>
#include <string>
#include <memory>

class IREEMNISTInference {
public:
    IREEMNISTInference(const std::string& modelPath);
    ~IREEMNISTInference();

    std::vector<float> infer(const std::vector<float>& pixels);

    bool isReady() const { return ready; }

private:
    bool ready = false;
    struct Impl;
    std::unique_ptr<Impl> impl;
};
