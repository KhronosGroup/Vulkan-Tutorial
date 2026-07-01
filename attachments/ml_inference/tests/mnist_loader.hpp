#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <stdint.h>
#include <stdexcept>
#include <arpa/inet.h> // for ntohl

class MNISTLoader {
public:
    static uint32_t swapEndian(uint32_t val) {
#if defined(__linux__) || defined(__APPLE__)
        return ntohl(val);
#else
        return ((val << 24) & 0xff000000) |
               ((val << 8) & 0x00ff0000) |
               ((val >> 8) & 0x0000ff00) |
               ((val >> 24) & 0x000000ff);
#endif
    }

    static std::vector<std::vector<float>> loadImages(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Failed to open " + path);

        uint32_t magic = 0, numImages = 0, rows = 0, cols = 0;
        file.read((char*)&magic, 4);
        file.read((char*)&numImages, 4);
        file.read((char*)&rows, 4);
        file.read((char*)&cols, 4);

        magic = swapEndian(magic);
        numImages = swapEndian(numImages);
        rows = swapEndian(rows);
        cols = swapEndian(cols);

        if (magic != 2051) throw std::runtime_error("Invalid MNIST image magic number");

        std::vector<std::vector<float>> images(numImages, std::vector<float>(rows * cols));
        for (uint32_t i = 0; i < numImages; ++i) {
            std::vector<uint8_t> buffer(rows * cols);
            file.read((char*)buffer.data(), rows * cols);
            for (uint32_t j = 0; j < rows * cols; ++j) {
                images[i][j] = buffer[j] / 255.0f;
            }
        }
        return images;
    }

    static std::vector<uint8_t> loadLabels(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Failed to open " + path);

        uint32_t magic = 0, numLabels = 0;
        file.read((char*)&magic, 4);
        file.read((char*)&numLabels, 4);

        magic = swapEndian(magic);
        numLabels = swapEndian(numLabels);

        if (magic != 2049) throw std::runtime_error("Invalid MNIST label magic number");

        std::vector<uint8_t> labels(numLabels);
        file.read((char*)labels.data(), numLabels);
        return labels;
    }
};

#endif
