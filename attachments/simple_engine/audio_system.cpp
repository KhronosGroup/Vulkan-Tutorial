#include "audio_system.h"

#include <cmath>
#include <cstring>
#include <iostream>

#include "renderer.h"

// Concrete implementation of AudioSource
class ConcreteAudioSource : public AudioSource {
public:
    explicit ConcreteAudioSource(const std::string& name) : name(name) {}
    ~ConcreteAudioSource() override = default;

    void Play() override {
        playing = true;
        std::cout << "Playing audio source: " << name << std::endl;
    }

    void Pause() override {
        playing = false;
        std::cout << "Pausing audio source: " << name << std::endl;
    }

    void Stop() override {
        playing = false;
        std::cout << "Stopping audio source: " << name << std::endl;
    }

    void SetVolume(float volume) override {
        this->volume = volume;
        std::cout << "Setting volume of audio source " << name << " to " << volume << std::endl;
    }

    void SetLoop(bool loop) override {
        this->loop = loop;
        std::cout << "Setting loop of audio source " << name << " to " << (loop ? "true" : "false") << std::endl;
    }

    void SetPosition(float x, float y, float z) override {
        position[0] = x;
        position[1] = y;
        position[2] = z;
        std::cout << "Setting position of audio source " << name << " to (" << x << ", " << y << ", " << z << ")" << std::endl;
    }

    void SetVelocity(float x, float y, float z) override {
        velocity[0] = x;
        velocity[1] = y;
        velocity[2] = z;
        std::cout << "Setting velocity of audio source " << name << " to (" << x << ", " << y << ", " << z << ")" << std::endl;
    }

    [[nodiscard]] bool IsPlaying() const override {
        return playing;
    }

private:
    std::string name;
    bool playing = false;
    bool loop = false;
    float volume = 1.0f;
    float position[3] = {0.0f, 0.0f, 0.0f};
    float velocity[3] = {0.0f, 0.0f, 0.0f};
};

AudioSystem::~AudioSystem() {
    // Destructor implementation
    sources.clear();
    audioData.clear();

    // Clean up HRTF buffers
    cleanupHRTFBuffers();
}

bool AudioSystem::Initialize(Renderer* renderer) {
    // This is a placeholder implementation
    // In a real implementation, this would initialize the audio API (e.g., OpenAL)

    std::cout << "Initializing audio system" << std::endl;

    // Store the renderer for compute shader support
    this->renderer = renderer;

    initialized = true;
    return true;
}

void AudioSystem::Update(float deltaTime) {
    // This is a placeholder implementation
    // In a real implementation, this would update the audio system

    // Update listener position, orientation, and velocity based on camera

    // Update audio sources
    for (auto& source : sources) {
        // Update source properties
    }
}

bool AudioSystem::LoadAudio(const std::string& filename, const std::string& name) {
    // This is a placeholder implementation
    // In a real implementation, this would load the audio file

    std::cout << "Loading audio file: " << filename << " as " << name << std::endl;

    // Simulate loading audio data
    std::vector<uint8_t> data(1024, 0); // Dummy data
    audioData[name] = data;

    return true;
}

AudioSource* AudioSystem::CreateAudioSource(const std::string& name) {
    // Check if the audio data exists
    auto it = audioData.find(name);
    if (it == audioData.end()) {
        std::cerr << "AudioSystem::CreateAudioSource: Audio data not found: " << name << std::endl;
        return nullptr;
    }

    // Create a new audio source
    auto source = std::make_unique<ConcreteAudioSource>(name);

    // Store the source
    AudioSource* sourcePtr = source.get();
    sources.push_back(std::move(source));

    std::cout << "Audio source created: " << name << std::endl;
    return sourcePtr;
}

void AudioSystem::SetListenerPosition(float x, float y, float z) {
    listenerPosition[0] = x;
    listenerPosition[1] = y;
    listenerPosition[2] = z;

    std::cout << "Setting listener position to (" << x << ", " << y << ", " << z << ")" << std::endl;
}

void AudioSystem::SetListenerOrientation(float forwardX, float forwardY, float forwardZ,
                                       float upX, float upY, float upZ) {
    listenerOrientation[0] = forwardX;
    listenerOrientation[1] = forwardY;
    listenerOrientation[2] = forwardZ;
    listenerOrientation[3] = upX;
    listenerOrientation[4] = upY;
    listenerOrientation[5] = upZ;

    std::cout << "Setting listener orientation to forward=(" << forwardX << ", " << forwardY << ", " << forwardZ << "), "
              << "up=(" << upX << ", " << upY << ", " << upZ << ")" << std::endl;
}

void AudioSystem::SetListenerVelocity(float x, float y, float z) {
    listenerVelocity[0] = x;
    listenerVelocity[1] = y;
    listenerVelocity[2] = z;

    std::cout << "Setting listener velocity to (" << x << ", " << y << ", " << z << ")" << std::endl;
}

void AudioSystem::SetMasterVolume(float volume) {
    masterVolume = volume;

    std::cout << "Setting master volume to " << volume << std::endl;
}

void AudioSystem::EnableHRTF(bool enable) {
    hrtfEnabled = enable;
    std::cout << "HRTF processing " << (enable ? "enabled" : "disabled") << std::endl;
}

bool AudioSystem::IsHRTFEnabled() const {
    return hrtfEnabled;
}

bool AudioSystem::LoadHRTFData(const std::string& filename) {
    // This is a placeholder implementation
    // In a real implementation, this would load HRTF data from a file

    std::cout << "Loading HRTF data from: " << filename << std::endl;

    // Simulate loading HRTF data
    // In a real implementation, this would parse a file containing HRTF impulse responses

    // Create some dummy HRTF data for testing
    // Typically, HRTF data consists of impulse responses for different directions
    const uint32_t hrtfSampleCount = 256;  // Number of samples per impulse response
    const uint32_t positionCount = 36 * 13; // 36 azimuths (10-degree steps) * 13 elevations (15-degree steps)
    const uint32_t channelCount = 2;       // Stereo (left and right ears)

    // Resize the HRTF data vector
    hrtfData.resize(hrtfSampleCount * positionCount * channelCount);

    // Fill with dummy data (simple exponential decay)
    for (uint32_t pos = 0; pos < positionCount; pos++) {
        for (uint32_t channel = 0; channel < channelCount; channel++) {
            for (uint32_t i = 0; i < hrtfSampleCount; i++) {
                float value = std::exp(-static_cast<float>(i) / 20.0f) * 0.5f;
                // Add some variation based on position and channel
                value *= (1.0f + 0.2f * std::sin(pos * 0.1f + channel * 3.14159f));

                uint32_t index = pos * hrtfSampleCount * channelCount + channel * hrtfSampleCount + i;
                hrtfData[index] = value;
            }
        }
    }

    // Store HRTF parameters
    hrtfSize = hrtfSampleCount;
    numHrtfPositions = positionCount;

    return true;
}

bool AudioSystem::ProcessHRTF(const float* inputBuffer, float* outputBuffer, uint32_t sampleCount, const float* sourcePosition) {
    if (!hrtfEnabled || !renderer || !renderer->IsInitialized()) {
        // If HRTF is disabled or renderer is not available, just copy input to output
        for (uint32_t i = 0; i < sampleCount; i++) {
            outputBuffer[i * 2] = inputBuffer[i];     // Left channel
            outputBuffer[i * 2 + 1] = inputBuffer[i]; // Right channel
        }
        return true;
    }

    // Create buffers for HRTF processing if they don't exist or if the sample count has changed
    if (!createHRTFBuffers(sampleCount)) {
        std::cerr << "Failed to create HRTF buffers" << std::endl;
        return false;
    }

    // Copy input data to input buffer
    void* data = inputBufferMemory.mapMemory(0, sampleCount * sizeof(float));
    memcpy(data, inputBuffer, sampleCount * sizeof(float));
    inputBufferMemory.unmapMemory();

    // Set up HRTF parameters
    struct HRTFParams {
        float sourcePosition[3];
        float listenerPosition[3];
        float listenerOrientation[6]; // Forward (3) and up (3) vectors
        uint32_t sampleCount;
        uint32_t hrtfSize;
        uint32_t numHrtfPositions;
        float padding; // For alignment
    } params;

    // Copy source and listener positions
    memcpy(params.sourcePosition, sourcePosition, sizeof(float) * 3);
    memcpy(params.listenerPosition, listenerPosition, sizeof(float) * 3);
    memcpy(params.listenerOrientation, listenerOrientation, sizeof(float) * 6);
    params.sampleCount = sampleCount;
    params.hrtfSize = hrtfSize;
    params.numHrtfPositions = numHrtfPositions;
    params.padding = 0.0f;

    // Copy parameters to parameter buffer
    data = paramsBufferMemory.mapMemory(0, sizeof(HRTFParams));
    memcpy(data, &params, sizeof(HRTFParams));
    paramsBufferMemory.unmapMemory();

    // Dispatch compute shader
    // In a real implementation, this would use a compute shader to perform HRTF convolution
    // For now, we'll simulate the HRTF processing on the CPU

    // Calculate direction from listener to source
    float direction[3];
    direction[0] = sourcePosition[0] - listenerPosition[0];
    direction[1] = sourcePosition[1] - listenerPosition[1];
    direction[2] = sourcePosition[2] - listenerPosition[2];

    // Normalize direction
    float length = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
    if (length > 0.0001f) {
        direction[0] /= length;
        direction[1] /= length;
        direction[2] /= length;
    } else {
        direction[0] = 0.0f;
        direction[1] = 0.0f;
        direction[2] = -1.0f; // Default to front
    }

    // Calculate azimuth and elevation
    float azimuth = std::atan2(direction[0], direction[2]);
    float elevation = std::asin(std::max(-1.0f, std::min(1.0f, direction[1])));

    // Convert to indices
    int azimuthIndex = static_cast<int>((azimuth + M_PI) / (2.0f * M_PI) * 36.0f) % 36;
    int elevationIndex = static_cast<int>((elevation + M_PI / 2.0f) / M_PI * 13.0f);
    elevationIndex = std::max(0, std::min(12, elevationIndex));

    // Get HRTF index
    int hrtfIndex = elevationIndex * 36 + azimuthIndex;
    hrtfIndex = std::min(hrtfIndex, static_cast<int>(numHrtfPositions) - 1);

    // Perform convolution for left and right ears
    for (uint32_t i = 0; i < sampleCount; i++) {
        float leftSample = 0.0f;
        float rightSample = 0.0f;

        // Convolve with HRTF impulse response
        for (uint32_t j = 0; j < hrtfSize && j <= i; j++) {
            uint32_t hrtfLeftIndex = hrtfIndex * hrtfSize * 2 + j;
            uint32_t hrtfRightIndex = hrtfIndex * hrtfSize * 2 + hrtfSize + j;

            if (hrtfLeftIndex < hrtfData.size() && hrtfRightIndex < hrtfData.size()) {
                leftSample += inputBuffer[i - j] * hrtfData[hrtfLeftIndex];
                rightSample += inputBuffer[i - j] * hrtfData[hrtfRightIndex];
            }
        }

        // Apply distance attenuation
        float distanceAttenuation = 1.0f / std::max(1.0f, length);
        leftSample *= distanceAttenuation;
        rightSample *= distanceAttenuation;

        // Write to output buffer
        outputBuffer[i * 2] = leftSample;
        outputBuffer[i * 2 + 1] = rightSample;
    }

    return true;
}

bool AudioSystem::createHRTFBuffers(uint32_t sampleCount) {
    // Clean up existing buffers
    cleanupHRTFBuffers();

    if (!renderer) {
        std::cerr << "AudioSystem::createHRTFBuffers: Renderer is null" << std::endl;
        return false;
    }

    const vk::raii::Device& device = renderer->GetRaiiDevice();
    try {
        // Create input buffer (mono audio)
        vk::BufferCreateInfo inputBufferInfo;
        inputBufferInfo.size = sampleCount * sizeof(float);
        inputBufferInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer;
        inputBufferInfo.sharingMode = vk::SharingMode::eExclusive;

        inputBuffer = vk::raii::Buffer(device, inputBufferInfo);

        vk::MemoryRequirements inputMemRequirements = inputBuffer.getMemoryRequirements();

        vk::MemoryAllocateInfo inputAllocInfo;
        inputAllocInfo.allocationSize = inputMemRequirements.size;
        inputAllocInfo.memoryTypeIndex = renderer->FindMemoryType(
            inputMemRequirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        inputBufferMemory = vk::raii::DeviceMemory(device, inputAllocInfo);
        inputBuffer.bindMemory(*inputBufferMemory, 0);

        // Create output buffer (stereo audio)
        vk::BufferCreateInfo outputBufferInfo;
        outputBufferInfo.size = sampleCount * 2 * sizeof(float); // Stereo (2 channels)
        outputBufferInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer;
        outputBufferInfo.sharingMode = vk::SharingMode::eExclusive;

        outputBuffer = vk::raii::Buffer(device, outputBufferInfo);

        vk::MemoryRequirements outputMemRequirements = outputBuffer.getMemoryRequirements();

        vk::MemoryAllocateInfo outputAllocInfo;
        outputAllocInfo.allocationSize = outputMemRequirements.size;
        outputAllocInfo.memoryTypeIndex = renderer->FindMemoryType(
            outputMemRequirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        outputBufferMemory = vk::raii::DeviceMemory(device, outputAllocInfo);
        outputBuffer.bindMemory(*outputBufferMemory, 0);

        // Create HRTF data buffer
        vk::BufferCreateInfo hrtfBufferInfo;
        hrtfBufferInfo.size = hrtfData.size() * sizeof(float);
        hrtfBufferInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer;
        hrtfBufferInfo.sharingMode = vk::SharingMode::eExclusive;

        hrtfBuffer = vk::raii::Buffer(device, hrtfBufferInfo);

        vk::MemoryRequirements hrtfMemRequirements = hrtfBuffer.getMemoryRequirements();

        vk::MemoryAllocateInfo hrtfAllocInfo;
        hrtfAllocInfo.allocationSize = hrtfMemRequirements.size;
        hrtfAllocInfo.memoryTypeIndex = renderer->FindMemoryType(
            hrtfMemRequirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        hrtfBufferMemory = vk::raii::DeviceMemory(device, hrtfAllocInfo);
        hrtfBuffer.bindMemory(*hrtfBufferMemory, 0);

        // Copy HRTF data to buffer
        void* hrtfMappedMemory = hrtfBufferMemory.mapMemory(0, hrtfData.size() * sizeof(float));
        memcpy(hrtfMappedMemory, hrtfData.data(), hrtfData.size() * sizeof(float));
        hrtfBufferMemory.unmapMemory();

        // Create parameters buffer
        vk::BufferCreateInfo paramsBufferInfo;
        paramsBufferInfo.size = 256; // Size large enough for all parameters
        paramsBufferInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
        paramsBufferInfo.sharingMode = vk::SharingMode::eExclusive;

        paramsBuffer = vk::raii::Buffer(device, paramsBufferInfo);

        vk::MemoryRequirements paramsMemRequirements = paramsBuffer.getMemoryRequirements();

        vk::MemoryAllocateInfo paramsAllocInfo;
        paramsAllocInfo.allocationSize = paramsMemRequirements.size;
        paramsAllocInfo.memoryTypeIndex = renderer->FindMemoryType(
            paramsMemRequirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        paramsBufferMemory = vk::raii::DeviceMemory(device, paramsAllocInfo);
        paramsBuffer.bindMemory(*paramsBufferMemory, 0);

        std::cout << "HRTF buffers created successfully" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error creating HRTF buffers: " << e.what() << std::endl;
        cleanupHRTFBuffers();
        return false;
    }
}

void AudioSystem::cleanupHRTFBuffers() {
    // With RAII, we just need to set the resources to nullptr
    // The destructors will handle the cleanup
    inputBuffer = nullptr;
    inputBufferMemory = nullptr;
    outputBuffer = nullptr;
    outputBufferMemory = nullptr;
    hrtfBuffer = nullptr;
    hrtfBufferMemory = nullptr;
    paramsBuffer = nullptr;
    paramsBufferMemory = nullptr;
}
