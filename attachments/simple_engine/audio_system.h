#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#ifdef __INTELLISENSE__
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#include <vulkan/vk_platform.h>

/**
 * @brief Class representing an audio source.
 */
class AudioSource {
public:
    /**
     * @brief Default constructor.
     */
    AudioSource() = default;

    /**
     * @brief Destructor for proper cleanup.
     */
    virtual ~AudioSource() = default;

    /**
     * @brief Play the audio source.
     */
    virtual void Play() = 0;

    /**
     * @brief Pause the audio source.
     */
    virtual void Pause() = 0;

    /**
     * @brief Stop the audio source.
     */
    virtual void Stop() = 0;

    /**
     * @brief Set the volume of the audio source.
     * @param volume The volume (0.0f to 1.0f).
     */
    virtual void SetVolume(float volume) = 0;

    /**
     * @brief Set whether the audio source should loop.
     * @param loop Whether to loop.
     */
    virtual void SetLoop(bool loop) = 0;

    /**
     * @brief Set the position of the audio source in 3D space.
     * @param x The x-coordinate.
     * @param y The y-coordinate.
     * @param z The z-coordinate.
     */
    virtual void SetPosition(float x, float y, float z) = 0;

    /**
     * @brief Set the velocity of the audio source in 3D space.
     * @param x The x-component.
     * @param y The y-component.
     * @param z The z-component.
     */
    virtual void SetVelocity(float x, float y, float z) = 0;

    /**
     * @brief Check if the audio source is playing.
     * @return True if playing, false otherwise.
     */
    virtual bool IsPlaying() const = 0;
};

// Forward declarations
class Renderer;

/**
 * @brief Interface for audio output devices.
 */
class AudioOutputDevice {
public:
    /**
     * @brief Default constructor.
     */
    AudioOutputDevice() = default;

    /**
     * @brief Virtual destructor for proper cleanup.
     */
    virtual ~AudioOutputDevice() = default;

    /**
     * @brief Initialize the audio output device.
     * @param sampleRate The sample rate (e.g., 44100).
     * @param channels The number of channels (typically 2 for stereo).
     * @param bufferSize The buffer size in samples.
     * @return True if initialization was successful, false otherwise.
     */
    virtual bool Initialize(uint32_t sampleRate, uint32_t channels, uint32_t bufferSize) = 0;

    /**
     * @brief Start audio playback.
     * @return True if successful, false otherwise.
     */
    virtual bool Start() = 0;

    /**
     * @brief Stop audio playback.
     * @return True if successful, false otherwise.
     */
    virtual bool Stop() = 0;

    /**
     * @brief Write audio data to the output device.
     * @param data Pointer to the audio data (interleaved stereo float samples).
     * @param sampleCount Number of samples per channel to write.
     * @return True if successful, false otherwise.
     */
    virtual bool WriteAudio(const float* data, uint32_t sampleCount) = 0;

    /**
     * @brief Check if the device is currently playing.
     * @return True if playing, false otherwise.
     */
    virtual bool IsPlaying() const = 0;

    /**
     * @brief Get the current playback position in samples.
     * @return Current position in samples.
     */
    virtual uint32_t GetPosition() const = 0;
};

/**
 * @brief Class for managing audio.
 */
class AudioSystem {
public:
    /**
     * @brief Default constructor.
     */
    AudioSystem() = default;

    /**
     * @brief Destructor for proper cleanup.
     */
    ~AudioSystem();

    /**
     * @brief Initialize the audio system.
     * @param renderer Pointer to the renderer for compute shader support.
     * @return True if initialization was successful, false otherwise.
     */
    bool Initialize(Renderer* renderer = nullptr);

    /**
     * @brief Update the audio system.
     * @param deltaTime The time elapsed since the last update.
     */
    void Update(float deltaTime);

    /**
     * @brief Load an audio file.
     * @param filename The path to the audio file.
     * @param name The name to assign to the audio.
     * @return True if loading was successful, false otherwise.
     */
    bool LoadAudio(const std::string& filename, const std::string& name);

    /**
     * @brief Create an audio source.
     * @param name The name of the audio to use.
     * @return Pointer to the created audio source, or nullptr if creation failed.
     */
    AudioSource* CreateAudioSource(const std::string& name);

    /**
     * @brief Create a sine wave ping audio source for debugging.
     * @param name The name to assign to the debug audio source.
     * @return Pointer to the created audio source, or nullptr if creation failed.
     */
    AudioSource* CreateDebugPingSource(const std::string& name);

    /**
     * @brief Set the listener position in 3D space.
     * @param x The x-coordinate.
     * @param y The y-coordinate.
     * @param z The z-coordinate.
     */
    void SetListenerPosition(float x, float y, float z);

    /**
     * @brief Set the listener orientation in 3D space.
     * @param forwardX The x-component of the forward vector.
     * @param forwardY The y-component of the forward vector.
     * @param forwardZ The z-component of the forward vector.
     * @param upX The x-component of the up vector.
     * @param upY The y-component of the up vector.
     * @param upZ The z-component of the up vector.
     */
    void SetListenerOrientation(float forwardX, float forwardY, float forwardZ,
                               float upX, float upY, float upZ);

    /**
     * @brief Set the listener velocity in 3D space.
     * @param x The x-component.
     * @param y The y-component.
     * @param z The z-component.
     */
    void SetListenerVelocity(float x, float y, float z);

    /**
     * @brief Set the master volume.
     * @param volume The volume (0.0f to 1.0f).
     */
    void SetMasterVolume(float volume);

    /**
     * @brief Enable HRTF (Head-Related Transfer Function) processing.
     * @param enable Whether to enable HRTF processing.
     */
    void EnableHRTF(bool enable);

    /**
     * @brief Check if HRTF processing is enabled.
     * @return True if HRTF processing is enabled, false otherwise.
     */
    bool IsHRTFEnabled() const;

    /**
     * @brief Set whether to force CPU-only HRTF processing.
     * @param cpuOnly Whether to force CPU-only processing (true) or allow Vulkan shader processing (false).
     */
    void SetHRTFCPUOnly(bool cpuOnly);

    /**
     * @brief Check if HRTF processing is set to CPU-only mode.
     * @return True if CPU-only mode is enabled, false if Vulkan shader processing is allowed.
     */
    bool IsHRTFCPUOnly() const;

    /**
     * @brief Load HRTF data from a file.
     * @param filename The path to the HRTF data file.
     * @return True if loading was successful, false otherwise.
     */
    bool LoadHRTFData(const std::string& filename);

    /**
     * @brief Process audio data with HRTF.
     * @param inputBuffer The input audio buffer.
     * @param outputBuffer The output audio buffer.
     * @param sampleCount The number of samples to process.
     * @param sourcePosition The position of the sound source.
     * @return True if processing was successful, false otherwise.
     */
    bool ProcessHRTF(const float* inputBuffer, float* outputBuffer, uint32_t sampleCount, const float* sourcePosition);

    /**
     * @brief Generate a sine wave ping for debugging purposes.
     * @param buffer The output buffer to fill with ping audio data.
     * @param sampleCount The number of samples to generate.
     * @param playbackPosition The current playback position for timing.
     */
    void GenerateSineWavePing(float* buffer, uint32_t sampleCount, uint32_t playbackPosition);

private:
    // Loaded audio data
    std::unordered_map<std::string, std::vector<uint8_t>> audioData;

    // Audio sources
    std::vector<std::unique_ptr<AudioSource>> sources;

    // Listener properties
    float listenerPosition[3] = {0.0f, 0.0f, 0.0f};
    float listenerOrientation[6] = {0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f};
    float listenerVelocity[3] = {0.0f, 0.0f, 0.0f};

    // Master volume
    float masterVolume = 1.0f;

    // Whether the audio system is initialized
    bool initialized = false;

    // HRTF processing
    bool hrtfEnabled = false;
    bool hrtfCPUOnly = false;
    std::vector<float> hrtfData;
    uint32_t hrtfSize = 0;
    uint32_t numHrtfPositions = 0;

    // Renderer for compute shader support
    Renderer* renderer = nullptr;

    // Audio output device for sending processed audio to speakers
    std::unique_ptr<AudioOutputDevice> outputDevice = nullptr;

    // Vulkan resources for HRTF processing
    vk::raii::Buffer inputBuffer = nullptr;
    vk::raii::DeviceMemory inputBufferMemory = nullptr;
    vk::raii::Buffer outputBuffer = nullptr;
    vk::raii::DeviceMemory outputBufferMemory = nullptr;
    vk::raii::Buffer hrtfBuffer = nullptr;
    vk::raii::DeviceMemory hrtfBufferMemory = nullptr;
    vk::raii::Buffer paramsBuffer = nullptr;
    vk::raii::DeviceMemory paramsBufferMemory = nullptr;

    /**
     * @brief Create buffers for HRTF processing.
     * @param sampleCount The number of samples to process.
     * @return True if creation was successful, false otherwise.
     */
    bool createHRTFBuffers(uint32_t sampleCount);

    /**
     * @brief Clean up HRTF buffers.
     */
    void cleanupHRTFBuffers();
};
