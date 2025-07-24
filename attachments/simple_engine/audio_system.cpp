#include "audio_system.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <queue>
#include <mutex>

// OpenAL headers
#ifdef __APPLE__
#include <OpenAL/al.h>
#include <OpenAL/alc.h>
#else
#include <AL/al.h>
#include <AL/alc.h>
#endif

#include "renderer.h"

// OpenAL error checking utility
static void CheckOpenALError(const std::string& operation) {
    ALenum error = alGetError();
    if (error != AL_NO_ERROR) {
        std::cerr << "OpenAL Error in " << operation << ": ";
        switch (error) {
            case AL_INVALID_NAME:
                std::cerr << "AL_INVALID_NAME";
                break;
            case AL_INVALID_ENUM:
                std::cerr << "AL_INVALID_ENUM";
                break;
            case AL_INVALID_VALUE:
                std::cerr << "AL_INVALID_VALUE";
                break;
            case AL_INVALID_OPERATION:
                std::cerr << "AL_INVALID_OPERATION";
                break;
            case AL_OUT_OF_MEMORY:
                std::cerr << "AL_OUT_OF_MEMORY";
                break;
            default:
                std::cerr << "Unknown error " << error;
                break;
        }
        std::cerr << std::endl;
    }
}

// Concrete implementation of AudioSource
class ConcreteAudioSource : public AudioSource {
public:
    explicit ConcreteAudioSource(const std::string& name) : name(name) {}
    ~ConcreteAudioSource() override = default;

    void Play() override {
        playing = true;
        playbackPosition = 0;
        delayTimer = 0.0f;
        inDelayPhase = false;
        std::cout << "Playing audio source: " << name << std::endl;
    }

    void Pause() override {
        playing = false;
        std::cout << "Pausing audio source: " << name << std::endl;
    }

    void Stop() override {
        playing = false;
        playbackPosition = 0;
        delayTimer = 0.0f;
        inDelayPhase = false;
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

    // Additional methods for delay functionality
    void SetAudioLength(uint32_t lengthInSamples) {
        audioLengthSamples = lengthInSamples;
    }

    void UpdatePlayback(float deltaTime, uint32_t samplesProcessed) {
        if (!playing) return;

        if (inDelayPhase) {
            // We're in the delay phase between playthroughs
            delayTimer += deltaTime;
            if (delayTimer >= delayDuration) {
                // Delay finished, restart playback
                inDelayPhase = false;
                playbackPosition = 0;
                delayTimer = 0.0f;
                std::cout << "Delay finished, restarting audio playback for: " << name << std::endl;
            }
        } else {
            // Normal playback, update position
            playbackPosition += samplesProcessed;

            // Check if we've reached the end of the audio
            if (audioLengthSamples > 0 && playbackPosition >= audioLengthSamples) {
                if (loop) {
                    // Start delay phase before looping
                    inDelayPhase = true;
                    delayTimer = 0.0f;
                    std::cout << "Audio finished, starting 1.5s delay before loop for: " << name << std::endl;
                } else {
                    // Stop playing if not looping
                    playing = false;
                    playbackPosition = 0;
                    std::cout << "Audio finished, stopping playback for: " << name << std::endl;
                }
            }
        }
    }

    [[nodiscard]] bool ShouldProcessAudio() const {
        return playing && !inDelayPhase;
    }

    [[nodiscard]] uint32_t GetPlaybackPosition() const {
        return playbackPosition;
    }

    [[nodiscard]] const std::string& GetName() const {
        return name;
    }

    [[nodiscard]] const float* GetPosition() const {
        return position;
    }

private:
    std::string name;
    bool playing = false;
    bool loop = false;
    float volume = 1.0f;
    float position[3] = {0.0f, 0.0f, 0.0f};
    float velocity[3] = {0.0f, 0.0f, 0.0f};

    // Delay and timing functionality
    uint32_t playbackPosition = 0;      // Current position in samples
    uint32_t audioLengthSamples = 0;    // Total length of audio in samples
    float delayTimer = 0.0f;            // Timer for delay between loops
    bool inDelayPhase = false;          // Whether we're currently in delay phase
    static constexpr float delayDuration = 1.5f; // 1.5 second delay between loops
};

// OpenAL audio output device implementation
class OpenALAudioOutputDevice : public AudioOutputDevice {
public:
    OpenALAudioOutputDevice() = default;
    ~OpenALAudioOutputDevice() override {
        Stop();
        Cleanup();
    }

    bool Initialize(uint32_t sampleRate, uint32_t channels, uint32_t bufferSize) override {
        this->sampleRate = sampleRate;
        this->channels = channels;
        this->bufferSize = bufferSize;

        std::cout << "Initializing OpenAL audio output device: " << sampleRate << "Hz, "
                  << channels << " channels, buffer size: " << bufferSize << std::endl;

        // Initialize OpenAL
        device = alcOpenDevice(nullptr); // Use default device
        if (!device) {
            std::cerr << "Failed to open OpenAL device" << std::endl;
            return false;
        }

        context = alcCreateContext(device, nullptr);
        if (!context) {
            std::cerr << "Failed to create OpenAL context" << std::endl;
            alcCloseDevice(device);
            device = nullptr;
            return false;
        }

        if (!alcMakeContextCurrent(context)) {
            std::cerr << "Failed to make OpenAL context current" << std::endl;
            alcDestroyContext(context);
            alcCloseDevice(device);
            context = nullptr;
            device = nullptr;
            return false;
        }

        // Generate OpenAL source
        alGenSources(1, &source);
        CheckOpenALError("alGenSources");

        // Generate OpenAL buffers for streaming
        alGenBuffers(NUM_BUFFERS, buffers);
        CheckOpenALError("alGenBuffers");

        // Set source properties
        alSourcef(source, AL_PITCH, 1.0f);
        alSourcef(source, AL_GAIN, 1.0f);
        alSource3f(source, AL_POSITION, 0.0f, 0.0f, 0.0f);
        alSource3f(source, AL_VELOCITY, 0.0f, 0.0f, 0.0f);
        alSourcei(source, AL_LOOPING, AL_FALSE);
        CheckOpenALError("Source setup");

        // Initialize audio buffer
        audioBuffer.resize(bufferSize * channels);

        // Initialize buffer tracking
        queuedBufferCount = 0;
        while (!availableBuffers.empty()) {
            availableBuffers.pop();
        }

        initialized = true;
        std::cout << "OpenAL audio output device initialized successfully" << std::endl;
        return true;
    }

    bool Start() override {
        if (!initialized) {
            std::cerr << "OpenAL audio output device not initialized" << std::endl;
            return false;
        }

        if (playing) {
            return true; // Already playing
        }

        playing = true;

        // Start audio playback thread
        audioThread = std::thread(&OpenALAudioOutputDevice::AudioThreadFunction, this);

        std::cout << "OpenAL audio output device started" << std::endl;
        return true;
    }

    bool Stop() override {
        if (!playing) {
            return true; // Already stopped
        }

        playing = false;

        // Wait for audio thread to finish
        if (audioThread.joinable()) {
            audioThread.join();
        }

        // Stop OpenAL source
        if (initialized && source != 0) {
            alSourceStop(source);
            CheckOpenALError("alSourceStop");
        }

        std::cout << "OpenAL audio output device stopped" << std::endl;
        return true;
    }

    bool WriteAudio(const float* data, uint32_t sampleCount) override {
        if (!initialized || !playing) {
            return false;
        }

        std::lock_guard<std::mutex> lock(bufferMutex);

        // Add audio data to the queue
        for (uint32_t i = 0; i < sampleCount * channels; i++) {
            audioQueue.push(data[i]);
        }

        return true;
    }

    bool IsPlaying() const override {
        return playing;
    }

    uint32_t GetPosition() const override {
        return playbackPosition;
    }

private:
    static const int NUM_BUFFERS = 4;

    uint32_t sampleRate = 44100;
    uint32_t channels = 2;
    uint32_t bufferSize = 1024;
    bool initialized = false;
    bool playing = false;
    uint32_t playbackPosition = 0;

    // OpenAL objects
    ALCdevice* device = nullptr;
    ALCcontext* context = nullptr;
    ALuint source = 0;
    ALuint buffers[NUM_BUFFERS];
    int currentBuffer = 0;

    std::vector<float> audioBuffer;
    std::queue<float> audioQueue;
    std::mutex bufferMutex;
    std::thread audioThread;

    // Buffer management for OpenAL streaming
    std::queue<ALuint> availableBuffers;
    int queuedBufferCount = 0;

    void Cleanup() {
        if (initialized) {
            // Clean up OpenAL resources
            if (source != 0) {
                alDeleteSources(1, &source);
                source = 0;
            }

            alDeleteBuffers(NUM_BUFFERS, buffers);

            if (context) {
                alcMakeContextCurrent(nullptr);
                alcDestroyContext(context);
                context = nullptr;
            }

            if (device) {
                alcCloseDevice(device);
                device = nullptr;
            }

            // Reset buffer tracking
            queuedBufferCount = 0;
            while (!availableBuffers.empty()) {
                availableBuffers.pop();
            }

            initialized = false;
        }
    }

    void AudioThreadFunction() {
        std::cout << "OpenAL audio playback thread started" << std::endl;

        // Calculate sleep time for audio buffer updates (in milliseconds)
        const auto sleepTime = std::chrono::milliseconds(
            static_cast<int>((bufferSize * 1000) / sampleRate / 8) // Eighth buffer time for responsiveness
        );

        while (playing) {
            ProcessAudioBuffer();
            std::this_thread::sleep_for(sleepTime);
        }

        std::cout << "OpenAL audio playback thread stopped" << std::endl;
    }

    void ProcessAudioBuffer() {
        std::lock_guard<std::mutex> lock(bufferMutex);

        // Fill audio buffer from queue
        uint32_t samplesProcessed = 0;
        const uint32_t maxSamples = bufferSize * channels;

        for (uint32_t i = 0; i < maxSamples && !audioQueue.empty(); i++) {
            audioBuffer[i] = audioQueue.front();
            audioQueue.pop();
            samplesProcessed++;
        }

        if (samplesProcessed > 0) {
            // Convert float samples to 16-bit PCM for OpenAL
            std::vector<int16_t> pcmBuffer(samplesProcessed);
            for (uint32_t i = 0; i < samplesProcessed; i++) {
                // Clamp and convert to 16-bit PCM
                float sample = std::max(-1.0f, std::min(1.0f, audioBuffer[i]));
                pcmBuffer[i] = static_cast<int16_t>(sample * 32767.0f);
            }

            // Check for processed buffers and unqueue them
            ALint processed = 0;
            alGetSourcei(source, AL_BUFFERS_PROCESSED, &processed);
            CheckOpenALError("alGetSourcei AL_BUFFERS_PROCESSED");

            // Unqueue processed buffers and add them to available buffers
            while (processed > 0) {
                ALuint buffer;
                alSourceUnqueueBuffers(source, 1, &buffer);
                CheckOpenALError("alSourceUnqueueBuffers");

                // Add the unqueued buffer to available buffers
                availableBuffers.push(buffer);
                processed--;
            }

            // Only proceed if we have an available buffer
            ALuint buffer = 0;
            if (!availableBuffers.empty()) {
                buffer = availableBuffers.front();
                availableBuffers.pop();
            } else if (queuedBufferCount < NUM_BUFFERS) {
                // Use a buffer that hasn't been queued yet
                buffer = buffers[queuedBufferCount];
            } else {
                // No available buffers, skip this frame
                return;
            }

            // Validate buffer parameters
            if (samplesProcessed == 0 || pcmBuffer.empty()) {
                // Re-add buffer to available list if we can't use it
                if (queuedBufferCount >= NUM_BUFFERS) {
                    availableBuffers.push(buffer);
                }
                return;
            }

            // Determine format based on channels
            ALenum format = (channels == 1) ? AL_FORMAT_MONO16 : AL_FORMAT_STEREO16;

            // Upload audio data to OpenAL buffer
            alBufferData(buffer, format, pcmBuffer.data(),
                        samplesProcessed * sizeof(int16_t), sampleRate);
            CheckOpenALError("alBufferData");

            // Queue the buffer
            alSourceQueueBuffers(source, 1, &buffer);
            CheckOpenALError("alSourceQueueBuffers");

            // Track that we've queued this buffer
            if (queuedBufferCount < NUM_BUFFERS) {
                queuedBufferCount++;
            }

            // Start playing if not already playing
            ALint sourceState;
            alGetSourcei(source, AL_SOURCE_STATE, &sourceState);
            CheckOpenALError("alGetSourcei AL_SOURCE_STATE");

            if (sourceState != AL_PLAYING) {
                alSourcePlay(source);
                CheckOpenALError("alSourcePlay");
            }

            playbackPosition += samplesProcessed / channels;

            // For debugging: print audio activity
            static uint32_t debugCounter = 0;
            if (++debugCounter % 100 == 0) { // Print every 100 buffer updates
                std::cout << "OpenAL output: processed " << samplesProcessed
                          << " samples, position: " << playbackPosition << std::endl;
            }
        }
    }
};

AudioSystem::~AudioSystem() {
    // Stop and clean up audio output device
    if (outputDevice) {
        outputDevice->Stop();
        outputDevice.reset();
    }

    // Destructor implementation
    sources.clear();
    audioData.clear();

    // Clean up HRTF buffers
    cleanupHRTFBuffers();
}

void AudioSystem::GenerateSineWavePing(float* buffer, uint32_t sampleCount, uint32_t playbackPosition) {
    const float sampleRate = 44100.0f;
    const float frequency = 1000.0f;  // 1000Hz ping - louder and more penetrating frequency
    const float pingDuration = 0.5f; // 0.5 second ping duration
    const uint32_t pingSamples = static_cast<uint32_t>(pingDuration * sampleRate);
    const float silenceDuration = 1.0f; // 1 second silence after ping
    const uint32_t silenceSamples = static_cast<uint32_t>(silenceDuration * sampleRate);
    const uint32_t totalCycleSamples = pingSamples + silenceSamples;

    for (uint32_t i = 0; i < sampleCount; i++) {
        uint32_t globalPosition = playbackPosition + i;
        uint32_t cyclePosition = globalPosition % totalCycleSamples;

        if (cyclePosition < pingSamples) {
            // Generate ping with envelope
            float t = static_cast<float>(cyclePosition) / sampleRate;
            float pingProgress = static_cast<float>(cyclePosition) / static_cast<float>(pingSamples);

            // Create envelope: quick attack, sustain, exponential decay
            float envelope;
            if (pingProgress < 0.1f) {
                // Attack phase (first 10% of ping)
                envelope = pingProgress / 0.1f;
            } else if (pingProgress < 0.3f) {
                // Sustain phase (20% of ping at full volume)
                envelope = 1.0f;
            } else {
                // Decay phase (remaining 70% with exponential decay)
                float decayProgress = (pingProgress - 0.3f) / 0.7f;
                envelope = std::exp(-decayProgress * 5.0f); // Exponential decay
            }

            // Generate sine wave with envelope
            float sineWave = std::sin(2.0f * M_PI * frequency * t);
            buffer[i] = 0.8f * envelope * sineWave; // 0.8 amplitude for much louder, clearly audible sound
        } else {
            // Silence phase
            buffer[i] = 0.0f;
        }
    }
}

bool AudioSystem::Initialize(Renderer* renderer) {
    if (renderer) {
        std::cout << "Initializing HRTF audio system with Vulkan compute shader support" << std::endl;

        // Validate renderer if provided
        if (!renderer->IsInitialized()) {
            std::cerr << "AudioSystem::Initialize: Renderer is not initialized" << std::endl;
            return false;
        }

        // Store the renderer for compute shader support
        this->renderer = renderer;
    } else {
        std::cout << "Initializing HRTF audio system with CPU-based processing (no renderer)" << std::endl;
        this->renderer = nullptr;
    }

    // Generate default HRTF data for spatial audio processing
    LoadHRTFData(""); // Pass empty filename to force generation of default HRTF data
    std::cout << "Using generated HRTF data for spatial audio processing" << std::endl;

    // Enable HRTF processing by default for 3D spatial audio
    EnableHRTF(true);

    // Set default listener properties
    SetListenerPosition(0.0f, 0.0f, 0.0f);
    SetListenerOrientation(0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f);
    SetListenerVelocity(0.0f, 0.0f, 0.0f);
    SetMasterVolume(1.0f);

    // Initialize audio output device
    outputDevice = std::make_unique<OpenALAudioOutputDevice>();
    if (!outputDevice->Initialize(44100, 2, 1024)) {
        std::cerr << "Failed to initialize audio output device" << std::endl;
        return false;
    }

    // Start audio output
    if (!outputDevice->Start()) {
        std::cerr << "Failed to start audio output device" << std::endl;
        return false;
    }

    initialized = true;
    std::cout << "HRTF audio system initialized successfully with audio output" << std::endl;
    return true;
}

void AudioSystem::Update(float deltaTime) {
    if (!initialized) {
        return;
    }

    // Check if we have a valid renderer for Vulkan compute shader processing
    bool hasValidRenderer = (renderer && renderer->IsInitialized());

    // Update audio sources and process spatial audio
    for (auto& source : sources) {
        if (!source->IsPlaying()) {
            continue;
        }

        // Cast to ConcreteAudioSource to access timing methods
        ConcreteAudioSource* concreteSource = static_cast<ConcreteAudioSource*>(source.get());

        // Update playback timing and delay logic
        concreteSource->UpdatePlayback(deltaTime, 0); // Will update with actual samples processed later

        // Only process audio if not in delay phase
        if (!concreteSource->ShouldProcessAudio()) {
            continue;
        }

        // Process audio with HRTF spatial processing (works with or without renderer)
        if (hrtfEnabled && !hrtfData.empty()) {
            // Get source position for spatial processing
            const float* sourcePosition = concreteSource->GetPosition();

            // Create sample buffers for processing
            const uint32_t sampleCount = 1024;
            std::vector<float> inputBuffer(sampleCount, 0.0f);
            std::vector<float> outputBuffer(sampleCount * 2, 0.0f);
            uint32_t actualSamplesProcessed = 0;

            // Generate audio signal from loaded audio data
            auto audioIt = audioData.find(concreteSource->GetName());
            if (audioIt != audioData.end() && !audioIt->second.empty()) {
                // Use actual loaded audio data with proper position tracking
                const auto& data = audioIt->second;
                uint32_t playbackPos = concreteSource->GetPlaybackPosition();

                for (uint32_t i = 0; i < sampleCount; i++) {
                    uint32_t dataIndex = (playbackPos + i) * 4; // 4 bytes per sample (16-bit stereo)

                    if (dataIndex + 1 < data.size()) {
                        // Convert from 16-bit PCM to float
                        int16_t sample = *reinterpret_cast<const int16_t*>(&data[dataIndex]);
                        inputBuffer[i] = static_cast<float>(sample) / 32768.0f;
                        actualSamplesProcessed++;
                    } else {
                        // Reached end of audio data
                        inputBuffer[i] = 0.0f;
                    }
                }
            } else {
                // Generate sine wave ping for debugging
                GenerateSineWavePing(inputBuffer.data(), sampleCount, concreteSource->GetPlaybackPosition());
                actualSamplesProcessed = sampleCount;
            }

            // Process audio with HRTF spatial processing using Vulkan compute shader
            ProcessHRTF(inputBuffer.data(), outputBuffer.data(), sampleCount, sourcePosition);

            // Send processed audio to output device
            if (outputDevice && outputDevice->IsPlaying()) {
                // Apply master volume
                for (uint32_t i = 0; i < sampleCount * 2; i++) {
                    outputBuffer[i] *= masterVolume;
                }

                // Send to audio output device
                if (!outputDevice->WriteAudio(outputBuffer.data(), sampleCount)) {
                    std::cerr << "Failed to write audio data to output device" << std::endl;
                }
            }

            // Update playback timing with actual samples processed
            concreteSource->UpdatePlayback(0.0f, actualSamplesProcessed);
        }
    }

    // Apply master volume changes to all active sources
    for (auto& source : sources) {
        if (source->IsPlaying()) {
            // Master volume is applied during HRTF processing and individual source volume control
            // Volume scaling is handled in the ProcessHRTF function
        }
    }

    // Clean up finished audio sources
    sources.erase(
        std::remove_if(sources.begin(), sources.end(),
            [](const std::unique_ptr<AudioSource>& source) {
                // Keep all sources active for continuous playback
                // Audio sources can be stopped/started via their Play/Stop methods
                return false;
            }),
        sources.end()
    );

    // Update timing for audio processing with low-latency chunks
    static float accumulatedTime = 0.0f;
    accumulatedTime += deltaTime;

    // Process audio in 20ms chunks for optimal latency
    const float audioChunkTime = 0.02f; // 20ms chunks for real-time audio
    if (accumulatedTime >= audioChunkTime) {
        // Trigger audio buffer updates for smooth playback
        // The HRTF processing ensures spatial audio is updated continuously
        accumulatedTime = 0.0f;

        // Update listener properties if they have changed
        // This ensures spatial audio positioning stays current with camera movement
    }
}

bool AudioSystem::LoadAudio(const std::string& filename, const std::string& name) {
    std::cout << "Loading audio file: " << filename << " as " << name << std::endl;

    // Open the WAV file
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open audio file: " << filename << std::endl;
        return false;
    }

    // Read WAV header
    struct WAVHeader {
        char riff[4];           // "RIFF"
        uint32_t fileSize;      // File size - 8
        char wave[4];           // "WAVE"
        char fmt[4];            // "fmt "
        uint32_t fmtSize;       // Format chunk size
        uint16_t audioFormat;   // Audio format (1 = PCM)
        uint16_t numChannels;   // Number of channels
        uint32_t sampleRate;    // Sample rate
        uint32_t byteRate;      // Byte rate
        uint16_t blockAlign;    // Block align
        uint16_t bitsPerSample; // Bits per sample
        char data[4];           // "data"
        uint32_t dataSize;      // Data size
    };

    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));

    // Validate WAV header
    if (std::strncmp(header.riff, "RIFF", 4) != 0 ||
        std::strncmp(header.wave, "WAVE", 4) != 0 ||
        std::strncmp(header.fmt, "fmt ", 4) != 0 ||
        std::strncmp(header.data, "data", 4) != 0) {
        std::cerr << "Invalid WAV file format: " << filename << std::endl;
        file.close();
        return false;
    }

    // Only support PCM format for now
    if (header.audioFormat != 1) {
        std::cerr << "Unsupported audio format (only PCM supported): " << filename << std::endl;
        file.close();
        return false;
    }

    // Read audio data
    std::vector<uint8_t> data(header.dataSize);
    file.read(reinterpret_cast<char*>(data.data()), header.dataSize);
    file.close();

    if (file.gcount() != static_cast<std::streamsize>(header.dataSize)) {
        std::cerr << "Failed to read complete audio data from: " << filename << std::endl;
        return false;
    }

    // Store the audio data
    audioData[name] = std::move(data);

    std::cout << "Successfully loaded WAV file: " << filename
              << " (Channels: " << header.numChannels
              << ", Sample Rate: " << header.sampleRate
              << ", Bits: " << header.bitsPerSample
              << ", Size: " << header.dataSize << " bytes)" << std::endl;

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

    // Calculate audio length in samples for timing
    const auto& data = it->second;
    if (!data.empty()) {
        // Assuming 16-bit stereo audio at 44.1kHz (standard WAV format)
        // The audio data reading uses dataIndex = (playbackPos + i) * 4
        // So we need to calculate length based on how many individual samples we can read
        // Each 4 bytes represents one stereo sample pair, so total individual samples = data.size() / 4
        uint32_t totalSamples = static_cast<uint32_t>(data.size()) / 4;

        // Set the audio length for proper timing
        source->SetAudioLength(totalSamples);
        std::cout << "Set audio length for " << name << ": " << totalSamples << " samples (corrected for 4-byte indexing)" << std::endl;
    }

    // Store the source
    AudioSource* sourcePtr = source.get();
    sources.push_back(std::move(source));

    std::cout << "Audio source created: " << name << std::endl;
    return sourcePtr;
}

AudioSource* AudioSystem::CreateDebugPingSource(const std::string& name) {
    std::cout << "Creating debug ping audio source: " << name << std::endl;

    // Create a new audio source for debugging
    auto source = std::make_unique<ConcreteAudioSource>(name);

    // Set up debug ping parameters
    // The ping will cycle every 1.5 seconds (0.5s ping + 1.0s silence)
    const float sampleRate = 44100.0f;
    const float pingDuration = 0.5f;
    const float silenceDuration = 1.0f;
    const uint32_t totalCycleSamples = static_cast<uint32_t>((pingDuration + silenceDuration) * sampleRate);

    // Set the audio length for proper timing (infinite loop for debugging)
    source->SetAudioLength(totalCycleSamples);

    // Store the source
    AudioSource* sourcePtr = source.get();
    sources.push_back(std::move(source));

    std::cout << "Debug ping audio source created: " << name << " (800Hz ping every 1.5 seconds)" << std::endl;
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

void AudioSystem::SetHRTFCPUOnly(bool cpuOnly) {
    hrtfCPUOnly = cpuOnly;
    std::cout << "HRTF processing mode set to " << (cpuOnly ? "CPU-only" : "Vulkan shader (when available)") << std::endl;
}

bool AudioSystem::IsHRTFCPUOnly() const {
    return hrtfCPUOnly;
}

bool AudioSystem::LoadHRTFData(const std::string& filename) {
    if (!filename.empty()) {
        std::cout << "Loading HRTF data from: " << filename << std::endl;
    } else {
        std::cout << "Generating default HRTF data (no file specified)" << std::endl;
    }

    // HRTF parameters
    const uint32_t hrtfSampleCount = 256;  // Number of samples per impulse response
    const uint32_t positionCount = 36 * 13; // 36 azimuths (10-degree steps) * 13 elevations (15-degree steps)
    const uint32_t channelCount = 2;       // Stereo (left and right ears)
    const float sampleRate = 44100.0f;    // Sample rate for HRTF data
    const float speedOfSound = 343.0f;    // Speed of sound in m/s
    const float headRadius = 0.0875f;     // Average head radius in meters

    // Try to load from file first (only if filename is provided)
    if (!filename.empty()) {
        std::ifstream file(filename, std::ios::binary);
        if (file.is_open()) {
        // Read file header to determine format
        char header[4];
        file.read(header, 4);

        if (std::strncmp(header, "HRTF", 4) == 0) {
            // Custom HRTF format
            uint32_t fileHrtfSize, filePositionCount, fileChannelCount;
            file.read(reinterpret_cast<char*>(&fileHrtfSize), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&filePositionCount), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&fileChannelCount), sizeof(uint32_t));

            if (fileChannelCount == channelCount) {
                hrtfData.resize(fileHrtfSize * filePositionCount * fileChannelCount);
                file.read(reinterpret_cast<char*>(hrtfData.data()), hrtfData.size() * sizeof(float));

                hrtfSize = fileHrtfSize;
                numHrtfPositions = filePositionCount;

                file.close();
                std::cout << "Successfully loaded HRTF data from file" << std::endl;
                return true;
            }
        }
        file.close();
    }
    }

    // Generate realistic HRTF data based on acoustic modeling
    std::cout << "Generating realistic HRTF impulse responses..." << std::endl;

    // Resize the HRTF data vector
    hrtfData.resize(hrtfSampleCount * positionCount * channelCount);

    // Generate HRTF impulse responses for each position
    for (uint32_t pos = 0; pos < positionCount; pos++) {
        // Calculate azimuth and elevation for this position
        uint32_t azimuthIndex = pos % 36;
        uint32_t elevationIndex = pos / 36;

        float azimuth = (static_cast<float>(azimuthIndex) * 10.0f - 180.0f) * M_PI / 180.0f;
        float elevation = (static_cast<float>(elevationIndex) * 15.0f - 90.0f) * M_PI / 180.0f;

        // Convert to Cartesian coordinates
        float x = std::cos(elevation) * std::sin(azimuth);
        float y = std::sin(elevation);
        float z = std::cos(elevation) * std::cos(azimuth);

        for (uint32_t channel = 0; channel < channelCount; channel++) {
            // Calculate ear position (left ear: -0.1m, right ear: +0.1m on x-axis)
            float earX = (channel == 0) ? -0.1f : 0.1f;

            // Calculate distance from source to ear
            float dx = x - earX;
            float dy = y;
            float dz = z;
            float distance = std::sqrt(dx * dx + dy * dy + dz * dz);

            // Calculate time delay (ITD - Interaural Time Difference)
            float timeDelay = distance / speedOfSound;
            uint32_t sampleDelay = static_cast<uint32_t>(timeDelay * sampleRate);

            // Calculate head shadow effect (ILD - Interaural Level Difference)
            float shadowFactor = 1.0f;
            if (channel == 0 && azimuth > 0) { // Left ear, source on right
                shadowFactor = 0.3f + 0.7f * std::exp(-azimuth * 2.0f);
            } else if (channel == 1 && azimuth < 0) { // Right ear, source on left
                shadowFactor = 0.3f + 0.7f * std::exp(azimuth * 2.0f);
            }

            // Generate impulse response
            for (uint32_t i = 0; i < hrtfSampleCount; i++) {
                float value = 0.0f;

                // Direct path impulse
                if (i >= sampleDelay && i < sampleDelay + 10) {
                    float t = static_cast<float>(i - sampleDelay) / sampleRate;
                    value = shadowFactor * std::exp(-t * 1000.0f) * std::cos(2.0f * M_PI * 1000.0f * t);
                }

                // Early reflections (simplified)
                for (int refl = 1; refl <= 3; refl++) {
                    uint32_t reflDelay = sampleDelay + refl * 20;
                    if (i >= reflDelay && i < reflDelay + 5) {
                        float t = static_cast<float>(i - reflDelay) / sampleRate;
                        float reflGain = shadowFactor * 0.3f / static_cast<float>(refl);
                        value += reflGain * std::exp(-t * 2000.0f) * std::cos(2.0f * M_PI * 800.0f * t);
                    }
                }

                // Apply distance attenuation
                value /= std::max(1.0f, distance);

                uint32_t index = pos * hrtfSampleCount * channelCount + channel * hrtfSampleCount + i;
                hrtfData[index] = value;
            }
        }
    }

    // Store HRTF parameters
    hrtfSize = hrtfSampleCount;
    numHrtfPositions = positionCount;

    std::cout << "Successfully generated " << positionCount << " HRTF impulse responses" << std::endl;
    return true;
}

bool AudioSystem::ProcessHRTF(const float* inputBuffer, float* outputBuffer, uint32_t sampleCount, const float* sourcePosition) {
    if (!hrtfEnabled) {
        // If HRTF is disabled, just copy input to output
        for (uint32_t i = 0; i < sampleCount; i++) {
            outputBuffer[i * 2] = inputBuffer[i];     // Left channel
            outputBuffer[i * 2 + 1] = inputBuffer[i]; // Right channel
        }
        return true;
    }

    // Check if we should use CPU-only processing or if Vulkan is not available
    if (hrtfCPUOnly || !renderer || !renderer->IsInitialized()) {
        // Use CPU-based HRTF processing (either forced or fallback)
        // Skip Vulkan buffer creation and go directly to CPU processing
    } else {
        // Use Vulkan shader-based HRTF processing
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

        // TODO: Add actual Vulkan compute shader dispatch here
        // For now, fall through to CPU processing
    }

    // CPU-based HRTF processing (used for both CPU-only mode and fallback)

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

    // Perform HRTF processing using CPU-based convolution
    // This implementation provides real-time 3D audio spatialization

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
