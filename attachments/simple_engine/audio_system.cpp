#include "audio_system.h"
#include "platform.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <chrono>
#include <queue>
#include <mutex>
#include <utility>
#include <unordered_map>
#include <algorithm>

#if defined(PLATFORM_ANDROID)
#include <SLES/OpenSLES.h>
#include <SLES/OpenSLES_Android.h>
#else
// OpenAL headers
#ifdef __APPLE__
#include <OpenAL/al.h>
#include <OpenAL/alc.h>
#else
#include <AL/al.h>
#include <AL/alc.h>
#endif

#endif


#if !defined(PLATFORM_ANDROID)
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
#endif

// Concrete implementation of AudioSource
class ConcreteAudioSource : public AudioSource {
public:
    explicit ConcreteAudioSource(std::string  name) : name(std::move(name)) {}
    ~ConcreteAudioSource() override = default;

    void Play() override {
        playing = true;
        playbackPosition = 0;
        delayTimer = std::chrono::milliseconds(0);
        inDelayPhase = false;
        sampleAccumulator = 0.0;
    }

    void Pause() override {
        playing = false;
    }

    void Stop() override {
        playing = false;
        playbackPosition = 0;
        delayTimer = std::chrono::milliseconds(0);
        inDelayPhase = false;
        sampleAccumulator = 0.0;
    }

    void SetVolume(float volume) override {
        this->volume = volume;
    }

    void SetLoop(bool loop) override {
        this->loop = loop;
    }

    void SetPosition(float x, float y, float z) override {
        position[0] = x;
        position[1] = y;
        position[2] = z;
    }

    void SetVelocity(float x, float y, float z) override {
        velocity[0] = x;
        velocity[1] = y;
        velocity[2] = z;
    }

    [[nodiscard]] bool IsPlaying() const override {
        return playing;
    }

    // Additional methods for delay functionality
    void SetAudioLength(uint32_t lengthInSamples) {
        audioLengthSamples = lengthInSamples;
    }

    void UpdatePlayback(std::chrono::milliseconds deltaTime, uint32_t samplesProcessed) {
        if (!playing) return;

        if (inDelayPhase) {
            // We're in the delay phase between playthroughs
            delayTimer += deltaTime;
            if (delayTimer >= delayDuration) {
                // Delay finished, restart playback
                inDelayPhase = false;
                playbackPosition = 0;
                delayTimer = std::chrono::milliseconds(0);
            }
        } else {
            // Normal playback, update position
            playbackPosition += samplesProcessed;

            // Check if we've reached the end of the audio
            if (audioLengthSamples > 0 && playbackPosition >= audioLengthSamples) {
                if (loop) {
                    // Start the delay phase before looping
                    inDelayPhase = true;
                    delayTimer = std::chrono::milliseconds(0);
                } else {
                    // Stop playing if not looping
                    playing = false;
                    playbackPosition = 0;
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

    [[nodiscard]] double GetSampleAccumulator() const {
        return sampleAccumulator;
    }

    void SetSampleAccumulator(double value) {
        sampleAccumulator = value;
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
    std::chrono::milliseconds delayTimer = std::chrono::milliseconds(0);            // Timer for delay between loops
    bool inDelayPhase = false;          // Whether we're currently in the delay phase
    static constexpr std::chrono::milliseconds delayDuration = std::chrono::milliseconds(1500); // 1.5-second delay between loops
    double sampleAccumulator = 0.0;     // Per-source sample accumulator for proper timing
};

#if defined(PLATFORM_ANDROID)

// OpenSL ES audio output device implementation
class OpenSLESAudioOutputDevice : public AudioOutputDevice {
public:
    OpenSLESAudioOutputDevice() = default;
    ~OpenSLESAudioOutputDevice() override { Stop(); }

    bool Initialize(uint32_t sampleRate, uint32_t channels, uint32_t bufferSize) override {
        this->sampleRate = sampleRate;
        this->channels = channels == 0 ? 2u : channels;
        this->bufferSize = bufferSize == 0 ? 1024u : bufferSize; // frames

        // Create and realize engine
        SLresult result = slCreateEngine(&engineObject, 0, nullptr, 0, nullptr, nullptr);
        if (result != SL_RESULT_SUCCESS) { LOGE("OpenSLES: slCreateEngine failed (%d)", result); return false; }
        result = (*engineObject)->Realize(engineObject, SL_BOOLEAN_FALSE);
        if (result != SL_RESULT_SUCCESS) { LOGE("OpenSLES: Engine Realize failed (%d)", result); Cleanup(); return false; }
        result = (*engineObject)->GetInterface(engineObject, SL_IID_ENGINE, &engineEngine);
        if (result != SL_RESULT_SUCCESS) { LOGE("OpenSLES: GetInterface SL_IID_ENGINE failed (%d)", result); Cleanup(); return false; }

        // Create output mix
        result = (*engineEngine)->CreateOutputMix(engineEngine, &outputMixObject, 0, nullptr, nullptr);
        if (result != SL_RESULT_SUCCESS) { LOGE("OpenSLES: CreateOutputMix failed (%d)", result); Cleanup(); return false; }
        result = (*outputMixObject)->Realize(outputMixObject, SL_BOOLEAN_FALSE);
        if (result != SL_RESULT_SUCCESS) { LOGE("OpenSLES: OutputMix Realize failed (%d)", result); Cleanup(); return false; }

        // Configure source: buffer queue + PCM format
        SLDataLocator_AndroidSimpleBufferQueue loc_bufq{ SL_DATALOCATOR_ANDROIDSIMPLEBUFFERQUEUE, (SLuint32)NUM_BUFFERS };
        SLDataFormat_PCM format_pcm{};
        format_pcm.formatType = SL_DATAFORMAT_PCM;
        format_pcm.numChannels = (SLuint32)this->channels;
        format_pcm.samplesPerSec = ToSLSampleRate(this->sampleRate);
        format_pcm.bitsPerSample = SL_PCMSAMPLEFORMAT_FIXED_16;
        format_pcm.containerSize = 16;
        format_pcm.channelMask = (this->channels == 1)
                                 ? (SL_SPEAKER_FRONT_CENTER)
                                 : (SL_SPEAKER_FRONT_LEFT | SL_SPEAKER_FRONT_RIGHT);
        format_pcm.endianness = SL_BYTEORDER_LITTLEENDIAN;

        SLDataSource audioSrc{ &loc_bufq, &format_pcm };

        // Sink: OutputMix
        SLDataLocator_OutputMix loc_outmix{ SL_DATALOCATOR_OUTPUTMIX, outputMixObject };
        SLDataSink audioSnk{ &loc_outmix, nullptr };

        // Create audio player; request buffer queue interface
        const SLInterfaceID ids[] = { SL_IID_BUFFERQUEUE };
        const SLboolean req[] = { SL_BOOLEAN_TRUE };
        result = (*engineEngine)->CreateAudioPlayer(engineEngine, &playerObject, &audioSrc, &audioSnk,
                                                    (SLuint32)(sizeof(ids)/sizeof(ids[0])), ids, req);
        if (result != SL_RESULT_SUCCESS) { LOGE("OpenSLES: CreateAudioPlayer failed (%d)", result); Cleanup(); return false; }
        result = (*playerObject)->Realize(playerObject, SL_BOOLEAN_FALSE);
        if (result != SL_RESULT_SUCCESS) { LOGE("OpenSLES: Player Realize failed (%d)", result); Cleanup(); return false; }

        // Interfaces
        result = (*playerObject)->GetInterface(playerObject, SL_IID_PLAY, &playItf);
        if (result != SL_RESULT_SUCCESS) { LOGE("OpenSLES: GetInterface SL_IID_PLAY failed (%d)", result); Cleanup(); return false; }
        result = (*playerObject)->GetInterface(playerObject, SL_IID_BUFFERQUEUE, &bufferQueueItf);
        if (result != SL_RESULT_SUCCESS) { LOGE("OpenSLES: GetInterface SL_IID_BUFFERQUEUE failed (%d)", result); Cleanup(); return false; }

        // Setup buffers
        pcmBuffers.assign(NUM_BUFFERS, std::vector<int16_t>(this->bufferSize * this->channels));
        nextBufferIndex = 0;

        // Register callback and clear queue
        (*bufferQueueItf)->Clear(bufferQueueItf);
        result = (*bufferQueueItf)->RegisterCallback(bufferQueueItf, &OpenSLESAudioOutputDevice::BufferQueueCallback, this);
        if (result != SL_RESULT_SUCCESS) { LOGE("OpenSLES: RegisterCallback failed (%d)", result); Cleanup(); return false; }

        // Reset state
        while (!audioQueue.empty()) audioQueue.pop();
        playbackPosition = 0;
        initialized = true;
        return true;
    }

    bool Start() override {
        if (!initialized) { LOGE("OpenSLES: device not initialized"); return false; }
        if (playing) return true;

        // Ensure queue empty in OpenSLES
        (*bufferQueueItf)->Clear(bufferQueueItf);

        // Prefill a few buffers to avoid initial underrun
        int prefill = std::min(3, NUM_BUFFERS);
        for (int i = 0; i < prefill; ++i) {
            if (!EnqueueNextBuffer()) {
                EnqueueSilence();
            }
        }

        SLresult result = (*playItf)->SetPlayState(playItf, SL_PLAYSTATE_PLAYING);
        if (result != SL_RESULT_SUCCESS) { LOGE("OpenSLES: SetPlayState PLAYING failed (%d)", result); return false; }

        playing = true;
        return true;
    }

    bool Stop() override {
        if (!initialized) return true;
        playing = false;

        if (playItf) {
            (*playItf)->SetPlayState(playItf, SL_PLAYSTATE_STOPPED);
        }
        if (bufferQueueItf) {
            (*bufferQueueItf)->Clear(bufferQueueItf);
        }

        Cleanup();
        initialized = false;
        return true;
    }

    bool WriteAudio(const float* data, uint32_t sampleCount) override {
        if (!initialized) return false;
        std::lock_guard<std::mutex> lock(bufferMutex);
        const uint64_t total = (uint64_t)sampleCount * (uint64_t)channels;
        for (uint64_t i = 0; i < total; ++i) {
            audioQueue.push(data[i]);
        }
        return true;
    }

    bool IsPlaying() const override { return playing; }
    uint32_t GetPosition() const override { return playbackPosition; }

private:
    static constexpr int NUM_BUFFERS = 8;

    uint32_t sampleRate = 44100;
    uint32_t channels = 2;
    uint32_t bufferSize = 1024; // frames per buffer
    bool initialized = false;
    std::atomic<bool> playing{false};
    uint32_t playbackPosition = 0; // frames

    // OpenSLES objects
    SLObjectItf engineObject = nullptr;
    SLEngineItf engineEngine = nullptr;
    SLObjectItf outputMixObject = nullptr;
    SLObjectItf playerObject = nullptr;
    SLPlayItf playItf = nullptr;
    SLAndroidSimpleBufferQueueItf bufferQueueItf = nullptr;

    // Buffers and queueing
    std::vector<std::vector<int16_t>> pcmBuffers; // NUM_BUFFERS x (bufferSize*channels)
    int nextBufferIndex = 0;
    std::queue<float> audioQueue; // interleaved float samples
    std::mutex bufferMutex;

    static SLuint32 ToSLSampleRate(uint32_t rate) {
        switch (rate) {
            case 8000: return SL_SAMPLINGRATE_8;
            case 11025: return SL_SAMPLINGRATE_11_025;
            case 12000: return SL_SAMPLINGRATE_12;
            case 16000: return SL_SAMPLINGRATE_16;
            case 22050: return SL_SAMPLINGRATE_22_05;
            case 24000: return SL_SAMPLINGRATE_24;
            case 32000: return SL_SAMPLINGRATE_32;
            case 44100: return SL_SAMPLINGRATE_44_1;
            case 48000: return SL_SAMPLINGRATE_48;
            case 64000: return SL_SAMPLINGRATE_64;
            case 88200: return SL_SAMPLINGRATE_88_2;
            case 96000: return SL_SAMPLINGRATE_96;
            case 192000: return SL_SAMPLINGRATE_192;
            default: return SL_SAMPLINGRATE_44_1;
        }
    }

    static void BufferQueueCallback(SLAndroidSimpleBufferQueueItf /*bq*/, void* context) {
        auto* self = static_cast<OpenSLESAudioOutputDevice*>(context);
        if (!self) return;
        if (!self->EnqueueNextBuffer()) {
            self->EnqueueSilence();
        }
    }

    bool EnqueueNextBuffer() {
        std::lock_guard<std::mutex> lock(bufferMutex);
        const uint32_t framesAvailable = static_cast<uint32_t>(audioQueue.size() / channels);
        if (framesAvailable == 0) {
            return false;
        }
        const uint32_t framesToSend = std::min<uint32_t>(bufferSize, framesAvailable);
        const uint32_t samplesToSend = framesToSend * channels;

        auto &buf = pcmBuffers[nextBufferIndex];
        // convert and copy
        for (uint32_t i = 0; i < samplesToSend; ++i) {
            float s = audioQueue.front();
            audioQueue.pop();
            if (s > 1.0f) s = 1.0f; else if (s < -1.0f) s = -1.0f;
            buf[i] = static_cast<int16_t>(s * 32767.0f);
        }
        // pad remaining with zeros if any
        const uint32_t totalSamples = bufferSize * channels;
        if (samplesToSend < totalSamples) {
            std::fill(buf.begin() + samplesToSend, buf.begin() + totalSamples, 0);
        }

        SLresult result = (*bufferQueueItf)->Enqueue(bufferQueueItf, buf.data(), totalSamples * sizeof(int16_t));
        if (result != SL_RESULT_SUCCESS) {
            LOGE("OpenSLES: Enqueue failed (%d)", result);
            return false;
        }
        playbackPosition += framesToSend;
        nextBufferIndex = (nextBufferIndex + 1) % NUM_BUFFERS;
        return true;
    }

    bool EnqueueSilence() {
        auto &buf = pcmBuffers[nextBufferIndex];
        const uint32_t totalSamples = bufferSize * channels;
        std::fill(buf.begin(), buf.begin() + totalSamples, 0);
        SLresult result = (*bufferQueueItf)->Enqueue(bufferQueueItf, buf.data(), totalSamples * sizeof(int16_t));
        if (result != SL_RESULT_SUCCESS) {
            LOGE("OpenSLES: Enqueue(silence) failed (%d)", result);
            return false;
        }
        nextBufferIndex = (nextBufferIndex + 1) % NUM_BUFFERS;
        return true;
    }

    void Cleanup() {
        if (playerObject) {
            (*playerObject)->Destroy(playerObject);
            playerObject = nullptr;
            playItf = nullptr;
            bufferQueueItf = nullptr;
        }
        if (outputMixObject) {
            (*outputMixObject)->Destroy(outputMixObject);
            outputMixObject = nullptr;
        }
        if (engineObject) {
            (*engineObject)->Destroy(engineObject);
            engineObject = nullptr;
            engineEngine = nullptr;
        }
        while (!audioQueue.empty()) audioQueue.pop();
        playbackPosition = 0;
        nextBufferIndex = 0;
        pcmBuffers.clear();
    }
};

#else

// OpenAL audio output device implementation
class OpenALAudioOutputDevice : public AudioOutputDevice {
public:
    OpenALAudioOutputDevice() = default;
    ~OpenALAudioOutputDevice() override {
        OpenALAudioOutputDevice::Stop();
        Cleanup();
    }

    bool Initialize(uint32_t sampleRate, uint32_t channels, uint32_t bufferSize) override {
        this->sampleRate = sampleRate;
        this->channels = channels;
        this->bufferSize = bufferSize;

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

        // Start an audio playback thread
        audioThread = std::thread(&OpenALAudioOutputDevice::AudioThreadFunction, this);

        return true;
    }

    bool Stop() override {
        if (!playing) {
            return true; // Already stopped
        }

        playing = false;

        // Wait for the audio thread to finish
        if (audioThread.joinable()) {
            audioThread.join();
        }

        // Stop OpenAL source
        if (initialized && source != 0) {
            alSourceStop(source);
            CheckOpenALError("alSourceStop");
        }

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

    [[nodiscard]] bool IsPlaying() const override {
        return playing;
    }

    [[nodiscard]] uint32_t GetPosition() const override {
        return playbackPosition;
    }

private:
    static constexpr int NUM_BUFFERS = 8;

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
    ALuint buffers[NUM_BUFFERS]{};
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
        // Calculate sleep time for audio buffer updates (in milliseconds)
        const auto sleepTime = std::chrono::milliseconds(
                static_cast<int>((bufferSize * 1000) / sampleRate / 8) // Eighth buffer time for responsiveness
        );

        while (playing) {
            ProcessAudioBuffer();
            std::this_thread::sleep_for(sleepTime);
        }
    }

    void ProcessAudioBuffer() {
        std::lock_guard<std::mutex> lock(bufferMutex);

        // Fill audio buffer from queue in whole stereo frames to preserve channel alignment
        uint32_t samplesProcessed = 0;
        const uint32_t framesAvailable = static_cast<uint32_t>(audioQueue.size() / channels);
        if (framesAvailable == 0) {
            // Not enough data for a whole frame yet
            return;
        }
        const uint32_t framesToSend = std::min(framesAvailable, bufferSize);
        const uint32_t samplesToSend = framesToSend * channels;
        for (uint32_t i = 0; i < samplesToSend; i++) {
            audioBuffer[i] = audioQueue.front();
            audioQueue.pop();
        }
        samplesProcessed = samplesToSend;

        if (samplesProcessed > 0) {
            // Convert float samples to 16-bit PCM for OpenAL
            std::vector<int16_t> pcmBuffer(samplesProcessed);
            for (uint32_t i = 0; i < samplesProcessed; i++) {
                // Clamp and convert to 16-bit PCM
                float sample = std::clamp(audioBuffer[i], -1.0f, 1.0f);
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
            if (pcmBuffer.empty()) {
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
                         static_cast<ALsizei>(samplesProcessed * sizeof(int16_t)), static_cast<ALsizei>(sampleRate));
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
        }
    }
};

#endif

// =============================
// AudioSystem implementation
// =============================

namespace {
    constexpr uint32_t kDefaultSampleRate = 44100;
    constexpr uint32_t kDefaultChannels   = 2; // stereo
}

AudioSystem::~AudioSystem() {
    // Stop any background work
    stopAudioThread();
    // Ensure device is destroyed last
    outputDevice.reset();
}

bool AudioSystem::Initialize(Engine* engine, Renderer* renderer) {
    if (initialized) return true;
    this->engine = engine;
    this->renderer = renderer;

    // Create output device per platform
#if defined(PLATFORM_ANDROID)
    outputDevice = std::make_unique<OpenSLESAudioOutputDevice>();
#else
    outputDevice = std::make_unique<OpenALAudioOutputDevice>();
#endif

    const uint32_t bufferSizeFrames = 1024; // frames per buffer for streaming
    if (!outputDevice->Initialize(kDefaultSampleRate, kDefaultChannels, bufferSizeFrames)) {
        LOGE("AudioSystem: Failed to initialize output device");
        outputDevice.reset();
        return false;
    }
    if (!outputDevice->Start()) {
        LOGE("AudioSystem: Failed to start output device");
        outputDevice.reset();
        return false;
    }

    initialized = true;
    return true;
}

void AudioSystem::FlushOutput() {
    if (!initialized) return;
    // Recreate device to clear queued buffers
    outputDevice.reset();
#if defined(PLATFORM_ANDROID)
    outputDevice = std::make_unique<OpenSLESAudioOutputDevice>();
#else
    outputDevice = std::make_unique<OpenALAudioOutputDevice>();
#endif
    const uint32_t bufferSizeFrames = 1024;
    if (outputDevice->Initialize(kDefaultSampleRate, kDefaultChannels, bufferSizeFrames)) {
        outputDevice->Start();
    }
}

void AudioSystem::Update(std::chrono::milliseconds deltaTime) {
    if (!initialized || !outputDevice) return;

    // Determine frames to mix for this update
    const double framesExact = (double)kDefaultSampleRate * (double)deltaTime.count() / 1000.0;
    const uint32_t framesToProcess = std::max(1u, (uint32_t)std::llround(framesExact));
    const uint32_t channels = kDefaultChannels;

    std::vector<float> mixBuffer(framesToProcess * channels, 0.0f);

    // Mix all sources
    for (auto &sp : sources) {
        auto* src = static_cast<ConcreteAudioSource*>(sp.get());
        if (!src) continue;
        if (!src->ShouldProcessAudio()) {
            src->UpdatePlayback(deltaTime, 0);
            continue;
        }

        const std::string& name = src->GetName();
        auto it = audioData.find(name);
        if (it == audioData.end()) {
            // Debug ping source
            if (name == "debug_ping") {
                std::vector<float> tmp(framesToProcess * channels, 0.0f);
                GenerateSineWavePing(tmp.data(), framesToProcess, src->GetPlaybackPosition());
                for (uint32_t i = 0; i < framesToProcess; ++i) {
                    mixBuffer[i*channels+0] += tmp[i*channels+0] * masterVolume;
                    mixBuffer[i*channels+1] += tmp[i*channels+1] * masterVolume;
                }
                src->UpdatePlayback(deltaTime, framesToProcess);
            }
            continue;
        }

        const std::vector<uint8_t>& bytes = it->second;
        if (bytes.empty()) continue;
        const int16_t* pcm = reinterpret_cast<const int16_t*>(bytes.data());
        const uint32_t totalSamples = (uint32_t)(bytes.size() / sizeof(int16_t));
        const uint32_t totalFrames = totalSamples / channels;

        uint32_t playPos = src->GetPlaybackPosition();
        src->SetAudioLength(totalFrames);

        uint32_t mixed = 0;
        for (; mixed < framesToProcess && playPos < totalFrames; ++mixed, ++playPos) {
            const uint32_t base = playPos * channels;
            float l, r;
            if (channels == 1) {
                l = r = (float)pcm[base] / 32767.0f;
            } else {
                l = (float)pcm[base+0] / 32767.0f;
                r = (float)pcm[base+1] / 32767.0f;
            }
            l *= masterVolume; r *= masterVolume;
            mixBuffer[mixed*channels+0] += l;
            mixBuffer[mixed*channels+1] += r;
        }
        src->UpdatePlayback(deltaTime, mixed);
    }

    // Clamp and send to device
    for (float &s : mixBuffer) {
        if (s > 1.0f) s = 1.0f; else if (s < -1.0f) s = -1.0f;
    }
    outputDevice->WriteAudio(mixBuffer.data(), framesToProcess);
}

bool AudioSystem::LoadAudio(const std::string& filename, const std::string& name) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) { std::cerr << "AudioSystem: failed to open " << filename << std::endl; return false; }
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (data.size() < 44) { std::cerr << "AudioSystem: file too small: " << filename << std::endl; return false; }

    const uint8_t* p = data.data();
    auto memeq4 = [](const uint8_t* a, const char* s){ return std::memcmp(a, s, 4)==0; };
    uint32_t dataOffset = 0, dataSize = 0;
    uint16_t fmt = 1, bits = 16; uint16_t ch = 2; uint32_t rate = kDefaultSampleRate;
    if (memeq4(p, "RIFF") && memeq4(p+8, "WAVE")) {
        size_t off = 12; // chunk start
        while (off + 8 <= data.size()) {
            const char* id = (const char*)(p + off);
            uint32_t sz = *reinterpret_cast<const uint32_t*>(p + off + 4);
            if (memeq4((const uint8_t*)id, "fmt ")) {
                if (off + 8 + sz > data.size()) break;
                fmt = *reinterpret_cast<const uint16_t*>(p + off + 8);
                ch  = *reinterpret_cast<const uint16_t*>(p + off + 10);
                rate= *reinterpret_cast<const uint32_t*>(p + off + 12);
                bits= *reinterpret_cast<const uint16_t*>(p + off + 22);
            } else if (memeq4((const uint8_t*)id, "data")) {
                dataOffset = (uint32_t)(off + 8);
                dataSize = sz; break;
            }
            off += 8 + sz;
        }
    }

    if (dataOffset == 0 || dataSize == 0 || fmt != 1 || bits != 16) {
        // Not a PCM16 WAV; store as-is and assume default format
        audioData[name] = std::move(data);
        return true;
    }
    if (dataOffset + dataSize > data.size()) return false;
    std::vector<uint8_t> pcm(data.begin()+dataOffset, data.begin()+dataOffset+dataSize);
    audioData[name] = std::move(pcm);
    return true;
}

AudioSource* AudioSystem::CreateAudioSource(const std::string& name) {
    auto src = std::make_unique<ConcreteAudioSource>(name);
    auto it = audioData.find(name);
    if (it != audioData.end()) {
        const uint32_t totalSamples = (uint32_t)(it->second.size() / sizeof(int16_t));
        const uint32_t totalFrames = totalSamples / kDefaultChannels;
        src->SetAudioLength(totalFrames);
    }
    sources.push_back(std::move(src));
    return sources.back().get();
}

AudioSource* AudioSystem::CreateDebugPingSource(const std::string& name) {
    auto src = std::make_unique<ConcreteAudioSource>(name);
    src->SetLoop(true);
    sources.push_back(std::move(src));
    return sources.back().get();
}

void AudioSystem::SetListenerPosition(float x, float y, float z) {
    listenerPosition[0] = x; listenerPosition[1] = y; listenerPosition[2] = z;
}
void AudioSystem::SetListenerOrientation(float fx, float fy, float fz, float ux, float uy, float uz) {
    listenerOrientation[0] = fx; listenerOrientation[1] = fy; listenerOrientation[2] = fz;
    listenerOrientation[3] = ux; listenerOrientation[4] = uy; listenerOrientation[5] = uz;
}
void AudioSystem::SetListenerVelocity(float x, float y, float z) {
    listenerVelocity[0] = x; listenerVelocity[1] = y; listenerVelocity[2] = z;
}
void AudioSystem::SetMasterVolume(float volume) { masterVolume = volume; }
void AudioSystem::EnableHRTF(bool enable) { hrtfEnabled = enable; }
bool AudioSystem::IsHRTFEnabled() const { return hrtfEnabled; }
void AudioSystem::SetHRTFCPUOnly(bool cpuOnly) { hrtfCPUOnly = cpuOnly; }
bool AudioSystem::IsHRTFCPUOnly() const { return hrtfCPUOnly; }

bool AudioSystem::LoadHRTFData(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return false;
    std::vector<float> buf((std::istreambuf_iterator<char>(f)), {});
    if (buf.empty()) return false;
    hrtfData = std::move(buf);
    hrtfSize = (uint32_t)hrtfData.size();
    numHrtfPositions = 0;
    return true;
}

bool AudioSystem::ProcessHRTF(const float* inputBuffer, float* outputBuffer, uint32_t sampleCount, const float* /*sourcePosition*/) {
    if (!inputBuffer || !outputBuffer) return false;
    std::memcpy(outputBuffer, inputBuffer, sampleCount * kDefaultChannels * sizeof(float));
    return true;
}

void AudioSystem::GenerateSineWavePing(float* buffer, uint32_t sampleCount, uint32_t playbackPosition) {
    if (!buffer) return;
    const float freq = 880.0f; // A5
    for (uint32_t i = 0; i < sampleCount; ++i) {
        float t = (float)(playbackPosition + i) / (float)kDefaultSampleRate;
        float s = std::sin(2.0f * 3.1415926535f * freq * t);
        float env = std::exp(-4.0f * t);
        float v = s * env;
        buffer[i*2+0] = v; buffer[i*2+1] = v;
    }
}

// Background processing stubs (no-op for now)
bool AudioSystem::createHRTFBuffers(uint32_t /*sampleCount*/) { return false; }
void AudioSystem::cleanupHRTFBuffers() {}
void AudioSystem::startAudioThread() {}
void AudioSystem::stopAudioThread() { audioThreadShouldStop = true; if (audioThread.joinable()) audioThread.join(); audioThreadRunning = false; }
void AudioSystem::audioThreadLoop() {}
void AudioSystem::processAudioTask(const std::shared_ptr<AudioTask>& /*task*/) {}
bool AudioSystem::submitAudioTask(const float* /*inputBuffer*/, uint32_t /*sampleCount*/, const float* /*sourcePosition*/, uint32_t /*actualSamplesProcessed*/, uint32_t /*trimFront*/) { return false; }