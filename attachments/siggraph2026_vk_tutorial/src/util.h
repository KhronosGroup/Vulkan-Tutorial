/* Copyright (c) 2026, Khronos Group and contributors
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <vulkan/vulkan.hpp>

namespace siggraph {
namespace util {

// Helper functions for logging.

// Small optimization to avoid calling std::format for messages without arguments.
inline void log_msg(std::format_string<> msg) { std::cout << msg.get() << '\n'; }

template <typename... T> inline void log_msg(std::format_string<T...> msg, T&&... args)
{
    // Format the message and write one complete log line.
    std::cout << std::format(msg, std::forward<T>(args)...) << '\n';
}

// Utility class to handle Vulkan binary buffers.
// Notice we have an offset since shader objects creation needs aligned buffers.
struct BinaryBuffer {
    std::vector<std::byte> storage;
    std::size_t offset = 0;
    std::size_t size = 0;

    BinaryBuffer(const BinaryBuffer&) = delete;
    BinaryBuffer(BinaryBuffer&&) = default;

    BinaryBuffer& operator=(const BinaryBuffer&) = delete;
    BinaryBuffer& operator=(BinaryBuffer&&) = delete;

    BinaryBuffer(std::size_t size, std::size_t alignment);

    // Return the aligned start of the useful byte range.
    [[nodiscard]] const std::byte* data() const { return storage.data() + offset; }

    [[nodiscard]] std::byte* data() { return storage.data() + offset; }

    // Most Vulkan-Hpp APIs accept spans, so expose the useful range directly.
    [[nodiscard]] std::span<const std::byte> as_byte_span() const { return {data(), size}; }
};

// CPU-side image data used before main.cpp uploads a texture to Vulkan.
struct ImageRgba8 {
    std::uint32_t m_width = 0;
    std::uint32_t m_height = 0;
    std::vector<std::uint32_t> m_pixels;
};

// Interleaved vertex layout used by the tutorial shaders.
struct PackedVertex {
    glm::vec3 position{};
    glm::vec2 uv{};
    glm::vec3 normal{};
    glm::vec4 tangent{1.0F, 0.0F, 0.0F, 1.0F};
};

// One renderable mesh after glTF accessors have been expanded for Vulkan upload.
struct MeshGeometryData {
    std::string name;
    std::vector<PackedVertex> vertices;
    std::vector<std::uint32_t> indices;
};

// Read a compiled SPIR-V binary and validate that it is aligned as 32-bit SPIR-V words.
[[nodiscard]] BinaryBuffer readSpirvFile(const std::filesystem::path& path);

// Read/write raw binary blobs. Missing or empty input files return nullopt so optional caches can fall back cleanly.
// Note: alignment = 0 means there are no alignment requirements.
[[nodiscard]] std::optional<BinaryBuffer> readBinaryFile(const std::filesystem::path& path, std::size_t alignment);
void writeBinaryFile(const std::filesystem::path& path, std::span<const std::uint8_t> data);

// Convert an 8-bit sRGB display color into linear RGB values for shader/render-target math.
[[nodiscard]] glm::vec3 sRgbToLinear(std::uint8_t r, std::uint8_t g, std::uint8_t b);

[[nodiscard]] std::uint64_t combineHash(std::span<const std::uint8_t> data, std::uint64_t seed = 0);
[[nodiscard]] std::uint64_t combineHash(std::span<const std::byte> data, std::uint64_t seed = 0);
[[nodiscard]] std::uint64_t combineHash(std::string_view value, std::uint64_t seed = 0);

// Hash integral and enum values by expanding them into stable little-endian bytes.
template <typename T> [[nodiscard]] inline std::uint64_t combineHash(T value, std::uint64_t seed = 0)
{
    if constexpr (std::is_enum_v<T>) {
        return combineHash(static_cast<std::underlying_type_t<T>>(value), seed);
    }
    else {
        static_assert(std::is_integral_v<T>, "combineHash supports integral values, enum values, strings, and bytes");
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned normalized = static_cast<Unsigned>(value);
        std::array<std::uint8_t, sizeof(Unsigned)> bytes{};
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<std::uint8_t>((normalized >> (i * 8U)) & static_cast<Unsigned>(0xFFU));
        }
        return combineHash(std::span<const std::uint8_t>{bytes.data(), bytes.size()}, seed);
    }
}
// Three or more values are hashed left-to-right. Two arguments keep the existing value+seed meaning.
template <typename First, typename Second, typename Third, typename... Rest>
[[nodiscard]] inline std::uint64_t combineHash(First first, Second second, Third third, Rest... rest)
{
    std::uint64_t seed = combineHash(first);
    seed = combineHash(second, seed);
    seed = combineHash(third, seed);
    ((seed = combineHash(rest, seed)), ...);
    return seed;
}

// Read an image from disk and convert it to tightly packed RGBA8 pixels.
// The pixels are stored as uint32_t words only to make Vulkan upload sizing simple;
// in memory the byte order is still R, G, B, A for VK_FORMAT_R8G8B8A8_UNORM.
[[nodiscard]] ImageRgba8 readImageFileRgba8(const std::filesystem::path& path);

// Convert common VkResult values into readable names for exception messages.
[[nodiscard]] const char* vkResultName(VkResult result);

// Throw when a Vulkan C API call fails. This keeps setup code readable in the tutorial.
void checkVk(VkResult result, std::string_view operation);

// General assertion helper for tutorial invariants that should fail loudly.
void require(bool condition, std::string_view message);

// Read the optional --frame-limit CLI value. Returns 0 when no limit is requested.
[[nodiscard]] std::uint32_t readFrameLimitCLI(int argc, char** argv);

// Cast integral values only after validating that the destination type can represent the value.
template <typename T, typename V> [[nodiscard]] inline T safeCastTo(V value)
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<V>, "safeCastTo supports integral values");
    require(std::in_range<T>(value), "Integer cast is out of range");
    return static_cast<T>(value);
}

template <typename V> [[nodiscard]] inline std::uint32_t safeCastToU32(V value)
{
    return safeCastTo<std::uint32_t>(value);
}

// Vulkan dispatchable handles are pointers; non-dispatchable handles are integer-like.
// Debug object naming wants either form packed into one uint64_t value.
template <typename RawHandle> [[nodiscard]] inline static std::uint64_t rawHandleToUint64(RawHandle handle)
{
    if constexpr (std::is_pointer_v<RawHandle>) {
        return reinterpret_cast<std::uint64_t>(handle);
    }
    else {
        return static_cast<std::uint64_t>(handle);
    }
}

// Round value up to the next multiple of alignment. Vulkan descriptor heap layouts
// use this for byte offsets where the alignment value comes from device properties.
template <typename T> [[nodiscard]] inline static constexpr T alignUp(T value, T alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

// Return the byte/count offset needed to move value to the next aligned value.
// This is useful when Vulkan returns an unaligned base address but later requires
// binding an aligned subrange inside the same allocation.
template <typename T, typename S> [[nodiscard]] inline static S alignedOffset(T value, S alignment)
{
    if (alignment == 0) {
        return 0;
    }

    const T alignedValue = alignUp(value, safeCastTo<T>(alignment));
    return safeCastTo<S>(alignedValue - value);
}

// Return a size large enough to contain an aligned subrange of rangeSize bytes.
// The extra alignment - 1 bytes cover the worst case where the base address is
// one byte past an aligned address.
template <typename T> [[nodiscard]] inline static constexpr T alignedAllocationSize(T rangeSize, T alignment)
{
    return rangeSize + ((alignment > 0) ? (alignment - 1) : 0);
}

} // namespace util

namespace util::math {

// Build a rotation matrix from tutorial Euler-angle controls.
[[nodiscard]] glm::mat4 generateRotation(const glm::vec3& eulerAngles);

// Build a full model matrix from position, rotation, and scale.
[[nodiscard]] glm::mat4 generateModel(const glm::vec3& pos, const glm::vec3& eulerAngles, const glm::vec3& scale);

// Convert a camera position and look-at point into yaw/pitch controls.
[[nodiscard]] std::pair<float, float> calculateYawPitch(const glm::vec3& cameraPos, const glm::vec3& cameraLookAt);

// Convert yaw/pitch controls back into a normalized forward direction.
[[nodiscard]] glm::vec3 calculateForward(float pitch, float yaw);

// Calculate the camera right vector from forward and world-up directions.
[[nodiscard]] glm::vec3 calculateRight(const glm::vec3& forward,
                                       const glm::vec3& worldUp = glm::vec3{0.0F, 1.0F, 0.0F});

// Build the projection * view matrix used by the vertex shader.
[[nodiscard]] glm::mat4 calculateViewProjection(float cameraPitch, float cameraYaw, const glm::vec3& cameraPos,
                                                float aspectRatio, float verticalFieldOfView = 0.78539816339F,
                                                float nearPlane = 0.1F, float farPlane = 100.0F);

} // namespace util::math

namespace util::slang {

// Descriptor binding information reflected from one Slang shader parameter.
struct ShaderResourceBinding {
    std::uint32_t set = 0;
    std::uint32_t binding = 0;
    VkSpirvResourceTypeFlagsEXT resourceMask = 0;
};

// Vertex shader input location and the Vulkan format that describes its data.
struct VertexInput {
    std::uint32_t location = 0;
    vk::Format format = vk::Format::eUndefined;
};

// Dynamic vertex-input state needed before drawing PackedVertex buffers.
struct PackedVertexInputLayout {
    std::vector<vk::VertexInputBindingDescription2EXT> bindings;
    std::vector<vk::VertexInputAttributeDescription2EXT> attributes;
};

// Read Slang reflection JSON and return descriptor-backed shader resources by name.
// CMake asks slangc to emit this JSON next to the compiled SPIR-V.
[[nodiscard]] std::unordered_map<std::string, ShaderResourceBinding>
calculateReflectionShaderResourceBindings(const std::filesystem::path& path);

// Merge descriptor-backed shader resources across multiple reflection files.
// A shared resource name must resolve to the same set, binding, and resource mask in every stage.
[[nodiscard]] std::unordered_map<std::string, ShaderResourceBinding>
collectShaderResourceBindings(std::span<const std::filesystem::path> reflectionPaths);

// Read Slang reflection JSON and return vertex inputs by field name.
// CMake asks slangc to emit this JSON next to the compiled SPIR-V.
[[nodiscard]] std::unordered_map<std::string, VertexInput>
calculateReflectionVertexInputs(const std::filesystem::path& path);

// Build Vulkan vertex binding/attribute descriptions from one reflection file.
[[nodiscard]] PackedVertexInputLayout calculatePackedVertexInputLayout(const std::filesystem::path& reflectionPath);

} // namespace util::slang

namespace util::gltf {

// Sentinel ID for optional glTF data that is not present.
inline constexpr std::uint32_t invalidGltfId = std::numeric_limits<std::uint32_t>::max();

// A parsed glTF mesh references shared accessor data by compact IDs.
struct Mesh {
    std::string name;
    // Required attribute/index streams.
    std::uint32_t verticesPositionId = invalidGltfId;
    std::uint32_t indicesId = invalidGltfId;
    // Optional attribute streams. Missing ones use invalidGltfId.
    std::uint32_t uvId = invalidGltfId;
    std::uint32_t normalId = invalidGltfId;
    std::uint32_t tangentId = invalidGltfId;
};

// One scene instance: transform, mesh ID, and the textures used by that draw.
struct Node {
    std::string name;
    // The parser stores transform components instead of a matrix so main.cpp can
    // build model matrices in the same way as the rest of the tutorial.
    glm::vec3 pos{};
    glm::vec3 eulerAngles{};
    glm::vec3 scale{1.0F};
    // References one Mesh entry in ParsedData::meshes.
    std::uint32_t meshId = invalidGltfId;
    // Texture IDs index ParsedData::textures, or invalidGltfId when absent.
    std::uint32_t albedoTextureId = invalidGltfId;
    std::uint32_t normalTextureId = invalidGltfId;
};

// Texture metadata from the glTF file. Image decoding happens later.
struct GltfTexture {
    std::string name;
    // Resolved filesystem path to the external texture image.
    std::string filename;
};

// Parsed scene data kept in simple arrays so main.cpp can upload it step by step.
struct ParsedData {
    // Camera values come from the single camera in the parsed glTF scene set.
    glm::vec3 cameraPos{};
    glm::vec3 cameraLookAt{};
    bool hasCamera = false;
    // Nodes are draw instances. Meshes are unique geometry descriptions.
    std::vector<Node> nodes;
    std::vector<Mesh> meshes;
    // Each vector below stores one copied glTF accessor stream.
    std::vector<std::vector<glm::vec3>> verticesPositions;
    std::vector<std::vector<std::uint32_t>> indices;
    std::vector<std::vector<glm::vec2>> uvs;
    std::vector<std::vector<glm::vec3>> normals;
    std::vector<std::vector<glm::vec4>> tangents;
    std::vector<GltfTexture> textures;
};

// Parse a basic glTF 2.0 scene into CPU arrays and shared geometry IDs.
// Texture output is metadata only; image decoding stays with util image readers.
void appendGltfFile(const std::filesystem::path& path, ParsedData& parsedData);
[[nodiscard]] ParsedData parseGltfFiles(std::span<const std::filesystem::path> paths);

// Expand one parsed mesh into the packed vertex/index arrays expected by Vulkan.
[[nodiscard]] util::MeshGeometryData buildMeshGeometryData(std::uint32_t meshId, const ParsedData& gltfData);

} // namespace util::gltf
} // namespace siggraph
