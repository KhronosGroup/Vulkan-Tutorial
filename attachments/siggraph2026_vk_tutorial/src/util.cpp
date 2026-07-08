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

// This file contains helper functions to make the Vulkan code in main.cpp easier to understand
// The idea is to abstract concepts like glTF parsing that are not relevant for a Vulkan application
// We suggest that people learning Vulkan ignore this file.

#include "util.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/quaternion.hpp>
#include <nlohmann/json.hpp>

// stb_image and tinygltf are header-only libraries. These macros make this one
// translation unit provide their implementation code for the whole program.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include <stb_image.h>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#include <tiny_gltf.h>

namespace siggraph {
namespace util {
namespace {

// Everything in this anonymous namespace is private to util.cpp.
// The public beginner-facing helpers are declared in util.h;
// these helpers keep their implementations small enough to read.

// Slang reflection names scalar/vector types such as float32x3.
// Vulkan needs the matching VkFormat so it knows how to fetch each vertex attribute from memory.
[[nodiscard]] vk::Format vertexFormatFromSlangType(std::string_view scalarType, int elementCount)
{
    if (scalarType == "float32") {
        switch (elementCount) {
        case 1:
            return vk::Format::eR32Sfloat;
        case 2:
            return vk::Format::eR32G32Sfloat;
        case 3:
            return vk::Format::eR32G32B32Sfloat;
        case 4:
            return vk::Format::eR32G32B32A32Sfloat;
        default:
            break;
        }
    }

    if (scalarType == "uint32") {
        switch (elementCount) {
        case 1:
            return vk::Format::eR32Uint;
        case 2:
            return vk::Format::eR32G32Uint;
        case 3:
            return vk::Format::eR32G32B32Uint;
        case 4:
            return vk::Format::eR32G32B32A32Uint;
        default:
            break;
        }
    }

    throw std::runtime_error(std::format("Unsupported Slang vertex input type: {}x{}", scalarType, elementCount));
}

// Reflection describes shader resources in Slang terms.
// Vulkan shader objects validate descriptor bindings with VkSpirvResourceTypeFlagsEXT masks instead.
// We use this function to convert between the two.
[[nodiscard]] VkSpirvResourceTypeFlagsEXT resourceMaskFromSlangType(const nlohmann::json& type)
{
    const std::string kind = type.value("kind", std::string{});
    if (kind == "constantBuffer") {
        return VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    }
    if (kind == "samplerState") {
        return VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT;
    }

    if (kind == "resource") {
        const std::string baseShape = type.value("baseShape", std::string{});
        if (baseShape == "structuredBuffer") {
            return VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT;
        }
        if (baseShape == "texture2D") {
            if (type.value("combined", false)) {
                return VK_SPIRV_RESOURCE_TYPE_COMBINED_SAMPLED_IMAGE_BIT_EXT;
            }
            return VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT;
        }
    }

    throw std::runtime_error(std::format("Unsupported Slang reflected resource type: {}", kind));
}

[[nodiscard]] float sRgbChannelToLinear(std::uint8_t value)
{
    // Convert one 8-bit display-encoded channel into a linear value.
    // sRGB transfer function:
    // https://www.w3.org/TR/css-color-4/#predefined-sRGB
    // https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#_srgb_transfer_functions
    const float normalized = static_cast<float>(value) / 255.0F;
    if (normalized <= 0.04045F) {
        return normalized / 12.92F;
    }
    return std::pow((normalized + 0.055F) / 1.055F, 2.4F);
}

// glTF is a Khronos Group format to efficiently store 3D data.
// We are using tinygltf to parse the current object to read it
//
// glTF stores raw bytes in buffers.
// A buffer view is a contiguous slice of one buffer, often holding one vertex or index stream.
// An accessor describes how to interpret bytes inside a buffer view: scalar/vector shape, component type, element
// count, offset, and optional stride.

// tinygltf stores glTF object references as signed indices, using -1 for optional missing references.
// Normalize those values at the parser boundary so internal parsed IDs and validated indices stay uint32_t.
[[nodiscard]] std::uint32_t optionalGltfIndex(int index)
{
    return index < 0 ? gltf::invalidGltfId : safeCastToU32(index);
}

[[nodiscard]] std::uint32_t requiredGltfIndex(int index, std::string_view what)
{
    require(index >= 0, "glTF " + std::string(what) + " index is missing");
    return safeCastToU32(index);
}

// These wrappers centralize bounds checks so the parsing code can stay readable.
[[nodiscard]] const tinygltf::Accessor& gltfAccessorAt(const tinygltf::Model& model, std::uint32_t index,
                                                       std::string_view what)
{
    require(index != gltf::invalidGltfId && index < model.accessors.size(),
            "glTF " + std::string(what) + " accessor index is out of range");
    return model.accessors[index];
}

[[nodiscard]] const tinygltf::BufferView& gltfBufferViewAt(const tinygltf::Model& model, std::uint32_t index,
                                                           std::string_view what)
{
    require(index != gltf::invalidGltfId && index < model.bufferViews.size(),
            "glTF " + std::string(what) + " buffer view index is out of range");
    return model.bufferViews[index];
}

[[nodiscard]] const tinygltf::Buffer& gltfBufferAt(const tinygltf::Model& model, std::uint32_t index,
                                                   std::string_view what)
{
    require(index != gltf::invalidGltfId && index < model.buffers.size(),
            "glTF " + std::string(what) + " buffer index is out of range");
    return model.buffers[index];
}

[[nodiscard]] std::size_t gltfAccessorByteSize(std::size_t elementCount, std::size_t byteStride,
                                               std::size_t elementSize, std::string_view what)
{
    if (elementCount == 0) {
        return 0;
    }

    const std::size_t lastElementIndex = elementCount - 1U;
    require(byteStride == 0 || lastElementIndex <= (std::numeric_limits<std::size_t>::max() - elementSize) / byteStride,
            "glTF " + std::string(what) + " accessor byte range overflows size_t");
    return lastElementIndex * byteStride + elementSize;
}

void requireGltfAccessorByteRange(std::span<const std::byte> accessorBytes, std::size_t byteOffset,
                                  std::size_t byteSize, std::string_view what)
{
    require(byteOffset <= accessorBytes.size() && byteSize <= accessorBytes.size() - byteOffset,
            "glTF " + std::string(what) + " accessor byte span read is out of bounds");
}

[[nodiscard]] std::span<const std::byte> gltfAccessorData(const tinygltf::Model& model,
                                                          const tinygltf::Accessor& accessor,
                                                          const tinygltf::BufferView& bufferView,
                                                          std::size_t elementSize, std::string_view what)
{
    // This tutorial handles normal, dense accessors only.
    // Sparse accessors are a valid glTF feature, but supporting them would obscure the basic data path.
    require(!accessor.sparse.isSparse, "glTF sparse accessors are not supported");

    const std::uint32_t bufferIndex = requiredGltfIndex(bufferView.buffer, "buffer");
    const tinygltf::Buffer& buffer = gltfBufferAt(model, bufferIndex, what);

    const std::size_t byteOffset = bufferView.byteOffset + accessor.byteOffset;
    require(byteOffset <= buffer.data.size(), "glTF " + std::string(what) + " accessor starts past its buffer");

    const int stride = accessor.ByteStride(bufferView);
    require(stride > 0, "glTF " + std::string(what) + " accessor has invalid byte stride");

    require(static_cast<std::size_t>(stride) >= elementSize,
            "glTF " + std::string(what) + " accessor stride is smaller than one element");

    // The final element may be strided, so validate the address of the last element plus its own byte size.
    const std::size_t byteSize =
        gltfAccessorByteSize(accessor.count, static_cast<std::size_t>(stride), elementSize, what);
    require(byteSize <= buffer.data.size() - byteOffset,
            "glTF " + std::string(what) + " accessor reads past its buffer");

    return std::span<const std::byte>{reinterpret_cast<const std::byte*>(buffer.data.data() + byteOffset), byteSize};
}

template <typename Value>
[[nodiscard]] std::vector<Value> copyGltfAccessorElements(const tinygltf::Accessor& accessor,
                                                          const tinygltf::BufferView& bufferView,
                                                          std::span<const std::byte> accessorBytes)
{
    // glTF attributes may be tightly packed or interleaved with other attributes.
    // ByteStride tells us how far to jump from one element to the next.
    const std::size_t byteStride = static_cast<std::size_t>(accessor.ByteStride(bufferView));
    require(byteStride > 0, "glTF accessor has invalid byte stride");
    require(byteStride >= sizeof(Value), "glTF accessor stride is smaller than one element");

    std::vector<Value> values(accessor.count);
    const std::size_t requiredByteSize = gltfAccessorByteSize(values.size(), byteStride, sizeof(Value), "attribute");
    require(accessorBytes.size() >= requiredByteSize, "glTF accessor byte span is smaller than the requested data");
    if (values.empty()) {
        return values;
    }

    if (byteStride == sizeof(Value)) {
        // Tightly packed accessors can be copied in one block.
        requireGltfAccessorByteRange(accessorBytes, 0, requiredByteSize, "attribute");
        std::memcpy(values.data(), accessorBytes.data(), requiredByteSize);
        return values;
    }

    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::size_t byteOffset = i * byteStride;
        requireGltfAccessorByteRange(accessorBytes, byteOffset, sizeof(Value), "attribute");
        // memcpy avoids alignment assumptions about the byte buffer returned by tinygltf.
        std::memcpy(&values[i], accessorBytes.data() + byteOffset, sizeof(Value));
    }
    return values;
}

template <typename Value>
[[nodiscard]] std::vector<Value> readGltfFloatVectorAccessor(const tinygltf::Model& model, std::uint32_t accessorIndex,
                                                             int expectedAccessorType,
                                                             std::string_view expectedTypeName, std::string_view what)
{
    // Positions, normals, tangents, and UVs are all float vectors in this sample.
    // The template keeps their validation and copy logic in one place.
    const tinygltf::Accessor& accessor = gltfAccessorAt(model, accessorIndex, what);
    require(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT,
            "glTF " + std::string(what) + " accessor must contain float components");
    require(accessor.type == expectedAccessorType,
            "glTF " + std::string(what) + " accessor must be " + std::string(expectedTypeName));

    const std::uint32_t bufferViewIndex = requiredGltfIndex(accessor.bufferView, std::string(what) + " buffer view");
    const tinygltf::BufferView& bufferView = gltfBufferViewAt(model, bufferViewIndex, what);
    const std::span<const std::byte> accessorBytes = gltfAccessorData(model, accessor, bufferView, sizeof(Value), what);
    return copyGltfAccessorElements<Value>(accessor, bufferView, accessorBytes);
}

[[nodiscard]] std::vector<glm::vec3> readGltfVec3Accessor(const tinygltf::Model& model, std::uint32_t accessorIndex,
                                                          std::string_view what)
{
    return readGltfFloatVectorAccessor<glm::vec3>(model, accessorIndex, TINYGLTF_TYPE_VEC3, "VEC3", what);
}

[[nodiscard]] std::vector<glm::vec4> readGltfVec4Accessor(const tinygltf::Model& model, std::uint32_t accessorIndex,
                                                          std::string_view what)
{
    return readGltfFloatVectorAccessor<glm::vec4>(model, accessorIndex, TINYGLTF_TYPE_VEC4, "VEC4", what);
}

[[nodiscard]] std::vector<glm::vec2> readGltfVec2Accessor(const tinygltf::Model& model, std::uint32_t accessorIndex,
                                                          std::string_view what)
{
    return readGltfFloatVectorAccessor<glm::vec2>(model, accessorIndex, TINYGLTF_TYPE_VEC2, "VEC2", what);
}

[[nodiscard]] std::size_t gltfIndexElementSize(int componentType)
{
    // glTF allows several integer widths for index buffers.
    // The renderer expands all of them to uint32_t so the rest of the tutorial has one index type.
    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return sizeof(std::uint8_t);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return sizeof(std::uint16_t);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        return sizeof(std::uint32_t);
    default:
        throw std::runtime_error(std::format("glTF index accessor uses unsupported component type {}", componentType));
    }
}

template <typename StoredIndex>
[[nodiscard]] std::vector<std::uint32_t> copyGltfIndexElements(const tinygltf::Accessor& accessor,
                                                               const tinygltf::BufferView& bufferView,
                                                               std::span<const std::byte> accessorBytes)
{
    const std::size_t byteStride = static_cast<std::size_t>(accessor.ByteStride(bufferView));
    require(byteStride > 0, "glTF index accessor has invalid byte stride");
    require(byteStride >= sizeof(StoredIndex), "glTF index accessor stride is smaller than one element");

    std::vector<std::uint32_t> values(accessor.count);
    const std::size_t requiredByteSize = gltfAccessorByteSize(values.size(), byteStride, sizeof(StoredIndex), "index");
    require(accessorBytes.size() >= requiredByteSize, "glTF index accessor byte span is smaller than the index data");
    if (values.empty()) {
        return values;
    }

    if (byteStride == sizeof(StoredIndex)) {
        // Tightly packed index accessors can be read as one byte range.
        requireGltfAccessorByteRange(accessorBytes, 0, requiredByteSize, "index");
        if constexpr (std::is_same_v<StoredIndex, std::uint32_t>) {
            std::memcpy(values.data(), accessorBytes.data(), requiredByteSize);
        }
        else {
            for (std::size_t i = 0; i < values.size(); ++i) {
                StoredIndex value = 0;
                std::memcpy(&value, accessorBytes.data() + i * sizeof(StoredIndex), sizeof(value));
                values[i] = static_cast<std::uint32_t>(value);
            }
        }
        return values;
    }

    // Interleaved index buffers are unusual, but glTF allows them through byteStride.
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::size_t byteOffset = i * byteStride;
        requireGltfAccessorByteRange(accessorBytes, byteOffset, sizeof(StoredIndex), "index");
        StoredIndex value = 0;
        std::memcpy(&value, accessorBytes.data() + byteOffset, sizeof(value));
        values[i] = static_cast<std::uint32_t>(value);
    }
    return values;
}

[[nodiscard]] std::vector<std::uint32_t> readGltfIndexAccessor(const tinygltf::Model& model,
                                                               std::uint32_t accessorIndex)
{
    // Index accessors are scalar integer streams. They point into the same glTF
    // buffer/bufferView system as vertex attributes.
    const tinygltf::Accessor& accessor = gltfAccessorAt(model, accessorIndex, "index");
    require(accessor.type == TINYGLTF_TYPE_SCALAR, "glTF index accessor must be SCALAR");

    const std::size_t elementSize = gltfIndexElementSize(accessor.componentType);
    const std::uint32_t bufferViewIndex = requiredGltfIndex(accessor.bufferView, "index buffer view");
    const tinygltf::BufferView& bufferView = gltfBufferViewAt(model, bufferViewIndex, "index");
    const std::span<const std::byte> accessorBytes =
        gltfAccessorData(model, accessor, bufferView, elementSize, "index");

    switch (accessor.componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return copyGltfIndexElements<std::uint8_t>(accessor, bufferView, accessorBytes);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return copyGltfIndexElements<std::uint16_t>(accessor, bufferView, accessorBytes);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        return copyGltfIndexElements<std::uint32_t>(accessor, bufferView, accessorBytes);
    default:
        throw std::runtime_error(
            std::format("glTF index accessor uses unsupported component type {}", accessor.componentType));
    }
}

[[nodiscard]] glm::mat4 gltfNodeLocalTransform(const tinygltf::Node& node)
{
    // glTF nodes can either store a full 4x4 matrix or separate translation,
    // rotation, and scale values. A matrix takes precedence when present.
    if (!node.matrix.empty()) {
        require(node.matrix.size() == 16, "glTF node matrix must contain 16 values");
        glm::mat4 transform{1.0F};
        // glTF stores matrices in column-major order, which matches GLM's matrix indexing.
        for (glm::length_t column = 0; column < 4; ++column) {
            for (glm::length_t row = 0; row < 4; ++row) {
                const std::size_t matrixIndex = static_cast<std::size_t>(column) * 4U + static_cast<std::size_t>(row);
                transform[column][row] = static_cast<float>(node.matrix[matrixIndex]);
            }
        }
        return transform;
    }

    glm::vec3 translation{0.0F};
    if (!node.translation.empty()) {
        require(node.translation.size() == 3, "glTF node translation must contain 3 values");
        translation = glm::vec3{static_cast<float>(node.translation[0]), static_cast<float>(node.translation[1]),
                                static_cast<float>(node.translation[2])};
    }

    glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
    if (!node.rotation.empty()) {
        require(node.rotation.size() == 4, "glTF node rotation must contain 4 values");
        // glTF writes quaternions as x,y,z,w; GLM's constructor takes w,x,y,z.
        rotation =
            glm::normalize(glm::quat{static_cast<float>(node.rotation[3]), static_cast<float>(node.rotation[0]),
                                     static_cast<float>(node.rotation[1]), static_cast<float>(node.rotation[2])});
    }

    glm::vec3 scale{1.0F};
    if (!node.scale.empty()) {
        require(node.scale.size() == 3, "glTF node scale must contain 3 values");
        scale = glm::vec3{static_cast<float>(node.scale[0]), static_cast<float>(node.scale[1]),
                          static_cast<float>(node.scale[2])};
    }

    // glTF composes local transforms in translation * rotation * scale order.
    return glm::translate(glm::mat4{1.0F}, translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4{1.0F}, scale);
}

// A scene may draw the same mesh data from multiple nodes.
// This key lets us store one copy of the shared geometry and let each node reference it by ID.
struct GeometryKey {
    std::uint32_t positionId = gltf::invalidGltfId;
    std::uint32_t indexId = gltf::invalidGltfId;
    std::uint32_t uvId = gltf::invalidGltfId;
    std::uint32_t normalId = gltf::invalidGltfId;
    std::uint32_t tangentId = gltf::invalidGltfId;

    [[nodiscard]] bool operator==(const GeometryKey& other) const = default;
};

struct GeometryKeyHash {
    [[nodiscard]] std::size_t operator()(const GeometryKey& key) const
    {
        return combineHash(key.positionId, key.indexId, key.uvId, key.normalId, key.tangentId);
    }
};

[[nodiscard]] glm::vec3 fallbackTangentForNormal(const glm::vec3& normal)
{
    // Pick any helper axis that is not almost parallel to the normal,
    // then cross it with the normal to get a stable perpendicular tangent.
    const glm::vec3 helper = std::abs(normal.y) < 0.999F ? glm::vec3{0.0F, 1.0F, 0.0F} : glm::vec3{1.0F, 0.0F, 0.0F};
    return glm::normalize(glm::cross(helper, normal));
}

void generateMissingTangents(MeshGeometryData& geometry)
{
    constexpr float tangentEpsilon = 0.000001F;

    // Normal mapping needs a tangent basis.
    // When a model has UVs and normals but no tangents,
    // accumulate one tangent/bitangent estimate per triangle.
    std::vector<glm::vec3> tangentSums(geometry.vertices.size(), glm::vec3{0.0F});
    std::vector<glm::vec3> bitangentSums(geometry.vertices.size(), glm::vec3{0.0F});

    for (std::size_t i = 0; i + 2 < geometry.indices.size(); i += 3) {
        const std::uint32_t i0 = geometry.indices[i + 0];
        const std::uint32_t i1 = geometry.indices[i + 1];
        const std::uint32_t i2 = geometry.indices[i + 2];

        const PackedVertex& v0 = geometry.vertices[i0];
        const PackedVertex& v1 = geometry.vertices[i1];
        const PackedVertex& v2 = geometry.vertices[i2];

        const glm::vec3 edge1 = v1.position - v0.position;
        const glm::vec3 edge2 = v2.position - v0.position;
        const glm::vec2 deltaUv1 = v1.uv - v0.uv;
        const glm::vec2 deltaUv2 = v2.uv - v0.uv;
        const float determinant = deltaUv1.x * deltaUv2.y - deltaUv1.y * deltaUv2.x;
        if (std::abs(determinant) <= tangentEpsilon) {
            // Degenerate UV triangles cannot define a useful tangent direction.
            continue;
        }

        const float inverseDeterminant = 1.0F / determinant;
        const glm::vec3 tangent = (edge1 * deltaUv2.y - edge2 * deltaUv1.y) * inverseDeterminant;
        const glm::vec3 bitangent = (edge2 * deltaUv1.x - edge1 * deltaUv2.x) * inverseDeterminant;

        tangentSums[i0] += tangent;
        tangentSums[i1] += tangent;
        tangentSums[i2] += tangent;
        bitangentSums[i0] += bitangent;
        bitangentSums[i1] += bitangent;
        bitangentSums[i2] += bitangent;
    }

    // Average the triangle estimates at each vertex and make the tangent
    // perpendicular to the normal with a Gram-Schmidt step.
    for (std::size_t i = 0; i < geometry.vertices.size(); ++i) {
        const glm::vec3 normal = glm::normalize(geometry.vertices[i].normal);
        glm::vec3 tangent = tangentSums[i] - normal * glm::dot(normal, tangentSums[i]);
        if (glm::dot(tangent, tangent) <= tangentEpsilon) {
            tangent = fallbackTangentForNormal(normal);
        }
        else {
            tangent = glm::normalize(tangent);
        }

        const float handedness = glm::dot(glm::cross(normal, tangent), bitangentSums[i]) < 0.0F ? -1.0F : 1.0F;
        // Store xyz as the tangent direction and w as handedness so the shader can reconstruct the bitangent.
        geometry.vertices[i].tangent = glm::vec4{tangent, handedness};
    }
}

[[nodiscard]] gltf::Node makeParsedNode(const glm::mat4& transform)
{
    // ParsedData stores transform components because main.cpp later builds model
    // matrices from position, Euler rotation, and scale sliders/values.
    gltf::Node parsedNode{};
    parsedNode.pos = glm::vec3{transform[3]};
    parsedNode.scale = glm::vec3{glm::length(glm::vec3{transform[0]}), glm::length(glm::vec3{transform[1]}),
                                 glm::length(glm::vec3{transform[2]})};

    // Remove scale from the matrix columns before converting the remaining
    // rotation matrix into Euler angles.
    glm::mat3 rotationMatrix{1.0F};
    for (glm::length_t column = 0; column < 3; ++column) {
        const float axisScale = parsedNode.scale[column];
        if (axisScale > 0.0F) {
            rotationMatrix[column] = glm::vec3{transform[column]} / axisScale;
        }
    }
    parsedNode.eulerAngles = glm::eulerAngles(glm::quat_cast(rotationMatrix));
    return parsedNode;
}

[[nodiscard]] std::string gltfNameOrFallback(const std::string& name, std::string_view fallbackPrefix,
                                             std::uint32_t index)
{
    // Names from the file are best for debugging;
    // generated names keep error messages useful when an asset omits them.
    if (!name.empty()) {
        return name;
    }
    return std::string{fallbackPrefix} + "[" + std::to_string(index) + "]";
}

[[nodiscard]] std::string gltfMeshPrimitiveName(const tinygltf::Mesh& mesh, std::uint32_t meshIndex,
                                                std::uint32_t primitiveIndex, std::size_t primitiveCount)
{
    // A glTF mesh can have several primitives; include the primitive index only when it matters.
    std::string name = gltfNameOrFallback(mesh.name, "Mesh", meshIndex);
    if (primitiveCount > 1) {
        name += "/Primitive[" + std::to_string(primitiveIndex) + "]";
    }
    return name;
}

[[nodiscard]] std::string gltfTextureName(const tinygltf::Texture& texture, const tinygltf::Image& image,
                                          std::uint32_t textureIndex)
{
    // Prefer names closest to the material reference, then fall back to the image or a generated name.
    if (!texture.name.empty()) {
        return texture.name;
    }
    if (!image.name.empty()) {
        return image.name;
    }
    return gltfNameOrFallback(std::string{}, "Texture", textureIndex);
}

[[nodiscard]] std::uint32_t parsedDataIdFromSize(std::size_t size, std::string_view what)
{
    const std::uint32_t id = safeCastToU32(size);
    require(id != gltf::invalidGltfId, std::format("glTF {} count uses the reserved invalid ID", what));
    return id;
}

[[nodiscard]] std::uint32_t
textureIdFromTextureIndex(const tinygltf::Model& model, const std::filesystem::path& gltfBaseDir,
                          std::uint32_t textureIndex, gltf::ParsedData& parsedData,
                          std::unordered_map<std::uint32_t, std::uint32_t>& imageToTextureId)
{
    // In glTF, materials reference textures, and textures reference images.
    // This tutorial stores one ParsedData texture per image URI and reuses it by ID.
    if (textureIndex == gltf::invalidGltfId) {
        return gltf::invalidGltfId;
    }

    require(textureIndex < model.textures.size(), "glTF texture index is out of range");

    const tinygltf::Texture& texture = model.textures[textureIndex];

    const std::uint32_t imageIndex = requiredGltfIndex(texture.source, "texture image");
    require(imageIndex < model.images.size(), "glTF texture image index is out of range");

    const auto existingIt = imageToTextureId.find(imageIndex);
    if (existingIt != imageToTextureId.end()) {
        // Multiple materials can point at the same image; keep one texture record and reuse its ID.
        return existingIt->second;
    }

    // Image not seen before, add it to the images in the scene.

    const tinygltf::Image& image = model.images[imageIndex];
    require(!image.uri.empty(), "glTF texture image must use an external URI");
    const std::filesystem::path texturePath =
        std::filesystem::path{image.uri}.is_absolute() ? std::filesystem::path{image.uri} : gltfBaseDir / image.uri;

    const std::uint32_t id = parsedDataIdFromSize(parsedData.textures.size(), "texture");
    parsedData.textures.push_back(gltf::GltfTexture{
        .name = gltfTextureName(texture, image, textureIndex),
        .filename = texturePath.lexically_normal().string(),
    });
    imageToTextureId.emplace(imageIndex, id);
    return id;
}

} // namespace

// BinaryBuffer is used to create buffers whose data have alignment requirements.
BinaryBuffer::BinaryBuffer(size_t size, size_t alignment) : size(size)
{
    // Allocate a little extra storage so an aligned subrange can always fit inside the vector.
    storage.resize(alignedAllocationSize(size, alignment));

    // Store how many bytes to skip before the aligned payload starts.
    const std::size_t storage_data = reinterpret_cast<std::size_t>(storage.data());
    offset = alignUp(storage_data, alignment) - storage_data;

    // Validate the two invariants this helper promises to callers.
    require(alignUp(storage_data, alignment) == reinterpret_cast<std::size_t>(data()),
            "Generated binary buffer is not aligned.");
    require((data() + this->size) <= (storage.data() + storage.size()), "Generated binary buffer is overflowing");
};

BinaryBuffer readSpirvFile(const std::filesystem::path& path)
{
    // Vulkan consumes shader modules as SPIR-V bytecode.
    // Keeping this as raw bytes lets the caller pass the data straight into Vulkan create-info structs.

    // Shader objects expect SPIR-V data to be 4-byte aligned.
    std::size_t spirvAlignment = 4;

    // readBinaryFile handles the filesystem details and gives us aligned storage.
    std::optional<BinaryBuffer> binBuffer = readBinaryFile(path, spirvAlignment);

    if (!binBuffer.has_value()) {
        throw std::runtime_error(std::format("Failed to read SPIR-V file: {}", path.string()));
    }

    // SPIR-V is defined as an array of 32-bit words. A non-multiple-of-4 file is invalid.
    if ((static_cast<std::uintmax_t>(binBuffer.value().size) % sizeof(std::uint32_t)) != 0U) {
        throw std::runtime_error(std::format("SPIR-V file size is not word aligned: {}", path.string()));
    }

    return std::move(binBuffer.value());
}

std::optional<BinaryBuffer> readBinaryFile(const std::filesystem::path& path, std::size_t alignment)
{
    // Missing cache files are not errors.
    // Other filesystem failures still need to stop execution because they may hide permission or path problems.
    std::error_code existsError;
    if (!std::filesystem::exists(path, existsError)) {
        if (existsError) {
            throw std::runtime_error(std::format("Failed to query binary file: {}", path.string()));
        }
        return std::nullopt;
    }

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error(std::format("Failed to open binary file: {}", path.string()));
    }

    // Opening at the end lets tellg() report the whole file size before allocation.
    const auto fileSize = file.tellg();
    if (fileSize < 0) {
        throw std::runtime_error(std::format("Failed to query binary file size: {}", path.string()));
    }
    else if (fileSize == 0) {
        log_msg("File: {} is empty", path.string());
        return std::nullopt;
    }
    BinaryBuffer binBuffer(fileSize, alignment);

    // Rewind after sizing the file and read the bytes into the aligned payload area.
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(binBuffer.data()), binBuffer.size);

    if (!file) {
        throw std::runtime_error(std::format("Failed to read binary file: {}", path.string()));
    }

    return binBuffer;
}

void writeBinaryFile(const std::filesystem::path& path, std::span<const std::uint8_t> data)
{
    // Cache/output paths may point into generated folders that do not exist yet.
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error(std::format("Failed to open binary file for writing: {}", path.string()));
    }

    if (!data.empty()) {
        // std::span keeps this helper flexible: callers can pass vectors, arrays,
        // or other contiguous byte ranges without copying first.
        file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
    if (!file) {
        throw std::runtime_error(std::format("Failed to write binary file: {}", path.string()));
    }
}

glm::vec3 sRgbToLinear(std::uint8_t r, std::uint8_t g, std::uint8_t b)
{
    // Apply the same sRGB transfer function independently to R, G, and B.
    return glm::vec3{sRgbChannelToLinear(r), sRgbChannelToLinear(g), sRgbChannelToLinear(b)};
}

std::uint64_t combineHash(std::span<const std::uint8_t> data, std::uint64_t seed)
{
    constexpr std::uint64_t fnvPrime = 1099511628211ULL;

    // Prefix the content with its length so ["ab", "c"] does not hash the same
    // way as ["a", "bc"] when several values are combined in sequence.
    const std::uint64_t dataSize = static_cast<std::uint64_t>(data.size());
    for (std::size_t i = 0; i < sizeof(dataSize); ++i) {
        const std::uint8_t value = static_cast<std::uint8_t>((dataSize >> (i * 8U)) & 0xFFU);
        seed ^= value;
        seed *= fnvPrime;
    }
    for (const std::uint8_t value : data) {
        seed ^= value;
        seed *= fnvPrime;
    }
    return seed;
}

std::uint64_t combineHash(std::span<const std::byte> data, std::uint64_t seed)
{
    // std::byte and uint8_t have the same storage, but the uint8_t overload does the actual FNV-style byte mixing.
    return combineHash(std::span<const std::uint8_t>{reinterpret_cast<const std::uint8_t*>(data.data()), data.size()},
                       seed);
}

std::uint64_t combineHash(std::string_view value, std::uint64_t seed)
{
    // Hash the bytes of the string view without requiring a null terminator.
    const std::span bytes{reinterpret_cast<const std::uint8_t*>(value.data()), value.size()};
    return combineHash(bytes, seed);
}

namespace slang {

std::unordered_map<std::string, ShaderResourceBinding>
calculateReflectionShaderResourceBindings(const std::filesystem::path& path)
{
    // The tutorial asks Slang to emit JSON reflection next to each compiled shader.
    // This lets the C++ code discover descriptor sets/bindings instead of duplicating those numbers by hand.
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error(std::format("Failed to open Slang reflection file: {}", path.string()));
    }

    // Parse reflection data as JSON
    const nlohmann::json reflection = nlohmann::json::parse(file, nullptr, true, true);

    // Store all bindings in a map
    std::unordered_map<std::string, ShaderResourceBinding> bindings;

    for (const nlohmann::json& parameter : reflection.value("parameters", nlohmann::json::array())) {
        const nlohmann::json& binding = parameter.value("binding", nlohmann::json::object());
        if (binding.value("kind", std::string{}) != "descriptorTableSlot") {
            continue;
        }

        bindings.emplace(parameter.at("name").get<std::string>(),
                         ShaderResourceBinding{
                             // Descriptor setId from [[vk::binding(bindingId, setId)]] in vertex/fragment shaders.
                             .set = binding.value("set", 0U),
                             // Descriptor bindingId from [[vk::binding(bindingId, setId)]].
                             .binding = binding.at("index").get<std::uint32_t>(),
                             .resourceMask = resourceMaskFromSlangType(parameter.at("type")),
                         });
    }

    if (bindings.empty()) {
        throw std::runtime_error(
            std::format("No shader resource bindings found in Slang reflection file: {}", path.string()));
    }

    return bindings;
}

std::unordered_map<std::string, ShaderResourceBinding>
collectShaderResourceBindings(std::span<const std::filesystem::path> reflectionPaths)
{
    require(!reflectionPaths.empty(), "At least one shader reflection path is required");

    // Vertex and fragment shaders can share descriptor names.
    // Shared names must agree exactly so one descriptor layout can be used for the full pipeline.
    std::unordered_map<std::string, ShaderResourceBinding> mergedBindings;
    for (const std::filesystem::path& reflectionPath : reflectionPaths) {
        for (const auto& [name, binding] : calculateReflectionShaderResourceBindings(reflectionPath)) {
            const auto [it, inserted] = mergedBindings.emplace(name, binding);
            if (!inserted) {
                require(it->second.set == binding.set && it->second.binding == binding.binding &&
                            it->second.resourceMask == binding.resourceMask,
                        "Slang reflection mismatch for shared resource binding: " + name);
            }
        }
    }

    return mergedBindings;
}

std::unordered_map<std::string, VertexInput> calculateReflectionVertexInputs(const std::filesystem::path& path)
{
    // Vertex inputs are reflected separately from descriptors, because Vulkan needs them in dynamic vertex-input state
    // instead of descriptor layouts.
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error(std::format("Failed to open Slang reflection file: {}", path.string()));
    }

    const nlohmann::json reflection = nlohmann::json::parse(file, nullptr, true, true);

    std::unordered_map<std::string, VertexInput> inputs;

    // Slang writes vertex input struct fields under entryPoints[0].parameters[0].type.fields.
    // Each field carries a varyingInput location plus scalar/vector type information.
    if (reflection.contains("entryPoints") && !reflection.at("entryPoints").empty()) {
        const nlohmann::json& entryPoint = reflection.at("entryPoints").at(0);
        if (entryPoint.contains("parameters") && !entryPoint.at("parameters").empty()) {
            const nlohmann::json& entryParameter = entryPoint.at("parameters").at(0);
            if (entryParameter.contains("type") && entryParameter.at("type").contains("fields")) {
                const nlohmann::json& fields = entryParameter.at("type").at("fields");
                for (const nlohmann::json& field : fields) {
                    const nlohmann::json& binding = field.value("binding", nlohmann::json::object());
                    if (binding.value("kind", std::string{}) != "varyingInput") {
                        continue;
                    }

                    const nlohmann::json& type = field.at("type");
                    const nlohmann::json& elementType =
                        type.value("kind", std::string{}) == "vector" ? type.at("elementType") : type;
                    int elementCount = type.value("elementCount", 1);
                    const std::string scalarType = elementType.at("scalarType").get<std::string>();

                    // Store by shader field name so the PackedVertex layout table
                    // below can match "position", "uv", "normal", and "tangent".
                    inputs.emplace(field.at("name").get<std::string>(),
                                   VertexInput{
                                       .location = binding.at("index").get<std::uint32_t>(),
                                       .format = vertexFormatFromSlangType(scalarType, elementCount),
                                   });
                }
            }
        }
    }

    if (inputs.empty()) {
        throw std::runtime_error(
            std::format("No vertex input variables found in Slang reflection file: {}", path.string()));
    }

    return inputs;
}

PackedVertexInputLayout calculatePackedVertexInputLayout(const std::filesystem::path& reflectionPath)
{
    // Vulkan needs one binding description for the buffer and one attribute description per shader input.
    PackedVertexInputLayout layout{};
    layout.bindings = {vk::VertexInputBindingDescription2EXT{
        // App binding slot: set by vkCmdSetVertexInputEXT and filled by vkCmdBindVertexBuffers2.
        .binding = 0,
        // Byte size of one interleaved CPU vertex.
        .stride = sizeof(PackedVertex),
        // Advance to the next PackedVertex for each vertex.
        .inputRate = vk::VertexInputRate::eVertex,
        // No instancing divisor; every vertex uses the next element.
        .divisor = 1,
    }};

    const std::unordered_map<std::string, VertexInput> vertexInputs = calculateReflectionVertexInputs(reflectionPath);

    // Keep the CPU vertex struct and shader input names in one visible table.
    // If the shader changes, this require() path explains which field no longer matches.
    struct VertexField {
        std::string_view name;
        std::uint32_t offset;
    };
    const std::array vertexFields{
        VertexField{.name = "position", .offset = safeCastToU32(offsetof(PackedVertex, position))},
        VertexField{.name = "uv", .offset = safeCastToU32(offsetof(PackedVertex, uv))},
        VertexField{.name = "normal", .offset = safeCastToU32(offsetof(PackedVertex, normal))},
        VertexField{.name = "tangent", .offset = safeCastToU32(offsetof(PackedVertex, tangent))},
    };

    layout.attributes.reserve(vertexFields.size());
    for (const VertexField& vertexField : vertexFields) {
        const std::string fieldName{vertexField.name};
        const auto inputIt = vertexInputs.find(fieldName);
        require(inputIt != vertexInputs.end(), "Slang reflection is missing vertex input: " + fieldName);

        layout.attributes.emplace_back(vk::VertexInputAttributeDescription2EXT{
            // Vertex input location, read from [[vk::location(locationId)]] in the vertex shader.
            .location = inputIt->second.location,
            // App binding slot: set by vkCmdSetVertexInputEXT and filled by vkCmdBindVertexBuffers2.
            .binding = 0,
            // Vulkan format inferred from the reflected Slang field type.
            .format = inputIt->second.format,
            // Byte offset of this field inside PackedVertex.
            .offset = vertexField.offset,
        });
    }

    return layout;
}

} // namespace slang

ImageRgba8 readImageFileRgba8(const std::filesystem::path& path)
{
    // stb_image gives this tutorial a small, dependency-light path from image bytes to CPU RGBA pixels.
    // Texture creation stays in main.cpp, where Vulkan code is easier to follow.
    int width = 0;
    int height = 0;
    int sourceChannels = 0;
    constexpr int requiredChannels = 4;
    stbi_uc* decodedPixels = stbi_load(path.string().c_str(), &width, &height, &sourceChannels, requiredChannels);
    if (decodedPixels == nullptr) {
        throw std::runtime_error(
            std::format("Failed to read image file: {} ({})", path.string(), stbi_failure_reason()));
    }

    ImageRgba8 image{};
    try {
        // Requesting four channels means stb_image converts RGB/gray input into
        // the RGBA layout expected by VK_FORMAT_R8G8B8A8_UNORM.
        require(width > 0 && height > 0, "Image dimensions must be positive");
        image.m_width = safeCastToU32(width);
        image.m_height = safeCastToU32(height);

        const std::size_t byteCount =
            static_cast<std::size_t>(image.m_width) * static_cast<std::size_t>(image.m_height) * requiredChannels;
        image.m_pixels.resize(byteCount / sizeof(std::uint32_t));
        // m_pixels is a uint32_t vector for convenient Vulkan upload sizing, but
        // the copied bytes remain ordered R, G, B, A.
        std::memcpy(image.m_pixels.data(), decodedPixels, byteCount);
    }
    catch (...) {
        // stbi_load uses malloc internally, so always release it before rethrowing.
        stbi_image_free(decodedPixels);
        throw;
    }

    // The decoded pixels have been copied into image.m_pixels, so stb's temporary allocation can be released.
    stbi_image_free(decodedPixels);
    return image;
}

namespace gltf {

void appendGltfFile(const std::filesystem::path& path, ParsedData& parsedData)
{
    // tinygltf can load both binary .glb files and text .gltf files.
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string error;
    std::string warning;
    const std::filesystem::path gltfBaseDir = path.parent_path();

    // Choose the loader from the extension so tutorial assets can use either glTF container format.
    const std::string extension = path.extension().string();
    const bool loaded = extension == ".glb" ? loader.LoadBinaryFromFile(&model, &error, &warning, path.string())
                                            : loader.LoadASCIIFromFile(&model, &error, &warning, path.string());
    if (!loaded) {
        throw std::runtime_error(std::format("Failed to parse glTF file: {} {}", path.string(), error));
    }

    const std::size_t nodeCountBefore = parsedData.nodes.size();
    const bool hadCameraBefore = parsedData.hasCamera;

    // These maps turn glTF accessor/image indices into compact IDs in ParsedData.
    // They also avoid copying shared buffers more than once.
    std::unordered_map<std::uint32_t, std::uint32_t> positionAccessorIds;
    std::unordered_map<std::uint32_t, std::uint32_t> uvAccessorIds;
    std::unordered_map<std::uint32_t, std::uint32_t> normalAccessorIds;
    std::unordered_map<std::uint32_t, std::uint32_t> tangentAccessorIds;
    std::unordered_map<std::uint32_t, std::uint32_t> indexAccessorIds;
    std::unordered_map<std::uint32_t, std::uint32_t> imageTextureIds;
    std::unordered_map<GeometryKey, std::uint32_t, GeometryKeyHash> meshIds;

    // Each get*Id lambda lazily copies one glTF accessor into ParsedData the
    // first time it is referenced, then returns the existing ID on later uses.
    const auto getPositionId = [&model, &parsedData, &positionAccessorIds](std::uint32_t accessorIndex) {
        // POSITION is required for every drawable primitive, so this helper never returns invalidGltfId.
        require(accessorIndex != gltf::invalidGltfId, "glTF primitive is missing POSITION");
        auto [it, inserted] = positionAccessorIds.emplace(
            accessorIndex, parsedDataIdFromSize(parsedData.verticesPositions.size(), "POSITION accessor"));
        if (inserted) {
            parsedData.verticesPositions.push_back(readGltfVec3Accessor(model, accessorIndex, "POSITION"));
        }
        return it->second;
    };

    const auto getUvId = [&model, &parsedData, &uvAccessorIds](std::uint32_t accessorIndex) {
        // Optional accessors use invalidGltfId so the caller can fill defaults later.
        if (accessorIndex == gltf::invalidGltfId) {
            return gltf::invalidGltfId;
        }
        auto [it, inserted] =
            uvAccessorIds.emplace(accessorIndex, parsedDataIdFromSize(parsedData.uvs.size(), "UV accessor"));
        if (inserted) {
            parsedData.uvs.push_back(readGltfVec2Accessor(model, accessorIndex, "TEXCOORD_0"));
        }
        return it->second;
    };

    const auto getNormalId = [&model, &parsedData, &normalAccessorIds](std::uint32_t accessorIndex) {
        // Optional accessors use invalidGltfId so the caller can fill defaults later.
        if (accessorIndex == gltf::invalidGltfId) {
            return gltf::invalidGltfId;
        }
        auto [it, inserted] = normalAccessorIds.emplace(
            accessorIndex, parsedDataIdFromSize(parsedData.normals.size(), "NORMAL accessor"));
        if (inserted) {
            parsedData.normals.push_back(readGltfVec3Accessor(model, accessorIndex, "NORMAL"));
        }
        return it->second;
    };

    const auto getTangentId = [&model, &parsedData, &tangentAccessorIds](std::uint32_t accessorIndex) {
        // Optional accessors use invalidGltfId so the caller can fill defaults later.
        if (accessorIndex == gltf::invalidGltfId) {
            return gltf::invalidGltfId;
        }
        auto [it, inserted] = tangentAccessorIds.emplace(
            accessorIndex, parsedDataIdFromSize(parsedData.tangents.size(), "TANGENT accessor"));
        if (inserted) {
            parsedData.tangents.push_back(readGltfVec4Accessor(model, accessorIndex, "TANGENT"));
        }
        return it->second;
    };

    const auto getIndexId = [&model, &parsedData, &indexAccessorIds](std::uint32_t accessorIndex) {
        // The tutorial renders indexed triangles only, so primitives without indices are rejected.
        require(accessorIndex != gltf::invalidGltfId, "glTF primitive must have indices");
        auto [it, inserted] =
            indexAccessorIds.emplace(accessorIndex, parsedDataIdFromSize(parsedData.indices.size(), "index accessor"));
        if (inserted) {
            parsedData.indices.push_back(readGltfIndexAccessor(model, accessorIndex));
        }
        return it->second;
    };

    bool foundCamera = parsedData.hasCamera;
    const auto handleCamera = [&model, &parsedData, &foundCamera](const tinygltf::Node& node, const glm::mat4& world) {
        const std::uint32_t cameraIndex = optionalGltfIndex(node.camera);
        if (cameraIndex == gltf::invalidGltfId) {
            return;
        }
        require(!foundCamera, "glTF scene set must contain exactly one camera");
        require(cameraIndex < model.cameras.size(), "glTF camera index is out of range");
        // This tutorial only needs an eye point and a look-at target.
        // A camera node's -Z axis is its forward direction in glTF.
        parsedData.cameraPos = glm::vec3{world[3]};
        const glm::vec3 forward = glm::normalize(glm::vec3{world * glm::vec4{0.0F, 0.0F, -1.0F, 0.0F}});
        parsedData.cameraLookAt = parsedData.cameraPos + forward;
        parsedData.hasCamera = true;
        foundCamera = true;
    };

    const auto textureIdForMaterial = [&gltfBaseDir, &model, &parsedData, &imageTextureIds](int materialIndex,
                                                                                            bool normalTexture) {
        // Materials may omit a texture. invalidGltfId is the renderer's simple "no texture" sentinel.
        const std::uint32_t materialIndexU32 = optionalGltfIndex(materialIndex);
        if (materialIndexU32 == gltf::invalidGltfId) {
            return gltf::invalidGltfId;
        }
        require(materialIndexU32 < model.materials.size(), "glTF material index is out of range");
        const tinygltf::Material& material = model.materials[materialIndexU32];
        // This tutorial uses base-color textures for albedo and normalTexture for normal mapping.
        const std::uint32_t textureIndex = optionalGltfIndex(
            normalTexture ? material.normalTexture.index : material.pbrMetallicRoughness.baseColorTexture.index);
        return textureIdFromTextureIndex(model, gltfBaseDir, textureIndex, parsedData, imageTextureIds);
    };

    // glTF scenes are trees.
    // Accumulate parent transforms as the traversal walks down the tree so each parsed node ends up in world space.
    std::function<void(std::uint32_t, const glm::mat4&)> traverseNode;
    traverseNode = [&getIndexId, &getNormalId, &getPositionId, &getTangentId, &getUvId, &handleCamera, &meshIds, &model,
                    &parsedData, &textureIdForMaterial,
                    &traverseNode](std::uint32_t nodeIndex, const glm::mat4& parentTransform) {
        require(nodeIndex != gltf::invalidGltfId && nodeIndex < model.nodes.size(),
                "glTF scene node index is out of range");
        const tinygltf::Node& gltfNode = model.nodes[nodeIndex];

        // Parent transform first, then this node's local transform, gives the node's world transform.
        const glm::mat4 worldTransform = parentTransform * gltfNodeLocalTransform(gltfNode);

        // A camera can be attached to any node, including nodes without meshes.
        handleCamera(gltfNode, worldTransform);

        const std::uint32_t gltfMeshIndex = optionalGltfIndex(gltfNode.mesh);
        if (gltfMeshIndex != gltf::invalidGltfId) {
            // A glTF mesh can contain multiple primitives, each with its own material.
            require(gltfMeshIndex < model.meshes.size(), "glTF mesh index is out of range");
            const tinygltf::Mesh& mesh = model.meshes[gltfMeshIndex];
            const std::uint32_t primitiveCount = safeCastToU32(mesh.primitives.size());

            for (std::uint32_t primitiveIndex = 0; primitiveIndex < primitiveCount; ++primitiveIndex) {

                const tinygltf::Primitive& primitive = mesh.primitives[primitiveIndex];

                // Each primitive is one indexed triangle stream with one material.
                require(primitive.mode == -1 || primitive.mode == TINYGLTF_MODE_TRIANGLES,
                        "Only glTF triangle primitives are supported");

                // POSITION is required to draw. UVs, normals, and tangents are optional;
                // buildMeshGeometryData fills simple defaults when they are absent.
                const auto positionIt = primitive.attributes.find("POSITION");
                require(positionIt != primitive.attributes.end(), "glTF primitive is missing POSITION");
                const auto uvIt = primitive.attributes.find("TEXCOORD_0");
                const auto normalIt = primitive.attributes.find("NORMAL");
                const auto tangentIt = primitive.attributes.find("TANGENT");

                // Store IDs to shared accessor data. buildMeshGeometryData expands these IDs later.
                const Mesh parsedMesh{
                    .name = gltfMeshPrimitiveName(mesh, gltfMeshIndex, primitiveIndex, mesh.primitives.size()),
                    .verticesPositionId = getPositionId(requiredGltfIndex(positionIt->second, "POSITION accessor")),
                    .indicesId = getIndexId(requiredGltfIndex(primitive.indices, "primitive indices accessor")),
                    .uvId = uvIt != primitive.attributes.end() ? getUvId(optionalGltfIndex(uvIt->second))
                                                               : gltf::invalidGltfId,
                    .normalId = normalIt != primitive.attributes.end()
                                    ? getNormalId(optionalGltfIndex(normalIt->second))
                                    : gltf::invalidGltfId,
                    .tangentId = tangentIt != primitive.attributes.end()
                                     ? getTangentId(optionalGltfIndex(tangentIt->second))
                                     : gltf::invalidGltfId,
                };
                const GeometryKey meshKey{
                    .positionId = parsedMesh.verticesPositionId,
                    .indexId = parsedMesh.indicesId,
                    .uvId = parsedMesh.uvId,
                    .normalId = parsedMesh.normalId,
                    .tangentId = parsedMesh.tangentId,
                };
                auto [meshIt, inserted] =
                    meshIds.emplace(meshKey, parsedDataIdFromSize(parsedData.meshes.size(), "mesh"));
                // Multiple nodes can instance the same primitive. Reusing the Mesh entry keeps geometry shared.
                if (inserted) {
                    // Only store a mesh description once even if multiple nodes
                    // instantiate the same primitive data.
                    parsedData.meshes.push_back(parsedMesh);
                }

                // A parsed node combines where the mesh is drawn with which textures
                // that primitive uses.
                Node parsedNode = makeParsedNode(worldTransform);
                parsedNode.name = gltfNameOrFallback(gltfNode.name, "Node", nodeIndex);
                parsedNode.meshId = meshIt->second;
                parsedNode.albedoTextureId = textureIdForMaterial(primitive.material, false);
                parsedNode.normalTextureId = textureIdForMaterial(primitive.material, true);
                // This sample's textured material path expects an albedo texture and a normal map together.
                parsedData.nodes.push_back(parsedNode);
            }
        }

        for (const int childIndex : gltfNode.children) {
            // Children inherit the current node transform through worldTransform.
            traverseNode(requiredGltfIndex(childIndex, "child node"), worldTransform);
        }
    };

    require(!model.scenes.empty(), "glTF file must contain at least one scene");
    // glTF can mark a default scene. If it does not, scene 0 is the usual fallback.
    const std::uint32_t sceneIndex = model.defaultScene >= 0 ? safeCastToU32(model.defaultScene) : std::uint32_t{0};
    require(sceneIndex < model.scenes.size(), "glTF default scene index is out of range");
    for (const int nodeIndex : model.scenes[sceneIndex].nodes) {
        // Root nodes start with the identity transform because they have no parent.
        traverseNode(requiredGltfIndex(nodeIndex, "root node"), glm::mat4{1.0F});
    }

    require(parsedData.nodes.size() > nodeCountBefore || parsedData.hasCamera != hadCameraBefore,
            "glTF scene contains no camera or renderable mesh primitives");
}

ParsedData parseGltfFiles(std::span<const std::filesystem::path> paths)
{
    require(!paths.empty(), "At least one glTF file path is required");

    ParsedData mergedData{};
    for (const std::filesystem::path& path : paths) {
        appendGltfFile(path, mergedData);
    }

    require(!mergedData.nodes.empty(), "glTF files contain no renderable mesh primitives");
    require(mergedData.hasCamera, "glTF files must contain exactly one camera");
    return mergedData;
}

util::MeshGeometryData buildMeshGeometryData(std::uint32_t meshId, const ParsedData& gltfData)
{
    // ParsedData keeps shared accessors by ID.
    // This function expands one mesh ID into the interleaved vertex/index arrays that Vulkan buffers will upload.
    require(meshId < gltfData.meshes.size(), "glTF node references an invalid mesh id");

    const Mesh& mesh = gltfData.meshes[meshId];

    require(mesh.verticesPositionId != gltf::invalidGltfId &&
                mesh.verticesPositionId < gltfData.verticesPositions.size(),
            "glTF node references an invalid POSITION id");
    require(mesh.indicesId != gltf::invalidGltfId && mesh.indicesId < gltfData.indices.size(),
            "glTF node references an invalid index id");

    const std::vector<glm::vec3>& positions = gltfData.verticesPositions[mesh.verticesPositionId];
    const std::vector<std::uint32_t>& sourceIndices = gltfData.indices[mesh.indicesId];

    // Optional attribute pointers stay null until the mesh proves that stream exists.
    const std::vector<glm::vec2>* uvs = nullptr;
    const std::vector<glm::vec3>* normals = nullptr;
    const std::vector<glm::vec4>* tangents = nullptr;

    // Missing optional attributes use simple defaults below. When an attribute is
    // present, glTF requires one value per POSITION for this tutorial path.
    if (mesh.uvId != gltf::invalidGltfId) {
        require(mesh.uvId < gltfData.uvs.size(), "glTF node references an invalid UV id");
        uvs = &gltfData.uvs[mesh.uvId];
        require(uvs->size() == positions.size(), "glTF UV count must match POSITION count");
    }
    if (mesh.normalId != gltf::invalidGltfId) {
        require(mesh.normalId < gltfData.normals.size(), "glTF node references an invalid NORMAL id");
        normals = &gltfData.normals[mesh.normalId];
        require(normals->size() == positions.size(), "glTF NORMAL count must match POSITION count");
    }
    if (mesh.tangentId != gltf::invalidGltfId) {
        require(mesh.tangentId < gltfData.tangents.size(), "glTF node references an invalid TANGENT id");
        tangents = &gltfData.tangents[mesh.tangentId];
        require(tangents->size() == positions.size(), "glTF TANGENT count must match POSITION count");
    }

    MeshGeometryData geometry{};
    geometry.name = gltfNameOrFallback(mesh.name, "Mesh", meshId);
    // Reserve storage because we know exactly how many vertices the POSITION accessor contains.
    geometry.vertices.reserve(positions.size());

    for (std::size_t i = 0; i < positions.size(); ++i) {
        // Interleave separate glTF attribute arrays into the PackedVertex layout
        // used by the shader's vertex input reflection.
        geometry.vertices.push_back(PackedVertex{
            .position = positions[i],
            .uv = uvs != nullptr ? (*uvs)[i] : glm::vec2{0.0F},
            .normal = normals != nullptr ? (*normals)[i] : glm::vec3{0.0F, 1.0F, 0.0F},
            .tangent = tangents != nullptr ? (*tangents)[i] : glm::vec4{1.0F, 0.0F, 0.0F, 1.0F},
        });
    }

    geometry.indices.reserve(sourceIndices.size());
    for (const std::uint32_t index : sourceIndices) {
        // Validate indices while copying so bad assets fail before Vulkan sees them.
        require(index < positions.size(), "glTF index references a vertex outside the POSITION accessor");
        // Indices are already normalized to uint32_t by readGltfIndexAccessor().
        geometry.indices.push_back(index);
    }

    if (tangents == nullptr && uvs != nullptr && normals != nullptr) {
        // Normal maps need tangents. If the asset did not provide them, generate
        // reasonable tangents from triangles, UVs, and normals.
        generateMissingTangents(geometry);
    }
    return geometry;
}

} // namespace gltf

namespace math {

glm::mat4 generateRotation(const glm::vec3& eulerAngles)
{
    // Angles are radians. The order here matches generateModel() and keeps the
    // tutorial's camera/object controls predictable: rotate around X, then Y, then Z.
    return glm::rotate(glm::mat4{1.0F}, eulerAngles.x, glm::vec3{1.0F, 0.0F, 0.0F}) *
           glm::rotate(glm::mat4{1.0F}, eulerAngles.y, glm::vec3{0.0F, 1.0F, 0.0F}) *
           glm::rotate(glm::mat4{1.0F}, eulerAngles.z, glm::vec3{0.0F, 0.0F, 1.0F});
}

glm::mat4 generateModel(const glm::vec3& pos, const glm::vec3& eulerAngles, const glm::vec3& scale)
{
    // With GLM's column-vector convention, this matrix scales the model first,
    // then rotates it, then moves it into world space.
    return glm::translate(glm::mat4{1.0F}, pos) * generateRotation(eulerAngles) * glm::scale(glm::mat4{1.0F}, scale);
}

std::pair<float, float> calculateYawPitch(const glm::vec3& cameraPos, const glm::vec3& cameraLookAt)
{
    // Convert a look-at target back into yaw/pitch controls. Forward -Z is the
    // tutorial camera's zero-yaw direction.
    const glm::vec3 forward = glm::normalize(cameraLookAt - cameraPos);
    return {
        std::atan2(forward.x, -forward.z),
        std::asin(std::clamp(forward.y, -1.0F, 1.0F)),
    };
}

glm::vec3 calculateForward(float pitch, float yaw)
{
    // Rebuild the camera forward vector from yaw around world Y and pitch above
    // or below the horizontal plane.
    return glm::normalize(glm::vec3{
        std::cos(pitch) * std::sin(yaw),
        std::sin(pitch),
        -std::cos(pitch) * std::cos(yaw),
    });
}

glm::vec3 calculateRight(const glm::vec3& forward, const glm::vec3& worldUp)
{
    // For a camera looking down -Z with +Y up, forward x up points to +X.
    return glm::normalize(glm::cross(forward, worldUp));
}

glm::mat4 calculateViewProjection(float cameraPitch, float cameraYaw, const glm::vec3& cameraPos, float aspectRatio,
                                  float verticalFieldOfView, float nearPlane, float farPlane)
{
    require(aspectRatio > 0.0F, "View-projection aspect ratio must be positive");
    require(verticalFieldOfView > 0.0F, "View-projection vertical field of view must be positive");
    require(verticalFieldOfView < glm::pi<float>(), "View-projection vertical field of view must be less than pi");
    require(nearPlane > 0.0F, "View-projection near plane must be positive");
    require(farPlane > nearPlane, "View-projection far plane must be greater than near plane");

    const glm::vec3 forward = calculateForward(cameraPitch, cameraYaw);
    const glm::mat4 view = glm::lookAt(cameraPos, cameraPos + forward, glm::vec3{0.0F, 1.0F, 0.0F});
    glm::mat4 projection = glm::perspective(verticalFieldOfView, aspectRatio, nearPlane, farPlane);
    // GLM's perspective helper follows OpenGL's clip-space Y convention. Vulkan's
    // viewport coordinates need the projection Y axis flipped.
    projection[1][1] *= -1.0F;
    // Shader math uses clipPosition = projection * view * model * localPosition.
    return projection * view;
}

} // namespace math

const char* vkResultName(VkResult result)
{
    // Vulkan's C API reports success/failure with VkResult integers. Names are easier
    // to read in tutorial error messages than raw numeric codes.
    switch (result) {
    case VK_SUCCESS:
        return "VK_SUCCESS";
    case VK_NOT_READY:
        return "VK_NOT_READY";
    case VK_TIMEOUT:
        return "VK_TIMEOUT";
    case VK_EVENT_SET:
        return "VK_EVENT_SET";
    case VK_EVENT_RESET:
        return "VK_EVENT_RESET";
    case VK_INCOMPLETE:
        return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
        return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
        return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
        return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED:
        return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:
        return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
        return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:
        return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
        return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS:
        return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
        return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL:
        return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_UNKNOWN:
        return "VK_ERROR_UNKNOWN";
    default:
        return "VK_RESULT_UNRECOGNIZED";
    }
}

void checkVk(VkResult result, std::string_view operation)
{
    // GLFW and a few Vulkan C APIs return VkResult directly instead of throwing exceptions.
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::format("{} failed with {}", operation, vkResultName(result)));
    }
}

void require(bool condition, std::string_view message)
{
    // Prefer explicit checks over undefined behavior when tutorial assumptions are violated.
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

std::uint32_t readFrameLimitCLI(int argc, char** argv)
{
    // The optional frame limit is used by automated runs/screenshots. Returning
    // 0 means "run normally with no fixed frame count".
    constexpr std::string_view frameLimitArgument = "--frame-limit";

    const auto parseFrameLimit = [](std::string_view value) {
        if (value.empty()) {
            throw std::runtime_error("Frame limit must not be empty");
        }

        std::uint64_t frameLimit = 0;
        const char* const begin = value.data();
        const char* const end = begin + value.size();
        // from_chars is locale-independent and reports where parsing stopped, so
        // values like "12abc" are rejected instead of partially accepted.
        const auto [parsedEnd, error] = std::from_chars(begin, end, frameLimit);
        if (error != std::errc{} || parsedEnd != end) {
            throw std::runtime_error("Frame limit must be a non-negative integer");
        }
        return safeCastToU32(frameLimit);
    };

    if (argc == 1) {
        return 0;
    }

    if (argc == 3 && std::string_view{argv[1]} == frameLimitArgument) {
        // Support --frame-limit 120.
        return parseFrameLimit(argv[2]);
    }

    throw std::runtime_error(std::format("Usage: {} [--frame-limit N]", argv[0]));
}

} // namespace util
} // namespace siggraph
