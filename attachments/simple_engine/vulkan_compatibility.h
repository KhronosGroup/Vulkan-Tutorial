/* Copyright (c) 2025 Holochip Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <vulkan/vulkan.h>

// Fallback defines for optional extension names (allow compiling against older headers)
#ifndef VK_EXT_ROBUSTNESS_2_EXTENSION_NAME
#	define VK_EXT_ROBUSTNESS_2_EXTENSION_NAME "VK_EXT_robustness2"
#endif
#ifndef VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME
#	define VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME "VK_KHR_dynamic_rendering_local_read"
#endif
#ifndef VK_EXT_SHADER_TILE_IMAGE_EXTENSION_NAME
#	define VK_EXT_SHADER_TILE_IMAGE_EXTENSION_NAME "VK_EXT_shader_tile_image"
#endif

// Opacity Micromap fallback (KHR vs EXT)
#ifndef VK_KHR_OPACITY_MICROMAP_EXTENSION_NAME
#	define VK_KHR_OPACITY_MICROMAP_EXTENSION_NAME "VK_KHR_opacity_micromap"
#endif

#if defined(PLATFORM_ANDROID) || defined(__ANDROID__)

// Only provide fallback mappings if the KHR types are not already defined by headers
#ifndef VK_KHR_opacity_micromap

// Map missing KHR types/enums to EXT equivalents or dummies for Android compilation
#ifndef VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_KHR
#define VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_KHR VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_EXT
#endif

#ifndef VkAccelerationStructureTrianglesOpacityMicromapKHR
#define VkAccelerationStructureTrianglesOpacityMicromapKHR VkAccelerationStructureTrianglesOpacityMicromapEXT
#endif

#ifndef VkMicromapUsageKHR
#define VkMicromapUsageKHR VkMicromapUsageEXT
#endif

#ifndef VkMicromapTriangleKHR
#define VkMicromapTriangleKHR VkMicromapTriangleEXT
#endif

#ifndef VK_OPACITY_MICROMAP_FORMAT_4_STATE_KHR
#define VK_OPACITY_MICROMAP_FORMAT_4_STATE_KHR VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT
#endif

#ifndef VK_GEOMETRY_TYPE_MICROMAP_KHR
#define VK_GEOMETRY_TYPE_MICROMAP_KHR (VkGeometryTypeKHR)1000396001
#endif

#ifndef VK_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_KHR
#define VK_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_KHR (VkAccelerationStructureTypeKHR)1000396000
#endif

#ifndef VK_BUILD_ACCELERATION_STRUCTURE_MICROMAP_LOSSY_BIT_KHR
#define VK_BUILD_ACCELERATION_STRUCTURE_MICROMAP_LOSSY_BIT_KHR 0
#endif

#ifndef VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR
#define VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR (VkStructureType)1000150002
#endif

#ifndef VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR
#define VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR (VkAccelerationStructureBuildTypeKHR)0
#endif

// Provide dummy structure for missing types in Android EXT version of OMM
#ifndef VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_2_KHR
#	define VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_2_KHR (VkStructureType)1000396003
typedef struct VkAccelerationStructureCreateInfo2KHR {
  VkStructureType sType;
  const void* pNext;
  VkAccelerationStructureCreateFlagsKHR createFlags;
  VkDeviceAddress addressRange;
  VkDeviceSize size;
  VkAccelerationStructureTypeKHR type;
} VkAccelerationStructureCreateInfo2KHR;
#endif

#ifndef VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MICROMAP_DATA_KHR
#	define VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MICROMAP_DATA_KHR (VkStructureType)1000396002
typedef struct VkAccelerationStructureGeometryMicromapDataKHR {
  VkStructureType sType;
  const void* pNext;
  uint32_t usageCountsCount;
  const VkMicromapUsageKHR* pUsageCounts;
  const VkMicromapUsageKHR* const* ppUsageCounts;
  VkDeviceOrHostAddressConstKHR data;
  VkDeviceOrHostAddressConstKHR triangleArray;
  VkDeviceSize triangleArrayStride;
} VkAccelerationStructureGeometryMicromapDataKHR;
#endif

#ifndef VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR
#	define VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR (VkStructureType)1000150000
#endif

#ifndef VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
#define VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR (VkBuildAccelerationStructureFlagBitsKHR)0x00000002
#endif

#ifndef VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
#define VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR (VkBuildAccelerationStructureModeKHR)0
#endif

// Map other missing symbols
#ifndef VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_KHR
#	define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_KHR VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT
#endif

#ifndef VkPhysicalDeviceOpacityMicromapFeaturesKHR
#define VkPhysicalDeviceOpacityMicromapFeaturesKHR VkPhysicalDeviceOpacityMicromapFeaturesEXT
#endif

#ifndef VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_KHR
#	define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_KHR VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_EXT
#endif

#ifndef VkPhysicalDeviceOpacityMicromapPropertiesKHR
#define VkPhysicalDeviceOpacityMicromapPropertiesKHR VkPhysicalDeviceOpacityMicromapPropertiesEXT
#endif

#ifndef VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_KHR
#	define VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_KHR VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT
#endif

#ifndef VkMicromapCreateInfoKHR
#define VkMicromapCreateInfoKHR VkMicromapCreateInfoEXT
#endif

// Vulkan-Hpp compatibility aliases for the vk:: namespace
namespace vk {

#ifndef VULKAN_HPP_DISABLE_OMM_KHR_ALIASES
using AccelerationStructureTrianglesOpacityMicromapKHR = ::VkAccelerationStructureTrianglesOpacityMicromapKHR;
using MicromapUsageKHR = ::VkMicromapUsageKHR;
using MicromapTriangleKHR = ::VkMicromapTriangleKHR;
using AccelerationStructureGeometryMicromapDataKHR = ::VkAccelerationStructureGeometryMicromapDataKHR;
using PhysicalDeviceOpacityMicromapFeaturesKHR = ::VkPhysicalDeviceOpacityMicromapFeaturesKHR;
using PhysicalDeviceOpacityMicromapPropertiesKHR = ::VkPhysicalDeviceOpacityMicromapPropertiesKHR;
using MicromapCreateInfoKHR = ::VkMicromapCreateInfoKHR;

#ifndef VkMicromapBuildInfoKHR
#	define VkMicromapBuildInfoKHR VkMicromapBuildInfoEXT
#endif
#ifndef VkMicromapBuildSizesInfoKHR
#	define VkMicromapBuildSizesInfoKHR VkMicromapBuildSizesInfoEXT
#endif
using MicromapBuildInfoKHR = ::VkMicromapBuildInfoKHR;
using MicromapBuildSizesInfoKHR = ::VkMicromapBuildSizesInfoKHR;
#endif
}

#endif // VK_KHR_opacity_micromap

#endif // PLATFORM_ANDROID
