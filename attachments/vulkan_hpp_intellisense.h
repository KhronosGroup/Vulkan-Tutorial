// This header provides a solution for intellisense issues with C++20 modules
// It allows IDEs to use traditional includes for intellisense while still using modules for compilation

#pragma once

// When using an IDE with intellisense (like Visual Studio, VS Code, CLion, etc.)
// the IDE will use this include path instead of the module import
#ifdef __INTELLISENSE__
#include <vulkan/vulkan.hpp>
#endif

// For actual compilation, the module import will be used
// This is just a dummy declaration to prevent intellisense errors
// The real definitions come from either the module or the include above
#ifdef __INTELLISENSE__
namespace vk {
    // Add any additional declarations needed for intellisense if required
}
#endif
