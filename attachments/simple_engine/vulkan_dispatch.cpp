#ifdef __INTELLISENSE__
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#include <vulkan/vk_platform.h>

// Define the defaultDispatchLoaderDynamic variable
namespace vk {
    namespace detail {
        vk::DispatchLoaderDynamic defaultDispatchLoaderDynamic;
    }
}
