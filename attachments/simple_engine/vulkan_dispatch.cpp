#ifdef __INTELLISENSE__
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

// Define the defaultDispatchLoaderDynamic variable
namespace vk::detail {
    DispatchLoaderDynamic defaultDispatchLoaderDynamic;
}
