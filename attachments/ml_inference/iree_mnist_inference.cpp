#include "iree_mnist_inference.h"
#include "vulkan_mnist_inference.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/api.h"
#include "iree/hal/drivers/vulkan/registration/driver_module.h"

#define CHECK_OK_RET(status, msg) \
  if (!iree_status_is_ok(status)) { \
    std::cerr << msg << " failed" << std::endl; \
    iree_status_free(status); \
    return; \
  }

#define CHECK_OK_RET_VAL(status, msg, val) \
  if (!iree_status_is_ok(status)) { \
    std::cerr << msg << " failed" << std::endl; \
    iree_status_free(status); \
    return val; \
  }

struct IREEMNISTInference::Impl {
    iree_runtime_instance_t* instance = nullptr;
    iree_hal_device_t* device = nullptr;
    iree_runtime_session_t* session = nullptr;
    std::string function_name;

    ~Impl() {
        if (session) iree_runtime_session_release(session);
        if (device) iree_hal_device_release(device);
        if (instance) iree_runtime_instance_release(instance);
    }
};

IREEMNISTInference::IREEMNISTInference(const std::string& modelPath) 
    : impl(std::make_unique<Impl>()) {
    
    iree_runtime_instance_options_t instance_options;
    iree_runtime_instance_options_initialize(&instance_options);
    iree_runtime_instance_options_use_all_available_drivers(&instance_options);
    
    CHECK_OK_RET(iree_runtime_instance_create(&instance_options, iree_allocator_system(), &impl->instance),
                 "iree_runtime_instance_create");

    CHECK_OK_RET(iree_hal_create_device(iree_runtime_instance_driver_registry(impl->instance),
                                      iree_make_cstring_view("vulkan"),
                                      iree_runtime_instance_host_allocator(impl->instance),
                                      &impl->device),
               "iree_hal_create_device");

    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    
    CHECK_OK_RET(iree_runtime_session_create_with_device(impl->instance, &session_options, impl->device, 
                                                       iree_runtime_instance_host_allocator(impl->instance), 
                                                       &impl->session),
               "iree_runtime_session_create_with_device");

    CHECK_OK_RET(iree_runtime_session_append_bytecode_module_from_file(impl->session, modelPath.c_str()),
               "iree_runtime_session_append_bytecode_module_from_file");

    // Try common function names to find the entry point
    const char* candidates[] = { "module.main_graph", "module.main", "main_graph", "main" };
    for (const char* candidate : candidates) {
        iree_runtime_call_t call;
        iree_status_t status = iree_runtime_call_initialize_by_name(impl->session, iree_make_cstring_view(candidate), &call);
        if (iree_status_is_ok(status)) {
            impl->function_name = candidate;
            iree_runtime_call_deinitialize(&call);
            break;
        }
        iree_status_free(status);
    }

    if (impl->function_name.empty()) {
        std::cerr << "IREE: Could not find entry point in " << modelPath << std::endl;
        return;
    }

    ready = true;
}

IREEMNISTInference::~IREEMNISTInference() = default;

std::vector<float> IREEMNISTInference::infer(const std::vector<float>& pixels) {
    if (!ready) return std::vector<float>(10, 0.0f);

    iree_runtime_call_t call;
    CHECK_OK_RET_VAL(iree_runtime_call_initialize_by_name(impl->session, 
                                                        iree_make_cstring_view(impl->function_name.c_str()), 
                                                        &call),
                    "iree_runtime_call_initialize_by_name", std::vector<float>(10, 0.0f));

    // 1. Preprocess: SAME as Vulkan/ONNX path (centering/scaling)
    std::vector<float> preprocessed = VulkanMNISTInference::preprocess(pixels);

    // 2. Normalize: MNIST model expects [1, 1, 28, 28] with Mean: 0.1307, Std: 0.3081
    std::vector<float> normalized(784);
    for (size_t i = 0; i < 784; ++i) {
        normalized[i] = (preprocessed[i] - 0.1307f) / 0.3081f;
    }

    iree_hal_buffer_view_t* input_view = nullptr;
    iree_hal_dim_t shape[] = {1, 1, 28, 28};
    
    iree_hal_buffer_params_t buffer_params;
    memset(&buffer_params, 0, sizeof(buffer_params));
    buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
    buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    iree_hal_buffer_params_canonicalize(&buffer_params);

    CHECK_OK_RET_VAL(iree_hal_buffer_view_allocate_buffer_copy(impl->device,
                                                             iree_hal_device_allocator(impl->device),
                                                             4, shape,
                                                             IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                                             IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                                             buffer_params,
                                                             iree_make_const_byte_span(normalized.data(), normalized.size() * sizeof(float)),
                                                             &input_view),
                    "iree_hal_buffer_view_allocate_buffer_copy", std::vector<float>(10, 0.0f));

    CHECK_OK_RET_VAL(iree_runtime_call_inputs_push_back_buffer_view(&call, input_view),
                    "iree_runtime_call_inputs_push_back_buffer_view", std::vector<float>(10, 0.0f));
    iree_hal_buffer_view_release(input_view);

    CHECK_OK_RET_VAL(iree_runtime_call_invoke(&call, 0), "iree_runtime_call_invoke", std::vector<float>(10, 0.0f));

    iree_hal_buffer_view_t* output_view = nullptr;
    CHECK_OK_RET_VAL(iree_runtime_call_outputs_pop_front_buffer_view(&call, &output_view),
                    "iree_runtime_call_outputs_pop_front_buffer_view", std::vector<float>(10, 0.0f));

    std::vector<float> logits(10);
    CHECK_OK_RET_VAL(iree_hal_device_transfer_d2h(impl->device,
                                                 iree_hal_buffer_view_buffer(output_view), 0,
                                                 logits.data(),
                                                 logits.size() * sizeof(float),
                                                 IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                                 iree_infinite_timeout()),
                    "iree_hal_device_transfer_d2h", std::vector<float>(10, 0.0f));

    iree_hal_buffer_view_release(output_view);
    iree_runtime_call_deinitialize(&call);

    // 3. Apply Softmax to logits to get probabilities
    std::vector<float> probabilities(10);
    float maxLogit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (size_t i = 0; i < 10; ++i) {
        probabilities[i] = std::exp(logits[i] - maxLogit);
        sum += probabilities[i];
    }
    for (size_t i = 0; i < 10; ++i) {
        probabilities[i] /= sum;
    }

    return probabilities;
}
