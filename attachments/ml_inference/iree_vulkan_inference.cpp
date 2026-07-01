#include <iostream>
#include <vector>
#include <string>
#include <cstring>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/api.h"
#include "iree/hal/drivers/vulkan/registration/driver_module.h"
#include "iree/io/stdio_stream.h"
#include "iree/tooling/numpy_io.h"

// Define a simple CHECK_OK macro for error handling
#define CHECK_OK(status, msg) \
  if (!iree_status_is_ok(status)) { \
    std::cerr << msg << " failed: " << std::endl; \
    iree_status_fprint(stderr, status); \
    iree_status_free(status); \
    return 1; \
  }

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.vmfb> [--input input.npy] [--output output.npy]" << std::endl;
    std::cerr << "Note: Only .npy files are supported for --input/--output in this minimal sample." << std::endl;
    return 1;
  }

  const char* vmfb_path = argv[1];
  const char* input_path = nullptr;
  const char* output_path = "iree_out.npy";
  const char* function_name = "module.main";
  bool function_specified = false;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--function" && i + 1 < argc) {
      function_name = argv[++i];
      function_specified = true;
    }
  }

  // Initialize IREE runtime
  iree_runtime_instance_t* instance = nullptr;
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  
  CHECK_OK(iree_runtime_instance_create(&instance_options, iree_allocator_system(), &instance), 
           "iree_runtime_instance_create");

  // Create a device (Vulkan)
  std::cout << "Creating Vulkan device..." << std::endl;
  iree_hal_device_t* device = nullptr;
  CHECK_OK(iree_hal_create_device(iree_runtime_instance_driver_registry(instance),
                                  iree_make_cstring_view("vulkan"),
                                  iree_runtime_instance_host_allocator(instance),
                                  &device),
           "iree_hal_create_device");

  // Create a session
  std::cout << "Creating session..." << std::endl;
  iree_runtime_session_t* session = nullptr;
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  
  CHECK_OK(iree_runtime_session_create_with_device(instance, &session_options, device, 
                                                   iree_runtime_instance_host_allocator(instance), 
                                                   &session),
           "iree_runtime_session_create_with_device");

  // Load the VMFB module
  std::cout << "Loading VMFB: " << vmfb_path << "..." << std::endl;
  CHECK_OK(iree_runtime_session_append_bytecode_module_from_file(session, vmfb_path),
           "iree_runtime_session_append_bytecode_module_from_file");

  // Identify the function to call
  iree_runtime_call_t call;
  iree_status_t status;
  
  if (function_specified) {
    status = iree_runtime_call_initialize_by_name(session, iree_make_cstring_view(function_name), &call);
    CHECK_OK(status, "iree_runtime_call_initialize_by_name");
  } else {
    // Try common function names
    const char* candidates[] = { "module.main", "module.main_graph", "main", "main_graph" };
    bool found = false;
    for (const char* candidate : candidates) {
      status = iree_runtime_call_initialize_by_name(session, iree_make_cstring_view(candidate), &call);
      if (iree_status_is_ok(status)) {
        std::cout << "Using function: " << candidate << std::endl;
        found = true;
        break;
      }
      iree_status_free(status);
    }
    if (!found) {
      std::cerr << "Error: Could not find a default entry point (tried module.main, module.main_graph, main, main_graph)." << std::endl;
      std::cerr << "Please specify one explicitly with --function <name>." << std::endl;
      return 1;
    }
  }

  // Load input if provided
  if (input_path) {
    iree_io_stream_t* input_stream = nullptr;
    CHECK_OK(iree_io_stdio_stream_open(IREE_IO_STDIO_STREAM_MODE_READ, 
                                       iree_make_cstring_view(input_path),
                                       iree_allocator_system(), 
                                       &input_stream),
             "iree_io_stdio_stream_open");
             
    iree_hal_buffer_view_t* buffer_view = nullptr;
    iree_hal_buffer_params_t buffer_params;
    memset(&buffer_params, 0, sizeof(buffer_params));
    buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
    iree_hal_buffer_params_canonicalize(&buffer_params);
    
    CHECK_OK(iree_numpy_npy_load_ndarray(input_stream, IREE_NUMPY_NPY_LOAD_OPTION_DEFAULT,
                                         buffer_params, device,
                                         iree_hal_device_allocator(device),
                                         &buffer_view),
             "iree_numpy_npy_load_ndarray");
             
    CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, buffer_view),
             "iree_runtime_call_inputs_push_back_buffer_view");
             
    iree_hal_buffer_view_release(buffer_view);
    iree_io_stream_release(input_stream);
  }

  // Invoke
  std::cout << "Invoking IREE model on Vulkan..." << std::endl;
  CHECK_OK(iree_runtime_call_invoke(&call, 0),
           "iree_runtime_call_invoke");

  // Get and save output
  iree_hal_buffer_view_t* output_view = nullptr;
  CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &output_view),
           "iree_runtime_call_outputs_pop_front_buffer_view");

  if (output_path) {
    // Copy output to host-visible buffer for saving
    iree_hal_buffer_params_t host_buffer_params;
    memset(&host_buffer_params, 0, sizeof(host_buffer_params));
    host_buffer_params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    host_buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_MAPPING | IREE_HAL_BUFFER_USAGE_TRANSFER;
    
    iree_hal_buffer_t* host_buffer = nullptr;
    CHECK_OK(iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                                host_buffer_params,
                                                iree_hal_buffer_view_byte_length(output_view),
                                                &host_buffer),
             "iree_hal_allocator_allocate_buffer (host)");
             
    CHECK_OK(iree_hal_device_transfer_d2d(device,
                                            iree_hal_buffer_view_buffer(output_view), 0,
                                            host_buffer, 0,
                                            iree_hal_buffer_view_byte_length(output_view),
                                            IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                            iree_infinite_timeout()),
             "iree_hal_device_transfer_d2d");
             
    iree_hal_buffer_view_t* host_output_view = nullptr;
    CHECK_OK(iree_hal_buffer_view_create_like(host_buffer, output_view, iree_allocator_system(), &host_output_view),
             "iree_hal_buffer_view_create_like");

    iree_io_stream_t* output_stream = nullptr;
    CHECK_OK(iree_io_stdio_stream_open(IREE_IO_STDIO_STREAM_MODE_WRITE | IREE_IO_STDIO_STREAM_MODE_DISCARD, 
                                       iree_make_cstring_view(output_path),
                                       iree_allocator_system(), 
                                       &output_stream),
             "iree_io_stdio_stream_open (write)");
             
    CHECK_OK(iree_numpy_npy_save_ndarray(output_stream, IREE_NUMPY_NPY_SAVE_OPTION_DEFAULT,
                                         host_output_view, iree_allocator_system()),
             "iree_numpy_npy_save_ndarray");
             
    std::cout << "Output saved to " << output_path << std::endl;
    iree_io_stream_release(output_stream);
    iree_hal_buffer_view_release(host_output_view);
    iree_hal_buffer_release(host_buffer);
  }

  // Clean up
  iree_hal_buffer_view_release(output_view);
  iree_runtime_call_deinitialize(&call);
  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);

  std::cout << "IREE Vulkan inference completed successfully." << std::endl;
  return 0;
}
