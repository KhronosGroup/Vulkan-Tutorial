
# SIGGRAPH 2026: How to write a Vulkan application

This is a simple tutorial with the objective of explaining how to write a Vulkan application in 2026.
The tutorial employs the latest recommended best practices for beginners:

- Descriptor heap
- Shader Object
- Dynamic Rendering

Shaders are written using Slang, a modern shading language hosted by the Khronos Group.

See the [course page](https://github.com/KhronosGroup/Vulkan-Tutorial/blob/main/en/courses/siggraph2026_vk_tutorial) for more information.

> [!NOTE]
> ChatGPT and Codex were used to generate code for this tutorial.
> However, all code has been reviewed, edited, and analyzed.

## Build

### Prerequisites

- Visual Studio 2022 or Clang and Ninja
- CMake 3.26 or newer
- Vulkan SDK
- `slangc`, either on `PATH` or supplied with `-DSLANGC_EXECUTABLE=...`

> [!NOTE]
> The tutorial builds and runs on Windows and Linux. Android is not supported.
> This tutorial requires a lot of modern Vulkan extensions, including `VK_EXT_descriptor_heap` and `VK_EXT_shader_object`.
> Old Vulkan drivers will not be able to run this tutorial.

### Setup

After cloning the repository, initialize the pinned dependencies with:

```bash
python .\init_submodules.py
```

The `commit = ...` entries in `.gitmodules` are a project convention used by
`init_submodules.py` to pin each dependency. Plain Git ignores this field.
The Sponza asset submodule also uses `sparse-path = Models/Sponza` so the setup
script checks out only the Sponza folder from `glTF-Sample-Assets`.

If you need to set up the dependencies manually, run the matching commands for
each dependency:

```bash
git init

git submodule add --force --name glfw https://github.com/glfw/glfw.git external/glfw
git submodule add --force --name glm https://github.com/g-truc/glm.git external/glm
git submodule add --force --name vk-bootstrap https://github.com/charles-lunarg/vk-bootstrap.git external/vk-bootstrap
git submodule add --force --name nlohmann https://github.com/nlohmann/json.git external/nlohmann
git submodule add --force --name stb https://github.com/nothings/stb.git external/stb
git submodule add --force --name tinygltf https://github.com/syoyo/tinygltf.git external/tinygltf

git clone --filter=blob:none --sparse --no-checkout https://github.com/KhronosGroup/glTF-Sample-Assets.git data/external/sponza
git -C data/external/sponza sparse-checkout set Models/Sponza
git -C data/external/sponza checkout 2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf
git submodule add --force --name gltf-sample-assets https://github.com/KhronosGroup/glTF-Sample-Assets.git data/external/sponza
git submodule absorbgitdirs data/external/sponza

git submodule init
git submodule update --init --recursive

git -C external/glfw checkout b00e6a8a88ad1b60c0a045e696301deb92c9a13e
git -C external/glm checkout 6f14f4792a0cde5d0cf2c910506724d61cb95834
git -C external/vk-bootstrap checkout 5ca6780498864ae4c12f3a594ee6a6c5133d4ce0
git -C external/nlohmann checkout 4fad4468974a7b1b26d374b1c5955d2ac7d449b0
git -C external/stb checkout 31c1ad37456438565541f4919958214b6e762fb4
git -C external/tinygltf checkout d31c16e333a6c8d593cad43f325f4e1825dd4776
git -C data/external/sponza sparse-checkout set Models/Sponza
git -C data/external/sponza checkout 2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf
```

### Compile with Visual Studio (Windows)

The repo includes a Visual Studio 2022 x64 preset:

```bash
cmake --preset vs2022-debug -DSLANGC_EXECUTABLE="C:/VulkanSDK/1.4.341.1/Bin/slangc.exe"
```

This writes Visual Studio project files and the executable under `build/vs2022`.

To build:
```bash
cmake --build --preset vs2022-debug
```

### Compile with Clang (Windows and Linux)

The repo includes CMake presets for this workflow:

```bash
cmake --preset clang-debug -DSLANGC_EXECUTABLE="C:/VulkanSDK/1.4.341.1/Bin/slangc.exe"
cmake --build --preset clang-debug
```

> [!NOTE]
> `-DSLANGC_EXECUTABLE` is optional. If not provided, CMake will use the version of `slangc` in `PATH`.

## Run

```bash
.\build\vs2022\Debug\vulkan_siggraph.exe
```

## Files

- `src/main.cpp`, `src/main.h`: Main application containing the Vulkan code.
- `src/util.cpp`, `src/util.h`: Collection of helper functions to make the tutorial easy to follow. Functions in these files parse reflection data, glTF objects, textures, and related inputs, so they are less relevant if your objective is to learn Vulkan.
- `shaders/`: Slang shaders.

## License

This SIGGRAPH tutorial sample is licensed under the [MIT License](LICENSE).
This license applies to the sample code and documentation in this directory.
Assets and third-party libraries remain under their own licenses.

## Assets

- **Sponza** - Cryengine Limited License Agreement (`LicenseRef-CRYENGINE-Agreement`). Building interior glTF scene used for lighting and rendering tests. See [glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Sponza).
- **Vulkan logo** - Khronos trademark. The Vulkan logo is a trademark of The Khronos Group Inc. The MIT license does not grant trademark rights. See the [Khronos trademark guidelines](https://www.khronos.org/legal/trademarks/).

## Third-party Libraries

- **GLFW** - [glfw/glfw](https://github.com/glfw/glfw): Window creation and Vulkan surface integration ([zlib License](https://github.com/glfw/glfw/blob/master/LICENSE.md)).
- **GLM** - [g-truc/glm](https://github.com/g-truc/glm): Header-only math library for vectors, matrices, and transforms ([MIT or Happy Bunny License](https://github.com/g-truc/glm/blob/master/copying.txt)).
- **vk-bootstrap** - [charles-lunarg/vk-bootstrap](https://github.com/charles-lunarg/vk-bootstrap): Helper library for Vulkan instance, device, queue, and swapchain setup ([MIT License](https://github.com/charles-lunarg/vk-bootstrap/blob/main/LICENSE.txt)).
- **nlohmann/json** - [nlohmann/json](https://github.com/nlohmann/json): Header-only JSON parser ([MIT License](https://github.com/nlohmann/json/blob/develop/LICENSE.MIT)).
- **stb** - [nothings/stb](https://github.com/nothings/stb): Header-only utility library; this tutorial uses `stb_image` for texture loading ([MIT or Public Domain / Unlicense](https://github.com/nothings/stb/blob/master/LICENSE)).
- **tinygltf** - [syoyo/tinygltf](https://github.com/syoyo/tinygltf): Header-only glTF 2.0 loader ([MIT License](https://github.com/syoyo/tinygltf/blob/release/LICENSE)).
