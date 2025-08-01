#include "model_loader.h"
#include "renderer.h"
#include "mesh_component.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <tiny_gltf.h>

// Include stb_image for proper texture loading (implementation is in renderer_resources.cpp)
#include <stb_image.h>

// Emissive scaling factor to convert from Blender units to engine units
#define EMISSIVE_SCALE_FACTOR (1.0f / 638.0f)

ModelLoader::~ModelLoader() {
    // Destructor implementation
    models.clear();
    materials.clear();
}

bool ModelLoader::Initialize(Renderer* renderer) {
    this->renderer = renderer;

    if (!renderer) {
        std::cerr << "ModelLoader::Initialize: Renderer is null" << std::endl;
        return false;
    }

    return true;
}

Model* ModelLoader::LoadGLTF(const std::string& filename) {
    // Check if the model is already loaded
    auto it = models.find(filename);
    if (it != models.end()) {
        return it->second.get();
    }

    // Create a new model
    auto model = std::make_unique<Model>(filename);

    // Parse the GLTF file
    if (!ParseGLTF(filename, model.get())) {
        std::cerr << "ModelLoader::LoadGLTF: Failed to parse GLTF file: " << filename << std::endl;
        return nullptr;
    }

    // Store the model
    models[filename] = std::move(model);

    return models[filename].get();
}

Model* ModelLoader::LoadGLTFWithPBR(const std::string& filename,
                                   const std::string& albedoMap,
                                   const std::string& normalMap,
                                   const std::string& metallicRoughnessMap,
                                   const std::string& aoMap,
                                   const std::string& emissiveMap) {
    // Check if the model is already loaded
    auto it = models.find(filename);
    if (it != models.end()) {
        return it->second.get();
    }

    // Create a new model
    auto model = std::make_unique<Model>(filename);

    // Parse the GLTF file
    if (!ParseGLTF(filename, model.get())) {
        std::cerr << "ModelLoader::LoadGLTFWithPBR: Failed to parse GLTF file: " << filename << std::endl;
        return nullptr;
    }

    // Create a PBR material
    auto material = std::make_unique<Material>(filename + "_material");

    // Load PBR textures
    if (!LoadPBRTextures(material.get(), albedoMap, normalMap, metallicRoughnessMap, aoMap, emissiveMap)) {
        std::cerr << "ModelLoader::LoadGLTFWithPBR: Failed to load PBR textures for model: " << filename << std::endl;
    }

    // Store the material
    materials[material->GetName()] = std::move(material);

    // Store the model
    models[filename] = std::move(model);

    return models[filename].get();
}

Model* ModelLoader::GetModel(const std::string& name) {
    auto it = models.find(name);
    if (it != models.end()) {
        return it->second.get();
    }
    return nullptr;
}

Material* ModelLoader::CreatePBRMaterial(const std::string& name,
                                        const glm::vec3& albedo,
                                        float metallic,
                                        float roughness,
                                        float ao,
                                        const glm::vec3& emissive) {
    // Check if the material already exists
    auto it = materials.find(name);
    if (it != materials.end()) {
        return it->second.get();
    }

    // Create a new material
    auto material = std::make_unique<Material>(name);

    // Set PBR properties
    material->albedo = albedo;
    material->metallic = metallic;
    material->roughness = roughness;
    material->ao = ao;
    material->emissive = emissive * EMISSIVE_SCALE_FACTOR;

    // Store the material
    materials[name] = std::move(material);

    std::cout << "PBR material created successfully: " << name << std::endl;
    return materials[name].get();
}

bool ModelLoader::ParseGLTF(const std::string& filename, Model* model) {
    std::cout << "Parsing GLTF file: " << filename << std::endl;

    // Extract the directory path from the model file to use as base path for textures
    std::filesystem::path modelPath(filename);
    std::string baseTexturePath = modelPath.parent_path().string();
    if (!baseTexturePath.empty() && baseTexturePath.back() != '/') {
        baseTexturePath += "/";
    }
    std::cout << "Using base texture path: " << baseTexturePath << std::endl;

    // Create tinygltf loader
    tinygltf::Model gltfModel;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    // Set up a proper image loader callback using stb_image
    loader.SetImageLoader([](tinygltf::Image* image, const int image_idx, std::string* err,
                            std::string* warn, int req_width, int req_height,
                            const unsigned char* bytes, int size, void* user_data) -> bool {
        // Use stb_image to decode the image data
        int width, height, channels;
        unsigned char* data = stbi_load_from_memory(bytes, size, &width, &height, &channels, 0);

        if (!data) {
            if (err) {
                *err = "Failed to load image with stb_image: " + std::string(stbi_failure_reason());
            }
            return false;
        }

        // Set image properties
        image->width = width;
        image->height = height;
        image->component = channels;
        image->bits = 8;
        image->pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;

        // Copy image data
        size_t image_size = width * height * channels;
        image->image.resize(image_size);
        std::memcpy(image->image.data(), data, image_size);

        // Free stb_image data
        stbi_image_free(data);

        return true;
    }, nullptr);

    // Load the GLTF file
    bool ret = false;
    if (filename.find(".glb") != std::string::npos) {
        ret = loader.LoadBinaryFromFile(&gltfModel, &err, &warn, filename);
    } else {
        ret = loader.LoadASCIIFromFile(&gltfModel, &err, &warn, filename);
    }

    if (!warn.empty()) {
        std::cout << "GLTF Warning: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << "GLTF Error: " << err << std::endl;
        return false;
    }

    if (!ret) {
        std::cerr << "Failed to parse GLTF file: " << filename << std::endl;
        return false;
    }

    // Extract mesh data from the first mesh (for now, we'll handle multiple meshes later)
    if (gltfModel.meshes.empty()) {
        std::cerr << "No meshes found in GLTF file" << std::endl;
        return false;
    }

    // Process materials first
    for (size_t i = 0; i < gltfModel.materials.size(); ++i) {
        const auto& gltfMaterial = gltfModel.materials[i];

        // Create PBR material
        auto material = std::make_unique<Material>(gltfMaterial.name.empty() ? ("material_" + std::to_string(i)) : gltfMaterial.name);

        // Extract PBR properties
        if (gltfMaterial.pbrMetallicRoughness.baseColorFactor.size() >= 3) {
            material->albedo = glm::vec3(
                gltfMaterial.pbrMetallicRoughness.baseColorFactor[0],
                gltfMaterial.pbrMetallicRoughness.baseColorFactor[1],
                gltfMaterial.pbrMetallicRoughness.baseColorFactor[2]
            );
        }
        material->metallic = static_cast<float>(gltfMaterial.pbrMetallicRoughness.metallicFactor);
        material->roughness = static_cast<float>(gltfMaterial.pbrMetallicRoughness.roughnessFactor);

        if (gltfMaterial.emissiveFactor.size() >= 3) {
            material->emissive = glm::vec3(
                gltfMaterial.emissiveFactor[0],
                gltfMaterial.emissiveFactor[1],
                gltfMaterial.emissiveFactor[2]
            ) * EMISSIVE_SCALE_FACTOR;
        }

        // Parse KHR_materials_emissive_strength extension
        auto extensionIt = gltfMaterial.extensions.find("KHR_materials_emissive_strength");
        if (extensionIt != gltfMaterial.extensions.end()) {
            const tinygltf::Value& extension = extensionIt->second;
            if (extension.Has("emissiveStrength") && extension.Get("emissiveStrength").IsNumber()) {
                material->emissiveStrength = static_cast<float>(extension.Get("emissiveStrength").Get<double>()) * EMISSIVE_SCALE_FACTOR;
            }
        } else {
            // Default emissive strength is 1.0 according to GLTF spec, scaled for engine units
            material->emissiveStrength = 1.0f * EMISSIVE_SCALE_FACTOR;
        }


        // Extract texture information and load embedded texture data
        if (gltfMaterial.pbrMetallicRoughness.baseColorTexture.index >= 0) {
            int texIndex = gltfMaterial.pbrMetallicRoughness.baseColorTexture.index;
            if (texIndex < gltfModel.textures.size()) {
                const auto& texture = gltfModel.textures[texIndex];
                if (texture.source >= 0 && texture.source < gltfModel.images.size()) {
                    std::string textureId = "gltf_texture_" + std::to_string(texIndex);
                    material->albedoTexturePath = textureId;

                    // Load texture data (embedded or external)
                    const auto& image = gltfModel.images[texture.source];
                    std::cout << "    Image data size: " << image.image.size() << ", URI: " << image.uri << std::endl;
                    if (!image.image.empty()) {
                        // Load embedded texture data
                        std::cout << "    Loading embedded base color texture: " << textureId << std::endl;
                        if (renderer->LoadTextureFromMemory(textureId, image.image.data(),
                                                          image.width, image.height, image.component)) {
                            std::cout << "    Successfully loaded embedded base color texture: " << textureId << std::endl;
                        } else {
                            std::cerr << "    Failed to load embedded base color texture: " << textureId << std::endl;
                        }
                    } else if (!image.uri.empty()) {
                        // Load external texture file
                        std::string texturePath = baseTexturePath + image.uri;
                        std::cout << "    Loading external base color texture: " << texturePath << std::endl;
                        if (renderer->LoadTexture(texturePath)) {
                            // Update the material to use the external texture path instead of gltf_texture_X
                            material->albedoTexturePath = texturePath;
                            std::cout << "    Successfully loaded external base color texture: " << texturePath << std::endl;
                        } else {
                            std::cerr << "    Failed to load external base color texture: " << texturePath << std::endl;
                        }
                    } else {
                        std::cout << "    No image data or URI available, skipping texture loading" << std::endl;
                    }
                }
            }
        }

        if (gltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index >= 0) {
            int texIndex = gltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index;
            if (texIndex < gltfModel.textures.size()) {
                const auto& texture = gltfModel.textures[texIndex];
                if (texture.source >= 0 && texture.source < gltfModel.images.size()) {
                    std::string textureId = "gltf_texture_" + std::to_string(texIndex);
                    material->metallicRoughnessTexturePath = textureId;

                    // Load texture data (embedded or external)
                    const auto& image = gltfModel.images[texture.source];
                    if (!image.image.empty()) {
                        // Load embedded texture data
                        if (renderer->LoadTextureFromMemory(textureId, image.image.data(),
                                                          image.width, image.height, image.component)) {
                            std::cout << "    Successfully loaded embedded metallic-roughness texture: " << textureId << std::endl;
                        } else {
                            std::cerr << "    Failed to load embedded metallic-roughness texture: " << textureId << std::endl;
                        }
                    } else if (!image.uri.empty()) {
                        // Load external texture file
                        std::string texturePath = baseTexturePath + image.uri;
                        if (renderer->LoadTexture(texturePath)) {
                            // Update the material to use the external texture path instead of gltf_texture_X
                            material->metallicRoughnessTexturePath = texturePath;
                            std::cout << "    Successfully loaded external metallic-roughness texture: " << texturePath << std::endl;
                        } else {
                            std::cerr << "    Failed to load external metallic-roughness texture: " << texturePath << std::endl;
                        }
                    }
                }
            }
        }

        if (gltfMaterial.normalTexture.index >= 0) {
            int texIndex = gltfMaterial.normalTexture.index;
            if (texIndex < gltfModel.textures.size()) {
                const auto& texture = gltfModel.textures[texIndex];
                if (texture.source >= 0 && texture.source < gltfModel.images.size()) {
                    std::string textureId = "gltf_texture_" + std::to_string(texIndex);
                    material->normalTexturePath = textureId;

                    // Load texture data (embedded or external)
                    const auto& image = gltfModel.images[texture.source];
                    if (!image.image.empty()) {
                        // Load embedded texture data
                        if (renderer->LoadTextureFromMemory(textureId, image.image.data(),
                                                          image.width, image.height, image.component)) {
                            std::cout << "    Successfully loaded embedded normal texture: " << textureId
                                      << " (" << image.width << "x" << image.height << ")" << std::endl;
                        } else {
                            std::cerr << "    Failed to load embedded normal texture: " << textureId << std::endl;
                        }
                    } else if (!image.uri.empty()) {
                        // Load external texture file
                        std::string texturePath = baseTexturePath + image.uri;
                        if (renderer->LoadTexture(texturePath)) {
                            // Update the material to use the external texture path instead of gltf_texture_X
                            material->normalTexturePath = texturePath;
                            std::cout << "    Successfully loaded external normal texture: " << texturePath << std::endl;
                        } else {
                            std::cerr << "    Failed to load external normal texture: " << texturePath << std::endl;
                        }
                    }
                }
            }
        }

        if (gltfMaterial.occlusionTexture.index >= 0) {
            int texIndex = gltfMaterial.occlusionTexture.index;
            if (texIndex < gltfModel.textures.size()) {
                const auto& texture = gltfModel.textures[texIndex];
                if (texture.source >= 0 && texture.source < gltfModel.images.size()) {
                    std::string textureId = "gltf_texture_" + std::to_string(texIndex);
                    material->occlusionTexturePath = textureId;

                    // Load texture data (embedded or external)
                    const auto& image = gltfModel.images[texture.source];
                    if (!image.image.empty()) {
                        // Load embedded texture data
                        if (renderer->LoadTextureFromMemory(textureId, image.image.data(),
                                                          image.width, image.height, image.component)) {
                            std::cout << "    Successfully loaded embedded occlusion texture: " << textureId
                                      << " (" << image.width << "x" << image.height << ")" << std::endl;
                        } else {
                            std::cerr << "    Failed to load embedded occlusion texture: " << textureId << std::endl;
                        }
                    } else if (!image.uri.empty()) {
                        // Load external texture file
                        std::string texturePath = baseTexturePath + image.uri;
                        if (renderer->LoadTexture(texturePath)) {
                            // Update the material to use the external texture path instead of gltf_texture_X
                            material->occlusionTexturePath = texturePath;
                            std::cout << "    Successfully loaded external occlusion texture: " << texturePath << std::endl;
                        } else {
                            std::cerr << "    Failed to load external occlusion texture: " << texturePath << std::endl;
                        }
                    }
                }
            }
        }

        if (gltfMaterial.emissiveTexture.index >= 0) {
            int texIndex = gltfMaterial.emissiveTexture.index;
            if (texIndex < gltfModel.textures.size()) {
                const auto& texture = gltfModel.textures[texIndex];
                if (texture.source >= 0 && texture.source < gltfModel.images.size()) {
                    std::string textureId = "gltf_texture_" + std::to_string(texIndex);
                    material->emissiveTexturePath = textureId;

                    // Load texture data (embedded or external)
                    const auto& image = gltfModel.images[texture.source];
                    if (!image.image.empty()) {
                        // Load embedded texture data
                        if (renderer->LoadTextureFromMemory(textureId, image.image.data(),
                                                          image.width, image.height, image.component)) {
                            std::cout << "    Successfully loaded embedded emissive texture: " << textureId
                                      << " (" << image.width << "x" << image.height << ")" << std::endl;
                        } else {
                            std::cerr << "    Failed to load embedded emissive texture: " << textureId << std::endl;
                        }
                    } else if (!image.uri.empty()) {
                        // Load external texture file
                        std::string texturePath = baseTexturePath + image.uri;
                        if (renderer->LoadTexture(texturePath)) {
                            // Update the material to use the external texture path instead of gltf_texture_X
                            material->emissiveTexturePath = texturePath;
                            std::cout << "    Successfully loaded external emissive texture: " << texturePath << std::endl;
                        } else {
                            std::cerr << "    Failed to load external emissive texture: " << texturePath << std::endl;
                        }
                    }
                }
            }
        }

        // Store the material
        materials[material->GetName()] = std::move(material);
    }

    // Process cameras from GLTF file
    if (!gltfModel.cameras.empty()) {
        std::cout << "Found " << gltfModel.cameras.size() << " camera(s) in GLTF file" << std::endl;

        for (size_t i = 0; i < gltfModel.cameras.size(); ++i) {
            const auto& gltfCamera = gltfModel.cameras[i];
            std::cout << "  Camera " << i << ": " << gltfCamera.name << std::endl;

            // Store camera data in the model for later use
            CameraData cameraData;
            cameraData.name = gltfCamera.name.empty() ? ("camera_" + std::to_string(i)) : gltfCamera.name;

            if (gltfCamera.type == "perspective") {
                cameraData.isPerspective = true;
                cameraData.fov = static_cast<float>(gltfCamera.perspective.yfov);
                cameraData.aspectRatio = static_cast<float>(gltfCamera.perspective.aspectRatio);
                cameraData.nearPlane = static_cast<float>(gltfCamera.perspective.znear);
                cameraData.farPlane = static_cast<float>(gltfCamera.perspective.zfar);
                std::cout << "    Perspective camera: FOV=" << cameraData.fov
                          << ", Aspect=" << cameraData.aspectRatio
                          << ", Near=" << cameraData.nearPlane
                          << ", Far=" << cameraData.farPlane << std::endl;
            } else if (gltfCamera.type == "orthographic") {
                cameraData.isPerspective = false;
                cameraData.orthographicSize = static_cast<float>(gltfCamera.orthographic.ymag);
                cameraData.nearPlane = static_cast<float>(gltfCamera.orthographic.znear);
                cameraData.farPlane = static_cast<float>(gltfCamera.orthographic.zfar);
                std::cout << "    Orthographic camera: Size=" << cameraData.orthographicSize
                          << ", Near=" << cameraData.nearPlane
                          << ", Far=" << cameraData.farPlane << std::endl;
            }

            // Find the node that uses this camera to get transform information
            for (size_t nodeIdx = 0; nodeIdx < gltfModel.nodes.size(); ++nodeIdx) {
                const auto& node = gltfModel.nodes[nodeIdx];
                if (node.camera == static_cast<int>(i)) {
                    // Extract transform from node
                    if (node.translation.size() == 3) {
                        cameraData.position = glm::vec3(
                            static_cast<float>(node.translation[0]),
                            static_cast<float>(node.translation[1]),
                            static_cast<float>(node.translation[2])
                        );
                    }

                    if (node.rotation.size() == 4) {
                        cameraData.rotation = glm::quat(
                            static_cast<float>(node.rotation[3]), // w
                            static_cast<float>(node.rotation[0]), // x
                            static_cast<float>(node.rotation[1]), // y
                            static_cast<float>(node.rotation[2])  // z
                        );
                    }

                    std::cout << "    Position: (" << cameraData.position.x << ", "
                              << cameraData.position.y << ", " << cameraData.position.z << ")" << std::endl;
                    break;
                }
            }

            model->cameras.push_back(cameraData);
        }
    }

    // Group primitives by material to create separate meshes for each material
    std::map<int, std::vector<Vertex>> materialVertices;
    std::map<int, std::vector<uint32_t>> materialIndices;
    std::map<int, std::string> materialNames;

    // Process all meshes and group by material
    for (const auto& mesh : gltfModel.meshes) {
        std::cout << "Processing mesh: " << mesh.name << std::endl;

        for (const auto& primitive : mesh.primitives) {
            // Get the material index for this primitive
            int materialIndex = primitive.material;
            if (materialIndex < 0) {
                materialIndex = -1; // Use -1 for primitives without materials
            }

            // Initialize vectors for this material if not already done
            if (!materialVertices.contains(materialIndex)) {
                materialVertices[materialIndex] = std::vector<Vertex>();
                materialIndices[materialIndex] = std::vector<uint32_t>();

                // Store material name for debugging
                if (materialIndex >= 0 && materialIndex < gltfModel.materials.size()) {
                    const auto& gltfMaterial = gltfModel.materials[materialIndex];
                    materialNames[materialIndex] = gltfMaterial.name.empty() ?
                        ("material_" + std::to_string(materialIndex)) : gltfMaterial.name;
                } else {
                    materialNames[materialIndex] = "no_material";
                }

                std::cout << "  Found material " << materialIndex << ": " << materialNames[materialIndex] << std::endl;
            }
            // Get indices for this material
            if (primitive.indices >= 0) {
                const tinygltf::Accessor& indexAccessor = gltfModel.accessors[primitive.indices];
                const tinygltf::BufferView& indexBufferView = gltfModel.bufferViews[indexAccessor.bufferView];
                const tinygltf::Buffer& indexBuffer = gltfModel.buffers[indexBufferView.buffer];

                const void* indexData = &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset];

                size_t indexOffset = materialVertices[materialIndex].size();

                // Handle different index types
                if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    const auto* buf = static_cast<const uint16_t*>(indexData);
                    for (size_t i = 0; i < indexAccessor.count; ++i) {
                        materialIndices[materialIndex].push_back(buf[i] + indexOffset);
                    }
                } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    const auto* buf = static_cast<const uint32_t*>(indexData);
                    for (size_t i = 0; i < indexAccessor.count; ++i) {
                        materialIndices[materialIndex].push_back(buf[i] + indexOffset);
                    }
                }
            }

            // Get vertex positions
            auto posIt = primitive.attributes.find("POSITION");
            if (posIt == primitive.attributes.end()) {
                std::cerr << "No POSITION attribute found in primitive" << std::endl;
                continue;
            }

            const tinygltf::Accessor& posAccessor = gltfModel.accessors[posIt->second];
            const tinygltf::BufferView& posBufferView = gltfModel.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer& posBuffer = gltfModel.buffers[posBufferView.buffer];

            const auto* positions = reinterpret_cast<const float*>(
                &posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]);

            // Get texture coordinates (if available)
            const float* texCoords = nullptr;
            auto texCoordIt = primitive.attributes.find("TEXCOORD_0");
            if (texCoordIt != primitive.attributes.end()) {
                const tinygltf::Accessor& texCoordAccessor = gltfModel.accessors[texCoordIt->second];
                const tinygltf::BufferView& texCoordBufferView = gltfModel.bufferViews[texCoordAccessor.bufferView];
                const tinygltf::Buffer& texCoordBuffer = gltfModel.buffers[texCoordBufferView.buffer];
                texCoords = reinterpret_cast<const float*>(
                    &texCoordBuffer.data[texCoordBufferView.byteOffset + texCoordAccessor.byteOffset]);
            }

            // Get normals (if available)
            const float* normals = nullptr;
            auto normalIt = primitive.attributes.find("NORMAL");
            if (normalIt != primitive.attributes.end()) {
                const tinygltf::Accessor& normalAccessor = gltfModel.accessors[normalIt->second];
                const tinygltf::BufferView& normalBufferView = gltfModel.bufferViews[normalAccessor.bufferView];
                const tinygltf::Buffer& normalBuffer = gltfModel.buffers[normalBufferView.buffer];
                normals = reinterpret_cast<const float*>(
                    &normalBuffer.data[normalBufferView.byteOffset + normalAccessor.byteOffset]);
            }

            // Create vertices for this material
            for (size_t i = 0; i < posAccessor.count; ++i) {
                Vertex vertex{};

                // Position
                vertex.position = glm::vec3(
                    positions[i * 3 + 0],
                    positions[i * 3 + 1],
                    positions[i * 3 + 2]
                );

                // Normal (use extracted normals if available, otherwise default up)
                if (normals) {
                    vertex.normal = glm::vec3(
                        normals[i * 3 + 0],
                        normals[i * 3 + 1],
                        normals[i * 3 + 2]
                    );
                } else {
                    vertex.normal = glm::vec3(0.0f, 0.0f, 1.0f); // Default forward normal
                }

                // Texture coordinates
                if (texCoords) {
                    vertex.texCoord = glm::vec2(
                        texCoords[i * 2 + 0],
                        texCoords[i * 2 + 1]
                    );
                } else {
                    vertex.texCoord = glm::vec2(0.0f, 0.0f);
                }

                // Tangent (default right tangent for now, could be extracted from GLTF if available)
                vertex.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);

                materialVertices[materialIndex].push_back(vertex);
            }
        }
    }

    // Store material meshes and combine for backward compatibility
    std::vector<MaterialMesh> modelMaterialMeshes;
    std::vector<Vertex> combinedVertices;
    std::vector<uint32_t> combinedIndices;

    std::cout << "Processing " << materialVertices.size() << " materials:" << std::endl;

    for (const auto& materialPair : materialVertices) {
        int materialIndex = materialPair.first;
        const auto& vertices = materialPair.second;
        const auto& indices = materialIndices[materialIndex];

        std::cout << "  Material " << materialIndex << " (" << materialNames[materialIndex]
                  << "): " << vertices.size() << " vertices, " << indices.size() << " indices" << std::endl;

        // Create MaterialMesh for this material
        MaterialMesh materialMesh;
        materialMesh.materialIndex = materialIndex;
        materialMesh.materialName = materialNames[materialIndex];
        materialMesh.vertices = vertices;
        materialMesh.indices = indices;

        // Get ALL texture paths for this material (same as ParseGLTFDataOnly)
        if (materialIndex >= 0 && materialIndex < gltfModel.materials.size()) {
            const auto& gltfMaterial = gltfModel.materials[materialIndex];

            // Extract base color texture
            if (gltfMaterial.pbrMetallicRoughness.baseColorTexture.index >= 0) {
                int texIndex = gltfMaterial.pbrMetallicRoughness.baseColorTexture.index;
                if (texIndex < gltfModel.textures.size()) {
                    const auto& texture = gltfModel.textures[texIndex];
                    if (texture.source >= 0 && texture.source < gltfModel.images.size()) {
                        std::string textureId = "gltf_texture_" + std::to_string(texIndex);
                        materialMesh.baseColorTexturePath = textureId;
                        materialMesh.texturePath = textureId; // Keep for backward compatibility

                        // Load texture data (embedded or external)
                        const auto& image = gltfModel.images[texture.source];
                        if (!image.image.empty()) {
                            // Load embedded texture data
                            if (renderer->LoadTextureFromMemory(textureId, image.image.data(),
                                                              image.width, image.height, image.component)) {
                                std::cout << "      Loaded embedded baseColor texture: " << textureId
                                          << " (" << image.width << "x" << image.height << ")" << std::endl;
                            } else {
                                std::cerr << "      Failed to load embedded baseColor texture: " << textureId << std::endl;
                            }
                        } else if (!image.uri.empty()) {
                            // Load external texture file
                            std::string texturePath = baseTexturePath + image.uri;
                            if (renderer->LoadTexture(texturePath)) {
                                // Update the MaterialMesh to use the external texture path instead of gltf_texture_X
                                materialMesh.baseColorTexturePath = texturePath;
                                materialMesh.texturePath = texturePath; // Keep for backward compatibility
                                std::cout << "      Loaded external baseColor texture: " << texturePath << std::endl;
                            } else {
                                std::cerr << "      Failed to load external baseColor texture: " << texturePath << std::endl;
                            }
                        }
                    }
                }
            } else {
                // Since texture indices are -1, try to find external texture files by material name
                std::string materialName = materialNames[materialIndex];

                // Look for external texture files that match this specific material
                for (const auto & image : gltfModel.images) {
                    if (!image.uri.empty()) {
                        std::string imageUri = image.uri;

                        // Check if this image belongs to this specific material based on naming patterns
                        // Look for BaseColor/Albedo textures that match the material name
                        if ((imageUri.find("BaseColor") != std::string::npos ||
                             imageUri.find("Albedo") != std::string::npos ||
                             imageUri.find("Diffuse") != std::string::npos) &&
                            (imageUri.find(materialName) != std::string::npos ||
                             materialName.find(imageUri.substr(0, imageUri.find('_'))) != std::string::npos)) {

                            // Use the relative path from the GLTF directory
                            std::string texturePath = baseTexturePath + imageUri;
                            materialMesh.baseColorTexturePath = texturePath;
                            materialMesh.texturePath = texturePath; // Keep for backward compatibility
                            std::cout << "      Found external baseColor texture for " << materialName << ": " << texturePath << std::endl;
                            break;
                        }
                    }
                }
            }

            // Extract normal texture
            if (gltfMaterial.normalTexture.index >= 0) {
                int texIndex = gltfMaterial.normalTexture.index;
                if (texIndex < gltfModel.textures.size()) {
                    const auto& texture = gltfModel.textures[texIndex];
                    if (texture.source >= 0 && texture.source < gltfModel.images.size()) {
                        std::string textureId = "gltf_texture_" + std::to_string(texIndex);
                        materialMesh.normalTexturePath = textureId;

                        // Load texture data (embedded or external)
                        const auto& image = gltfModel.images[texture.source];
                        if (!image.image.empty()) {
                            // Load embedded texture data
                            if (renderer->LoadTextureFromMemory(textureId, image.image.data(),
                                                              image.width, image.height, image.component)) {
                                std::cout << "      Loaded embedded normal texture: " << textureId
                                          << " (" << image.width << "x" << image.height << ")" << std::endl;
                            } else {
                                std::cerr << "      Failed to load embedded normal texture: " << textureId << std::endl;
                            }
                        } else if (!image.uri.empty()) {
                            // Load external texture file
                            std::string texturePath = baseTexturePath + image.uri;
                            if (renderer->LoadTexture(texturePath)) {
                                // Update the MaterialMesh to use the external texture path instead of gltf_texture_X
                                materialMesh.normalTexturePath = texturePath;
                                std::cout << "      Loaded external normal texture: " << texturePath << std::endl;
                            } else {
                                std::cerr << "      Failed to load external normal texture: " << texturePath << std::endl;
                            }
                        }
                    }
                }
            } else {
                // Look for external normal texture files that match this specific material
                std::string materialName = materialNames[materialIndex];
                for (const auto & image : gltfModel.images) {
                    if (!image.uri.empty()) {
                        std::string imageUri = image.uri;
                        if (imageUri.find("Normal") != std::string::npos &&
                            (imageUri.find(materialName) != std::string::npos ||
                             materialName.find(imageUri.substr(0, imageUri.find('_'))) != std::string::npos)) {
                            std::string texturePath = baseTexturePath + imageUri;
                            materialMesh.normalTexturePath = texturePath;
                            std::cout << "      Found external normal texture for " << materialName << ": " << texturePath << std::endl;
                            break;
                        }
                    }
                }
            }

            // Extract metallic-roughness texture
            if (gltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index >= 0) {
                int texIndex = gltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index;
                if (texIndex < gltfModel.textures.size()) {
                    const auto& texture = gltfModel.textures[texIndex];
                    if (texture.source >= 0 && texture.source < gltfModel.images.size()) {
                        std::string textureId = "gltf_texture_" + std::to_string(texIndex);
                        materialMesh.metallicRoughnessTexturePath = textureId;

                        // Load texture data (embedded or external)
                        const auto& image = gltfModel.images[texture.source];
                        if (!image.image.empty()) {
                            // Load embedded texture data
                            if (renderer->LoadTextureFromMemory(textureId, image.image.data(),
                                                              image.width, image.height, image.component)) {
                                std::cout << "      Loaded embedded metallic-roughness texture: " << textureId
                                          << " (" << image.width << "x" << image.height << ")" << std::endl;
                            } else {
                                std::cerr << "      Failed to load embedded metallic-roughness texture: " << textureId << std::endl;
                            }
                        } else if (!image.uri.empty()) {
                            // Load external texture file
                            std::string texturePath = baseTexturePath + image.uri;
                            if (renderer->LoadTexture(texturePath)) {
                                // Update the MaterialMesh to use the external texture path instead of gltf_texture_X
                                materialMesh.metallicRoughnessTexturePath = texturePath;
                                std::cout << "      Loaded external metallic-roughness texture: " << texturePath << std::endl;
                            } else {
                                std::cerr << "      Failed to load external metallic-roughness texture: " << texturePath << std::endl;
                            }
                        }
                    }
                }
            } else {
                // Look for external metallic-roughness texture files that match this specific material
                std::string materialName = materialNames[materialIndex];
                for (const auto & image : gltfModel.images) {
                    if (!image.uri.empty()) {
                        std::string imageUri = image.uri;
                        if ((imageUri.find("Metallic") != std::string::npos ||
                             imageUri.find("Roughness") != std::string::npos ||
                             imageUri.find("Specular") != std::string::npos) &&
                            (imageUri.find(materialName) != std::string::npos ||
                             materialName.find(imageUri.substr(0, imageUri.find('_'))) != std::string::npos)) {
                            std::string texturePath = baseTexturePath + imageUri;
                            materialMesh.metallicRoughnessTexturePath = texturePath;
                            std::cout << "      Found external metallic-roughness texture for " << materialName << ": " << texturePath << std::endl;
                            break;
                        }
                    }
                }
            }

            // Extract occlusion texture
            if (gltfMaterial.occlusionTexture.index >= 0) {
                int texIndex = gltfMaterial.occlusionTexture.index;
                if (texIndex < gltfModel.textures.size()) {
                    const auto& texture = gltfModel.textures[texIndex];
                    if (texture.source >= 0 && texture.source < gltfModel.images.size()) {
                        std::string textureId = "gltf_texture_" + std::to_string(texIndex);
                        materialMesh.occlusionTexturePath = textureId;

                        // Load texture data (embedded or external)
                        const auto& image = gltfModel.images[texture.source];
                        if (!image.image.empty()) {
                            // Load embedded texture data
                            if (renderer->LoadTextureFromMemory(textureId, image.image.data(),
                                                              image.width, image.height, image.component)) {
                                std::cout << "      Loaded embedded occlusion texture: " << textureId
                                          << " (" << image.width << "x" << image.height << ")" << std::endl;
                            } else {
                                std::cerr << "      Failed to load embedded occlusion texture: " << textureId << std::endl;
                            }
                        } else if (!image.uri.empty()) {
                            // Load external texture file
                            std::string texturePath = baseTexturePath + image.uri;
                            if (renderer->LoadTexture(texturePath)) {
                                // Update the MaterialMesh to use the external texture path instead of gltf_texture_X
                                materialMesh.occlusionTexturePath = texturePath;
                                std::cout << "      Loaded external occlusion texture: " << texturePath << std::endl;
                            } else {
                                std::cerr << "      Failed to load external occlusion texture: " << texturePath << std::endl;
                            }
                        }
                    }
                }
            } else {
                // Look for external occlusion texture files that match this specific material
                std::string materialName = materialNames[materialIndex];
                for (const auto & image : gltfModel.images) {
                    if (!image.uri.empty()) {
                        std::string imageUri = image.uri;
                        if ((imageUri.find("Occlusion") != std::string::npos ||
                             imageUri.find("AO") != std::string::npos) &&
                            (imageUri.find(materialName) != std::string::npos ||
                             materialName.find(imageUri.substr(0, imageUri.find('_'))) != std::string::npos)) {
                            std::string texturePath = baseTexturePath + imageUri;
                            materialMesh.occlusionTexturePath = texturePath;
                            std::cout << "      Found external occlusion texture for " << materialName << ": " << texturePath << std::endl;
                            break;
                        }
                    }
                }
            }

            // Extract emissive texture
            if (gltfMaterial.emissiveTexture.index >= 0) {
                int texIndex = gltfMaterial.emissiveTexture.index;
                if (texIndex < gltfModel.textures.size()) {
                    const auto& texture = gltfModel.textures[texIndex];
                    if (texture.source >= 0 && texture.source < gltfModel.images.size()) {
                        std::string textureId = "gltf_texture_" + std::to_string(texIndex);
                        materialMesh.emissiveTexturePath = textureId;

                        // Load texture data (embedded or external)
                        const auto& image = gltfModel.images[texture.source];
                        if (!image.image.empty()) {
                            // Load embedded texture data
                            if (renderer->LoadTextureFromMemory(textureId, image.image.data(),
                                                              image.width, image.height, image.component)) {
                                std::cout << "      Loaded embedded emissive texture: " << textureId
                                          << " (" << image.width << "x" << image.height << ")" << std::endl;
                            } else {
                                std::cerr << "      Failed to load embedded emissive texture: " << textureId << std::endl;
                            }
                        } else if (!image.uri.empty()) {
                            // Load external texture file
                            std::string texturePath = baseTexturePath + image.uri;
                            if (renderer->LoadTexture(texturePath)) {
                                // Update the MaterialMesh to use the external texture path instead of gltf_texture_X
                                materialMesh.emissiveTexturePath = texturePath;
                                std::cout << "      Loaded external emissive texture: " << texturePath << std::endl;
                            } else {
                                std::cerr << "      Failed to load external emissive texture: " << texturePath << std::endl;
                            }
                        }
                    }
                }
            } else {
                // Look for external emissive texture files that match this specific material
                std::string materialName = materialNames[materialIndex];
                for (const auto & image : gltfModel.images) {
                    if (!image.uri.empty()) {
                        std::string imageUri = image.uri;
                        if ((imageUri.find("Emissive") != std::string::npos ||
                             imageUri.find("Emission") != std::string::npos) &&
                            (imageUri.find(materialName) != std::string::npos ||
                             materialName.find(imageUri.substr(0, imageUri.find('_'))) != std::string::npos)) {
                            std::string texturePath = baseTexturePath + imageUri;
                            materialMesh.emissiveTexturePath = texturePath;
                            std::cout << "      Found external emissive texture for " << materialName << ": " << texturePath << std::endl;
                            break;
                        }
                    }
                }
            }
        }

        modelMaterialMeshes.push_back(materialMesh);

        // Also add to combined mesh for backward compatibility
        size_t vertexOffset = combinedVertices.size();
        combinedVertices.insert(combinedVertices.end(), vertices.begin(), vertices.end());

        for (uint32_t index : indices) {
            combinedIndices.push_back(index + vertexOffset);
        }
    }

    // Store material meshes for this model
    materialMeshes[filename] = modelMaterialMeshes;

    // Set the combined mesh data in the model for backward compatibility
    model->SetVertices(combinedVertices);
    model->SetIndices(combinedIndices);

    // Extract lights from the GLTF model
    std::cout << "Extracting lights from GLTF model..." << std::endl;

    // Extract punctual lights (KHR_lights_punctual extension)
    if (!ExtractPunctualLights(gltfModel, filename)) {
        std::cerr << "Warning: Failed to extract punctual lights from " << filename << std::endl;
    }

    std::cout << "GLTF model loaded successfully with " << combinedVertices.size() << " vertices and " << combinedIndices.size() << " indices" << std::endl;
    return true;
}

bool ModelLoader::LoadPBRTextures(Material* material,
                                 const std::string& albedoMap,
                                 const std::string& normalMap,
                                 const std::string& metallicRoughnessMap,
                                 const std::string& aoMap,
                                 const std::string& emissiveMap) const {
    if (!material) {
        std::cerr << "ModelLoader::LoadPBRTextures: Material is null" << std::endl;
        return false;
    }

    if (!renderer) {
        std::cerr << "ModelLoader::LoadPBRTextures: Renderer is null" << std::endl;
        return false;
    }

    std::cout << "Loading PBR textures for material: " << material->GetName() << std::endl;

    bool success = true;

    // Load albedo map or create default
    if (!albedoMap.empty()) {
        std::cout << "  Loading albedo map: " << albedoMap << std::endl;
        material->albedoTexturePath = albedoMap;
        if (!renderer->LoadTexture(albedoMap)) {
            std::cerr << "  Failed to load albedo texture: " << albedoMap << std::endl;
            success = false;
        }
    } else {
        // Use shared default albedo texture (much more efficient than creating per-material textures)
        std::cout << "  Using shared default albedo texture" << std::endl;
        material->albedoTexturePath = Renderer::SHARED_DEFAULT_ALBEDO_ID;
    }

    // Load normal map or create default
    if (!normalMap.empty()) {
        std::cout << "  Loading normal map: " << normalMap << std::endl;
        material->normalTexturePath = normalMap;
        if (!renderer->LoadTexture(normalMap)) {
            std::cerr << "  Failed to load normal texture: " << normalMap << std::endl;
            success = false;
        }
    } else {
        // Use shared default normal texture (much more efficient than creating per-material textures)
        std::cout << "  Using shared default normal texture" << std::endl;
        material->normalTexturePath = Renderer::SHARED_DEFAULT_NORMAL_ID;
    }

    // Load metallic-roughness map or create default
    if (!metallicRoughnessMap.empty()) {
        std::cout << "  Loading metallic-roughness map: " << metallicRoughnessMap << std::endl;
        material->metallicRoughnessTexturePath = metallicRoughnessMap;
        if (!renderer->LoadTexture(metallicRoughnessMap)) {
            std::cerr << "  Failed to load metallic-roughness texture: " << metallicRoughnessMap << std::endl;
            success = false;
        }
    } else {
        // Use shared default metallic-roughness texture (much more efficient than creating per-material textures)
        std::cout << "  Using shared default metallic-roughness texture" << std::endl;
        material->metallicRoughnessTexturePath = Renderer::SHARED_DEFAULT_METALLIC_ROUGHNESS_ID;
    }

    // Load ambient occlusion map or create default
    if (!aoMap.empty()) {
        std::cout << "  Loading ambient occlusion map: " << aoMap << std::endl;
        material->occlusionTexturePath = aoMap;
        if (!renderer->LoadTexture(aoMap)) {
            std::cerr << "  Failed to load occlusion texture: " << aoMap << std::endl;
            success = false;
        }
    } else {
        // Use shared default occlusion texture (much more efficient than creating per-material textures)
        std::cout << "  Using shared default occlusion texture" << std::endl;
        material->occlusionTexturePath = Renderer::SHARED_DEFAULT_OCCLUSION_ID;
    }

    // Load emissive map or create default
    if (!emissiveMap.empty()) {
        std::cout << "  Loading emissive map: " << emissiveMap << std::endl;
        material->emissiveTexturePath = emissiveMap;
        if (!renderer->LoadTexture(emissiveMap)) {
            std::cerr << "  Failed to load emissive texture: " << emissiveMap << std::endl;
            success = false;
        }
    } else {
        // Use shared default emissive texture (much more efficient than creating per-material textures)
        std::cout << "  Using shared default emissive texture" << std::endl;
        material->emissiveTexturePath = Renderer::SHARED_DEFAULT_EMISSIVE_ID;
    }

    std::cout << "PBR texture paths stored for material: " << material->GetName() << std::endl;
    return success;
}

std::string ModelLoader::GetFirstMaterialTexturePath(const std::string& modelName) {
    // Get material meshes for this specific model
    auto it = materialMeshes.find(modelName);
    if (it == materialMeshes.end()) {
        std::cout << "No material meshes found for model: " << modelName << std::endl;
        return "";
    }

    const auto& modelMaterialMeshes = it->second;

    // First, try to find a material mesh with a texture path (prioritizing base color)
    for (const auto& materialMesh : modelMaterialMeshes) {
        if (!materialMesh.texturePath.empty()) {
            std::cout << "Found texture path for model " << modelName << ": " << materialMesh.texturePath << std::endl;
            return materialMesh.texturePath;
        }
    }

    // If no texture path found in MaterialMesh, try to get from the actual materials
    // Only look for albedo textures to ensure non-PBR rendering doesn't use normal/metallic maps
    for (const auto& materialMesh : modelMaterialMeshes) {
        const std::string& materialName = materialMesh.materialName;
        if (materialName.empty()) continue;

        auto materialIt = materials.find(materialName);
        if (materialIt != materials.end()) {
            const auto& material = materialIt->second;

            // Only return albedo texture for non-PBR rendering compatibility
            if (!material->albedoTexturePath.empty()) {
                std::cout << "Found albedo texture path for model " << modelName << ": " << material->albedoTexturePath << std::endl;
                return material->albedoTexturePath;
            }
            // Don't fall back to normal or metallic-roughness textures to avoid
            // using them in non-PBR rendering where they would be incorrect
        }
    }

    std::cout << "No texture path found for model: " << modelName << std::endl;
    return "";
}

std::vector<ExtractedLight> ModelLoader::GetExtractedLights(const std::string& modelName) const {
    std::vector<ExtractedLight> lights;

    // First, try to get punctual lights from the extracted lights storage
    auto lightIt = extractedLights.find(modelName);
    if (lightIt != extractedLights.end()) {
        lights = lightIt->second;
        std::cout << "Found " << lights.size() << " punctual lights for model: " << modelName << std::endl;
    }

    // Now extract emissive materials as light sources
    auto materialMeshIt = materialMeshes.find(modelName);
    if (materialMeshIt != materialMeshes.end()) {
        for (const auto& materialMesh : materialMeshIt->second) {
            // Get the material for this mesh
            auto materialIt = materials.find(materialMesh.materialName);
            if (materialIt != materials.end()) {
                const Material* material = materialIt->second.get();

                // Check if this material has emissive properties (no threshold filtering)
                float emissiveIntensity = glm::length(material->emissive) * material->emissiveStrength;
                if (emissiveIntensity >= 0.0f) { // Accept all emissive materials, including zero intensity
                    // Calculate the center position of the emissive surface
                    glm::vec3 center(0.0f);
                    if (!materialMesh.vertices.empty()) {
                        for (const auto& vertex : materialMesh.vertices) {
                            center += vertex.position;
                        }
                        center /= static_cast<float>(materialMesh.vertices.size());
                    }

                    // Calculate a reasonable direction (average normal of the surface)
                    glm::vec3 avgNormal(0.0f);
                    if (!materialMesh.vertices.empty()) {
                        for (const auto& vertex : materialMesh.vertices) {
                            avgNormal += vertex.normal;
                        }
                        avgNormal = glm::normalize(avgNormal / static_cast<float>(materialMesh.vertices.size()));
                    } else {
                        avgNormal = glm::vec3(0.0f, -1.0f, 0.0f); // Default downward direction
                    }

                    // CRITICAL FIX: Offset the light position away from the surface
                    // This allows the emissive light to properly illuminate the surface from outside
                    float offsetDistance = 0.5f; // Offset distance from surface
                    glm::vec3 lightPosition = center + avgNormal * offsetDistance;

                    // Create an emissive light source
                    ExtractedLight emissiveLight;
                    emissiveLight.type = ExtractedLight::Type::Emissive;
                    emissiveLight.position = lightPosition; // Use offset position
                    emissiveLight.color = material->emissive;
                    emissiveLight.intensity = material->emissiveStrength;
                    emissiveLight.range = 10.0f; // Default range for emissive lights
                    emissiveLight.sourceMaterial = material->GetName();
                    emissiveLight.direction = avgNormal;

                    lights.push_back(emissiveLight);

                    std::cout << "Created emissive light from material '" << material->GetName()
                              << "' at position (" << center.x << ", " << center.y << ", " << center.z
                              << ") with intensity " << emissiveIntensity << std::endl;
                }
            }
        }
    }

    std::cout << "Total lights extracted for model '" << modelName << "': " << lights.size()
              << " (including emissive-derived lights)" << std::endl;

    return lights;
}

const std::vector<MaterialMesh>& ModelLoader::GetMaterialMeshes(const std::string& modelName) const {
    auto it = materialMeshes.find(modelName);
    if (it != materialMeshes.end()) {
        return it->second;
    }
    // Return a static empty vector to avoid creating temporary objects
    static constexpr std::vector<MaterialMesh> emptyVector;
    return emptyVector;
}

Material* ModelLoader::GetMaterial(const std::string& materialName) const {
    auto it = materials.find(materialName);
    if (it != materials.end()) {
        return it->second.get();
    }
    return nullptr;
}

bool ModelLoader::ExtractPunctualLights(const tinygltf::Model& gltfModel, const std::string& modelName) {
    std::cout << "Extracting punctual lights from model: " << modelName << std::endl;

    std::vector<ExtractedLight> lights;

    // Check if the model has the KHR_lights_punctual extension
    auto extensionIt = gltfModel.extensions.find("KHR_lights_punctual");
    if (extensionIt != gltfModel.extensions.end()) {
        std::cout << "  Found KHR_lights_punctual extension" << std::endl;

        // Parse the punctual lights from the extension
        const tinygltf::Value& extension = extensionIt->second;
        if (extension.Has("lights") && extension.Get("lights").IsArray()) {
            const tinygltf::Value::Array& lightsArray = extension.Get("lights").Get<tinygltf::Value::Array>();

            for (size_t i = 0; i < lightsArray.size(); ++i) {
                const tinygltf::Value& lightValue = lightsArray[i];
                if (!lightValue.IsObject()) continue;

                ExtractedLight light;

                // Parse light type
                if (lightValue.Has("type") && lightValue.Get("type").IsString()) {
                    std::string type = lightValue.Get("type").Get<std::string>();
                    if (type == "directional") {
                        light.type = ExtractedLight::Type::Directional;
                    } else if (type == "point") {
                        light.type = ExtractedLight::Type::Point;
                    } else if (type == "spot") {
                        light.type = ExtractedLight::Type::Spot;
                    }
                }

                // Parse light color
                if (lightValue.Has("color") && lightValue.Get("color").IsArray()) {
                    const tinygltf::Value::Array& colorArray = lightValue.Get("color").Get<tinygltf::Value::Array>();
                    if (colorArray.size() >= 3) {
                        light.color = glm::vec3(
                            colorArray[0].IsNumber() ? static_cast<float>(colorArray[0].Get<double>()) : 1.0f,
                            colorArray[1].IsNumber() ? static_cast<float>(colorArray[1].Get<double>()) : 1.0f,
                            colorArray[2].IsNumber() ? static_cast<float>(colorArray[2].Get<double>()) : 1.0f
                        );
                    }
                }

                // Parse light intensity
                if (lightValue.Has("intensity") && lightValue.Get("intensity").IsNumber()) {
                    light.intensity = static_cast<float>(lightValue.Get("intensity").Get<double>());
                }

                // Parse light range (for point and spot lights)
                if (lightValue.Has("range") && lightValue.Get("range").IsNumber()) {
                    light.range = static_cast<float>(lightValue.Get("range").Get<double>());
                }

                // Parse spot light specific parameters
                if (light.type == ExtractedLight::Type::Spot && lightValue.Has("spot")) {
                    const tinygltf::Value& spotValue = lightValue.Get("spot");
                    if (spotValue.Has("innerConeAngle") && spotValue.Get("innerConeAngle").IsNumber()) {
                        light.innerConeAngle = static_cast<float>(spotValue.Get("innerConeAngle").Get<double>());
                    }
                    if (spotValue.Has("outerConeAngle") && spotValue.Get("outerConeAngle").IsNumber()) {
                        light.outerConeAngle = static_cast<float>(spotValue.Get("outerConeAngle").Get<double>());
                    }
                }

                lights.push_back(light);
                std::cout << "    Parsed punctual light " << i << ": type=" << static_cast<int>(light.type)
                          << ", intensity=" << light.intensity << std::endl;
            }
        }
    } else {
        std::cout << "  No KHR_lights_punctual extension found" << std::endl;
    }

    // Now find light nodes in the scene to get positions and directions
    for (const auto& node : gltfModel.nodes) {
        if (node.extensions.contains("KHR_lights_punctual")) {
            const tinygltf::Value& nodeExtension = node.extensions.at("KHR_lights_punctual");
            if (nodeExtension.Has("light") && nodeExtension.Get("light").IsInt()) {
                int lightIndex = nodeExtension.Get("light").Get<int>();
                if (lightIndex >= 0 && lightIndex < static_cast<int>(lights.size())) {
                    // Extract position from node transform
                    if (node.translation.size() >= 3) {
                        lights[lightIndex].position = glm::vec3(
                            static_cast<float>(node.translation[0]),
                            static_cast<float>(node.translation[1]),
                            static_cast<float>(node.translation[2])
                        );
                    }

                    // Extract direction from node rotation (for directional and spot lights)
                    if (node.rotation.size() >= 4 &&
                        (lights[lightIndex].type == ExtractedLight::Type::Directional ||
                         lights[lightIndex].type == ExtractedLight::Type::Spot)) {
                        // Convert quaternion to direction vector
                        glm::quat rotation(
                            static_cast<float>(node.rotation[3]), // w
                            static_cast<float>(node.rotation[0]), // x
                            static_cast<float>(node.rotation[1]), // y
                            static_cast<float>(node.rotation[2])  // z
                        );
                        // Default forward direction in glTF is -Z
                        lights[lightIndex].direction = rotation * glm::vec3(0.0f, 0.0f, -1.0f);
                    }

                    std::cout << "    Light " << lightIndex << " positioned at ("
                              << lights[lightIndex].position.x << ", "
                              << lights[lightIndex].position.y << ", "
                              << lights[lightIndex].position.z << ")" << std::endl;
                }
            }
        }
    }

    // Store the extracted lights
    extractedLights[modelName] = lights;

    std::cout << "  Extracted " << lights.size() << " total lights from model" << std::endl;
    return lights.empty();
}

bool ModelLoader::LoadEmbeddedGLTFTextures(const std::string& filename) const {
    std::cout << "Loading embedded GLTF textures from: " << filename << std::endl;

    if (!renderer) {
        std::cerr << "LoadEmbeddedGLTFTextures: Renderer is null" << std::endl;
        return false;
    }

    // Create a tinygltf loader with proper image loading
    tinygltf::Model gltfModel;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    // Set up a proper image loader callback using stb_image (same as ParseGLTF)
    loader.SetImageLoader([](tinygltf::Image* image, const int image_idx, std::string* err,
                            std::string* warn, int req_width, int req_height,
                            const unsigned char* bytes, int size, void* user_data) -> bool {
        // Use stb_image to decode the image data
        int width, height, channels;
        unsigned char* data = stbi_load_from_memory(bytes, size, &width, &height, &channels, 0);

        if (!data) {
            if (err) {
                *err = "Failed to load image with stb_image: " + std::string(stbi_failure_reason());
            }
            return false;
        }

        // Set image properties
        image->width = width;
        image->height = height;
        image->component = channels;
        image->bits = 8;
        image->pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;

        // Copy image data
        size_t image_size = width * height * channels;
        image->image.resize(image_size);
        std::memcpy(image->image.data(), data, image_size);

        // Free stb_image data
        stbi_image_free(data);

        std::cout << "  Loaded embedded texture: " << width << "x" << height << " with " << channels << " channels" << std::endl;
        return true;
    }, nullptr);

    // Load the GLTF file
    bool ret = false;
    if (filename.find(".glb") != std::string::npos) {
        ret = loader.LoadBinaryFromFile(&gltfModel, &err, &warn, filename);
    } else {
        ret = loader.LoadASCIIFromFile(&gltfModel, &err, &warn, filename);
    }

    if (!warn.empty()) {
        std::cout << "GLTF Warning: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << "GLTF Error: " << err << std::endl;
        return false;
    }

    if (!ret) {
        std::cerr << "Failed to parse GLTF file for texture loading: " << filename << std::endl;
        return false;
    }

    std::cout << "Successfully loaded GLTF file for texture extraction" << std::endl;

    // Load all embedded textures using LoadTextureFromMemory
    int texturesLoaded = 0;
    for (size_t texIndex = 0; texIndex < gltfModel.textures.size(); ++texIndex) {
        const auto& texture = gltfModel.textures[texIndex];
        if (texture.source >= 0 && texture.source < gltfModel.images.size()) {
            std::string textureId = "gltf_texture_" + std::to_string(texIndex);
            const auto& image = gltfModel.images[texture.source];

            if (!image.image.empty()) {
                if (renderer->LoadTextureFromMemory(textureId, image.image.data(),
                                                  image.width, image.height, image.component)) {
                    std::cout << "  Loaded embedded texture: " << textureId
                              << " (" << image.width << "x" << image.height << ")" << std::endl;
                    texturesLoaded++;
                } else {
                    std::cerr << "  Failed to load embedded texture: " << textureId << std::endl;
                }
            } else {
                std::cerr << "  Empty image data for texture: " << textureId << std::endl;
            }
        }
    }

    std::cout << "Successfully loaded " << texturesLoaded << " embedded GLTF textures" << std::endl;
    return texturesLoaded > 0;
}
