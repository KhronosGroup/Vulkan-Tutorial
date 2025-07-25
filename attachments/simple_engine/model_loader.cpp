#include "model_loader.h"
#include "renderer.h"
#include "mesh_component.h"
#include <iostream>
#include <fstream>
#include <tiny_gltf.h>

// Include stb_image for proper texture loading (implementation is in renderer_resources.cpp)
#include <stb_image.h>

// Forward declarations for classes that will be defined in separate files

class Material {
public:
    Material(const std::string& name) : name(name) {}
    ~Material() = default;

    const std::string& GetName() const { return name; }

    // PBR properties
    glm::vec3 albedo = glm::vec3(1.0f);
    float metallic = 0.0f;
    float roughness = 1.0f;
    float ao = 1.0f;
    glm::vec3 emissive = glm::vec3(0.0f);

    // Texture paths for PBR materials
    std::string albedoTexturePath;
    std::string normalTexturePath;
    std::string metallicRoughnessTexturePath;
    std::string occlusionTexturePath;
    std::string emissiveTexturePath;

private:
    std::string name;
};

ModelLoader::ModelLoader() {
    // Constructor implementation
}

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

    std::cout << "ModelLoader initialized successfully" << std::endl;
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
    Model* modelPtr = model.get();
    models[filename] = std::move(model);

    std::cout << "Model loaded successfully: " << filename << std::endl;
    return modelPtr;
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
    Model* modelPtr = model.get();
    models[filename] = std::move(model);

    std::cout << "Model with PBR materials loaded successfully: " << filename << std::endl;
    return modelPtr;
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
    material->emissive = emissive;

    // Store the material
    Material* materialPtr = material.get();
    materials[name] = std::move(material);

    std::cout << "PBR material created successfully: " << name << std::endl;
    return materialPtr;
}

bool ModelLoader::ParseGLTF(const std::string& filename, Model* model) {
    std::cout << "Parsing GLTF file: " << filename << std::endl;

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

        std::cout << "Loaded texture: " << width << "x" << height << " with " << channels << " channels" << std::endl;
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

    std::cout << "Successfully loaded GLTF file with " << gltfModel.meshes.size() << " meshes" << std::endl;

    // Extract mesh data from the first mesh (for now, we'll handle multiple meshes later)
    if (gltfModel.meshes.empty()) {
        std::cerr << "No meshes found in GLTF file" << std::endl;
        return false;
    }

    // Process materials first
    std::cout << "Processing " << gltfModel.materials.size() << " materials" << std::endl;
    for (size_t i = 0; i < gltfModel.materials.size(); ++i) {
        const auto& gltfMaterial = gltfModel.materials[i];
        std::cout << "  Material " << i << ": " << gltfMaterial.name << std::endl;

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
        material->metallic = gltfMaterial.pbrMetallicRoughness.metallicFactor;
        material->roughness = gltfMaterial.pbrMetallicRoughness.roughnessFactor;

        if (gltfMaterial.emissiveFactor.size() >= 3) {
            material->emissive = glm::vec3(
                gltfMaterial.emissiveFactor[0],
                gltfMaterial.emissiveFactor[1],
                gltfMaterial.emissiveFactor[2]
            );
        }

        // Extract texture information and load embedded texture data
        if (gltfMaterial.pbrMetallicRoughness.baseColorTexture.index >= 0) {
            int texIndex = gltfMaterial.pbrMetallicRoughness.baseColorTexture.index;
            if (texIndex < gltfModel.textures.size()) {
                const auto& texture = gltfModel.textures[texIndex];
                if (texture.source >= 0 && texture.source < gltfModel.images.size()) {
                    std::string textureId = "gltf_texture_" + std::to_string(texIndex);
                    material->albedoTexturePath = textureId;

                    // Load embedded texture data
                    const auto& image = gltfModel.images[texture.source];
                    if (!image.image.empty()) {
                        if (renderer->LoadTextureFromMemory(textureId, image.image.data(),
                                                          image.width, image.height, image.component)) {
                            std::cout << "    Loaded base color texture: " << textureId
                                      << " (" << image.width << "x" << image.height << ")" << std::endl;
                        } else {
                            std::cerr << "    Failed to load base color texture: " << textureId << std::endl;
                        }
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

                    // Load embedded texture data
                    const auto& image = gltfModel.images[texture.source];
                    if (!image.image.empty()) {
                        if (renderer->LoadTextureFromMemory(textureId, image.image.data(),
                                                          image.width, image.height, image.component)) {
                            std::cout << "    Loaded metallic-roughness texture: " << textureId
                                      << " (" << image.width << "x" << image.height << ")" << std::endl;
                        } else {
                            std::cerr << "    Failed to load metallic-roughness texture: " << textureId << std::endl;
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

                    // Load embedded texture data
                    const auto& image = gltfModel.images[texture.source];
                    if (!image.image.empty()) {
                        if (renderer->LoadTextureFromMemory(textureId, image.image.data(),
                                                          image.width, image.height, image.component)) {
                            std::cout << "    Loaded normal texture: " << textureId
                                      << " (" << image.width << "x" << image.height << ")" << std::endl;
                        } else {
                            std::cerr << "    Failed to load normal texture: " << textureId << std::endl;
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

                    // Load embedded texture data
                    const auto& image = gltfModel.images[texture.source];
                    if (!image.image.empty()) {
                        if (renderer->LoadTextureFromMemory(textureId, image.image.data(),
                                                          image.width, image.height, image.component)) {
                            std::cout << "    Loaded occlusion texture: " << textureId
                                      << " (" << image.width << "x" << image.height << ")" << std::endl;
                        } else {
                            std::cerr << "    Failed to load occlusion texture: " << textureId << std::endl;
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

                    // Load embedded texture data
                    const auto& image = gltfModel.images[texture.source];
                    if (!image.image.empty()) {
                        if (renderer->LoadTextureFromMemory(textureId, image.image.data(),
                                                          image.width, image.height, image.component)) {
                            std::cout << "    Loaded emissive texture: " << textureId
                                      << " (" << image.width << "x" << image.height << ")" << std::endl;
                        } else {
                            std::cerr << "    Failed to load emissive texture: " << textureId << std::endl;
                        }
                    }
                }
            }
        }

        // Store the material
        materials[material->GetName()] = std::move(material);
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
            if (materialVertices.find(materialIndex) == materialVertices.end()) {
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
                    const uint16_t* buf = static_cast<const uint16_t*>(indexData);
                    for (size_t i = 0; i < indexAccessor.count; ++i) {
                        materialIndices[materialIndex].push_back(buf[i] + indexOffset);
                    }
                } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    const uint32_t* buf = static_cast<const uint32_t*>(indexData);
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

            const float* positions = reinterpret_cast<const float*>(
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

                // Color (use normal as color if available, otherwise white)
                if (normals) {
                    vertex.color = glm::vec3(
                        std::abs(normals[i * 3 + 0]),
                        std::abs(normals[i * 3 + 1]),
                        std::abs(normals[i * 3 + 2])
                    );
                } else {
                    vertex.color = glm::vec3(0.8f, 0.8f, 0.8f);
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

        // Get texture path for this material
        if (materialIndex >= 0 && materialIndex < gltfModel.materials.size()) {
            const auto& gltfMaterial = gltfModel.materials[materialIndex];

            // Try to get base color texture first
            if (gltfMaterial.pbrMetallicRoughness.baseColorTexture.index >= 0) {
                int texIndex = gltfMaterial.pbrMetallicRoughness.baseColorTexture.index;
                materialMesh.texturePath = "gltf_texture_" + std::to_string(texIndex);
            }
            // Fall back to other texture types if no base color
            else if (gltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index >= 0) {
                int texIndex = gltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index;
                materialMesh.texturePath = "gltf_texture_" + std::to_string(texIndex);
            }
            else if (gltfMaterial.normalTexture.index >= 0) {
                int texIndex = gltfMaterial.normalTexture.index;
                materialMesh.texturePath = "gltf_texture_" + std::to_string(texIndex);
            }
        }

        std::cout << "    Texture path: " << (materialMesh.texturePath.empty() ? "none" : materialMesh.texturePath) << std::endl;

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

    // Extract lights from emissive materials
    if (!ExtractEmissiveLights(gltfModel, filename)) {
        std::cerr << "Warning: Failed to extract emissive lights from " << filename << std::endl;
    }

    std::cout << "GLTF model loaded successfully with " << combinedVertices.size() << " vertices and " << combinedIndices.size() << " indices" << std::endl;
    return true;
}

bool ModelLoader::LoadPBRTextures(Material* material,
                                 const std::string& albedoMap,
                                 const std::string& normalMap,
                                 const std::string& metallicRoughnessMap,
                                 const std::string& aoMap,
                                 const std::string& emissiveMap) {
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

    // Load albedo map
    if (!albedoMap.empty()) {
        std::cout << "  Loading albedo map: " << albedoMap << std::endl;
        material->albedoTexturePath = albedoMap;
        if (!renderer->LoadTexture(albedoMap)) {
            std::cerr << "  Failed to load albedo texture: " << albedoMap << std::endl;
            success = false;
        }
    }

    // Load normal map
    if (!normalMap.empty()) {
        std::cout << "  Loading normal map: " << normalMap << std::endl;
        material->normalTexturePath = normalMap;
        if (!renderer->LoadTexture(normalMap)) {
            std::cerr << "  Failed to load normal texture: " << normalMap << std::endl;
            success = false;
        }
    }

    // Load metallic-roughness map
    if (!metallicRoughnessMap.empty()) {
        std::cout << "  Loading metallic-roughness map: " << metallicRoughnessMap << std::endl;
        material->metallicRoughnessTexturePath = metallicRoughnessMap;
        if (!renderer->LoadTexture(metallicRoughnessMap)) {
            std::cerr << "  Failed to load metallic-roughness texture: " << metallicRoughnessMap << std::endl;
            success = false;
        }
    }

    // Load ambient occlusion map
    if (!aoMap.empty()) {
        std::cout << "  Loading ambient occlusion map: " << aoMap << std::endl;
        material->occlusionTexturePath = aoMap;
        if (!renderer->LoadTexture(aoMap)) {
            std::cerr << "  Failed to load occlusion texture: " << aoMap << std::endl;
            success = false;
        }
    }

    // Load emissive map
    if (!emissiveMap.empty()) {
        std::cout << "  Loading emissive map: " << emissiveMap << std::endl;
        material->emissiveTexturePath = emissiveMap;
        if (!renderer->LoadTexture(emissiveMap)) {
            std::cerr << "  Failed to load emissive texture: " << emissiveMap << std::endl;
            success = false;
        }
    }

    std::cout << "PBR texture paths stored for material: " << material->GetName() << std::endl;
    return success;
}

std::string ModelLoader::GetFirstMaterialTexturePath(const std::string& modelName) {
    // Iterate through all materials to find the first one with an albedo texture path
    for (const auto& materialPair : materials) {
        const auto& material = materialPair.second;
        if (!material->albedoTexturePath.empty()) {
            std::cout << "Found texture path for model " << modelName << ": " << material->albedoTexturePath << std::endl;
            return material->albedoTexturePath;
        }
    }

    // If no albedo texture found, try other texture types
    for (const auto& materialPair : materials) {
        const auto& material = materialPair.second;
        if (!material->normalTexturePath.empty()) {
            std::cout << "Found normal texture path for model " << modelName << ": " << material->normalTexturePath << std::endl;
            return material->normalTexturePath;
        }
        if (!material->metallicRoughnessTexturePath.empty()) {
            std::cout << "Found metallic-roughness texture path for model " << modelName << ": " << material->metallicRoughnessTexturePath << std::endl;
            return material->metallicRoughnessTexturePath;
        }
    }

    std::cout << "No texture path found for model: " << modelName << std::endl;
    return "";
}

std::vector<ExtractedLight> ModelLoader::GetExtractedLights(const std::string& modelName) const {
    auto it = extractedLights.find(modelName);
    if (it != extractedLights.end()) {
        return it->second;
    }
    return std::vector<ExtractedLight>();
}

std::vector<MaterialMesh> ModelLoader::GetMaterialMeshes(const std::string& modelName) const {
    auto it = materialMeshes.find(modelName);
    if (it != materialMeshes.end()) {
        return it->second;
    }
    return std::vector<MaterialMesh>();
}

bool ModelLoader::ExtractPunctualLights(const tinygltf::Model& gltfModel, const std::string& modelName) {
    std::cout << "Extracting punctual lights from model: " << modelName << std::endl;

    // Check if the model has the KHR_lights_punctual extension
    auto extensionIt = gltfModel.extensions.find("KHR_lights_punctual");
    if (extensionIt == gltfModel.extensions.end()) {
        std::cout << "  No KHR_lights_punctual extension found" << std::endl;
        return true; // Not an error, just no punctual lights
    }

    std::cout << "  Found KHR_lights_punctual extension" << std::endl;

    // TODO: Parse the punctual lights from the extension
    // This would require parsing the JSON structure of the extension
    // For now, we'll focus on emissive lights

    return true;
}

bool ModelLoader::ExtractEmissiveLights(const tinygltf::Model& gltfModel, const std::string& modelName) {
    std::cout << "Extracting emissive lights from model: " << modelName << std::endl;

    std::vector<ExtractedLight> lights;

    // Iterate through materials to find emissive ones
    for (size_t i = 0; i < gltfModel.materials.size(); ++i) {
        const auto& gltfMaterial = gltfModel.materials[i];

        // Check if material has emissive properties
        bool hasEmissiveFactor = gltfMaterial.emissiveFactor.size() >= 3;
        bool hasEmissiveTexture = gltfMaterial.emissiveTexture.index >= 0;

        if (!hasEmissiveFactor && !hasEmissiveTexture) {
            continue; // No emissive properties
        }

        // Calculate emissive intensity
        glm::vec3 emissiveColor(0.0f);
        if (hasEmissiveFactor) {
            emissiveColor = glm::vec3(
                gltfMaterial.emissiveFactor[0],
                gltfMaterial.emissiveFactor[1],
                gltfMaterial.emissiveFactor[2]
            );
        }

        // Calculate luminance to determine if this should be a light source
        float luminance = 0.299f * emissiveColor.r + 0.587f * emissiveColor.g + 0.114f * emissiveColor.b;

        // Only create lights for materials with significant emissive values
        if (luminance > 0.1f) { // Threshold for creating a light
            ExtractedLight light;
            light.type = ExtractedLight::Type::Emissive;
            light.color = emissiveColor;
            light.intensity = luminance * 10.0f; // Scale up for lighting
            light.range = 50.0f; // Default range for emissive lights
            light.sourceMaterial = gltfMaterial.name.empty() ? ("material_" + std::to_string(i)) : gltfMaterial.name;

            // For now, place the light at the origin - we'll improve this later
            // by calculating positions from mesh geometry
            light.position = glm::vec3(0.0f, 5.0f, 0.0f);

            lights.push_back(light);

            std::cout << "  Created emissive light from material '" << light.sourceMaterial
                      << "' with intensity " << light.intensity << std::endl;
        }
    }

    // Store the extracted lights
    extractedLights[modelName] = lights;

    std::cout << "  Extracted " << lights.size() << " emissive lights" << std::endl;
    return true;
}

std::vector<MaterialMesh> ModelLoader::ParseGLTFDataOnly(const std::string& filename) {
    std::cout << "Thread-safe parsing GLTF file: " << filename << std::endl;

    // Create tinygltf loader
    tinygltf::Model gltfModel;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    // Set up a dummy image loader callback that doesn't load actual image data
    loader.SetImageLoader([](tinygltf::Image* image, const int image_idx, std::string* err,
                            std::string* warn, int req_width, int req_height,
                            const unsigned char* bytes, int size, void* user_data) -> bool {
        // Just set basic image properties without loading actual data
        // This avoids any potential file I/O or memory allocation issues in background thread
        image->width = 1;
        image->height = 1;
        image->component = 4; // RGBA
        image->bits = 8;
        image->pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
        image->image.resize(4, 255); // Dummy white pixel
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
        return std::vector<MaterialMesh>();
    }

    if (!ret) {
        std::cerr << "Failed to parse GLTF file: " << filename << std::endl;
        return std::vector<MaterialMesh>();
    }

    std::cout << "Successfully loaded GLTF file with " << gltfModel.meshes.size() << " meshes (thread-safe)" << std::endl;

    // Extract mesh data from the first mesh (for now, we'll handle multiple meshes later)
    if (gltfModel.meshes.empty()) {
        std::cerr << "No meshes found in GLTF file" << std::endl;
        return std::vector<MaterialMesh>();
    }

    // Group primitives by material to create separate meshes for each material
    std::map<int, std::vector<Vertex>> materialVertices;
    std::map<int, std::vector<uint32_t>> materialIndices;
    std::map<int, std::string> materialNames;

    // Process all meshes and group by material
    for (const auto& mesh : gltfModel.meshes) {
        std::cout << "Processing mesh: " << mesh.name << " (thread-safe)" << std::endl;

        for (const auto& primitive : mesh.primitives) {
            // Get the material index for this primitive
            int materialIndex = primitive.material;
            if (materialIndex < 0) {
                materialIndex = -1; // Use -1 for primitives without materials
            }

            // Initialize vectors for this material if not already done
            if (materialVertices.find(materialIndex) == materialVertices.end()) {
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

                std::cout << "  Found material " << materialIndex << ": " << materialNames[materialIndex] << " (thread-safe)" << std::endl;
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
                    const uint16_t* buf = static_cast<const uint16_t*>(indexData);
                    for (size_t i = 0; i < indexAccessor.count; ++i) {
                        materialIndices[materialIndex].push_back(buf[i] + indexOffset);
                    }
                } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    const uint32_t* buf = static_cast<const uint32_t*>(indexData);
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

            const float* positions = reinterpret_cast<const float*>(
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

                // Color (use normal as color if available, otherwise white)
                if (normals) {
                    vertex.color = glm::vec3(
                        std::abs(normals[i * 3 + 0]),
                        std::abs(normals[i * 3 + 1]),
                        std::abs(normals[i * 3 + 2])
                    );
                } else {
                    vertex.color = glm::vec3(0.8f, 0.8f, 0.8f);
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

                materialVertices[materialIndex].push_back(vertex);
            }
        }
    }

    // Create material meshes with texture path information (but don't load textures)
    std::vector<MaterialMesh> modelMaterialMeshes;

    std::cout << "Processing " << materialVertices.size() << " materials (thread-safe):" << std::endl;

    for (const auto& materialPair : materialVertices) {
        int materialIndex = materialPair.first;
        const auto& vertices = materialPair.second;
        const auto& indices = materialIndices[materialIndex];

        std::cout << "  Material " << materialIndex << " (" << materialNames[materialIndex]
                  << "): " << vertices.size() << " vertices, " << indices.size() << " indices (thread-safe)" << std::endl;

        // Create MaterialMesh for this material
        MaterialMesh materialMesh;
        materialMesh.materialIndex = materialIndex;
        materialMesh.materialName = materialNames[materialIndex];
        materialMesh.vertices = vertices;
        materialMesh.indices = indices;

        // Get texture path for this material (but don't load the texture)
        if (materialIndex >= 0 && materialIndex < gltfModel.materials.size()) {
            const auto& gltfMaterial = gltfModel.materials[materialIndex];

            // Try to get base color texture first
            if (gltfMaterial.pbrMetallicRoughness.baseColorTexture.index >= 0) {
                int texIndex = gltfMaterial.pbrMetallicRoughness.baseColorTexture.index;
                materialMesh.texturePath = "gltf_texture_" + std::to_string(texIndex);
            }
            // Fall back to other texture types if no base color
            else if (gltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index >= 0) {
                int texIndex = gltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index;
                materialMesh.texturePath = "gltf_texture_" + std::to_string(texIndex);
            }
            else if (gltfMaterial.normalTexture.index >= 0) {
                int texIndex = gltfMaterial.normalTexture.index;
                materialMesh.texturePath = "gltf_texture_" + std::to_string(texIndex);
            }
        }

        std::cout << "    Texture path: " << (materialMesh.texturePath.empty() ? "none" : materialMesh.texturePath) << " (thread-safe)" << std::endl;

        modelMaterialMeshes.push_back(materialMesh);
    }

    std::cout << "Thread-safe GLTF parsing completed with " << modelMaterialMeshes.size() << " material meshes" << std::endl;
    return modelMaterialMeshes;
}
