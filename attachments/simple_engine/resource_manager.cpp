#include "resource_manager.h"

#include <ranges>

// Most of the ResourceManager class implementation is in the header file
// This file is mainly for any methods that might need additional implementation
//
// This implementation corresponds to the Engine_Architecture chapter in the tutorial:
// @see en/Building_a_Simple_Engine/Engine_Architecture/04_resource_management.adoc

bool Resource::Load() {
    loaded = true;
    return true;
}

void Resource::Unload() {
    loaded = false;
}

void ResourceManager::UnloadAllResources() {
    for (auto& val : resources | std::views::values) {
        for (auto& loadedResource : val | std::views::values) {
            loadedResource->Unload();
        }
        val.clear();
    }
    resources.clear();
}
