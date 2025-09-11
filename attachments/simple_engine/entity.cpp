#include "entity.h"

// Most of the Entity class implementation is in the header file
// This file is mainly for any methods that might need additional implementation

void Entity::Initialize() {
    for (auto& component : components) {
        component->Initialize();
    }
}

void Entity::Update(std::chrono::milliseconds deltaTime) {
    if (!active) return;

    for (auto& component : components) {
        if (component->IsActive()) {
            component->Update(deltaTime);
        }
    }
}

void Entity::Render() {
    if (!active) return;

    for (auto& component : components) {
        if (component->IsActive()) {
            component->Render();
        }
    }
}
