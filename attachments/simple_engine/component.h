#pragma once

#include <memory>
#include <string>

// Forward declaration
class Entity;

/**
 * @brief Base class for all components in the engine.
 *
 * Components are the building blocks of the entity-component system.
 * Each component encapsulates a specific behavior or property.
 *
 * This class implements the component system as described in the Engine_Architecture chapter:
 * https://github.com/KhronosGroup/Vulkan-Tutorial/blob/master/en/Building_a_Simple_Engine/Engine_Architecture/03_component_systems.adoc
 */
class Component {
protected:
    Entity* owner = nullptr;
    std::string name;
    bool active = true;

public:
    /**
     * @brief Constructor with optional name.
     * @param componentName The name of the component.
     */
    explicit Component(const std::string& componentName = "Component") : name(componentName) {}

    /**
     * @brief Virtual destructor for proper cleanup.
     */
    virtual ~Component() = default;

    /**
     * @brief Initialize the component.
     * Called when the component is added to an entity.
     */
    virtual void Initialize() {}

    /**
     * @brief Update the component.
     * Called every frame.
     * @param deltaTime The time elapsed since the last frame.
     */
    virtual void Update(float deltaTime) {}

    /**
     * @brief Render the component.
     * Called during the rendering phase.
     */
    virtual void Render() {}

    /**
     * @brief Set the owner entity of this component.
     * @param entity The entity that owns this component.
     */
    void SetOwner(Entity* entity) { owner = entity; }

    /**
     * @brief Get the owner entity of this component.
     * @return The entity that owns this component.
     */
    Entity* GetOwner() const { return owner; }

    /**
     * @brief Get the name of the component.
     * @return The name of the component.
     */
    const std::string& GetName() const { return name; }

    /**
     * @brief Check if the component is active.
     * @return True if the component is active, false otherwise.
     */
    bool IsActive() const { return active; }

    /**
     * @brief Set the active state of the component.
     * @param isActive The new active state.
     */
    void SetActive(bool isActive) { active = isActive; }
};
