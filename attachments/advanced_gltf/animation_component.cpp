/* Copyright (c) 2025 Holochip Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <algorithm>
#include <iostream>
#include <functional>
#include <chrono>
#include <sstream>
#include <shared_mutex>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <string>


#include "animation_component.h"

#include "entity.h"
#include "transform_component.h"
#include "renderer_advanced_types.h"
#include "mesh_component.h"

void AnimationComponent_SetHierarchy(AnimationComponent* anim,
                                    const std::unordered_map<int, std::vector<int>>& nodeChildren,
                                    const std::unordered_map<int, glm::mat4>& initialLocalTransforms,
                                    const std::unordered_map<int, glm::vec3>& initialLocalTranslations,
                                    const std::unordered_map<int, glm::quat>& initialLocalRotations,
                                    const std::unordered_map<int, glm::vec3>& initialLocalScales,
                                    const std::vector<int>& rootNodes)
{
    std::unique_lock<std::shared_mutex> lock(g_advancedStateMutex);
    auto& state = g_animationAdvancedStates[anim];
    state.nodeChildren = nodeChildren;
    state.initialLocalTransforms = initialLocalTransforms;
    state.initialLocalTranslations = initialLocalTranslations;
    state.initialLocalRotations = initialLocalRotations;
    state.initialLocalScales = initialLocalScales;
    state.rootNodes = rootNodes;
}

void AnimationComponent::Update(std::chrono::milliseconds deltaTime)
{
	if (!playing || currentAnimationIndex < 0 ||
	    currentAnimationIndex >= static_cast<int>(animations.size()))
	{
		return;
	}

    AdvancedAnimationState state;
    {
        std::shared_lock<std::shared_mutex> lock(g_advancedStateMutex);
        auto stateIt = g_animationAdvancedStates.find(this);
        if (stateIt == g_animationAdvancedStates.end()) return;
        state = stateIt->second;
    }

	const Animation &anim     = animations[currentAnimationIndex];
	float            duration = anim.GetDuration();

	if (duration <= 0.0f)
	{
		return;
	}

	// Advance time
	float dt = static_cast<float>(deltaTime.count()) * 0.001f * playbackSpeed;
	currentTime += dt;

	// Handle looping or stopping at the end
	if (currentTime >= duration)
	{
		if (looping)
		{
			currentTime = std::fmod(currentTime, duration);
		}
		else
		{
			currentTime = duration;
			playing     = false;
		}
	}

	// 1. Collect all LOCAL transforms for the current time
	std::unordered_map<int, glm::vec3> currentTranslations = state.initialLocalTranslations;
	std::unordered_map<int, glm::quat> currentRotations    = state.initialLocalRotations;
	std::unordered_map<int, glm::vec3> currentScales       = state.initialLocalScales;
    std::unordered_map<int, std::vector<float>> currentWeights;

	for (const auto &channel : anim.channels)
	{
		if (channel.samplerIndex < 0 || channel.samplerIndex >= static_cast<int>(anim.samplers.size()))
			continue;

		const AnimationSampler &sampler = anim.samplers[channel.samplerIndex];
		
		switch (channel.path)
		{
			case AnimationPath::Translation:
				currentTranslations[channel.targetNode] = SampleVec3(sampler, currentTime);
				break;
			case AnimationPath::Rotation:
				currentRotations[channel.targetNode] = SampleQuat(sampler, currentTime);
				break;
			case AnimationPath::Scale:
				currentScales[channel.targetNode] = SampleVec3(sampler, currentTime);
				break;
			case AnimationPath::Weights:
				{
					int numTargets = 0;
					auto it = nodeToEntities.find(channel.targetNode);
					if (it != nodeToEntities.end() && !it->second.empty() && it->second[0]) {
						if (auto* mesh = it->second[0]->GetComponent<MeshComponent>()) {
							numTargets = GetMeshComponentMorphTargets(mesh);
						}
					}
					if (numTargets > 0) {
						currentWeights[channel.targetNode] = SampleWeights(sampler, currentTime, numTargets);
					}
				}
				break;
			default: break;
		}
	}

	// 2. Compute world transforms by traversing hierarchy
	std::unordered_map<int, glm::mat4> worldTransforms;
	
	std::function<void(int, const glm::mat4&)> computeWorldTransforms = [&](int nodeIndex, const glm::mat4& parentTransform) {
		glm::mat4 localTransform;
		if (currentTranslations.count(nodeIndex)) {
            glm::mat4 T = glm::translate(glm::mat4(1.0f), currentTranslations[nodeIndex]);
            glm::mat4 R = glm::mat4_cast(currentRotations[nodeIndex]);
            glm::mat4 S = glm::scale(glm::mat4(1.0f), currentScales[nodeIndex]);
            localTransform = T * R * S;
		} else if (state.initialLocalTransforms.count(nodeIndex)) {
            localTransform = state.initialLocalTransforms.at(nodeIndex);
        } else {
            localTransform = glm::mat4(1.0f);
        }
        
		glm::mat4 worldTransform = parentTransform * localTransform;
		worldTransforms[nodeIndex] = worldTransform;
		
		if (state.nodeChildren.count(nodeIndex)) {
			for (int childIndex : state.nodeChildren.at(nodeIndex)) {
				computeWorldTransforms(childIndex, worldTransform);
			}
		}
	};

	glm::mat4 rootTransform = glm::mat4(1.0f);
	if (owner) {
		auto* transform = owner->GetComponent<TransformComponent>();
		if (transform) {
			rootTransform = transform->GetModelMatrix();
		}
	}

	if (state.rootNodes.empty()) {
		// If no root nodes defined, we can't traverse. This shouldn't happen for glTF.
	} else {
		for (int rootIndex : state.rootNodes) {
			computeWorldTransforms(rootIndex, rootTransform);
		}
	}

	// 3. Apply world transforms to entities AND compute matrix palettes for skins
	for (const auto& [nodeIndex, entities] : nodeToEntities) {
		for (Entity* entity : entities) {
			// Once the physics system owns this entity (e.g. the Fox after it is thrown),
			// it drives the transform; skip it here so we don't fight physics and cause
			// the object to oscillate between its physics pose and its animated pose.
			if (IsEntityPhysicsOwned(entity)) continue;
			if (entity && worldTransforms.count(nodeIndex)) {
				auto* transform = entity->GetComponent<TransformComponent>();
				if (transform) {
					glm::mat4 worldMatrix = worldTransforms[nodeIndex];
					
					// Extract position, rotation, scale from world matrix
					glm::vec3 worldPos = glm::vec3(worldMatrix[3]);
					transform->SetPosition(worldPos);
					
					// Extract rotation by normalizing axes to remove scale
					glm::mat4 rotationMatrix = worldMatrix;
					rotationMatrix[0] = glm::normalize(rotationMatrix[0]);
					rotationMatrix[1] = glm::normalize(rotationMatrix[1]);
					rotationMatrix[2] = glm::normalize(rotationMatrix[2]);
					glm::quat worldRot = glm::quat_cast(rotationMatrix);
					transform->SetRotation(glm::eulerAngles(worldRot));
					
					float sx = glm::length(glm::vec3(worldMatrix[0]));
					float sy = glm::length(glm::vec3(worldMatrix[1]));
					float sz = glm::length(glm::vec3(worldMatrix[2]));
					transform->SetScale(glm::vec3(sx, sy, sz));
	
					// If this entity has a deformable mesh, compute its matrix palette
					auto* mesh = entity->GetComponent<MeshComponent>();
					if (mesh) {
						// Update morph weights if present
						auto wIt = currentWeights.find(nodeIndex);
						if (wIt != currentWeights.end()) {
							SetMeshComponentMorphWeights(mesh, wIt->second);
						}
	
						std::vector<int> jointNodes;
						std::vector<glm::mat4> ibms;
						{
							std::shared_lock<std::shared_mutex> lock(g_advancedStateMutex);
							auto meshIt = g_meshComponentData.find(mesh);
							if (meshIt != g_meshComponentData.end() && meshIt->second.isDeformable) {
								jointNodes = meshIt->second.joints;
								ibms = meshIt->second.inverseBindMatrices;
							}
						}
							
						if (!jointNodes.empty() && jointNodes.size() == ibms.size()) {
							std::vector<glm::mat4> palette(jointNodes.size());
							for (size_t i = 0; i < jointNodes.size(); ++i) {
								int jointNodeIdx = jointNodes[i];
								glm::mat4 jointWorld;
								if (worldTransforms.count(jointNodeIdx)) {
									jointWorld = worldTransforms[jointNodeIdx];
								} else {
									jointWorld = glm::mat4(1.0f);
								}
								palette[i] = glm::inverse(worldMatrix) * jointWorld * ibms[i];
							}
							SetMeshComponentJointMatrices(mesh, palette);
						} else if (IsMeshComponentDeformable(mesh)) {
							// Morph-only or incomplete skinning: provide identity joint matrix for joint 0
							SetMeshComponentJointMatrices(mesh, {glm::mat4(1.0f)});
						}
					}
				}
			}
		}
	}
}

void AnimationComponent::FindKeyframes(const std::vector<float> &times, float time,
                                       size_t &outIndex0, size_t &outIndex1, float &outT) const
{
	if (times.empty())
	{
		outIndex0 = 0;
		outIndex1 = 0;
		outT      = 0.0f;
		return;
	}

	if (times.size() == 1 || time <= times.front())
	{
		outIndex0 = 0;
		outIndex1 = 0;
		outT      = 0.0f;
		return;
	}

	if (time >= times.back())
	{
		outIndex0 = times.size() - 1;
		outIndex1 = times.size() - 1;
		outT      = 0.0f;
		return;
	}

	// Binary search for the keyframe
	auto it = std::lower_bound(times.begin(), times.end(), time);
	if (it == times.begin())
	{
		outIndex0 = 0;
		outIndex1 = 0;
		outT      = 0.0f;
		return;
	}

	outIndex1 = static_cast<size_t>(std::distance(times.begin(), it));
	outIndex0 = outIndex1 - 1;

	float t0 = times[outIndex0];
	float t1 = times[outIndex1];
	float dt = t1 - t0;

	if (dt > 0.0f)
	{
		outT = (time - t0) / dt;
	}
	else
	{
		outT = 0.0f;
	}
}

glm::vec3 AnimationComponent::SampleVec3(const AnimationSampler &sampler, float time) const
{
	if (sampler.inputTimes.empty() || sampler.outputValues.size() < 3)
	{
		return glm::vec3(0.0f);
	}

	size_t index0, index1;
	float  t;
	FindKeyframes(sampler.inputTimes, time, index0, index1, t);

	// Get values at keyframes (3 floats per vec3)
	size_t offset0 = index0 * 3;
	size_t offset1 = index1 * 3;

	if (offset0 + 2 >= sampler.outputValues.size())
	{
		offset0 = sampler.outputValues.size() - 3;
	}
	if (offset1 + 2 >= sampler.outputValues.size())
	{
		offset1 = sampler.outputValues.size() - 3;
	}

	glm::vec3 v0(sampler.outputValues[offset0],
	             sampler.outputValues[offset0 + 1],
	             sampler.outputValues[offset0 + 2]);
	glm::vec3 v1(sampler.outputValues[offset1],
	             sampler.outputValues[offset1 + 1],
	             sampler.outputValues[offset1 + 2]);

	// Interpolate based on interpolation type
	switch (sampler.interpolation)
	{
		case AnimationInterpolation::Step:
			return v0;
		case AnimationInterpolation::Linear:
			return glm::mix(v0, v1, t);
		case AnimationInterpolation::CubicSpline:
			// For cubic spline, the output has in-tangent, value, out-tangent
			// Simplified: just use linear interpolation for now
			// Full cubic spline would require reading tangents from output data
			return glm::mix(v0, v1, t);
		default:
			return glm::mix(v0, v1, t);
	}
}

glm::quat AnimationComponent::SampleQuat(const AnimationSampler &sampler, float time) const
{
	if (sampler.inputTimes.empty() || sampler.outputValues.size() < 4)
	{
		return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
	}

	size_t index0, index1;
	float  t;
	FindKeyframes(sampler.inputTimes, time, index0, index1, t);

	// Get values at keyframes (4 floats per quaternion: x, y, z, w)
	size_t offset0 = index0 * 4;
	size_t offset1 = index1 * 4;

	if (offset0 + 3 >= sampler.outputValues.size())
	{
		offset0 = sampler.outputValues.size() - 4;
	}
	if (offset1 + 3 >= sampler.outputValues.size())
	{
		offset1 = sampler.outputValues.size() - 4;
	}

	// glTF quaternions are stored as (x, y, z, w)
	glm::quat q0(sampler.outputValues[offset0 + 3],         // w
	             sampler.outputValues[offset0],             // x
	             sampler.outputValues[offset0 + 1],         // y
	             sampler.outputValues[offset0 + 2]);        // z
	glm::quat q1(sampler.outputValues[offset1 + 3],         // w
	             sampler.outputValues[offset1],             // x
	             sampler.outputValues[offset1 + 1],         // y
	             sampler.outputValues[offset1 + 2]);        // z

	// Interpolate based on interpolation type
	switch (sampler.interpolation)
	{
		case AnimationInterpolation::Step:
			return q0;
		case AnimationInterpolation::Linear:
			return glm::slerp(q0, q1, t);
		case AnimationInterpolation::CubicSpline:
			// Simplified: use slerp for now
			return glm::slerp(q0, q1, t);
		default:
			return glm::slerp(q0, q1, t);
	}
}

std::vector<float> AnimationComponent::SampleWeights(const AnimationSampler &sampler, float time, size_t numTargets) const
{
	if (sampler.inputTimes.empty() || sampler.outputValues.size() < numTargets)
	{
		return std::vector<float>(numTargets, 0.0f);
	}

	size_t index0, index1;
	float  t;
	FindKeyframes(sampler.inputTimes, time, index0, index1, t);

	std::vector<float> result(numTargets);
	if (sampler.interpolation == AnimationInterpolation::CubicSpline) {
		// CubicSpline for weights: each keyframe has (in-tangent, value, out-tangent) per target
		size_t stride = 3 * numTargets;
		if (sampler.outputValues.size() < (index1 + 1) * stride) return std::vector<float>(numTargets, 0.0f);

		float dt = sampler.inputTimes[index1] - sampler.inputTimes[index0];
		for (size_t i = 0; i < numTargets; ++i) {
			float p0 = sampler.outputValues[index0 * stride + numTargets + i];
			float m0 = sampler.outputValues[index0 * stride + 2 * numTargets + i] * dt;
			float p1 = sampler.outputValues[index1 * stride + numTargets + i];
			float m1 = sampler.outputValues[index1 * stride + i] * dt;

			float t2 = t * t;
			float t3 = t2 * t;
			result[i] = (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1;
		}
	} else {
		size_t stride = numTargets;
		if (sampler.outputValues.size() < (index1 + 1) * stride) return std::vector<float>(numTargets, 0.0f);
		
		for (size_t i = 0; i < numTargets; ++i) {
			float v0 = sampler.outputValues[index0 * stride + i];
			float v1 = sampler.outputValues[index1 * stride + i];
			if (sampler.interpolation == AnimationInterpolation::Step) {
				result[i] = v0;
			} else {
				result[i] = glm::mix(v0, v1, t);
			}
		}
	}

	return result;
}
