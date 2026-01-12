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
#include "mesh_component.h"
#include "model_loader.h"
#include <cmath>
#include <limits>

// Helper to transform an AABB by a matrix
static void transformAABBLocal(const glm::mat4 &M,
                               const glm::vec3 &localMin,
                               const glm::vec3 &localMax,
                               glm::vec3       &outMin,
                               glm::vec3       &outMax)
{
	const glm::vec3 c = 0.5f * (localMin + localMax);
	const glm::vec3 e = 0.5f * (localMax - localMin);

	const glm::vec3 worldCenter = glm::vec3(M * glm::vec4(c, 1.0f));
	const glm::mat3 A            = glm::mat3(M);
	const glm::mat3 AbsA         = glm::mat3(glm::abs(A[0]), glm::abs(A[1]), glm::abs(A[2]));
	const glm::vec3 worldExtents = AbsA * e;

	outMin = worldCenter - worldExtents;
	outMax = worldCenter + worldExtents;
}

void MeshComponent::RecomputeMeshAABB()
{
	if (meshAABBValid)
		return;

	if (vertices.empty())
	{
		meshAABBMin   = glm::vec3(0.0f);
		meshAABBMax   = glm::vec3(0.0f);
		meshAABBValid = false;
		return;
	}
	glm::vec3 minB = vertices[0].position;
	glm::vec3 maxB = vertices[0].position;
	for (const auto &v : vertices)
	{
		minB = glm::min(minB, v.position);
		maxB = glm::max(maxB, v.position);
	}
	meshAABBMin   = minB;
	meshAABBMax   = maxB;
	meshAABBValid = true;
}

void MeshComponent::RecomputeLocalAABB()
{
	// First ensure base mesh AABB is up to date
	RecomputeMeshAABB();

	if (!meshAABBValid)
	{
		localAABBMin   = glm::vec3(0.0f);
		localAABBMax   = glm::vec3(0.0f);
		localAABBValid = false;
		return;
	}

	if (instances.empty())
	{
		// No instances: local AABB is just the mesh AABB
		localAABBMin   = meshAABBMin;
		localAABBMax   = meshAABBMax;
		localAABBValid = true;
	}
	else
	{
		// Union of all transformed instance AABBs
		glm::vec3 fullMin(std::numeric_limits<float>::max());
		glm::vec3 fullMax(-std::numeric_limits<float>::max());

		for (const auto &inst : instances)
		{
			glm::vec3 instMin, instMax;
			transformAABBLocal(inst.modelMatrix, meshAABBMin, meshAABBMax, instMin, instMax);
			fullMin = glm::min(fullMin, instMin);
			fullMax = glm::max(fullMax, instMax);
		}

		localAABBMin   = fullMin;
		localAABBMax   = fullMax;
		localAABBValid = true;
	}
}

// Most of the MeshComponent class implementation is in the header file
// This file is mainly for any methods that might need additional implementation

void MeshComponent::CreateSphere(float radius, const glm::vec3 &color, int segments)
{
	vertices.clear();
	indices.clear();

	// Generate sphere vertices using parametric equations
	for (int lat = 0; lat <= segments; ++lat)
	{
		const auto  theta    = static_cast<float>(lat * M_PI / segments);        // Latitude angle (0 to PI)
		const float sinTheta = sinf(theta);
		const float cosTheta = cosf(theta);

		for (int lon = 0; lon <= segments; ++lon)
		{
			const auto  phi    = static_cast<float>(lon * 2.0 * M_PI / segments);        // Longitude angle (0 to 2*PI)
			const float sinPhi = sinf(phi);
			const float cosPhi = cosf(phi);

			// Calculate position
			glm::vec3 position = {
			    radius * sinTheta * cosPhi,
			    radius * cosTheta,
			    radius * sinTheta * sinPhi};

			// Normal is the same as normalized position for a sphere centered at origin
			glm::vec3 normal = glm::normalize(position);

			// Texture coordinates
			const glm::vec2 texCoord = {
			    static_cast<float>(lon) / static_cast<float>(segments),
			    static_cast<float>(lat) / static_cast<float>(segments)};

			// Calculate tangent (derivative with respect to longitude). Handle poles robustly.
			glm::vec3 tangent = {
			    -sinTheta * sinPhi,
			    0.0f,
			    sinTheta * cosPhi};
			float len2 = glm::dot(tangent, tangent);
			if (len2 < 1e-12f)
			{
				// At poles sinTheta ~ 0 -> fallback tangent orthogonal to normal
				glm::vec3 t = glm::cross(normal, glm::vec3(0.0f, 0.0f, 1.0f));
				if (glm::length(t) < 1e-12f)
				{
					t = glm::cross(normal, glm::vec3(1.0f, 0.0f, 0.0f));
				}
				tangent = glm::normalize(t);
			}
			else
			{
				tangent = glm::normalize(tangent);
			}

			vertices.push_back({position,
			                    normal,
			                    texCoord,
			                    glm::vec4(tangent, 1.0f)});
		}
	}

	// Generate indices for triangles
	for (int lat = 0; lat < segments; ++lat)
	{
		for (int lon = 0; lon < segments; ++lon)
		{
			const int current = lat * (segments + 1) + lon;
			const int next    = current + segments + 1;

			// Create two triangles for each quad
			indices.push_back(current);
			indices.push_back(next);
			indices.push_back(current + 1);

			indices.push_back(current + 1);
			indices.push_back(next);
			indices.push_back(next + 1);
		}
	}

	RecomputeLocalAABB();
}

void MeshComponent::LoadFromModel(const Model *model)
{
	if (!model)
	{
		return;
	}

	// Copy vertex and index data from the model
	vertices = model->GetVertices();
	indices  = model->GetIndices();

	RecomputeLocalAABB();
}
