struct VSInput {
    float3 inPosition;
    float3 inColor;
    float2 inTexCoord;
    float3 inNormal;
};

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float3   cameraPos;
};
[[vk::binding(0,0)]]
ConstantBuffer<UniformBuffer> ubo;

// TASK05: Acceleration structure binding
[[vk::binding(1,0)]]
RaytracingAccelerationStructure accelerationStructure;

[[vk::binding(2,0)]]
StructuredBuffer<uint> indexBuffer;

[[vk::binding(3,0)]]
StructuredBuffer<float2> uvBuffer;

// TASK09: Instance look-up table
struct InstanceLUT {
    uint materialID;
    uint indexBufferOffset;
};
[[vk::binding(4,0)]]
StructuredBuffer<InstanceLUT> instanceLUTBuffer;

struct VSOutput
{
    float4 pos : SV_Position;
    float3 fragColor;
    float2 fragTexCoord;
    float3 fragNormal;
    float3 worldPos;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.pos = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.inPosition, 1.0))));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    output.fragNormal = input.inNormal;
    output.worldPos = mul(ubo.model, float4(input.inPosition, 1.0)).xyz;
    return output;
}

// TASK09: Bindless resources
[[vk::binding(0,1)]]
SamplerState textureSampler;

[[vk::binding(1,1)]]
Texture2D<float4> textures[];

// TASK11
struct PushConstant {
    uint materialIndex;
    uint reflective;
};
[push_constant]
PushConstant pc;

static const float3 lightDir = float3(-6.0, 0.0, 6.0);

// Small epsilon to avoid self-intersection
static const float EPSILON = 0.01;

float2 intersection_uv(uint instanceID, uint primIndex, float2 barycentrics) {
    uint indexOffset = instanceLUTBuffer[NonUniformResourceIndex(instanceID)].indexBufferOffset;

    uint i0 = indexBuffer[indexOffset + (primIndex * 3 + 0)];
    uint i1 = indexBuffer[indexOffset + (primIndex * 3 + 1)];
    uint i2 = indexBuffer[indexOffset + (primIndex * 3 + 2)];

    float2 uv0 = uvBuffer[i0];
    float2 uv1 = uvBuffer[i1];
    float2 uv2 = uvBuffer[i2];

    float w0 = 1.0 - barycentrics.x - barycentrics.y;
    float w1 = barycentrics.x;
    float w2 = barycentrics.y;

    return w0 * uv0 + w1 * uv1 + w2 * uv2;
}

void apply_reflection(float3 P, float3 N, inout float4 baseColor) {
    // Build the reflections ray
    float3 V = normalize(ubo.cameraPos - P);
    float3 R = reflect(-V, N);

    RayDesc reflectionRayDesc;
    reflectionRayDesc.Origin = P;
    reflectionRayDesc.Direction = R;
    reflectionRayDesc.TMin = EPSILON;
    reflectionRayDesc.TMax = 1e4;

    // Initialize a ray query for reflections
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> rq;
    let rayFlags = RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;

    rq.TraceRayInline(accelerationStructure, rayFlags, 0xFF, reflectionRayDesc);

    while (rq.Proceed())
    {
        uint instanceID = rq.CandidateRayInstanceCustomIndex();
        uint primIndex = rq.CandidatePrimitiveIndex();

        float2 uv = intersection_uv(instanceID, primIndex, rq.CandidateTriangleBarycentrics());

        uint materialID = instanceLUTBuffer[NonUniformResourceIndex(instanceID)].materialID;
        float4 intersectionColor = textures[NonUniformResourceIndex(materialID)].SampleLevel(textureSampler, uv, 0);

        if (intersectionColor.a < 0.5) {
            // If the triangle is transparent, we continue to trace
            // to find the next opaque triangle.
        } else {
            // If we hit an opaque triangle, we stop tracing.
            rq.CommitNonOpaqueTriangleHit();
        }
    }

    bool hit = (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT);

    if (hit)
    {
        uint instanceID = rq.CommittedRayInstanceCustomIndex();
        uint primIndex = rq.CommittedPrimitiveIndex();

        float2 uv = intersection_uv(instanceID, primIndex, rq.CommittedTriangleBarycentrics());

        uint materialID = instanceLUTBuffer[NonUniformResourceIndex(instanceID)].materialID;
        float4 intersectionColor = textures[NonUniformResourceIndex(materialID)].SampleLevel(textureSampler, uv, 0);

        baseColor.rgb = lerp(baseColor.rgb, intersectionColor.rgb, 0.7);
    }
}

// TASK05: Implement ray query shadows
bool in_shadow(float3 P)
{
    // Build the shadow ray from the world position toward the light
    RayDesc shadowRayDesc;
    shadowRayDesc.Origin = P;
    shadowRayDesc.Direction = normalize(lightDir);
    shadowRayDesc.TMin = EPSILON;
    shadowRayDesc.TMax = 1e4;

    // Initialize a ray query for shadows
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES |
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
    let rayFlags = RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES |
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

    sq.TraceRayInline(accelerationStructure, rayFlags, 0xFF, shadowRayDesc);

    while (sq.Proceed())
    {
        uint instanceID = sq.CandidateRayInstanceCustomIndex();
        uint primIndex = sq.CandidatePrimitiveIndex();

        float2 uv = intersection_uv(instanceID, primIndex, sq.CandidateTriangleBarycentrics());

        uint materialID = instanceLUTBuffer[NonUniformResourceIndex(instanceID)].materialID;
        float4 intersectionColor = textures[NonUniformResourceIndex(materialID)].SampleLevel(textureSampler, uv, 0);

        if (intersectionColor.a < 0.5) {
            // If the triangle is transparent, we continue to trace
            // to find the next opaque triangle.
        } else {
            // If we hit an opaque triangle, we stop tracing.
            sq.CommitNonOpaqueTriangleHit();
        }
    }

    // If the shadow ray intersects an opaque triangle, we consider the pixel in shadow
    bool hit = (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT);

    return hit;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
   float4 baseColor = textures[pc.materialIndex].Sample(textureSampler, vertIn.fragTexCoord);

   // Alpha test
   if (baseColor.a < 0.5) discard;

   float3 P = vertIn.worldPos;
   float3 N = vertIn.fragNormal;

   if (pc.reflective > 0) {
       apply_reflection(P, N, baseColor);
   }

   bool inShadow = in_shadow(P);

   // Darken if in shadow
   if (inShadow) {
       baseColor.rgb *= 0.2;
   }

   return baseColor;
}