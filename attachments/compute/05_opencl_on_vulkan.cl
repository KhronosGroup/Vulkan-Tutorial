// ===========================================================================
// 05_opencl_on_vulkan.cl  —  Instanced forest renderer in OpenCL C
// ===========================================================================
// This is the ONE kernel that drives the entire Chapter 05 demo. The exact
// same source file is fed onto Vulkan two different ways:
//
//   1. clspv (AOT):  the build system compiles this file to `forest.spv`, which
//                    the host loads into a *raw Vulkan compute pipeline*.
//   2. clvk (layer): the host hands this source to the OpenCL 3.0 runtime at
//                    run time; clvk uses clspv internally to produce SPIR-V and
//                    dispatches it on the Vulkan driver.
//
// Both paths must produce a byte-for-byte identical image — that is the whole
// point of the demo, and it is what "OpenCL on Vulkan" guarantees.
//
// The scene is a forest grown by *instancing a single tree*: one tree SDF is
// repeated across an infinite grid using domain repetition (round(p/cell)), and
// a per-cell hash gives each instance its own height, canopy size, colour, and
// the occasional clearing. This is the SDF analogue of instanced rendering — one
// primitive, drawn thousands of times with per-instance variation.
//
// Portability rules followed here (see "Kernel Portability" in the chapter):
//   * Only __global *buffer* arguments are used (no scalar/POD kernel args), so
//     clspv's descriptor mapping is fully deterministic:
//         arg 0 (params)  -> set 0, binding 0   (storage buffer)
//         arg 1 (output)  -> set 0, binding 1   (storage buffer)
//   * The output is a __global uint* (one packed RGBA8 word per pixel), so only
//     32-bit storage access is needed — no 8-bit storage Vulkan feature.
//   * reqd_work_group_size pins the local size at compile time, and the host
//     rounds the global size up to a multiple of it, so the NDRange stays
//     uniform (required by the clspv default path).
//   * Every invocation computes one pixel independently — no atomics, no shared
//     state, no cross-invocation races — so the result is bit-deterministic and
//     the two compile paths agree exactly.
// ===========================================================================

// Layout MUST match the C++ `Params` struct on the host. All members are 4-byte
// scalars, so std430 / scalar layout places them at offsets 0,4,8,12,16,20.
typedef struct {
    int   width;
    int   height;
    float camX;       // camera position (for fly-through navigation)
    float camY;
    float camZ;
    float camYaw;     // camera heading (radians)
    float camPitch;   // camera pitch (radians)
    float fog;        // exponential fog density
} Params;

#define CELL 2.2f       // grid spacing of the instanced forest
// Fixed raymarch step budget. NOTE: this is a compile-time constant on purpose —
// clspv miscompiles a raymarch loop whose bound is loaded from a storage buffer
// (the structured-control-flow pass cannot bound it), so do NOT make this dynamic.
#define MAX_STEPS 128

static float frac1(float x)       { return x - floor(x); }
static float hash21(float2 id)    { return frac1(sin(id.x * 127.1f + id.y * 311.7f) * 43758.5453f); }

// --- Signed distance primitives -------------------------------------------
static float sdEllipsoid(float3 p, float3 r) {
    float k0 = length(p / r);
    float k1 = length(p / (r * r));
    return k0 * (k0 - 1.0f) / k1;            // good approximation; under-relax marching
}
static float sdCappedCylinder(float3 p, float h, float r) {  // axis = y
    float2 d = (float2)(length(p.xz) - r, fabs(p.y) - h);
    return fmin(fmax(d.x, d.y), 0.0f) + length(fmax(d, (float2)(0.0f)));
}

// --- Scene: ground plane + one tree instanced across a grid ----------------
// Returns (distance, materialId): 0 = ground, 1 = trunk, 2 = canopy.
static float2 map(float3 p) {
    float2 res = (float2)(p.y, 0.0f);                  // ground plane y = 0

    float2 id  = round(p.xz / CELL);                   // which grid cell
    float2 lp  = p.xz - CELL * id;                      // position within the cell
    float  h   = hash21(id);                            // per-instance random

    if (h > 0.12f) {                                   // ~12% of cells are clearings
        float3 q  = (float3)(lp.x, p.y, lp.y);
        float  th = 0.9f + h * 0.9f;                   // trunk height varies per instance

        float trunk = sdCappedCylinder(q - (float3)(0.0f, th * 0.5f, 0.0f),
                                       th * 0.5f, 0.06f + 0.03f * h);
        if (trunk < res.x) res = (float2)(trunk, 1.0f);

        float  cr = 0.55f + 0.35f * frac1(h * 7.3f);   // canopy radius (< half a cell)
        float3 cc = q - (float3)(0.0f, th + cr * 0.55f, 0.0f);
        float  canopy = sdEllipsoid(cc, (float3)(cr, cr * 1.3f, cr));
        if (canopy < res.x) res = (float2)(canopy, 2.0f);
    }
    return res;
}

static float3 calcNormal(float3 p) {
    float2 e = (float2)(0.0015f, 0.0f);
    float  d = map(p).x;
    float3 n = (float3)(map(p + e.xyy).x - d,
                        map(p + e.yxy).x - d,
                        map(p + e.yyx).x - d);
    return normalize(n);
}

static float softShadow(float3 ro, float3 rd) {
    float res = 1.0f, t = 0.05f;
    for (int i = 0; i < 24 && t < 12.0f; ++i) {
        float h = map(ro + rd * t).x;
        if (h < 0.001f) return 0.0f;
        res = fmin(res, 10.0f * h / t);
        t  += clamp(h, 0.02f, 0.35f);
    }
    return clamp(res, 0.0f, 1.0f);
}

static float3 skyColor(float3 rd) {
    float t = clamp(rd.y * 0.5f + 0.5f, 0.0f, 1.0f);
    return mix((float3)(0.70f, 0.78f, 0.86f), (float3)(0.22f, 0.40f, 0.72f), t);
}

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void render(__global const Params* P, __global uint* outRGBA) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= P->width || y >= P->height)
        return;  // guard the padding lanes from the rounded-up global size

    // Pixel -> normalised camera-plane coordinates (y up).
    float2 uv = (2.0f * (float2)((float)x, (float)y) - (float2)((float)P->width, (float)P->height))
                / (float)P->height;
    uv.y = -uv.y;

    // Free-fly camera: the host feeds position + yaw/pitch so the scene can be
    // navigated interactively. The forest grid is infinite, so you can fly forever.
    const float cy = cos(P->camYaw), sy = sin(P->camYaw);
    const float cp = cos(P->camPitch), sp = sin(P->camPitch);
    float3 ro  = (float3)(P->camX, P->camY, P->camZ);
    float3 fwd = (float3)(sy * cp, sp, cy * cp);
    float3 rgt = normalize(cross(fwd, (float3)(0.0f, 1.0f, 0.0f)));
    float3 up  = cross(rgt, fwd);
    float3 rd  = normalize(uv.x * rgt + uv.y * up + 1.35f * fwd);   // ~73° wide FOV

    const float3 sun = normalize((float3)(0.55f, 0.70f, 0.40f));

    // March the primary ray.
    float t = 0.0f, mat = -1.0f;
    for (int i = 0; i < MAX_STEPS; ++i) {
        float3 p = ro + rd * t;
        float2 h = map(p);
        if (h.x < 0.0015f * t) { mat = h.y; break; }
        if (t > 60.0f) break;
        t += h.x * 0.8f;                  // under-relax for the approximate ellipsoid SDF
    }

    float3 col;
    if (mat < 0.0f) {
        col = skyColor(rd);               // missed everything
    } else {
        float3 p = ro + rd * t;
        float3 n = calcNormal(p);

        // Per-material albedo. Canopy hue varies per instance (some autumn trees).
        float3 albedo;
        if (mat < 0.5f) {                 // ground
            float g = hash21(round(p.xz / CELL));
            albedo = mix((float3)(0.12f, 0.17f, 0.08f), (float3)(0.20f, 0.24f, 0.10f), g);
        } else if (mat < 1.5f) {          // trunk
            albedo = (float3)(0.23f, 0.15f, 0.09f);
        } else {                          // canopy
            float a = frac1(hash21(round(p.xz / CELL)) * 3.7f);
            albedo = mix((float3)(0.13f, 0.38f, 0.12f), (float3)(0.62f, 0.36f, 0.07f),
                         smoothstep(0.6f, 0.95f, a));
        }

        float  sh   = softShadow(p + n * 0.02f, sun);
        float  diff = max(dot(n, sun), 0.0f) * sh;
        float  sky  = 0.5f + 0.5f * n.y;                 // hemispheric ambient
        float  bnc  = max(dot(n, (float3)(-sun.x, 0.0f, -sun.z)), 0.0f) * 0.3f;  // ground bounce fill
        col = albedo * ((float3)(1.35f, 1.25f, 1.0f) * diff
                      + (float3)(0.38f, 0.44f, 0.55f) * sky
                      + (float3)(0.30f, 0.28f, 0.20f) * bnc);

        // Distance fog blends toward the sky colour.
        float f = 1.0f - exp(-P->fog * t);
        col = mix(col, skyColor(rd), f);
    }

    // Tone-map + gamma.
    col = col / (col + (float3)(1.0f));
    col = pow(col, (float3)(1.0f / 2.2f));

    const uint r = (uint)(clamp(col.x, 0.0f, 1.0f) * 255.0f);
    const uint g = (uint)(clamp(col.y, 0.0f, 1.0f) * 255.0f);
    const uint b = (uint)(clamp(col.z, 0.0f, 1.0f) * 255.0f);

    // Packed little-endian RGBA: R in the low byte, A=0xFF in the high byte.
    outRGBA[y * P->width + x] = r | (g << 8) | (b << 16) | (0xFFu << 24);
}
