// Advanced Compute Tutorial – Chapter 3: Memory Models
// Interactive 2-D incompressible fluid — Stam's "Stable Fluids" (Navier–Stokes).
//
// This is an EULERIAN (grid-based) fluid solver.  Each frame the velocity field
// is advected, made divergence-free by a pressure projection, and used to carry
// a coloured "dye" that you can paint by dragging the mouse.  The result looks
// and behaves like ink swirling in water.
//
// Why this lives in the "Memory Models" chapter:
//   The pressure projection solves a Poisson equation with a JACOBI iteration.
//   Every iteration reads its neighbours' values from the PREVIOUS iteration, so
//   the C++ driver places a memory barrier between each dispatch — a textbook
//   demonstration of inter-dispatch visibility.  Advection, divergence, and the
//   gradient subtraction are likewise separate passes separated by barriers.
//
// Entry points (one SPIR-V module):
//   splatPass     – inject velocity + dye under the mouse
//   advectVelPass – semi-Lagrangian advection of velocity      (velA → velB)
//   divergencePass– divergence of velB, clears pressure
//   jacobiPass    – one Jacobi pressure iteration (ping-ponged)
//   gradientPass  – subtract ∇pressure, enforce walls          (velB → velA)
//   advectDyePass – semi-Lagrangian advection of dye           (dyeA → dyeB)
//   renderPass    – colour the dye field into the output image

// ===========================================================================
// Bindings (set 0)
// ===========================================================================
[[vk::binding(0, 0)]] RWStructuredBuffer<float2> velA;    // velocity (current)
[[vk::binding(1, 0)]] RWStructuredBuffer<float2> velB;    // velocity (scratch)
[[vk::binding(2, 0)]] RWStructuredBuffer<float>  pres0;   // pressure ping
[[vk::binding(3, 0)]] RWStructuredBuffer<float>  pres1;   // pressure pong
[[vk::binding(4, 0)]] RWStructuredBuffer<float>  diverg;  // velocity divergence
[[vk::binding(5, 0)]] RWStructuredBuffer<float>  dyeA;    // dye (current)
[[vk::binding(6, 0)]] RWStructuredBuffer<float>  dyeB;    // dye (scratch)
[[vk::binding(7, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> outImage;

// ===========================================================================
// Push constants — byte-identical to FluidPush in the C++ driver
// ===========================================================================
struct FluidPush
{
    uint  nx;          // grid width  (cells)
    uint  ny;          // grid height (cells)
    float dt;          // time step
    float dissipation; // dye fade per step (≈0.997)
    float velFade;     // velocity damping per step (≈0.999)
    float mouseX;      // current mouse, in grid coordinates
    float mouseY;
    float mousePx;     // previous mouse, in grid coordinates
    float mousePy;
    float splatRadius; // injection radius (cells)
    float forceScale;  // velocity injected per unit mouse motion
    float dyeAmount;   // dye injected at the mouse
    uint  mouseDown;   // 1 while dragging
    uint  jacobiSrc;   // 0 → read pres0/write pres1, 1 → read pres1/write pres0
    uint  imgWidth;
    uint  imgHeight;
};
[[vk::push_constant]] FluidPush pc;

// ===========================================================================
// Grid helpers
// ===========================================================================
int IX(int x, int y)
{
    x = clamp(x, 0, (int)pc.nx - 1);
    y = clamp(y, 0, (int)pc.ny - 1);
    return y * (int)pc.nx + x;
}

bool onBoundary(int x, int y)
{
    return x == 0 || y == 0 || x == (int)pc.nx - 1 || y == (int)pc.ny - 1;
}

// Bilinear sample of the velocity field at grid position p (cell units).
float2 sampleVel(float2 p)
{
    p = clamp(p, float2(0.0f, 0.0f), float2((float)pc.nx - 1.0f, (float)pc.ny - 1.0f));
    int   x0 = (int)floor(p.x), y0 = (int)floor(p.y);
    float fx = p.x - (float)x0, fy = p.y - (float)y0;
    float2 v00 = velA[IX(x0,     y0)];
    float2 v10 = velA[IX(x0 + 1, y0)];
    float2 v01 = velA[IX(x0,     y0 + 1)];
    float2 v11 = velA[IX(x0 + 1, y0 + 1)];
    return lerp(lerp(v00, v10, fx), lerp(v01, v11, fx), fy);
}

// Bilinear sample of the dye field.
float sampleDye(float2 p)
{
    p = clamp(p, float2(0.0f, 0.0f), float2((float)pc.nx - 1.0f, (float)pc.ny - 1.0f));
    int   x0 = (int)floor(p.x), y0 = (int)floor(p.y);
    float fx = p.x - (float)x0, fy = p.y - (float)y0;
    float d00 = dyeA[IX(x0,     y0)];
    float d10 = dyeA[IX(x0 + 1, y0)];
    float d01 = dyeA[IX(x0,     y0 + 1)];
    float d11 = dyeA[IX(x0 + 1, y0 + 1)];
    return lerp(lerp(d00, d10, fx), lerp(d01, d11, fx), fy);
}

// Distance from cell centre to the mouse drag segment (prev → current),
// used so a fast drag paints a continuous streak rather than dotted blobs.
float distToMouseSegment(float2 p)
{
    float2 a = float2(pc.mousePx, pc.mousePy);
    float2 b = float2(pc.mouseX,  pc.mouseY);
    float2 ab = b - a;
    float  len2 = max(dot(ab, ab), 1e-6f);
    float  t = clamp(dot(p - a, ab) / len2, 0.0f, 1.0f);
    return length(p - (a + t * ab));
}

// ===========================================================================
// PASS 1 – splat: inject velocity + dye under the mouse
// ===========================================================================
[numthreads(16, 16, 1)]
[shader("compute")]
void splatPass(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= pc.nx || id.y >= pc.ny) return;
    int idx = (int)id.y * (int)pc.nx + (int)id.x;

    if (pc.mouseDown != 0u)
    {
        float2 p = float2((float)id.x + 0.5f, (float)id.y + 0.5f);
        float  d = distToMouseSegment(p);
        float  w = exp(-(d * d) / (pc.splatRadius * pc.splatRadius));

        float2 drag = float2(pc.mouseX - pc.mousePx, pc.mouseY - pc.mousePy);
        velA[idx] += drag * (pc.forceScale * w);
        dyeA[idx]  = min(dyeA[idx] + pc.dyeAmount * w, 1.5f);
    }
}

// ===========================================================================
// PASS 2 – advect velocity (semi-Lagrangian back-trace).  velA → velB
// ===========================================================================
[numthreads(16, 16, 1)]
[shader("compute")]
void advectVelPass(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= pc.nx || id.y >= pc.ny) return;
    int idx = (int)id.y * (int)pc.nx + (int)id.x;

    float2 p   = float2((float)id.x + 0.5f, (float)id.y + 0.5f);
    float2 vel = velA[idx];
    float2 src = p - vel * pc.dt;          // trace backwards
    velB[idx]  = sampleVel(src) * pc.velFade;
}

// ===========================================================================
// PASS 3 – divergence of velB; clear the pressure field to zero.
// ===========================================================================
[numthreads(16, 16, 1)]
[shader("compute")]
void divergencePass(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= pc.nx || id.y >= pc.ny) return;
    int x = (int)id.x, y = (int)id.y;
    int idx = y * (int)pc.nx + x;

    float l = velB[IX(x - 1, y)].x;
    float r = velB[IX(x + 1, y)].x;
    float b = velB[IX(x, y - 1)].y;
    float t = velB[IX(x, y + 1)].y;

    diverg[idx] = 0.5f * ((r - l) + (t - b));
    pres0[idx]  = 0.0f;
    pres1[idx]  = 0.0f;
}

// ===========================================================================
// PASS 4 – one Jacobi iteration of the pressure Poisson solve.
//   ∇²p = divergence  →  p_new = (Σ neighbours − divergence) / 4
// Ping-ponged between pres0 and pres1 by pc.jacobiSrc; the C++ driver issues a
// memory barrier between every invocation so each reads the previous result.
// ===========================================================================
[numthreads(16, 16, 1)]
[shader("compute")]
void jacobiPass(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= pc.nx || id.y >= pc.ny) return;
    int x = (int)id.x, y = (int)id.y;
    int idx = y * (int)pc.nx + x;

    float l, r, b, t, c;
    if (pc.jacobiSrc == 0u)
    {
        l = pres0[IX(x - 1, y)]; r = pres0[IX(x + 1, y)];
        b = pres0[IX(x, y - 1)]; t = pres0[IX(x, y + 1)];
    }
    else
    {
        l = pres1[IX(x - 1, y)]; r = pres1[IX(x + 1, y)];
        b = pres1[IX(x, y - 1)]; t = pres1[IX(x, y + 1)];
    }
    c = (l + r + b + t - diverg[idx]) * 0.25f;

    if (pc.jacobiSrc == 0u) pres1[idx] = c;
    else                    pres0[idx] = c;
}

// ===========================================================================
// PASS 5 – subtract the pressure gradient to project velocity to be
// divergence-free, and enforce no-slip walls.  velB → velA
// pc.jacobiSrc here names the buffer holding the FINAL pressure.
// ===========================================================================
[numthreads(16, 16, 1)]
[shader("compute")]
void gradientPass(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= pc.nx || id.y >= pc.ny) return;
    int x = (int)id.x, y = (int)id.y;
    int idx = y * (int)pc.nx + x;

    float pl, pr, pb, pt;
    if (pc.jacobiSrc == 0u)
    {
        pl = pres0[IX(x - 1, y)]; pr = pres0[IX(x + 1, y)];
        pb = pres0[IX(x, y - 1)]; pt = pres0[IX(x, y + 1)];
    }
    else
    {
        pl = pres1[IX(x - 1, y)]; pr = pres1[IX(x + 1, y)];
        pb = pres1[IX(x, y - 1)]; pt = pres1[IX(x, y + 1)];
    }

    float2 v = velB[idx] - 0.5f * float2(pr - pl, pt - pb);
    if (onBoundary(x, y)) v = float2(0.0f, 0.0f);   // no-slip walls keep fluid in
    velA[idx] = v;
}

// ===========================================================================
// PASS 6 – advect dye by the (now divergence-free) velocity.  dyeA → dyeB
// ===========================================================================
[numthreads(16, 16, 1)]
[shader("compute")]
void advectDyePass(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= pc.nx || id.y >= pc.ny) return;
    int idx = (int)id.y * (int)pc.nx + (int)id.x;

    float2 p   = float2((float)id.x + 0.5f, (float)id.y + 0.5f);
    float2 vel = velA[idx];
    float2 src = p - vel * pc.dt;
    dyeB[idx]  = sampleDye(src) * pc.dissipation;
}

// ===========================================================================
// PASS 7 – render the dye field to the screen as illuminated water.
// ===========================================================================
[numthreads(16, 16, 1)]
[shader("compute")]
void renderPass(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= pc.imgWidth || id.y >= pc.imgHeight) return;

    // Pixel → grid coordinate (bilinear).  dyeB holds the freshest dye.
    float gx = ((float)id.x + 0.5f) / (float)pc.imgWidth  * (float)pc.nx - 0.5f;
    float gy = ((float)id.y + 0.5f) / (float)pc.imgHeight * (float)pc.ny - 0.5f;
    float2 g = clamp(float2(gx, gy), float2(0.0f, 0.0f),
                     float2((float)pc.nx - 1.0f, (float)pc.ny - 1.0f));

    int   x0 = (int)floor(g.x), y0 = (int)floor(g.y);
    float fx = g.x - (float)x0, fy = g.y - (float)y0;
    float d00 = dyeB[IX(x0, y0)],     d10 = dyeB[IX(x0 + 1, y0)];
    float d01 = dyeB[IX(x0, y0 + 1)], d11 = dyeB[IX(x0 + 1, y0 + 1)];
    float dye = lerp(lerp(d00, d10, fx), lerp(d01, d11, fx), fy);

    float2 v00 = velA[IX(x0, y0)],     v10 = velA[IX(x0 + 1, y0)];
    float2 v01 = velA[IX(x0, y0 + 1)], v11 = velA[IX(x0 + 1, y0 + 1)];
    float2 vel = lerp(lerp(v00, v10, fx), lerp(v01, v11, fx), fy);
    float  speed = length(vel);

    // Water look: dark teal base → bright cyan/white where dye is dense or fast.
    float  amt    = clamp(dye, 0.0f, 1.0f);
    float3 deep   = float3(0.01f, 0.05f, 0.10f);   // dark water
    float3 body   = float3(0.05f, 0.45f, 0.75f);   // lit water blue
    float3 bright = float3(0.75f, 0.95f, 1.00f);   // foam / highlight
    float3 col    = lerp(deep, body, smoothstep(0.0f, 0.5f, amt));
    col           = lerp(col, bright, smoothstep(0.5f, 1.0f, amt));

    // Speed adds a subtle bright sheen so motion reads as flowing water.
    col += float3(0.10f, 0.20f, 0.30f) * clamp(speed * 0.05f, 0.0f, 1.0f) * amt;

    outImage[uint2(id.x, id.y)] = float4(col, 1.0f);
}
