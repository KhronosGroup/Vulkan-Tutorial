// Chapter 2 – Compute Architecture: Mandelbrot Explorer
//
// Compiled with:   slangc 02_compute_architecture.slang -o shaders/slang.spv \
//                      -profile glsl_460 -target spirv -entry compMain
//
// Uses double-float (two-word float) arithmetic for the full Mandelbrot
// iteration, enabling zoom depths around 1e-13 without float64 hardware.
// The center is split into (hi, lo) float pairs on the CPU; each shader
// invocation reconstructs c with full double-float precision before iterating.

// Storage image – rgba8 matches the R8G8B8A8Unorm storage image on the C++ side
[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> outputImage;

// Reference orbit (perturbation theory).  Element n holds the reference point's
// orbit value X_n as TWO double-floats packed into a float4:
//   .xy = X_n.real (hi, lo),   .zw = X_n.imag (hi, lo)
// X_n is computed once per frame on the CPU at the view centre, in extended
// precision, then iterated per-pixel only as a small delta.  See compMain.
[[vk::binding(1, 0)]] StructuredBuffer<float4> refOrbit;

// Push constants – must be byte-identical to the C++ MandelbrotPush struct.
//
// The view CENTRE is no longer pushed: it IS the reference orbit, so the shader
// never needs its absolute coordinates.  Only the per-pixel offset scale (zoom)
// and the iteration window are required.
struct MandelbrotPush
{
    float zoomHi;     // zoom hi-word: float(zoom)
    float zoomLo;     // zoom lo-word: float(zoom - hi) — sub-ULP residual
    uint  width;
    uint  height;
    uint  minIter;    // iteration window floor — coarser layers are above this
    uint  maxIter;    // auto-scaled total iterations for this zoom depth
    uint  refLen;     // valid length of refOrbit (where the reference escaped)
    float colorPhase; // animated offset into palette cycle [0,1)
};
[[vk::push_constant]] MandelbrotPush pc;

// ---------------------------------------------------------------------------
// Double-float arithmetic
//
// A "double-float" (df) is a pair (hi, lo) satisfying |lo| <= 0.5*ulp(hi).
// hi + lo equals the intended value to ~48 bits of precision.
//
// All operations follow Shewchuk's exact arithmetic algorithms and rely on
// IEEE 754 round-to-nearest semantics, which Vulkan mandates for compute.
// ---------------------------------------------------------------------------

// Quick two-sum: exact split of (a + b) when |a| >= |b|.
float2 df_quick_two_sum(float a, float b)
{
    float s = a + b;
    float e = b - (s - a);
    return float2(s, e);
}

// General two-sum: exact split of (a + b) for arbitrary a, b.
float2 df_two_sum(float a, float b)
{
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return float2(s, e);
}

// Veltkamp split: split a 24-bit float into two exact 12-bit halves.
// Used by df_two_product.
float2 df_split(float a)
{
    float t  = 4097.0f * a;   // (2^12 + 1) * a
    float hi = t - (t - a);
    float lo = a - hi;
    return float2(hi, lo);
}

// Two-product: compute (a*b, err) such that a*b + err = exact product.
float2 df_two_product(float a, float b)
{
    float  p  = a * b;
    float2 as = df_split(a);
    float2 bs = df_split(b);
    float  e  = ((as.x * bs.x - p) + as.x * bs.y + as.y * bs.x) + as.y * bs.y;
    return float2(p, e);
}

// Add a double-float and a single float.
float2 df_add_f(float2 a, float b)
{
    float2 t = df_two_sum(a.x, b);
    t.y += a.y;
    return df_quick_two_sum(t.x, t.y);
}

// Add two double-floats.
float2 df_add(float2 a, float2 b)
{
    float2 s = df_two_sum(a.x, b.x);
    s.y += a.y + b.y;
    return df_quick_two_sum(s.x, s.y);
}

// Subtract two double-floats.
float2 df_sub(float2 a, float2 b)
{
    return df_add(a, float2(-b.x, -b.y));
}

// Square a double-float (cheaper than general multiply).
float2 df_sqr(float2 a)
{
    float2 t = df_two_product(a.x, a.x);
    t.y += 2.0f * a.x * a.y;
    return df_quick_two_sum(t.x, t.y);
}

// Multiply two double-floats.
float2 df_mul(float2 a, float2 b)
{
    float2 t = df_two_product(a.x, b.x);
    t.y += a.x * b.y + a.y * b.x;
    return df_quick_two_sum(t.x, t.y);
}

// Scale a double-float by a power-of-2 scalar (exact, no rounding).
float2 df_scale2(float2 a, float s)
{
    return float2(a.x * s, a.y * s);
}

// ---------------------------------------------------------------------------
// IQ cosine palette
// Inigo Quilez's palette function: a + b*cos(2π*(c*t + d))
// ---------------------------------------------------------------------------
float3 palette(float t)
{
    float3 a = float3(0.5f, 0.5f, 0.5f);
    float3 b = float3(0.5f, 0.5f, 0.5f);
    float3 c = float3(1.0f, 1.0f, 1.0f);
    float3 d = float3(0.0f, 0.33f, 0.67f);
    return a + b * cos(6.28318530718f * (c * t + d));
}

[numthreads(16, 16, 1)]
[shader("compute")]
void compMain(uint3 dispatchID : SV_DispatchThreadID)
{
    uint px = dispatchID.x;
    uint py = dispatchID.y;

    if (px >= pc.width || py >= pc.height)
        return;

    // -----------------------------------------------------------------------
    // PERTURBATION THEORY — the key to "infinite" zoom.
    //
    // Direct iteration computes c = centre + pixelOffset.  Once the zoom is so
    // deep that pixelOffset is smaller than one ULP of `centre`, the addition
    // throws the offset away: every pixel in a region collapses to the same c
    // and the image degrades into flat rectangles.
    //
    // Instead we iterate one REFERENCE orbit X_m (the view centre, computed in
    // extended precision on the CPU) and, per pixel, only the SMALL deviation
    //     δ = z − X_m,     δc = pixelOffset · zoom
    // via  δ ← 2·X_m·δ + δ² + δc.   δ never gets added to a large number, so it
    // keeps full precision far past the centre's ULP — no more pixel rectangles.
    //
    // REBASING (Zhuoran's method) makes this robust: whenever the running orbit
    // value z = X_m + δ becomes smaller than δ itself — or the stored reference
    // runs out (it only spans until the centre escaped) — we "rebase", adopting
    // z as the new δ and restarting the reference index m at 0.  This is what
    // prevents the whole screen collapsing to one colour when the view centre
    // lies outside the set (a short reference): the orbit simply wraps around
    // the reference instead of being capped.  δ is carried in double-float.
    // -----------------------------------------------------------------------
    float pxOff = (float)px - (float)pc.width  * 0.5f;  // exact: integer offset
    float pyOff = (float)py - (float)pc.height * 0.5f;

    // δc = pixel offset · zoom, in double-float (pixel offset is exact).
    float2 dcr = df_mul(float2(pxOff, 0.0f), float2(pc.zoomHi, pc.zoomLo));
    float2 dci = df_mul(float2(pyOff, 0.0f), float2(pc.zoomHi, pc.zoomLo));

    float2 dr = float2(0.0f, 0.0f);   // δ real (double-float)
    float2 di = float2(0.0f, 0.0f);   // δ imag
    uint   m    = 0u;                 // reference index
    uint   iter = 0u;                 // true iteration count
    float  zr_h = 0.0f, zi_h = 0.0f;  // hi parts of escaped z (for smooth colour)

    while (iter < pc.maxIter)
    {
        // δ ← 2·X_m·δ + δ² + δc   (complex, double-float)
        float4 X  = refOrbit[m];
        float2 Xr = X.xy;
        float2 Xi = X.zw;
        float2 XrDr = df_mul(Xr, dr);
        float2 XiDi = df_mul(Xi, di);
        float2 XrDi = df_mul(Xr, di);
        float2 XiDr = df_mul(Xi, dr);
        float2 twoXd_r = df_scale2(df_sub(XrDr, XiDi), 2.0f);
        float2 twoXd_i = df_scale2(df_add(XrDi, XiDr), 2.0f);
        float2 dsq_r = df_sub(df_sqr(dr), df_sqr(di));
        float2 dsq_i = df_scale2(df_mul(dr, di), 2.0f);
        dr = df_add(df_add(twoXd_r, dsq_r), dcr);
        di = df_add(df_add(twoXd_i, dsq_i), dci);
        ++m;
        ++iter;

        // Full orbit value z = X_m + δ (hi parts are enough for the tests).
        float4 Xm = refOrbit[m];
        float2 zr = df_add(Xm.xy, dr);
        float2 zi = df_add(Xm.zw, di);
        zr_h = zr.x;
        zi_h = zi.x;
        float z2 = zr_h * zr_h + zi_h * zi_h;
        if (z2 > 4.0f)
            break;   // escaped at `iter`

        // Rebase when z is closer to the origin than δ, or the reference ran out.
        float d2 = dr.x * dr.x + di.x * di.x;
        if (m >= pc.refLen - 1u || z2 < d2)
        {
            dr = zr;
            di = zi;
            m  = 0u;
        }
    }

    float4 color;
    if (iter == pc.maxIter)
    {
        // Never escaped within the window: interior, draw black.
        color = float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    else
    {
        // Smooth (continuous) iteration count from the escaped magnitude |z|.
        float mag2   = max(zr_h * zr_h + zi_h * zi_h, 1.0f);
        float smooth = (float)iter + 1.0f - log2(log2(mag2) * 0.5f);

        if (iter < pc.minIter)
        {
            // Escaped before the window floor: this pixel belongs to a coarser
            // zoom layer that is "above" the current view.  Dim it so it recedes
            // into the background, leaving the window band as the visual focus.
            float t = frac(smooth * 0.015f + pc.colorPhase);
            color = float4(palette(t) * 0.18f, 1.0f);
        }
        else
        {
            // Escaped within [minIter, maxIter]: this is the detail layer for
            // the current zoom.  Map the window to the full colour cycle so the
            // palette is always used at full contrast regardless of depth.
            float windowSize = (float)(pc.maxIter - pc.minIter);
            float t = frac((smooth - (float)pc.minIter) / windowSize + pc.colorPhase);
            color = float4(palette(t), 1.0f);
        }
    }

    outputImage[uint2(px, py)] = color;
}
