// Chapter 12 – Embedded Compute: Parallel FABRIK IK Solver
//
// This shader solves 3-bone Inverse Kinematics (IK) for N robot arm
// configurations in parallel, one GPU thread per arm.  It demonstrates:
//
//   • Purely headless compute dispatch — no images, no swapchain, no GLFW.
//     The same kernel runs unchanged on Raspberry Pi (v3dv) and NVIDIA Jetson.
//
//   • FABRIK (Forward And Backward Reaching IK): an iterative, link-length-
//     preserving algorithm widely used in robotics and animation rigs.
//     Each iteration is O(n) in the number of bones (3 here), so 20 iterations
//     is a very tight cap that still converges to <1 mm for most poses.
//
//   • Embedded-safe workgroup size: 64 threads, well within the 256-invocation
//     limit on Raspberry Pi (VideoCore VI/VII) and within Jetson's 1024 limit.
//
// Bindings:
//   0 – StructuredBuffer<IKJob>         (read-only input, one per arm)
//   1 – RWStructuredBuffer<IKSolution>  (write-only output, one per arm)
//
// Push constant:
//   count – total number of IK jobs (may be < dispatch threads, guarded below)
//
// Entry point: ikMain [numthreads(64,1,1)]

// ---------------------------------------------------------------------------
// Structs  (must be byte-identical to the host-side CPU structs)
// ---------------------------------------------------------------------------

struct IKJob
{
    float4 target;   // xyz = world-space target for the end effector, w unused
    float4 base;     // xyz = fixed root joint position, w unused
    float4 reach;    // xyz = bone lengths [root→j1, j1→j2, j2→end], w unused
    uint   maxIter;
    float  tolerance;
    uint   _pad0;
    uint   _pad1;
};
// sizeof(IKJob) == 64 bytes (3×float4 + 4×uint/float)

struct IKSolution
{
    float4 joint0;        // xyz = root (copy of base after solver)
    float4 joint1;        // xyz = joint 1 solved position
    float4 joint2;        // xyz = joint 2 solved position
    float4 endEffector;   // xyz = end-effector solved position
    uint   iterations;    // FABRIK iterations used (0 if unreachable branch taken)
    float  residual;      // |endEffector - target| in metres after last iteration
    uint   reachable;     // 1 if residual < tolerance, 0 otherwise
    uint   _pad;
};
// sizeof(IKSolution) == 80 bytes (4×float4 + 4×uint/float)

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------
[[vk::binding(0, 0)]] StructuredBuffer<IKJob>          g_jobs;
[[vk::binding(1, 0)]] RWStructuredBuffer<IKSolution>   g_solutions;

// ---------------------------------------------------------------------------
// Push constants
// ---------------------------------------------------------------------------
struct PushConst
{
    uint count;   // total IK jobs; threads with id.x >= count are no-ops
};
[[vk::push_constant]] PushConst pc;

// ---------------------------------------------------------------------------
// Helper: safe normalise — returns zero for near-degenerate vectors
// ---------------------------------------------------------------------------
float3 safeNorm(float3 v)
{
    float len = length(v);
    return (len > 1e-9f) ? (v / len) : float3(0.0f, 0.0f, 0.0f);
}

// ---------------------------------------------------------------------------
// Entry point: ikMain
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void ikMain(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= pc.count) return;

    IKJob job = g_jobs[i];

    // Bone lengths from the reach.xyz channel
    float r0 = job.reach.x;   // root  → joint1
    float r1 = job.reach.y;   // joint1 → joint2
    float r2 = job.reach.z;   // joint2 → end effector

    float totalReach = r0 + r1 + r2;
    float reachArr[3] = { r0, r1, r2 };

    // Initialise joint chain in rest pose: stretched along +X from the base.
    float3 p[4];
    p[0] = job.base.xyz;
    p[1] = p[0] + float3(r0, 0.0f, 0.0f);
    p[2] = p[1] + float3(r1, 0.0f, 0.0f);
    p[3] = p[2] + float3(r2, 0.0f, 0.0f);

    float3 target = job.target.xyz;
    float  distToTarget = length(target - p[0]);

    IKSolution sol;
    sol.iterations = 0u;
    sol.reachable  = 0u;

    if (distToTarget >= totalReach * 0.9999f)
    {
        // Target is out of reach: stretch the arm as a straight chain toward target.
        float3 dir = safeNorm(target - p[0]);
        p[1] = p[0] + dir * r0;
        p[2] = p[1] + dir * r1;
        p[3] = p[2] + dir * r2;
        sol.residual = length(p[3] - target);
    }
    else
    {
        // FABRIK iterative solver.
        float residual = length(p[3] - target);
        uint  iters    = 0u;

        while (residual > job.tolerance && iters < job.maxIter)
        {
            // --- Forward pass: pull end-effector to target, propagate toward root ---
            p[3] = target;
            for (int j = 2; j >= 0; --j)
            {
                float3 delta = p[j] - p[j + 1];
                float  d     = length(delta);
                float  lam   = (d > 1e-9f) ? (reachArr[j] / d) : 0.0f;
                p[j] = lerp(p[j + 1], p[j], lam);
            }

            // --- Backward pass: re-pin root, propagate toward end-effector ---
            p[0] = job.base.xyz;
            for (int j = 0; j < 3; ++j)
            {
                float3 delta = p[j + 1] - p[j];
                float  d     = length(delta);
                float  lam   = (d > 1e-9f) ? (reachArr[j] / d) : 0.0f;
                p[j + 1] = lerp(p[j], p[j + 1], lam);
            }

            residual = length(p[3] - target);
            iters++;
        }

        sol.iterations = iters;
        sol.residual   = residual;
        sol.reachable  = (residual < job.tolerance) ? 1u : 0u;
    }

    // Write results
    sol.joint0       = float4(p[0], 0.0f);
    sol.joint1       = float4(p[1], 0.0f);
    sol.joint2       = float4(p[2], 0.0f);
    sol.endEffector  = float4(p[3], 0.0f);

    g_solutions[i] = sol;
}
