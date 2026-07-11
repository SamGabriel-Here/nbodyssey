#include "forces.cuh"
#include "cuda_check.cuh"

// Naive O(n^2) all-pairs gravity with shared-memory tiling.
//
// Each thread owns one target particle i and accumulates the acceleration on it
// from every source j. Sources are streamed a block at a time through shared
// memory: the block cooperatively loads kForceBlock source particles, every
// thread reuses them from fast shared memory, then the block advances to the
// next tile. This turns O(n) global-memory loads per target into O(n / block).
//
// Softened, matching the reference integrator:
//   a_i = G * sum_j m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
// The j == i term contributes zero (its separation vector is zero), and padded
// lanes carry mass 0, so neither needs a branch in the inner loop.

__device__ __forceinline__ float3 body_body(float4 bi, float4 bj, float3 ai,
                                             float eps2) {
  float3 r;
  r.x = bj.x - bi.x;
  r.y = bj.y - bi.y;
  r.z = bj.z - bi.z;
  float dist2 = r.x * r.x + r.y * r.y + r.z * r.z + eps2;
  float inv = rsqrtf(dist2);
  float inv3 = inv * inv * inv;
  float s = bj.w * inv3;   // bj.w carries source mass
  ai.x += r.x * s;
  ai.y += r.y * s;
  ai.z += r.z * s;
  return ai;
}

__global__ void force_kernel(const float4* __restrict__ pos, float4* __restrict__ acc,
                             int n, float eps2, float G) {
  extern __shared__ float4 tile[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float4 bi = (i < n) ? pos[i] : make_float4(0.f, 0.f, 0.f, 0.f);
  float3 ai = {0.f, 0.f, 0.f};

  for (int t = 0; t < gridDim.x; ++t) {
    int src = t * blockDim.x + threadIdx.x;
    tile[threadIdx.x] = (src < n) ? pos[src] : make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
#pragma unroll 8
    for (int j = 0; j < blockDim.x; ++j) {
      ai = body_body(bi, tile[j], ai, eps2);
    }
    __syncthreads();
  }

  if (i < n) acc[i] = make_float4(G * ai.x, G * ai.y, G * ai.z, 0.f);
}

void forces_naive(const ParticleSystem& sys, float4* d_acc, const SimParams& p) {
  int blocks = (sys.n + kForceBlock - 1) / kForceBlock;
  size_t shmem = kForceBlock * sizeof(float4);
  float eps2 = p.epsilon * p.epsilon;
  force_kernel<<<blocks, kForceBlock, shmem>>>(sys.d_pos, d_acc, sys.n, eps2, p.G);
  CUDA_CHECK(cudaGetLastError());
}

namespace {
double g_ms = 0.0;
long g_calls = 0;
cudaEvent_t g_start = nullptr, g_stop = nullptr;
}  // namespace

void compute_forces(const ParticleSystem& sys, float4* d_acc, const SimParams& p) {
  if (!g_start) {
    CUDA_CHECK(cudaEventCreate(&g_start));
    CUDA_CHECK(cudaEventCreate(&g_stop));
  }
  CUDA_CHECK(cudaEventRecord(g_start));
  if (p.force == ForceMethod::kBarnesHut) {
    forces_barnes_hut(sys, d_acc, p);
  } else {
    forces_naive(sys, d_acc, p);
  }
  CUDA_CHECK(cudaEventRecord(g_stop));
  CUDA_CHECK(cudaEventSynchronize(g_stop));

  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, g_start, g_stop));
  g_ms += ms;
  ++g_calls;
}

double force_kernel_ms_total() { return g_ms; }
long force_kernel_calls() { return g_calls; }
void reset_force_timing() { g_ms = 0.0; g_calls = 0; }
