#include "integrator.cuh"
#include "forces.cuh"
#include "cuda_check.cuh"

__global__ void kick_kernel(float4* vel, const float4* __restrict__ acc, int n,
                            float half_dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 v = vel[i], a = acc[i];
  v.x += half_dt * a.x;
  v.y += half_dt * a.y;
  v.z += half_dt * a.z;
  vel[i] = v;
}

__global__ void drift_kernel(float4* pos, const float4* __restrict__ vel, int n,
                             float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p = pos[i], v = vel[i];
  p.x += dt * v.x;
  p.y += dt * v.y;
  p.z += dt * v.z;   // p.w (mass) left untouched
  pos[i] = p;
}

void leapfrog_step(ParticleSystem& sys, float4* d_acc, const SimParams& p) {
  int block = kForceBlock;
  int blocks = (sys.n + block - 1) / block;
  float half_dt = 0.5f * p.dt;

  kick_kernel<<<blocks, block>>>(sys.d_vel, d_acc, sys.n, half_dt);
  drift_kernel<<<blocks, block>>>(sys.d_pos, sys.d_vel, sys.n, p.dt);
  compute_forces(sys, d_acc, p);                 // d_acc <- a(x_new)
  kick_kernel<<<blocks, block>>>(sys.d_vel, d_acc, sys.n, half_dt);
  CUDA_CHECK(cudaGetLastError());
}

// Per-particle kinetic and potential energy, tiled like the force kernel, then
// block-reduced and atomically summed into a double accumulator.
//   accum[0] = KE = sum_i 1/2 m_i |v_i|^2
//   accum[1] = PE = -1/2 G sum_{i!=j} m_i m_j / sqrt(r_ij^2 + eps^2)
__global__ void energy_kernel(const float4* __restrict__ pos,
                              const float4* __restrict__ vel, int n, float eps2,
                              float G, double* accum) {
  extern __shared__ float4 tile[];
  __shared__ double s_ke[kForceBlock];
  __shared__ double s_pe[kForceBlock];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float4 bi = (i < n) ? pos[i] : make_float4(0.f, 0.f, 0.f, 0.f);
  float4 vi = (i < n) ? vel[i] : make_float4(0.f, 0.f, 0.f, 0.f);

  float phi = 0.f;   // sum_j m_j / sqrt(r_ij^2 + eps^2), includes the j==i term
  for (int t = 0; t < gridDim.x; ++t) {
    int src = t * blockDim.x + threadIdx.x;
    tile[threadIdx.x] = (src < n) ? pos[src] : make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
    for (int j = 0; j < blockDim.x; ++j) {
      float dx = tile[j].x - bi.x;
      float dy = tile[j].y - bi.y;
      float dz = tile[j].z - bi.z;
      phi += tile[j].w * rsqrtf(dx * dx + dy * dy + dz * dz + eps2);
    }
    __syncthreads();
  }

  double ke = 0.0, pe = 0.0;
  if (i < n) {
    float m = bi.w;
    ke = 0.5 * (double)m * (vi.x * vi.x + vi.y * vi.y + vi.z * vi.z);
    float self = m * rsqrtf(eps2);                 // remove the j==i self term
    pe = -0.5 * (double)G * (double)m * ((double)phi - (double)self);
  }

  int tid = threadIdx.x;
  s_ke[tid] = ke;
  s_pe[tid] = pe;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_ke[tid] += s_ke[tid + s];
      s_pe[tid] += s_pe[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(&accum[0], s_ke[0]);
    atomicAdd(&accum[1], s_pe[0]);
  }
}

Energy total_energy(const ParticleSystem& sys, const SimParams& p) {
  double* d_accum = nullptr;
  CUDA_CHECK(cudaMalloc(&d_accum, 2 * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_accum, 0, 2 * sizeof(double)));

  int block = kForceBlock;
  int blocks = (sys.n + block - 1) / block;
  size_t shmem = block * sizeof(float4);
  float eps2 = p.epsilon * p.epsilon;
  energy_kernel<<<blocks, block, shmem>>>(sys.d_pos, sys.d_vel, sys.n, eps2, p.G,
                                          d_accum);
  CUDA_CHECK(cudaGetLastError());

  double h[2] = {0.0, 0.0};
  CUDA_CHECK(cudaMemcpy(h, d_accum, 2 * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_accum));
  return Energy{h[0] + h[1], h[0], h[1]};
}
