#pragma once
#include "simulation.hpp"

// One softened interaction, shared by the naive kernel and both tree walks:
//   a_i += m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
// bj.w carries the source mass. G is applied once per particle by the caller.
__device__ __forceinline__ float3 body_body(float4 bi, float4 bj, float3 ai,
                                             float eps2) {
  float rx = bj.x - bi.x, ry = bj.y - bi.y, rz = bj.z - bi.z;
  float inv = rsqrtf(rx * rx + ry * ry + rz * rz + eps2);
  float s = bj.w * (inv * inv * inv);
  return make_float3(ai.x + rx * s, ai.y + ry * s, ai.z + rz * s);
}

// Swappable force-computation interface. Fills d_acc with the acceleration on
// each particle and dispatches on p.force, so either implementation drops in
// without changing the integrator.
//   d_acc: device array of length sys.n, one float4 per particle (w unused).
void compute_forces(const ParticleSystem& sys, float4* d_acc, const SimParams& p);

// the implementations behind the dispatcher
void forces_naive(const ParticleSystem& sys, float4* d_acc, const SimParams& p);
void forces_barnes_hut(const ParticleSystem& sys, float4* d_acc, const SimParams& p);

// Force timing, accumulated by the dispatcher around whichever module runs
// (CUDA events). For Barnes-Hut this covers the whole tree pipeline, which is
// the fair unit to compare against the naive kernel.
double force_kernel_ms_total();
long   force_kernel_calls();

// Barnes-Hut extras: per-phase timing breakdown and scratch cleanup.
void bh_report_phase_timing();
void bh_release();

constexpr int kForceBlock = 256;
