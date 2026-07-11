#pragma once
#include "simulation.hpp"

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
void   reset_force_timing();

// Barnes-Hut extras: per-phase timing breakdown and scratch cleanup.
void bh_report_phase_timing();
void bh_release();

constexpr int kForceBlock = 256;
