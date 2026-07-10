#pragma once
#include "simulation.hpp"

// Swappable force-computation interface. Fills d_acc with the acceleration on
// each particle. Any implementation (naive all-pairs, Barnes-Hut) satisfies this
// signature so it drops in without changing the integrator.
//   d_acc: device array of length sys.n, one float4 per particle (w unused).
void compute_forces(const ParticleSystem& sys, float4* d_acc, const SimParams& p);

// Force-kernel timing, isolated to this module (CUDA events). Lets the writeup
// report force-kernel ms/step independent of memcpy and integration overhead.
double force_kernel_ms_total();
long   force_kernel_calls();
void   reset_force_timing();

constexpr int kForceBlock = 256;
