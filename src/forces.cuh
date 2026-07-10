#pragma once
#include "simulation.hpp"

// Swappable force-computation interface. The integrator calls compute_forces to
// fill d_acc with the acceleration on each particle. Any implementation
// (naive all-pairs, Barnes-Hut) satisfies this signature so it can be dropped in
// without changing the integrator.
//
// d_acc: device array of length sys.n, one float4 acceleration per particle
//        (w component unused).
void compute_forces(const ParticleSystem& sys, float4* d_acc, const SimParams& p);

// --- naive O(n^2) all-pairs, shared-memory tiled ---
// TODO: __global__ kernel where each thread accumulates the force on one
// particle by streaming blocks of source particles through shared memory.
// Softened: a_i = G * sum_j m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
