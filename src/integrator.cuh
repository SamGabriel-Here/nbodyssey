#pragma once
#include "simulation.hpp"

// Total system energy, the correctness diagnostic. Accumulated in double on the
// device for an accurate readout even though the state is single precision.
struct Energy {
  double total;
  double kinetic;
  double potential;
};

// Leapfrog kick-drift-kick, symplectic. Advances the system by one dt:
//   kick:  v += a(x)     * dt/2   (using d_acc from the previous step)
//   drift: x += v        * dt
//   force: d_acc = a(x_new)       (swappable force module)
//   kick:  v += a(x_new) * dt/2
// On entry d_acc must hold a(x) for the current positions; on exit it holds
// a(x_new), ready to be reused as the next step's first kick.
void leapfrog_step(ParticleSystem& sys, float4* d_acc, const SimParams& p);

Energy total_energy(const ParticleSystem& sys, const SimParams& p);
