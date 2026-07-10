#pragma once
#include "simulation.hpp"

// Leapfrog kick-drift-kick, symplectic. One call advances the system by dt:
//   kick:  v += a(x)     * dt/2
//   drift: x += v        * dt
//   kick:  v += a(x_new) * dt/2
// Acceleration comes from the swappable force module (see forces.cuh), so the
// stepper is independent of how forces are computed.
void leapfrog_step(ParticleSystem& sys, float4* d_acc, const SimParams& p);

// Total system energy (kinetic + softened potential), the correctness
// diagnostic. TODO: reduction on device; logged against sim time.
double total_energy(const ParticleSystem& sys, const SimParams& p);
