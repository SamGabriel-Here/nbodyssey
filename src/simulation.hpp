#pragma once
#include <vector_types.h>

// Simulation parameters shared across host and device code.
struct SimParams {
  float dt;         // timestep
  float epsilon;    // softening length
  float G = 1.0f;   // gravitational constant in simulation units
};

// Particle state in structure-of-arrays form, resident on the device.
// pos packs position + mass as (x, y, z, m); vel is (vx, vy, vz, _).
struct ParticleSystem {
  float4* d_pos = nullptr;   // device: position + mass
  float4* d_vel = nullptr;   // device: velocity
  int n = 0;                 // particle count

  void allocate(int count);
  void release();
  void upload(const float4* h_pos, const float4* h_vel);   // host -> device, once
  void download_positions(float4* h_pos) const;            // device -> host, per frame
};
