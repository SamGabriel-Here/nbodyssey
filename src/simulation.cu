#include "simulation.hpp"
#include "cuda_check.cuh"

void ParticleSystem::allocate(int count) {
  n = count;
  CUDA_CHECK(cudaMalloc(&d_pos, n * sizeof(float4)));
  CUDA_CHECK(cudaMalloc(&d_vel, n * sizeof(float4)));
}

void ParticleSystem::release() {
  if (d_pos) CUDA_CHECK(cudaFree(d_pos));
  if (d_vel) CUDA_CHECK(cudaFree(d_vel));
  d_pos = d_vel = nullptr;
  n = 0;
}

void ParticleSystem::upload(const float4* h_pos, const float4* h_vel) {
  CUDA_CHECK(cudaMemcpy(d_pos, h_pos, n * sizeof(float4), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vel, h_vel, n * sizeof(float4), cudaMemcpyHostToDevice));
}

void ParticleSystem::download_positions(float4* h_pos) const {
  CUDA_CHECK(cudaMemcpy(h_pos, d_pos, n * sizeof(float4), cudaMemcpyDeviceToHost));
}
