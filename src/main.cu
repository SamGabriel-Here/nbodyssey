#include <cstdio>
#include <string>
#include "simulation.hpp"
#include "forces.cuh"
#include "integrator.cuh"

// Host driver. Loads initial conditions, uploads once, integrates on the GPU,
// and dumps positions to disk periodically. Force computation and integration
// are not implemented yet; this establishes the run loop and the timing/energy
// instrumentation points.

struct Args {
  std::string ic_path;
  std::string out_dir = "frames";
  int steps = 1000;
  int dump_every = 10;
  SimParams params{/*dt=*/0.01f, /*epsilon=*/0.05f};
};

static Args parse_args(int argc, char** argv);   // TODO
static ParticleSystem load_ic(const std::string& path); // TODO: read IC binary, upload
static void dump_frame(const ParticleSystem& sys, const std::string& dir, int frame); // TODO

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  ParticleSystem sys = load_ic(args.ic_path);

  float4* d_acc = nullptr;
  // TODO: cudaMalloc d_acc, sys.n

  // TODO: CUDA events for force-kernel timing; energy log file in benchmarks/

  int frame = 0;
  for (int step = 0; step < args.steps; ++step) {
    leapfrog_step(sys, d_acc, args.params);   // kick-drift-kick

    if (step % args.dump_every == 0) {
      dump_frame(sys, args.out_dir, frame++);
      // TODO: log total_energy(sys, args.params) vs step
    }
  }

  // TODO: report aggregate kernel timing; free device memory
  printf("scaffold: run loop in place, physics pending\n");
  return 0;
}
