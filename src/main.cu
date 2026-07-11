#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "forces.cuh"
#include "integrator.cuh"
#include "simulation.hpp"

namespace fs = std::filesystem;

struct Args {
  std::string ic_path;
  std::string out_dir = "frames";
  std::string energy_log = "benchmarks/energy_gpu.csv";
  int steps = 1000;
  int dump_every = 5;
  bool compare = false;   // run both force modules once, report, exit
  SimParams params{/*dt=*/0.01f, /*epsilon=*/0.05f, /*G=*/1.0f};
};

static void die(const std::string& msg) {
  std::fprintf(stderr, "%s\n", msg.c_str());
  std::exit(EXIT_FAILURE);
}

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) die("missing value for " + k);
      return argv[++i];
    };
    if (k == "--ic") a.ic_path = next();
    else if (k == "--out") a.out_dir = next();
    else if (k == "--energy-log") a.energy_log = next();
    else if (k == "--steps") a.steps = std::stoi(next());
    else if (k == "--dump-every") a.dump_every = std::stoi(next());
    else if (k == "--dt") a.params.dt = std::stof(next());
    else if (k == "--eps") a.params.epsilon = std::stof(next());
    else if (k == "--G") a.params.G = std::stof(next());
    else if (k == "--theta") a.params.theta = std::stof(next());
    else if (k == "--compare-forces") a.compare = true;
    else if (k == "--force") {
      std::string v = next();
      if (v == "naive") a.params.force = ForceMethod::kNaive;
      else if (v == "bh") a.params.force = ForceMethod::kBarnesHut;
      else die("--force must be naive or bh, got: " + v);
    }
    else die("unknown argument: " + k);
  }
  if (a.ic_path.empty()) die("usage: galaxy_sim --ic ic.bin [--steps N] "
                             "[--dt f] [--eps f] [--dump-every N] [--out dir] "
                             "[--force naive|bh] [--theta f] [--compare-forces]");
  return a;
}

// Run both force modules on the loaded state and report how far apart they
// are. theta = 0 must agree to float round-off; this is the on-GPU port of the
// CPU oracle's exactness gate.
static int compare_forces(ParticleSystem& sys, const SimParams& base) {
  int n = sys.n;
  float4 *d_a = nullptr, *d_b = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float4)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float4)));

  SimParams pn = base, pb = base;
  pn.force = ForceMethod::kNaive;
  pb.force = ForceMethod::kBarnesHut;
  compute_forces(sys, d_a, pn);
  compute_forces(sys, d_b, pb);

  std::vector<float4> a(n), b(n);
  CUDA_CHECK(cudaMemcpy(a.data(), d_a, n * sizeof(float4),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(b.data(), d_b, n * sizeof(float4),
                        cudaMemcpyDeviceToHost));

  double max_rel = 0.0, sum_rel = 0.0;
  for (int i = 0; i < n; ++i) {
    double dx = a[i].x - b[i].x, dy = a[i].y - b[i].y, dz = a[i].z - b[i].z;
    double an = std::sqrt((double)a[i].x * a[i].x + (double)a[i].y * a[i].y +
                          (double)a[i].z * a[i].z);
    double rel = std::sqrt(dx * dx + dy * dy + dz * dz) / (an + 1e-30);
    sum_rel += rel;
    if (rel > max_rel) max_rel = rel;
  }
  std::printf("naive vs barnes-hut, theta=%.3g, n=%d\n", base.theta, n);
  std::printf("relative force error: max %.3e, mean %.3e\n", max_rel,
              sum_rel / n);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  bh_release();
  return 0;
}

// Read ic.bin: int32 n, then n float4 positions, then n float4 velocities.
static int load_ic(const std::string& path, std::vector<float4>& pos,
                   std::vector<float4>& vel) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open IC file: " + path);
  std::int32_t n = 0;
  f.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (!f || n <= 0) die("bad particle count in " + path);
  pos.resize(n);
  vel.resize(n);
  f.read(reinterpret_cast<char*>(pos.data()), n * sizeof(float4));
  f.read(reinterpret_cast<char*>(vel.data()), n * sizeof(float4));
  if (!f) die("truncated IC file: " + path);
  return n;
}

static void dump_frame(const std::vector<float4>& pos, const std::string& dir,
                       int frame) {
  char name[64];
  std::snprintf(name, sizeof(name), "frame_%05d.bin", frame);
  std::ofstream f(fs::path(dir) / name, std::ios::binary);
  std::int32_t n = static_cast<std::int32_t>(pos.size());
  f.write(reinterpret_cast<const char*>(&n), sizeof(n));
  f.write(reinterpret_cast<const char*>(pos.data()), n * sizeof(float4));
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  std::vector<float4> h_pos, h_vel;
  int n = load_ic(args.ic_path, h_pos, h_vel);

  ParticleSystem sys;
  sys.allocate(n);
  sys.upload(h_pos.data(), h_vel.data());

  if (args.compare) return compare_forces(sys, args.params);

  float4* d_acc = nullptr;
  CUDA_CHECK(cudaMalloc(&d_acc, n * sizeof(float4)));

  fs::create_directories(args.out_dir);
  if (fs::path(args.energy_log).has_parent_path())
    fs::create_directories(fs::path(args.energy_log).parent_path());

  std::ofstream elog(args.energy_log);
  elog << "step,time,energy,kinetic,potential,rel_error\n";

  bool bh = args.params.force == ForceMethod::kBarnesHut;
  Energy e0 = total_energy(sys, args.params);
  std::printf("n=%d dt=%.4g eps=%.4g steps=%d force=%s", n, args.params.dt,
              args.params.epsilon, args.steps, bh ? "barnes-hut" : "naive");
  if (bh) std::printf(" theta=%.3g", args.params.theta);
  std::printf("\n");
  std::printf("E0=%.6e KE0=%.6e PE0=%.6e\n", e0.total, e0.kinetic, e0.potential);

  auto log_energy = [&](int step) {
    Energy e = total_energy(sys, args.params);
    double rel = (e.total - e0.total) / std::abs(e0.total);
    char line[256];
    std::snprintf(line, sizeof(line), "%d,%.5f,%.8e,%.8e,%.8e,%.3e\n", step,
                  step * args.params.dt, e.total, e.kinetic, e.potential, rel);
    elog << line;
  };

  std::vector<float4> frame_buf(n);
  int frame = 0;
  reset_force_timing();
  compute_forces(sys, d_acc, args.params);   // a(x0) for the first kick

  for (int step = 0; step < args.steps; ++step) {
    leapfrog_step(sys, d_acc, args.params);
    if (step % args.dump_every == 0) {
      sys.download_positions(frame_buf.data());
      dump_frame(frame_buf, args.out_dir, frame++);
      log_energy(step);
    }
  }
  log_energy(args.steps);

  double ms = force_kernel_ms_total();
  long calls = force_kernel_calls();
  std::printf("force pipeline: %ld calls, %.1f ms total, %.3f ms/call\n", calls,
              ms, ms / calls);
  if (bh) {
    bh_report_phase_timing();
  } else {
    double interactions = static_cast<double>(n) * n * calls;
    double gips = interactions / (ms * 1e-3) / 1e9;   // billion interactions/s
    std::printf(
        "throughput: %.2f Ginteractions/s (~%.1f GFLOP/s at 20 flop/pair)\n",
        gips, gips * 20.0);
  }
  std::printf("wrote %d frames to %s\n", frame, args.out_dir.c_str());

  CUDA_CHECK(cudaFree(d_acc));
  bh_release();
  sys.release();
  return 0;
}
