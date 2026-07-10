# galaxy-collision-cuda

[![build](https://github.com/SamGabriel-Here/galaxy-collision-cuda/actions/workflows/build.yml/badge.svg)](https://github.com/SamGabriel-Here/galaxy-collision-cuda/actions/workflows/build.yml)

A GPU-accelerated N-body simulator for galaxy collisions, written in CUDA C++.
Two disk galaxies are seeded with realistic rotation curves, released toward each
other, and integrated forward under mutual gravity. Particle state lives on the
GPU for the whole run; frames are dumped to disk and rendered offline.

The point of the project is the GPU work: a naive all-pairs force kernel with
shared-memory tiling first, then a Barnes-Hut tree code, with timing and energy
diagnostics built in so the optimization story is measurable rather than
asserted.

![galaxy collision](docs/collision.gif)

*Two disk galaxies on a grazing encounter, colored by their galaxy of origin.
12k particles integrated with leapfrog under softened gravity; the pass strips
material into tidal tails and bridges before the cores fall back together.*

## Status

The full naive pipeline is in place: generate two colliding disks, integrate on
the GPU, dump frames, render. The physics is validated against a CPU reference
integrator (see below), and the CUDA sources are compiled in CI. Next up is the
Barnes-Hut force module and, with it, the performance writeup.

Milestones:

- [x] Repo, build system, architecture, benchmarking harness scaffold
- [x] Initial-condition generator (two exponential disks)
- [x] Naive O(n^2) force kernel with shared-memory tiling
- [x] Leapfrog (kick-drift-kick) integrator
- [x] CUDA-event kernel timing + energy-vs-time log
- [x] Frame dumps + offline matplotlib renderer
- [x] CPU reference integrator + energy-conservation validation
- [ ] Barnes-Hut O(n log n) force module
- [ ] Performance writeup comparing the two force modules

## Validation

Because the state is single precision, correctness is not obvious, so the physics
is pinned two ways.

A CPU reference integrator (`scripts/reference_nbody.py`) implements the same
force law, kick-drift-kick scheme, and energy diagnostic in NumPy. Running a full
two-galaxy collision and tracking total energy gives the plot below: kinetic
energy peaks at pericenter as the disks fall together, the potential well deepens
in step, and total energy stays flat. The relative energy error stays bounded
within about 0.02% across the encounter and oscillates rather than drifting --
the signature of a symplectic integrator.

![energy conservation](docs/energy_conservation.png)

The reference also doubles as a GPU-free way to exercise the whole pipeline: it
writes frames and energy logs in the same on-disk formats the CUDA path uses, so
the renderer and downstream tooling are validated end to end. The GPU kernels are
written to mirror the reference and are compiled in CI (`nvcc` targets a device
architecture without needing a physical GPU on the runner).

## Architecture

The design is fixed up front so the force computation can be swapped without
touching the rest of the code.

**Data resident on the GPU.** The host builds the initial conditions, uploads
them once, and thereafter only copies particle positions back when it is time to
write a frame. Nothing about the integration touches host memory.

**Structure-of-arrays layout.** Positions and velocities are stored in separate
arrays rather than an array of particle structs, so memory accesses across a warp
are coalesced. Position and mass are packed together as a `float4` (`x, y, z, m`);
velocity is a `float4` (`vx, vy, vz, unused`). Mass rides in the position array
because the force kernel needs mass and position together and nothing else on the
same access.

**Single precision.** All particle state is `float`. Single precision is the
right trade for throughput on the target hardware; the cost is energy drift,
which is exactly what the energy diagnostic is there to catch.

**Leapfrog integrator.** Kick-drift-kick leapfrog, which is symplectic and
time-reversible, so total energy oscillates around a constant rather than
drifting secularly. This is what makes single precision defensible.

**Softened gravity.** A softening length `epsilon` is added in quadrature to the
pairwise separation so that close encounters do not produce singular forces and
blow up the integrator. This is standard for collisionless disk simulations where
the particles model a smooth mass distribution rather than real point stars.

**Force computation is a swappable module.** Everything above is stable across
force implementations. The first implementation is the naive all-pairs kernel:
each thread accumulates the force on one particle from all others, staging blocks
of source particles through shared memory (tiling) to cut global-memory traffic.
The second is Barnes-Hut, which builds an octree and approximates distant groups
of particles by their center of mass, taking the asymptotic cost from O(n^2) to
O(n log n). Both implement the same interface so the integrator does not know or
care which is loaded.

**Offline visualization.** The simulator does not render. It writes particle
positions per frame to disk in a simple binary format; a separate Python script
turns those frames into images or an animation. Rendering is decoupled from the
simulation so neither constrains the other.

## Benchmarking

Timing and correctness instrumentation are part of the simulator from the start,
not bolted on later:

- **Kernel timing** with CUDA events around the force kernel, reported per step
  and aggregated, so the naive-vs-Barnes-Hut comparison is a measured speedup.
- **Energy log** writing total kinetic + potential energy versus simulation time.
  A correct symplectic integrator keeps this bounded; a bug shows up as drift or
  a blowup.

Logs land in `benchmarks/`.

## Repository layout

```
src/         CUDA kernels and host driver
scripts/     initial-condition generator and offline renderer (Python)
benchmarks/  timing and energy logs
docs/        architecture notes and the performance writeup
```

## Building

The build targets a CUDA-capable Linux machine with the CUDA Toolkit and a recent
CMake. Note that macOS has no CUDA support, so the code is developed on macOS but
built and run on a GPU host.

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Set `-DCMAKE_CUDA_ARCHITECTURES=<sm>` to match the target GPU (for example `86`
for Ampere, `89` for Ada).

## Running

```
# generate two colliding disk galaxies
python scripts/generate_ic.py --particles 50000 --out ic.bin

# integrate, dumping a frame every 10 steps
./build/galaxy_sim --ic ic.bin --steps 2000 --dump-every 10 --out frames/

# render the dumped frames
python scripts/render.py --frames frames/ --out collision.mp4
```

Flags and formats will be documented here as they land.

## License

MIT. See [LICENSE](LICENSE).
