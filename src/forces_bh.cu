#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cub/cub.cuh>
#include "forces.cuh"
#include "cuda_check.cuh"

// Barnes-Hut on the GPU via a Karras LBVH, rebuilt from scratch every step
// since the particles move. Six phases, each timed separately:
//
//   1. bbox      reduce positions to the global bounding box
//   2. morton    quantize each particle to a 63-bit Morton key (21 bits/axis)
//   3. sort      CUB radix sort of (key, particle index) pairs
//   4. build     Karras 2012: binary radix tree over the sorted keys, fully
//                parallel, one thread per internal node, no locks
//   5. com       bottom-up centers of mass + AABBs; each leaf walks toward the
//                root, the first arrival at a node dies, the second combines
//                both finished children (threadfence + atomic counter pattern)
//   6. traverse  one thread per particle walks the tree with an explicit
//                stack; a node whose AABB size s and COM distance d satisfy
//                s/d < theta is accepted as a point mass, otherwise descend
//
// The force law matches the naive kernel and the CPU oracle exactly, so
// theta = 0 (never accept a node) must reproduce the naive forces to float
// round-off -- that is the correctness gate, runnable via --compare-forces.
//
// Differences from the CPU oracle, both documented in the writeup: the tree is
// binary rather than 8-way (a radix tree over Morton keys encodes the same
// spatial hierarchy), and node size s is the longest edge of the tight AABB
// rather than the octree cell edge, which only makes acceptance conservative
// for the same theta.
//
// Duplicate Morton keys (particles closer than the 2^21 grid) are handled by
// breaking delta() ties on the sorted index, which keeps the tree well formed
// with bounded depth. Traversal order follows the Morton sort, so neighboring
// threads walk nearly identical paths -- the locality that makes the per-thread
// stack viable before the warp-cooperative version.

namespace {

// Radix trees over 63-bit keys with index tie-breaking cannot exceed ~96
// levels, and the stack grows by at most one entry per level.
constexpr int kBhStack = 128;

constexpr int kBboxBlocks = 256;

// Tree scratch, persistent across steps, reallocated only when n changes.
// Node ids: internal nodes are [0, n-2] with node 0 the root; leaf i (in
// sorted order) is id n-1+i.
struct Scratch {
  int cap = 0;
  uint64_t* keys = nullptr;      // sort double buffers
  uint64_t* keys_alt = nullptr;
  int* vals = nullptr;
  int* vals_alt = nullptr;
  float4* pos_sorted = nullptr;  // positions gathered into Morton order
  int2* children = nullptr;      // internal: left/right child ids
  int* parent = nullptr;         // all 2n-1 nodes; root's parent is -1
  int* visit = nullptr;          // internal: bottom-up arrival counters
  float4* com = nullptr;         // internal: center of mass (xyz) + mass (w)
  float4* box_lo = nullptr;      // internal: AABB
  float4* box_hi = nullptr;
  float* size = nullptr;         // internal: longest AABB edge
  float4* aabb = nullptr;        // [0] = global lo, [1] = global hi
  float4* blk = nullptr;         // per-block bbox partials (lo, hi pairs)
  void* tmp = nullptr;           // CUB temp storage
  size_t tmp_bytes = 0;
};
Scratch g_s;

enum Phase { kBbox, kMorton, kSort, kBuild, kCom, kTraverse, kPhases };
const char* kPhaseNames[kPhases] = {"bbox",  "morton", "sort",
                                    "build", "com",    "traverse"};
double g_phase_ms[kPhases] = {};
long g_bh_calls = 0;
cudaEvent_t g_pa = nullptr, g_pb = nullptr;

void free_scratch() {
  cudaFree(g_s.keys);
  cudaFree(g_s.keys_alt);
  cudaFree(g_s.vals);
  cudaFree(g_s.vals_alt);
  cudaFree(g_s.pos_sorted);
  cudaFree(g_s.children);
  cudaFree(g_s.parent);
  cudaFree(g_s.visit);
  cudaFree(g_s.com);
  cudaFree(g_s.box_lo);
  cudaFree(g_s.box_hi);
  cudaFree(g_s.size);
  cudaFree(g_s.aabb);
  cudaFree(g_s.blk);
  cudaFree(g_s.tmp);
  g_s = Scratch{};
}

void ensure_scratch(int n) {
  if (n <= g_s.cap) return;
  free_scratch();
  g_s.cap = n;
  CUDA_CHECK(cudaMalloc(&g_s.keys, n * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&g_s.keys_alt, n * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&g_s.vals, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&g_s.vals_alt, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&g_s.pos_sorted, n * sizeof(float4)));
  CUDA_CHECK(cudaMalloc(&g_s.children, (n - 1) * sizeof(int2)));
  CUDA_CHECK(cudaMalloc(&g_s.parent, (2 * n - 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&g_s.visit, (n - 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&g_s.com, (n - 1) * sizeof(float4)));
  CUDA_CHECK(cudaMalloc(&g_s.box_lo, (n - 1) * sizeof(float4)));
  CUDA_CHECK(cudaMalloc(&g_s.box_hi, (n - 1) * sizeof(float4)));
  CUDA_CHECK(cudaMalloc(&g_s.size, (n - 1) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&g_s.aabb, 2 * sizeof(float4)));
  CUDA_CHECK(cudaMalloc(&g_s.blk, 2 * kBboxBlocks * sizeof(float4)));

  cub::DoubleBuffer<uint64_t> dk(g_s.keys, g_s.keys_alt);
  cub::DoubleBuffer<int> dv(g_s.vals, g_s.vals_alt);
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, g_s.tmp_bytes, dk, dv, n,
                                             0, 63));
  CUDA_CHECK(cudaMalloc(&g_s.tmp, g_s.tmp_bytes));
}

// --- phase 1: bounding box ---

__global__ void bbox_partial_kernel(const float4* __restrict__ pos, int n,
                                    float4* blk) {
  __shared__ float3 s_lo[kForceBlock];
  __shared__ float3 s_hi[kForceBlock];
  float3 lo = {FLT_MAX, FLT_MAX, FLT_MAX};
  float3 hi = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    float4 p = pos[i];
    lo.x = fminf(lo.x, p.x); lo.y = fminf(lo.y, p.y); lo.z = fminf(lo.z, p.z);
    hi.x = fmaxf(hi.x, p.x); hi.y = fmaxf(hi.y, p.y); hi.z = fmaxf(hi.z, p.z);
  }
  int tid = threadIdx.x;
  s_lo[tid] = lo;
  s_hi[tid] = hi;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_lo[tid].x = fminf(s_lo[tid].x, s_lo[tid + s].x);
      s_lo[tid].y = fminf(s_lo[tid].y, s_lo[tid + s].y);
      s_lo[tid].z = fminf(s_lo[tid].z, s_lo[tid + s].z);
      s_hi[tid].x = fmaxf(s_hi[tid].x, s_hi[tid + s].x);
      s_hi[tid].y = fmaxf(s_hi[tid].y, s_hi[tid + s].y);
      s_hi[tid].z = fmaxf(s_hi[tid].z, s_hi[tid + s].z);
    }
    __syncthreads();
  }
  if (tid == 0) {
    blk[2 * blockIdx.x] = make_float4(s_lo[0].x, s_lo[0].y, s_lo[0].z, 0.f);
    blk[2 * blockIdx.x + 1] = make_float4(s_hi[0].x, s_hi[0].y, s_hi[0].z, 0.f);
  }
}

__global__ void bbox_final_kernel(const float4* __restrict__ blk, int nblk,
                                  float4* aabb) {
  __shared__ float3 s_lo[kForceBlock];
  __shared__ float3 s_hi[kForceBlock];
  float3 lo = {FLT_MAX, FLT_MAX, FLT_MAX};
  float3 hi = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
  for (int i = threadIdx.x; i < nblk; i += blockDim.x) {
    float4 l = blk[2 * i], h = blk[2 * i + 1];
    lo.x = fminf(lo.x, l.x); lo.y = fminf(lo.y, l.y); lo.z = fminf(lo.z, l.z);
    hi.x = fmaxf(hi.x, h.x); hi.y = fmaxf(hi.y, h.y); hi.z = fmaxf(hi.z, h.z);
  }
  int tid = threadIdx.x;
  s_lo[tid] = lo;
  s_hi[tid] = hi;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_lo[tid].x = fminf(s_lo[tid].x, s_lo[tid + s].x);
      s_lo[tid].y = fminf(s_lo[tid].y, s_lo[tid + s].y);
      s_lo[tid].z = fminf(s_lo[tid].z, s_lo[tid + s].z);
      s_hi[tid].x = fmaxf(s_hi[tid].x, s_hi[tid + s].x);
      s_hi[tid].y = fmaxf(s_hi[tid].y, s_hi[tid + s].y);
      s_hi[tid].z = fmaxf(s_hi[tid].z, s_hi[tid + s].z);
    }
    __syncthreads();
  }
  if (tid == 0) {
    aabb[0] = make_float4(s_lo[0].x, s_lo[0].y, s_lo[0].z, 0.f);
    aabb[1] = make_float4(s_hi[0].x, s_hi[0].y, s_hi[0].z, 0.f);
  }
}

// --- phase 2: Morton codes ---

__device__ __forceinline__ uint64_t spread21(uint64_t v) {
  v &= 0x1fffffULL;
  v = (v | v << 32) & 0x1f00000000ffffULL;
  v = (v | v << 16) & 0x1f0000ff0000ffULL;
  v = (v | v << 8) & 0x100f00f00f00f00fULL;
  v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
  v = (v | v << 2) & 0x1249249249249249ULL;
  return v;
}

__global__ void morton_kernel(const float4* __restrict__ pos, int n,
                              const float4* __restrict__ aabb, uint64_t* keys,
                              int* vals) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 lo = aabb[0], hi = aabb[1];
  // one cubic scale for all axes keeps the curve's cells isotropic
  float ext = fmaxf(hi.x - lo.x, fmaxf(hi.y - lo.y, hi.z - lo.z));
  float inv = (ext > 0.f) ? 1.f / ext : 0.f;
  float4 p = pos[i];
  uint32_t qx = (uint32_t)fminf(fmaxf((p.x - lo.x) * inv, 0.f) * 2097152.f,
                                2097151.f);
  uint32_t qy = (uint32_t)fminf(fmaxf((p.y - lo.y) * inv, 0.f) * 2097152.f,
                                2097151.f);
  uint32_t qz = (uint32_t)fminf(fmaxf((p.z - lo.z) * inv, 0.f) * 2097152.f,
                                2097151.f);
  keys[i] = (spread21(qx) << 2) | (spread21(qy) << 1) | spread21(qz);
  vals[i] = i;
}

__global__ void gather_kernel(const float4* __restrict__ pos,
                              const int* __restrict__ perm, int n,
                              float4* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = pos[perm[i]];
}

// --- phase 4: Karras radix-tree build ---

// Length of the common key prefix of sorted leaves i and j; -1 when j is out
// of range. Equal keys fall back to the sorted index, which acts as extra
// low-order key bits and keeps every internal node's split well defined.
__device__ __forceinline__ int delta_fn(const uint64_t* keys, int n, int i,
                                        int j) {
  if (j < 0 || j >= n) return -1;
  uint64_t x = keys[i] ^ keys[j];
  if (x) return __clzll((long long)x);
  return 64 + __clz(i ^ j);
}

__global__ void build_kernel(const uint64_t* __restrict__ keys, int n,
                             int2* children, int* parent) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n - 1) return;

  // direction of this node's range, and a lower bound on its prefix
  int d = (delta_fn(keys, n, i, i + 1) - delta_fn(keys, n, i, i - 1)) >= 0 ? 1
                                                                           : -1;
  int dmin = delta_fn(keys, n, i, i - d);

  // grow, then binary-search, the other end of the range
  int lmax = 2;
  while (delta_fn(keys, n, i, i + lmax * d) > dmin) lmax <<= 1;
  int l = 0;
  for (int t = lmax >> 1; t > 0; t >>= 1)
    if (delta_fn(keys, n, i, i + (l + t) * d) > dmin) l += t;
  int j = i + l * d;

  // binary-search the split: the last position sharing the node's prefix
  int first = min(i, j), last = max(i, j);
  int common = delta_fn(keys, n, first, last);
  int split = first;
  int step = last - first;
  do {
    step = (step + 1) >> 1;
    if (split + step < last &&
        delta_fn(keys, n, first, split + step) > common)
      split += step;
  } while (step > 1);

  int left = (split == first) ? (n - 1 + split) : split;
  int right = (split + 1 == last) ? (n - 1 + split + 1) : split + 1;
  children[i] = make_int2(left, right);
  parent[left] = i;
  parent[right] = i;
}

// --- phase 5: bottom-up centers of mass ---

__global__ void com_kernel(const float4* __restrict__ pos_sorted, int n,
                           const int2* __restrict__ children,
                           const int* __restrict__ parent, int* visit,
                           float4* com, float4* box_lo, float4* box_hi,
                           float* size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int node = parent[n - 1 + i];
  while (node >= 0) {
    // first arrival stops; the second finds both children complete
    if (atomicAdd(&visit[node], 1) == 0) return;

    int2 ch = children[node];
    float4 cl, ll, hl;
    if (ch.x >= n - 1) {
      cl = pos_sorted[ch.x - (n - 1)];
      ll = hl = cl;
    } else {
      cl = com[ch.x];
      ll = box_lo[ch.x];
      hl = box_hi[ch.x];
    }
    float4 cr, lr, hr;
    if (ch.y >= n - 1) {
      cr = pos_sorted[ch.y - (n - 1)];
      lr = hr = cr;
    } else {
      cr = com[ch.y];
      lr = box_lo[ch.y];
      hr = box_hi[ch.y];
    }

    float m = cl.w + cr.w;
    float inv = 1.f / m;   // particle masses are strictly positive
    com[node] = make_float4((cl.x * cl.w + cr.x * cr.w) * inv,
                            (cl.y * cl.w + cr.y * cr.w) * inv,
                            (cl.z * cl.w + cr.z * cr.w) * inv, m);
    float4 lo = make_float4(fminf(ll.x, lr.x), fminf(ll.y, lr.y),
                            fminf(ll.z, lr.z), 0.f);
    float4 hi = make_float4(fmaxf(hl.x, hr.x), fmaxf(hl.y, hr.y),
                            fmaxf(hl.z, hr.z), 0.f);
    box_lo[node] = lo;
    box_hi[node] = hi;
    size[node] = fmaxf(hi.x - lo.x, fmaxf(hi.y - lo.y, hi.z - lo.z));

    // publish this node before touching the parent's counter, so whichever
    // thread continues from there is guaranteed to see it
    __threadfence();
    node = parent[node];
  }
}

// --- phase 6: traversal ---

__global__ void traverse_kernel(const float4* __restrict__ pos_sorted,
                                const int* __restrict__ perm, int n,
                                const int2* __restrict__ children,
                                const float4* __restrict__ com,
                                const float* __restrict__ size, float theta2,
                                float eps2, float G, float4* acc) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  float4 bi = pos_sorted[t];
  float3 ai = {0.f, 0.f, 0.f};

  int stack[kBhStack];
  int sp = 0;
  stack[sp++] = 0;
  while (sp > 0) {
    int id = stack[--sp];
    float4 src;
    if (id >= n - 1) {
      src = pos_sorted[id - (n - 1)];   // leaf; the self term is zero anyway
    } else {
      float4 c = com[id];
      float dx = c.x - bi.x, dy = c.y - bi.y, dz = c.z - bi.z;
      float d2 = dx * dx + dy * dy + dz * dz;
      float s = size[id];
      if (s * s < theta2 * d2) {
        src = c;                        // far enough: the node is a point mass
      } else {
        int2 ch = children[id];
        stack[sp++] = ch.x;
        stack[sp++] = ch.y;
        continue;
      }
    }
    float dx = src.x - bi.x, dy = src.y - bi.y, dz = src.z - bi.z;
    float dist2 = dx * dx + dy * dy + dz * dz + eps2;
    float invr = rsqrtf(dist2);
    float f = src.w * invr * invr * invr;
    ai.x += dx * f;
    ai.y += dy * f;
    ai.z += dz * f;
  }
  acc[perm[t]] = make_float4(G * ai.x, G * ai.y, G * ai.z, 0.f);
}

// Warp-cooperative variant: one warp walks one shared path. A node is opened
// if any lane needs it open (__any_sync); otherwise every lane accepts it as a
// point mass. A lane that would have accepted an ancestor instead interacts
// with its descendants, which only refines that lane's result, so accuracy at
// a given theta is never worse than the per-thread walk. The trade: the warp's
// loads of children/COM/size become one uniform access instead of 32 divergent
// ones, at the cost of lanes doing interactions they did not strictly need.
// Morton order makes neighboring lanes want nearly the same path, which is
// what keeps the extra work small. Which effect wins is measured, not
// asserted (docs/PERFORMANCE.md).
//
// The stack is per-warp in shared memory, written by lane 0 only. Control flow
// is warp-uniform (id and sp are identical across lanes), and the two
// __syncwarp() calls order the pop-read against the reuse of that slot by the
// next push.

__global__ void traverse_warp_kernel(const float4* __restrict__ pos_sorted,
                                     const int* __restrict__ perm, int n,
                                     const int2* __restrict__ children,
                                     const float4* __restrict__ com,
                                     const float* __restrict__ size,
                                     float theta2, float eps2, float G,
                                     float4* acc) {
  extern __shared__ int warp_stacks[];
  int lane = threadIdx.x & 31;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int* stack = warp_stacks + (threadIdx.x >> 5) * kBhStack;

  bool live = t < n;
  float4 bi = pos_sorted[live ? t : n - 1];
  float3 ai = {0.f, 0.f, 0.f};

  if (lane == 0) stack[0] = 0;
  __syncwarp();
  int sp = 1;
  while (sp > 0) {
    int id = stack[--sp];
    __syncwarp();   // every lane has read the slot before a push can reuse it
    float4 src;
    if (id >= n - 1) {
      src = pos_sorted[id - (n - 1)];
    } else {
      float4 c = com[id];
      float dx = c.x - bi.x, dy = c.y - bi.y, dz = c.z - bi.z;
      float d2 = dx * dx + dy * dy + dz * dz;
      float s = size[id];
      bool open = live && !(s * s < theta2 * d2);
      if (__any_sync(0xffffffff, open)) {
        int2 ch = children[id];
        if (lane == 0) {
          stack[sp] = ch.x;
          stack[sp + 1] = ch.y;
        }
        sp += 2;
        __syncwarp();
        continue;
      }
      src = c;
    }
    float dx = src.x - bi.x, dy = src.y - bi.y, dz = src.z - bi.z;
    float dist2 = dx * dx + dy * dy + dz * dz + eps2;
    float invr = rsqrtf(dist2);
    float f = src.w * invr * invr * invr;
    ai.x += dx * f;
    ai.y += dy * f;
    ai.z += dz * f;
  }
  if (live) acc[perm[t]] = make_float4(G * ai.x, G * ai.y, G * ai.z, 0.f);
}

template <typename F>
void timed_phase(Phase ph, F&& launch) {
  CUDA_CHECK(cudaEventRecord(g_pa));
  launch();
  CUDA_CHECK(cudaEventRecord(g_pb));
  CUDA_CHECK(cudaEventSynchronize(g_pb));
  CUDA_CHECK(cudaGetLastError());
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, g_pa, g_pb));
  g_phase_ms[ph] += ms;
}

}  // namespace

void forces_barnes_hut(const ParticleSystem& sys, float4* d_acc,
                       const SimParams& p) {
  int n = sys.n;
  if (n < 2) {
    CUDA_CHECK(cudaMemset(d_acc, 0, n * sizeof(float4)));
    return;
  }
  ensure_scratch(n);
  if (!g_pa) {
    CUDA_CHECK(cudaEventCreate(&g_pa));
    CUDA_CHECK(cudaEventCreate(&g_pb));
  }

  int blocks = (n + kForceBlock - 1) / kForceBlock;
  float eps2 = p.epsilon * p.epsilon;
  float theta2 = p.theta * p.theta;

  timed_phase(kBbox, [&] {
    bbox_partial_kernel<<<kBboxBlocks, kForceBlock>>>(sys.d_pos, n, g_s.blk);
    bbox_final_kernel<<<1, kForceBlock>>>(g_s.blk, kBboxBlocks, g_s.aabb);
  });

  timed_phase(kMorton, [&] {
    morton_kernel<<<blocks, kForceBlock>>>(sys.d_pos, n, g_s.aabb, g_s.keys,
                                           g_s.vals);
  });

  cub::DoubleBuffer<uint64_t> dk(g_s.keys, g_s.keys_alt);
  cub::DoubleBuffer<int> dv(g_s.vals, g_s.vals_alt);
  timed_phase(kSort, [&] {
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(g_s.tmp, g_s.tmp_bytes, dk, dv,
                                               n, 0, 63));
  });
  const uint64_t* keys = dk.Current();
  const int* perm = dv.Current();

  timed_phase(kBuild, [&] {
    gather_kernel<<<blocks, kForceBlock>>>(sys.d_pos, perm, n, g_s.pos_sorted);
    CUDA_CHECK(cudaMemsetAsync(g_s.parent, 0xFF, (2 * n - 1) * sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(g_s.visit, 0, (n - 1) * sizeof(int)));
    int nb = (n - 1 + kForceBlock - 1) / kForceBlock;
    build_kernel<<<nb, kForceBlock>>>(keys, n, g_s.children, g_s.parent);
  });

  timed_phase(kCom, [&] {
    com_kernel<<<blocks, kForceBlock>>>(g_s.pos_sorted, n, g_s.children,
                                        g_s.parent, g_s.visit, g_s.com,
                                        g_s.box_lo, g_s.box_hi, g_s.size);
  });

  timed_phase(kTraverse, [&] {
    if (p.traverse == BhTraversal::kWarp) {
      size_t shmem = (kForceBlock / 32) * kBhStack * sizeof(int);
      traverse_warp_kernel<<<blocks, kForceBlock, shmem>>>(
          g_s.pos_sorted, perm, n, g_s.children, g_s.com, g_s.size, theta2,
          eps2, p.G, d_acc);
    } else {
      traverse_kernel<<<blocks, kForceBlock>>>(g_s.pos_sorted, perm, n,
                                               g_s.children, g_s.com, g_s.size,
                                               theta2, eps2, p.G, d_acc);
    }
  });

  ++g_bh_calls;
}

void bh_report_phase_timing() {
  if (g_bh_calls == 0) return;
  double total = 0.0;
  for (int i = 0; i < kPhases; ++i) total += g_phase_ms[i];
  std::printf("barnes-hut pipeline, per-call average over %ld calls:\n",
              g_bh_calls);
  for (int i = 0; i < kPhases; ++i) {
    std::printf("  %-9s %8.3f ms  (%4.1f%%)\n", kPhaseNames[i],
                g_phase_ms[i] / g_bh_calls, 100.0 * g_phase_ms[i] / total);
  }
}

void bh_release() {
  free_scratch();
  if (g_pa) {
    cudaEventDestroy(g_pa);
    cudaEventDestroy(g_pb);
    g_pa = g_pb = nullptr;
  }
}
