// The engine's physics in WebGPU: the tiled all-pairs force kernel from
// engine/forces.cu, the kick-drift-kick leapfrog from engine/integrator.cu, and
// the same energy diagnostic, all on the GPU. The renderer draws straight from
// the position buffer, so particle state never leaves the device except for the
// periodic diagnostics readback.

const WG = 256;   // workgroup size = shared-memory tile, as kForceBlock in CUDA

const physicsWGSL = /* wgsl */ `
struct Params { n: u32, dt: f32, eps2: f32, _pad: f32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4f>;   // xyz, mass
@group(0) @binding(2) var<storage, read_write> vel: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> acc: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> energy: array<vec2f>; // ke, pe

var<workgroup> tile: array<vec4f, ${WG}>;

fn load(j: u32) -> vec4f {   // padded lanes carry mass 0, so no inner-loop branch
  return select(vec4f(0.0), pos[min(j, P.n - 1u)], j < P.n);
}

@compute @workgroup_size(${WG})
fn forces(@builtin(global_invocation_id) gid: vec3u,
          @builtin(local_invocation_id) lid: vec3u,
          @builtin(num_workgroups) groups: vec3u) {
  let bi = load(gid.x);
  var a = vec3f(0.0);
  for (var t = 0u; t < groups.x; t++) {
    tile[lid.x] = load(t * ${WG}u + lid.x);
    workgroupBarrier();
    for (var j = 0u; j < ${WG}u; j++) {
      let bj = tile[j];
      let r = bj.xyz - bi.xyz;
      let inv = inverseSqrt(dot(r, r) + P.eps2);
      a += r * (bj.w * (inv * inv * inv));
    }
    workgroupBarrier();
  }
  if (gid.x < P.n) { acc[gid.x] = vec4f(a, 0.0); }
}

@compute @workgroup_size(${WG})
fn kick_drift(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let v = vel[i].xyz + acc[i].xyz * (0.5 * P.dt);
  vel[i] = vec4f(v, 0.0);
  pos[i] = vec4f(pos[i].xyz + v * P.dt, pos[i].w);
}

@compute @workgroup_size(${WG})
fn kick(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  vel[i] = vec4f(vel[i].xyz + acc[i].xyz * (0.5 * P.dt), 0.0);
}

// Per-particle kinetic and potential energy; summed in double on the CPU.
//   pe_i = -1/2 m_i sum_{j != i} m_j / sqrt(r_ij^2 + eps^2)
@compute @workgroup_size(${WG})
fn diagnostics(@builtin(global_invocation_id) gid: vec3u,
               @builtin(local_invocation_id) lid: vec3u,
               @builtin(num_workgroups) groups: vec3u) {
  let bi = load(gid.x);
  var phi = 0.0;   // includes the j == i self term, removed below
  for (var t = 0u; t < groups.x; t++) {
    tile[lid.x] = load(t * ${WG}u + lid.x);
    workgroupBarrier();
    for (var j = 0u; j < ${WG}u; j++) {
      let r = tile[j].xyz - bi.xyz;
      phi += tile[j].w * inverseSqrt(dot(r, r) + P.eps2);
    }
    workgroupBarrier();
  }
  if (gid.x < P.n) {
    let v = vel[gid.x].xyz;
    let self_term = bi.w * inverseSqrt(P.eps2);
    energy[gid.x] = vec2f(0.5 * bi.w * dot(v, v), -0.5 * bi.w * (phi - self_term));
  }
}
`;

const pointsWGSL = /* wgsl */ `
struct Cam { view_proj: mat4x4f, size: vec2f, gain: f32, split: u32 }
@group(0) @binding(0) var<uniform> C: Cam;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;

struct Out { @builtin(position) clip: vec4f, @location(0) uv: vec2f, @location(1) tint: vec3f }

const A = vec3f(0.37, 0.78, 1.0);    // galaxy A, cool blue  (#5ec8ff)
const B = vec3f(1.0, 0.62, 0.24);    // galaxy B, warm gold  (#ff9d3c)

@vertex
fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
  var corners = array(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0));
  let corner = corners[v];
  var o: Out;
  let c = C.view_proj * vec4f(pos[i].xyz, 1.0);
  o.clip = vec4f(c.xy + corner * C.size * c.w, c.zw);   // fixed size in pixels
  o.uv = corner;
  o.tint = select(B, A, i < C.split);
  return o;
}

@fragment
fn fs(o: Out) -> @location(0) vec4f {
  let fall = exp(-3.5 * dot(o.uv, o.uv));
  return vec4f(o.tint * fall * C.gain, 1.0);
}
`;

// Tone map the additive light: dense cores saturate gracefully to white-hot
// instead of clipping, and faint tidal tails stay visible.
const tonemapWGSL = /* wgsl */ `
@group(0) @binding(0) var hdr: texture_2d<f32>;

@vertex
fn full(@builtin(vertex_index) v: u32) -> @builtin(position) vec4f {
  var p = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[v], 0.0, 1.0);
}

@fragment
fn tonemap(@builtin(position) at: vec4f) -> @location(0) vec4f {
  let c = textureLoad(hdr, vec2u(at.xy), 0).rgb;
  let lum = dot(c, vec3f(0.3, 0.5, 0.2));
  return vec4f(1.0 - exp(-1.6 * c) + vec3f(0.55) * (1.0 - exp(-0.12 * lum)), 1.0);
}
`;

export async function createSim(canvas) {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  // One explicit layout so every kernel shares one bind group.
  const storage = { type: "storage" };
  const layout = device.createBindGroupLayout({
    entries: [{ type: "uniform" }, storage, storage, storage, storage].map((buffer, binding) =>
      ({ binding, visibility: GPUShaderStage.COMPUTE, buffer })),
  });
  const physics = device.createShaderModule({ code: physicsWGSL });
  const kernels = Object.fromEntries(["forces", "kick_drift", "kick", "diagnostics"].map((entryPoint) =>
    [entryPoint, device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: physics, entryPoint } })]));

  const pointsModule = device.createShaderModule({ code: pointsWGSL });
  const points = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: pointsModule, entryPoint: "vs" },
    fragment: {
      module: pointsModule, entryPoint: "fs",
      targets: [{ format: "rgba16float", blend: {
        color: { srcFactor: "one", dstFactor: "one" },
        alpha: { srcFactor: "one", dstFactor: "one" } } }],
    },
    primitive: { topology: "triangle-strip" },
  });
  const tonemapModule = device.createShaderModule({ code: tonemapWGSL });
  const tonemap = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: tonemapModule, entryPoint: "full" },
    fragment: { module: tonemapModule, entryPoint: "tonemap", targets: [{ format }] },
  });

  const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const cam = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  let n = 0, groups = 0, buf = {}, bind, pointsBind, hdr, hdrBind;
  let readback = null;   // in-flight diagnostics, at most one at a time

  function load({ n: count, pos, vel }, dt, eps) {
    for (const b of Object.values(buf)) b.destroy();
    n = count;
    groups = Math.ceil(n / WG);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    buf = {
      pos: device.createBuffer({ size: 16 * n, usage }),
      vel: device.createBuffer({ size: 16 * n, usage }),
      acc: device.createBuffer({ size: 16 * n, usage }),
      energy: device.createBuffer({ size: 8 * n, usage }),
    };
    device.queue.writeBuffer(buf.pos, 0, pos);
    device.queue.writeBuffer(buf.vel, 0, vel);
    device.queue.writeBuffer(params, 0, new Uint32Array([n]));
    device.queue.writeBuffer(params, 4, new Float32Array([dt, eps * eps]));
    bind = device.createBindGroup({ layout, entries: [params, buf.pos, buf.vel, buf.acc, buf.energy]
      .map((buffer, binding) => ({ binding, resource: { buffer } })) });
    pointsBind = device.createBindGroup({ layout: points.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: cam } }, { binding: 1, resource: { buffer: buf.pos } }] });
    readback = null;

    const enc = device.createCommandEncoder();   // a(x0) for the first kick
    const pass = enc.beginComputePass();
    dispatch(pass, "forces");
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  function dispatch(pass, name) {
    pass.setPipeline(kernels[name]);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(groups);
  }

  function resizeTarget(w, h) {
    if (hdr && hdr.width === w && hdr.height === h) return;
    hdr?.destroy();
    hdr = device.createTexture({ size: [w, h], format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    hdrBind = device.createBindGroup({ layout: tonemap.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: hdr.createView() }] });
  }

  // Advance `steps` leapfrog steps (each dispatch sees the previous one's
  // writes), render, and when asked queue a diagnostics readback -- energies
  // plus positions for tracking the two galaxies. Returns that readback's
  // promise, or null.
  function frame({ steps, viewProj, pixelSize, gain, sample }) {
    const enc = device.createCommandEncoder();
    if (steps) {
      const pass = enc.beginComputePass();
      for (let s = 0; s < steps; s++) {
        dispatch(pass, "kick_drift");
        dispatch(pass, "forces");
        dispatch(pass, "kick");
      }
      pass.end();
    }

    const w = canvas.width, h = canvas.height;
    resizeTarget(w, h);
    const u = new ArrayBuffer(80);
    new Float32Array(u, 0, 19).set([...viewProj, pixelSize / w, pixelSize / h, gain]);
    new Uint32Array(u, 76, 1)[0] = n >> 1;
    device.queue.writeBuffer(cam, 0, u);
    let pass = enc.beginRenderPass({ colorAttachments: [{ view: hdr.createView(),
      loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(points);
    pass.setBindGroup(0, pointsBind);
    pass.draw(4, n);
    pass.end();
    pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(),
      loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(tonemap);
    pass.setBindGroup(0, hdrBind);
    pass.draw(3);
    pass.end();

    let staging = null;
    if (sample && !readback) {
      const p = enc.beginComputePass();
      dispatch(p, "diagnostics");
      p.end();
      staging = device.createBuffer({ size: 24 * n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      enc.copyBufferToBuffer(buf.energy, 0, staging, 0, 8 * n);
      enc.copyBufferToBuffer(buf.pos, 0, staging, 8 * n, 16 * n);
    }
    device.queue.submit([enc.finish()]);
    if (!staging) return null;

    const count = n;
    const mine = staging.mapAsync(GPUMapMode.READ).then(() => {
      const raw = staging.getMappedRange();
      const out = summarize(new Float32Array(raw, 0, 2 * count), new Float32Array(raw, 8 * count), count);
      staging.destroy();
      return out;
    }).finally(() => { if (readback === mine) readback = null; });
    readback = mine;
    return mine;
  }

  return { load, frame, adapter };
}

// Total energy (summed in double) and each galaxy's center of mass.
function summarize(e, p, n) {
  let ke = 0, pe = 0;
  for (let i = 0; i < 2 * n; i += 2) { ke += e[i]; pe += e[i + 1]; }
  const half = n >> 1, com = [[0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < n; i++) {
    const c = com[i < half ? 0 : 1];
    c[0] += p[4 * i]; c[1] += p[4 * i + 1]; c[2] += p[4 * i + 2];
  }
  for (const c of com) for (let k = 0; k < 3; k++) c[k] /= half;
  return { ke, pe, energy: ke + pe, com };
}
