// Initial conditions: two exponential disks on a grazing encounter.
// A port of tools/generate_ic.py (same profile, orbits, tilt and units: G = 1,
// mass 1 and scale radius 1 per galaxy), with a seeded RNG so a run is
// reproducible from its URL. Galaxy A's particles come first, then galaxy B's.

export const DEFAULTS = { n: 16384, impact: 3, inclination: 30, approach: 0.55 };
export const EPS = 0.05;   // softening length, shared with the simulator

// mulberry32: tiny, fast, good enough for sampling a disk
function rng(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function disk(n, offset, { center, bulk, inclination, spin }, rand, pos, vel) {
  const m = 1 / n;
  const rMax = 4;   // truncate at four scale lengths, resampling the tail
  // Gamma(2, 1) is the exponential disk's radial profile r e^-r: the sum of
  // two unit exponentials. Sorted, so enclosed mass is a running count.
  const r = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    do r[i] = -Math.log((1 - rand()) * (1 - rand())); while (r[i] > rMax);
  }
  r.sort();

  const c = Math.cos(inclination), s = Math.sin(inclination);
  for (let i = 0; i < n; i++) {
    const phi = 2 * Math.PI * rand();
    // thin disk: Box-Muller normal, 5% of the scale radius
    const z = 0.05 * Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
    const x = r[i] * Math.cos(phi), y = r[i] * Math.sin(phi);

    // circular speed under the softened monopole force the integrator uses;
    // half a particle of interior mass so the innermost star still orbits
    const mEnc = m * (i + 0.5);
    const a = (mEnc * r[i]) / Math.pow(r[i] * r[i] + EPS * EPS, 1.5);
    const v = Math.sqrt(r[i] * a);
    const vx = -spin * v * Math.sin(phi), vy = spin * v * Math.cos(phi);

    // tilt about the x-axis, then place and launch the disk
    const k = 4 * (offset + i);
    pos[k] = x + center[0];
    pos[k + 1] = c * y - s * z + center[1];
    pos[k + 2] = s * y + c * z + center[2];
    pos[k + 3] = m;
    vel[k] = vx + bulk[0];
    vel[k + 1] = c * vy + bulk[1];
    vel[k + 2] = s * vy + bulk[2];
    vel[k + 3] = 0;
  }
}

export function galaxies({ n, impact, inclination, approach }, seed = 0) {
  const each = n >> 1;
  const pos = new Float32Array(8 * each), vel = new Float32Array(8 * each);
  const rand = rng(seed);
  const halfSep = 8, halfImp = impact / 2;
  disk(each, 0, { center: [-halfSep, -halfImp, 0], bulk: [approach, 0, 0],
                  inclination: 0, spin: 1 }, rand, pos, vel);
  disk(each, each, { center: [halfSep, halfImp, 0], bulk: [-approach, 0, 0],
                     inclination: (inclination * Math.PI) / 180, spin: -1 }, rand, pos, vel);
  return { n: 2 * each, pos, vel };
}
