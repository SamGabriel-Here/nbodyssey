import { DEFAULTS, EPS, galaxies } from "./ic.js";
import { createSim } from "./sim.js";

const DT = 0.01;   // the engine's default timestep
const $ = (id) => document.getElementById(id);
const canvas = $("sky");
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

// --- run settings, mirrored in the URL so a run can be shared -------------

const query = new URLSearchParams(location.search);
const num = (k, fallback, lo, hi) => {
  const v = Number(query.get(k));
  return query.has(k) && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : fallback;
};
const SIZES = [8192, 16384, 32768, 65536];
const settings = {
  n: SIZES.includes(num("n", 0, 0, 1e9)) ? num("n", 0, 0, 1e9)
     : matchMedia("(pointer: coarse)").matches ? 8192 : DEFAULTS.n,
  impact: num("b", DEFAULTS.impact, 0, 8),
  inclination: num("i", DEFAULTS.inclination, 0, 180),
  approach: num("v", DEFAULTS.approach, 0.1, 1.2),
};
let speed = speedFor(settings.n);
let playing = !reduceMotion;

// enough steps per frame to keep the story moving, without starving big runs
function speedFor(n) { return n <= 16384 ? 4 : n <= 32768 ? 2 : 1; }

function syncUrl() {
  const q = new URLSearchParams({ n: settings.n, b: settings.impact,
                                  i: settings.inclination, v: settings.approach });
  history.replaceState(null, "", `?${q}`);
}

// --- the encounter, narrated --------------------------------------------
// Phases are read off the separation of the two galaxies' centers of mass,
// so the story follows whatever orbit the sliders set up.

const PHASES = [
  ["Approach", (n) => `Two disk galaxies, ${(n / 2).toLocaleString()} stars each, fall toward each other.`],
  ["First contact", () => "The outer disks touch first. Stars at the rims start to feel the other galaxy's pull."],
  ["Pericenter", () => "Closest approach. The cores swing past each other at their fastest."],
  ["Tidal bridge", () => "Tides stretch a bridge of stars between the cores and fling long tails outward."],
  ["Fall back", () => "Gravity wins. The cores turn around and fall back toward each other."],
  ["Merger", () => "The cores merge. One disturbed system remains, ringed by its own debris."],
  ["Escape", () => "The cores separate for good, each trailing the stars the other tore away."],
];

const tracker = () => ({ phase: 0, minSep: Infinity, maxSep: 0, tPeri: 0 });

function advancePhase(s, sep, t) {
  switch (s.phase) {
    case 0: if (sep < 8) s.phase = 1; break;   // two disks of radius ~4 touch
    case 1:
      s.minSep = Math.min(s.minSep, sep);
      if (sep > s.minSep + 0.05) { s.tPeri = t; s.phase = 2; }
      break;
    case 2: if (t - s.tPeri > 1.5) s.phase = 3; break;
    case 3:
    case 6:   // escape is only final if they never turn around
      s.maxSep = Math.max(s.maxSep, sep);
      if (sep < s.maxSep - 0.4) s.phase = 4;
      else if (s.phase === 3 && t - s.tPeri > 14) s.phase = 6;
      break;
    case 4: if (sep < 1.2) s.phase = 5; break;
  }
}

let shownPhase = -1;
function showPhase(p, n) {
  if (p === shownPhase) return;
  shownPhase = p;
  const line = $("phase-line");
  const apply = () => {
    $("phase-no").textContent = String(p + 1).padStart(2, "0");
    $("phase-name").textContent = PHASES[p][0];
    line.textContent = PHASES[p][1](n);
    line.classList.remove("swap");
  };
  if (reduceMotion) return apply();
  line.classList.add("swap");
  setTimeout(apply, 350);
}

// --- camera: orbit around the system's center of mass (z is up) ---------

const VIEWS = { face: 1.45, three: 0.62, edge: 0.04 };
// dist follows the galaxies' separation until the viewer zooms by hand
const camera = { yaw: -1.2, pitch: VIEWS.three, dist: 30, goal: null, framing: 30, manualZoom: false };

function viewProj(aspect) {
  const { yaw, pitch, dist } = camera;
  const e = [dist * Math.cos(pitch) * Math.cos(yaw), dist * Math.cos(pitch) * Math.sin(yaw), dist * Math.sin(pitch)];
  const z = e.map((v) => v / dist);                         // camera looks down -z
  const lx = Math.hypot(z[0], z[1]);
  const x = [-z[1] / lx, z[0] / lx, 0];                    // right = up(+z) × z
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const view = [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
                -dot(x, e), -dot(y, e), -dot(z, e), 1];
  const f = 1 / Math.tan(0.375), near = 0.1, far = 400, r = 1 / (near - far);
  const proj = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far * r, -1, 0, 0, near * far * r, 0];
  const out = new Array(16).fill(0);   // column-major proj × view
  for (let c = 0; c < 4; c++)
    for (let row = 0; row < 4; row++)
      for (let k = 0; k < 4; k++) out[c * 4 + row] += proj[k * 4 + row] * view[c * 4 + k];
  return out;
}

function setView(name) {
  camera.goal = VIEWS[name];
  for (const b of document.querySelectorAll("#views button"))
    b.setAttribute("aria-pressed", String(b.dataset.view === name));
}

let lastInteraction = -Infinity;
function bindOrbit() {
  const pointers = new Map();
  let pinch = 0;
  const zoom = (k) => {
    camera.manualZoom = true;
    camera.dist = Math.min(120, Math.max(6, camera.dist * k));
  };
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
  });
  canvas.addEventListener("pointermove", (e) => {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    lastInteraction = performance.now();
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pinch) zoom(pinch / d);
      pinch = d;
      return;
    }
    camera.goal = null;
    for (const b of document.querySelectorAll("#views button")) b.setAttribute("aria-pressed", "false");
    camera.yaw -= (e.clientX - prev[0]) * 0.006;
    camera.pitch = Math.min(1.5, Math.max(-1.5, camera.pitch + (e.clientY - prev[1]) * 0.006));
  });
  const up = (e) => { pointers.delete(e.pointerId); pinch = 0; };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoom(Math.exp(e.deltaY * 0.001));
  }, { passive: false });
}

const sci = (x) => {
  if (!Number.isFinite(x)) return "—";
  const [m, e] = x.toExponential(1).split("e");
  const sup = e.replace("+", "").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]).replace("-", "⁻");
  return `${m}×10${sup}`;
};

// --- main ------------------------------------------------------------------

async function main() {
  const fallback = () => {
    $("fallback").hidden = false;
    for (const el of [$("console"), canvas, document.querySelector(".caption")]) el.hidden = true;
  };
  const sim = await createSim(canvas).catch(() => null);
  if (!sim) return fallback();
  sim.lost.then(fallback);   // a GPU reset or driver crash ends the live run
  if (matchMedia("(max-width: 720px)").matches) $("console-details").open = false;

  let run;   // everything that resets with a restart
  function restart() {
    const ic = galaxies(settings);
    sim.load(ic, DT, EPS);
    run = { n: ic.n, step: 0, e0: null, story: tracker(), needSample: true };
    $("r-bodies").textContent = ic.n.toLocaleString();
    $("r-energy").textContent = "—";
    shownPhase = -1;
    showPhase(0, ic.n);
    syncUrl();
  }

  const radios = (name, value, onChange) => {
    for (const r of document.querySelectorAll(`input[name="${name}"]`)) {
      r.checked = Number(r.value) === value;
      r.addEventListener("change", () => onChange(Number(r.value)));
    }
  };
  radios("speed", speed, (v) => { speed = v; });
  radios("n", settings.n, (v) => {
    settings.n = v;
    speed = speedFor(v);
    for (const r of document.querySelectorAll('input[name="speed"]')) r.checked = Number(r.value) === speed;
    restart();
  });

  const fmt = { impact: (v) => v.toFixed(1), inclination: (v) => `${v}°`, approach: (v) => v.toFixed(2) };
  for (const key of ["impact", "inclination", "approach"]) {
    const input = $(key), out = $(`o-${key}`);
    input.value = settings[key];
    out.textContent = fmt[key](settings[key]);
    input.addEventListener("input", () => { out.textContent = fmt[key](Number(input.value)); });
    input.addEventListener("change", () => { settings[key] = Number(input.value); restart(); });
  }

  const play = $("play");
  const setPlaying = (p) => { playing = p; play.textContent = p ? "Pause" : "Play"; };
  setPlaying(playing);
  play.addEventListener("click", () => setPlaying(!playing));
  $("restart").addEventListener("click", restart);
  for (const b of document.querySelectorAll("#views button"))
    b.addEventListener("click", () => setView(b.dataset.view));
  setView("three");

  addEventListener("keydown", (e) => {
    if (e.target.closest?.("input, button, summary")) return;
    if (e.code === "Space") { e.preventDefault(); setPlaying(!playing); }
    else if (e.key === "r") restart();
    else if (["1", "2", "3"].includes(e.key)) setView(["face", "three", "edge"][e.key - 1]);
  });
  bindOrbit();

  new ResizeObserver(([entry]) => {   // device pixels, scaled down together past 4096
    const box = entry.devicePixelContentBoxSize?.[0];
    const w = box ? box.inlineSize : Math.round(entry.contentRect.width * devicePixelRatio);
    const h = box ? box.blockSize : Math.round(entry.contentRect.height * devicePixelRatio);
    const k = Math.min(1, 4096 / Math.max(w, h, 1));
    canvas.width = Math.max(1, Math.round(w * k));
    canvas.height = Math.max(1, Math.round(h * k));
  }).observe(canvas);

  restart();

  let lastSample = 0, last = performance.now(), rateSteps = 0, rateStart = last;
  function frame(now) {
    const dtWall = Math.min(0.1, (now - last) / 1000);
    last = now;
    // ease toward a chosen view; otherwise drift slowly while nobody is driving
    if (camera.goal !== null) camera.pitch += (camera.goal - camera.pitch) * Math.min(1, dtWall * 3);
    if (!reduceMotion && playing && now - lastInteraction > 4000) camera.yaw += dtWall * 0.03;
    if (!camera.manualZoom) camera.dist += (camera.framing - camera.dist) * Math.min(1, dtWall * 0.8);

    const steps = playing ? speed : 0;
    const pending = sim.frame({
      steps,
      viewProj: viewProj(canvas.width / canvas.height),
      pixelSize: 2.6 * devicePixelRatio,
      gain: 0.5 * Math.pow(16384 / run.n, 0.8),   // same total light at any n
      sample: run.needSample || now - lastSample > 250,
    });
    run.step += steps;
    rateSteps += steps;

    if (pending) {
      lastSample = now;
      run.needSample = false;
      const r = run, step = r.step;
      pending.then(({ energy, com }) => {
        if (r !== run) return;   // restarted meanwhile
        r.e0 ??= energy;
        const t = step * DT;
        const [a, b] = com;
        const sep = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        camera.framing = Math.min(110, 18 + 1.2 * sep);
        advancePhase(r.story, sep, t);
        showPhase(r.story.phase, r.n);
        $("r-time").textContent = t.toFixed(2);
        $("r-step").textContent = step.toLocaleString();
        $("r-energy").textContent = step ? sci((energy - r.e0) / Math.abs(r.e0)) : "—";
      }).catch(() => {});
    }

    if (now - rateStart > 1000) {
      $("r-rate").textContent = rateSteps ? sci((rateSteps * run.n * run.n) / ((now - rateStart) / 1000)) : "paused";
      rateSteps = 0;
      rateStart = now;
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main();
