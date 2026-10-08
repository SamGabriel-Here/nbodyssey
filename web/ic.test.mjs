// node --test web/ -- checks the JS initial conditions match tools/generate_ic.py's setup
import { test } from "node:test";
import assert from "node:assert/strict";
import { DEFAULTS, galaxies } from "./ic.js";

const sum = (a, k, from, to) => { let s = 0; for (let i = from; i < to; i++) s += a[4 * i + k]; return s; };

test("two unit-mass disks, launched head-on along x, at +-8", () => {
  const { n, pos, vel } = galaxies(DEFAULTS);
  const h = n / 2;
  assert.equal(n, DEFAULTS.n);
  assert.ok(Math.abs(sum(pos, 3, 0, h) - 1) < 1e-4 && Math.abs(sum(pos, 3, h, n) - 1) < 1e-4);
  // centers of mass sit where the generator put them
  assert.ok(Math.abs(sum(pos, 0, 0, h) / h + 8) < 0.1 && Math.abs(sum(pos, 0, h, n) / h - 8) < 0.1);
  assert.ok(Math.abs(sum(pos, 1, h, n) / h - DEFAULTS.impact / 2) < 0.1);
  // rotation averages out, leaving each disk's bulk velocity
  assert.ok(Math.abs(sum(vel, 0, 0, h) / h - DEFAULTS.approach) < 0.02);
  assert.ok(Math.abs(sum(vel, 0, h, n) / h + DEFAULTS.approach) < 0.02);
  // net momentum is zero up to sampling noise (~v / sqrt(n))
  assert.ok(Math.abs(sum(vel, 0, 0, n)) / n < 0.02);
});

test("same seed, same galaxies", () => {
  assert.deepEqual(galaxies(DEFAULTS).pos.slice(0, 64), galaxies(DEFAULTS).pos.slice(0, 64));
});
