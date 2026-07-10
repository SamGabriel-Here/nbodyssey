#!/usr/bin/env python3
"""Generate initial conditions: two disk galaxies on a collision course.

Each galaxy is an exponential-profile disk whose stars are placed on softened
circular orbits, then given a bulk velocity so the two disks fall toward each
other on a grazing encounter. The output is a flat binary the simulator uploads
straight to the GPU (see docs/FORMATS.md).

Units: G = 1, and by default each galaxy has total mass 1 and scale radius 1, so
the dynamical time is order unity and velocities are order unity.
"""
import argparse
import numpy as np


def build_disk(n, m_total, scale_radius, center, bulk_velocity,
               inclination, spin, eps, rng, r_max_factor=4.0):
    """One exponential disk of `n` particles.

    Radii are drawn from the exponential-disk radial profile p(r) ~ r exp(-r/Rd),
    which is a Gamma(shape=2, scale=Rd) distribution. Circular speeds come from
    the enclosed mass under the same softened monopole force the integrator uses,
    so the disk starts close to equilibrium instead of collapsing.
    """
    Rd = scale_radius
    m_particle = m_total / n

    # radial samples from the exponential disk, truncated at a few scale lengths.
    # Resample the tail rather than clipping, which would pile particles into a
    # ring at exactly r_max.
    r_max = r_max_factor * Rd
    r = rng.gamma(shape=2.0, scale=Rd, size=n)
    over = r > r_max
    while np.any(over):
        r[over] = rng.gamma(shape=2.0, scale=Rd, size=int(over.sum()))
        over = r > r_max
    r = np.sort(r)  # sorted so enclosed mass is just a cumulative count

    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    # thin disk: small vertical scatter, ~5% of the scale radius
    z = rng.normal(0.0, 0.05 * Rd, size=n)

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # enclosed mass at each (sorted) radius, offset by half a particle so the
    # innermost star still feels some interior mass
    m_enclosed = m_particle * (np.arange(n) + 0.5)

    # softened circular speed: v^2 = r * a, a = G m_enc r / (r^2 + eps^2)^(3/2)
    a_mag = m_enclosed * r / np.power(r * r + eps * eps, 1.5)
    v_circ = np.sqrt(np.maximum(r * a_mag, 0.0))

    # tangential velocity in the disk plane, sign sets the spin direction
    vx = -spin * v_circ * np.sin(phi)
    vy = spin * v_circ * np.cos(phi)
    vz = np.zeros(n)

    pos = np.stack([x, y, z], axis=1)
    vel = np.stack([vx, vy, vz], axis=1)

    # tilt the disk out of the xy-plane (rotation about the x-axis)
    if inclination != 0.0:
        c, s = np.cos(inclination), np.sin(inclination)
        rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        pos = pos @ rot.T
        vel = vel @ rot.T

    pos += np.asarray(center)
    vel += np.asarray(bulk_velocity)

    mass = np.full(n, m_particle)
    pos4 = np.column_stack([pos, mass]).astype("<f4")
    vel4 = np.column_stack([vel, np.zeros(n)]).astype("<f4")
    return pos4, vel4


def write_ic(path, pos4, vel4):
    n = pos4.shape[0]
    with open(path, "wb") as f:
        f.write(np.int32(n).astype("<i4").tobytes())
        pos4.astype("<f4").tofile(f)
        vel4.astype("<f4").tofile(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--particles", type=int, default=40000,
                    help="total particles across both galaxies")
    ap.add_argument("--out", default="ic.bin", help="output binary path")
    ap.add_argument("--separation", type=float, default=16.0,
                    help="initial separation between galaxy centers")
    ap.add_argument("--impact", type=float, default=3.0,
                    help="impact parameter (transverse offset) for a grazing pass")
    ap.add_argument("--approach", type=float, default=0.55,
                    help="approach speed of each galaxy toward the other")
    ap.add_argument("--mass", type=float, default=1.0, help="mass per galaxy")
    ap.add_argument("--scale-radius", type=float, default=1.0, help="disk scale radius")
    ap.add_argument("--eps", type=float, default=0.05, help="softening length")
    ap.add_argument("--inclination", type=float, default=30.0,
                    help="tilt of the second disk, degrees")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_each = args.particles // 2
    half_sep = 0.5 * args.separation
    half_imp = 0.5 * args.impact
    incl = np.radians(args.inclination)

    # galaxy A: left, moving right and slightly up; spin +
    posA, velA = build_disk(
        n_each, args.mass, args.scale_radius,
        center=[-half_sep, -half_imp, 0.0],
        bulk_velocity=[args.approach, 0.0, 0.0],
        inclination=0.0, spin=+1.0, eps=args.eps, rng=rng)

    # galaxy B: right, moving left and slightly down; opposite spin, tilted
    posB, velB = build_disk(
        n_each, args.mass, args.scale_radius,
        center=[half_sep, half_imp, 0.0],
        bulk_velocity=[-args.approach, 0.0, 0.0],
        inclination=incl, spin=-1.0, eps=args.eps, rng=rng)

    pos4 = np.concatenate([posA, posB], axis=0)
    vel4 = np.concatenate([velA, velB], axis=0)
    write_ic(args.out, pos4, vel4)

    print(f"wrote {pos4.shape[0]} particles to {args.out} "
          f"({pos4.nbytes + vel4.nbytes + 4} bytes)")


if __name__ == "__main__":
    main()
