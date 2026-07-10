# Binary formats

All values little-endian. Floats are 32-bit (`float`), matching the device
single-precision state. Layout matches the device `float4` arrays exactly, so the
host reads a file straight into the upload buffer with no repacking.

## Initial conditions (`ic.bin`)

Produced by `scripts/generate_ic.py`, consumed by the simulator at startup.

```
int32   n                     particle count
float4  pos[n]                x, y, z, m   (position + mass)
float4  vel[n]                vx, vy, vz, _ (w unused, written as 0)
```

`pos` is the full array first, then the full `vel` array (structure-of-arrays on
disk as well as on the device).

## Frame dump (`frame_%05d.bin`)

Written each dump interval by the simulator, consumed by `scripts/render.py`.
Positions only — velocity is not needed to render.

```
int32   n                     particle count
float4  pos[n]                x, y, z, m
```

The mass in `w` lets the renderer size or color particles, and lets a viewer tell
the two galaxies apart if their particle masses differ.
