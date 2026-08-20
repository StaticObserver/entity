# 2D polar-cap

Local two-dimensional pulsar polar-cap discharge on Entity v1.4.4, targeting
the 2D setup of Chernoglazov, Philippov & Timokhin (2024, arXiv:2409.15409).
The authoritative design contract is [`docs/design.md`](docs/design.md).

This PGen is a port of the validated [`pgens/1d_polar_cap`](../1d_polar_cap):
the initial/boundary physics is lifted to 2D (x1 along the background field,
x2 transverse and periodic), plus an x2-dependent external-current profile.
A transversely uniform 2D run must reduce column-by-column to the 1D results.

Differences from the 1D PGen:

- Dimension: `Dim::_2D` only (traits and static_asserts updated everywhere);
- `setup.polar_cap.current_profile`: `"uniform"` (default, identical to 1D)
  or `"philippov2024"` (CPT2024 Eq. 4/5 transverse profile, mapped with
  `current_profile_center` / `current_profile_halfwidth`);
- the deprecated boundary-flux compensation is dropped entirely —
  `CustomParticleUpdate` returns the photon-opacity update directly;
- the 1D `debug_edge_fields` diagnostic is removed;
- the MATCH-layer `Ex1` damping is generalized to 2D: it damps the last
  `boundary_ex_damping.cells` active x1 faces of every transverse column
  (`boundary_damping.hpp`);
- pair creation places the converted pair in the photon's `(i1, i2)` cell.

Reference checks that do not compile Entity:

```bash
python3 pgens/2d_polar_cap/tests/reference_models.py
```

The curvature spectrum table is shared with the 1D case; regenerate it with
`python3 pgens/2d_polar_cap/generate_curvature_table.py`.
