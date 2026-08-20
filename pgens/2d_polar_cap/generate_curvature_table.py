#!/usr/bin/env python3
"""Generate the normalized curvature-radiation photon-number CCDF table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import kv


NORMALIZATION = 5.0 * np.pi / 3.0


def curvature_ccdf(x: float) -> float:
    value, _ = quad(
        lambda t: (t - x) * kv(5.0 / 3.0, t),
        x,
        np.inf,
        epsabs=1.0e-13,
        epsrel=2.0e-11,
        limit=500,
    )
    return value / NORMALIZATION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xmin", type=float, default=1.0e-8)
    parser.add_argument("--xmax", type=float, default=30.0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "curvature_ccdf.tsv",
    )
    args = parser.parse_args()

    if args.xmin <= 0.0 or args.xmax <= args.xmin or args.size < 2:
        raise ValueError("require 0 < xmin < xmax and size >= 2")

    x_values = np.geomspace(args.xmin, args.xmax, args.size)
    ccdf_values = np.asarray([curvature_ccdf(float(x)) for x in x_values])
    if not np.all(np.diff(ccdf_values) < 0.0):
        raise RuntimeError("generated CCDF is not strictly decreasing")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output,
        np.column_stack((x_values, ccdf_values)),
        fmt="%.16e",
        delimiter="\t",
    )


if __name__ == "__main__":
    main()
