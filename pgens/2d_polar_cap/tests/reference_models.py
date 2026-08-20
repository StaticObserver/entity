#!/usr/bin/env python3
"""Reference checks for the 2D polar-cap PGen that do not compile Entity.

Covers the analytic contract of the "philippov2024" transverse current
profile and the integrity of the tabulated curvature CCDF. See
docs/design.md for the full design contract.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


CASE_DIR = Path(__file__).resolve().parents[1]

# CPT2024 (arXiv:2409.15409) Eq. 4/5 constraint solution.
C1 = 100.0 / 3.0
C2 = 560.0 / 17.0
C3 = -82.0 / 51.0
TOLERANCE = 1.0e-12


def profile(x: float) -> float:
    """Dimensionless philippov2024 profile j(x) / j_scale on [-1, 1]."""
    x = max(-1.0, min(1.0, x))
    return 2.0 - C1 * x**4 + C2 * x**6 + C3 * x**40


class Philippov2024ProfileTests(unittest.TestCase):
    def test_profile_vanishes_at_cap_edge(self) -> None:
        self.assertAlmostEqual(profile(1.0), 0.0, delta=TOLERANCE)
        self.assertAlmostEqual(profile(-1.0), 0.0, delta=TOLERANCE)

    def test_profile_derivative_vanishes_at_cap_edge(self) -> None:
        # j'(x) = -4 C1 x^3 + 6 C2 x^5 + 40 C3 x^39, evaluated analytically.
        derivative_at_one = -4.0 * C1 + 6.0 * C2 + 40.0 * C3
        self.assertAlmostEqual(derivative_at_one, 0.0, delta=TOLERANCE)

    def test_profile_carries_zero_net_current(self) -> None:
        # Analytic integral of each even monomial over [-1, 1].
        integral = 4.0 - C1 * (2.0 / 5.0) + C2 * (2.0 / 7.0) + C3 * (2.0 / 41.0)
        self.assertAlmostEqual(integral, 0.0, delta=TOLERANCE)
        # Cross-check against a dense numerical quadrature.
        grid = np.linspace(-1.0, 1.0, 1_000_001)
        values = 2.0 - C1 * grid**4 + C2 * grid**6 + C3 * grid**40
        numerical = float(np.trapezoid(values, grid))
        self.assertAlmostEqual(numerical, 0.0, delta=1.0e-10)

    def test_profile_center_is_twice_the_scale(self) -> None:
        self.assertAlmostEqual(profile(0.0), 2.0, delta=TOLERANCE)


class CurvatureSpectrumTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.table_path = CASE_DIR / "data" / "curvature_ccdf.tsv"
        cls.x, cls.ccdf = np.loadtxt(cls.table_path, unpack=True)

    def test_spectrum_table_contract(self) -> None:
        self.assertTrue(self.table_path.is_file())
        self.assertGreaterEqual(self.x.size, 2)
        self.assertTrue(np.all(self.x > 0.0))
        self.assertTrue(np.all(np.diff(self.x) > 0.0))
        self.assertTrue(np.all((self.ccdf > 0.0) & (self.ccdf <= 1.0)))
        self.assertTrue(np.all(np.diff(self.ccdf) < 0.0))

    def test_inverse_ccdf_round_trip(self) -> None:
        probabilities = np.geomspace(self.ccdf[-1], self.ccdf[0], 2000)
        inverse = np.exp(
            np.interp(
                np.log(probabilities),
                np.log(self.ccdf[::-1]),
                np.log(self.x[::-1]),
            )
        )
        reconstructed = np.exp(
            np.interp(np.log(inverse), np.log(self.x), np.log(self.ccdf))
        )
        np.testing.assert_allclose(reconstructed, probabilities, rtol=2.0e-12)

    def test_spectrum_upper_tail_has_no_point_mass(self) -> None:
        tail_slope = (
            np.log(self.ccdf[-1]) - np.log(self.ccdf[-2])
        ) / (self.x[-1] - self.x[-2])
        self.assertLess(tail_slope, 0.0)


if __name__ == "__main__":
    unittest.main()
