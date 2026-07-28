#!/usr/bin/env python3
"""Reference checks for the pgen-local polar-cap QED formulas."""

from __future__ import annotations

import math
import re
import tomllib
import unittest
from pathlib import Path

import numpy as np


CASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_ROOT = CASE_DIR.parents[1]


class PolarCapReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.x, cls.ccdf = np.loadtxt(
            CASE_DIR / "data" / "curvature_ccdf.tsv", unpack=True
        )
        with (CASE_DIR / "1d_polar_cap.toml").open("rb") as stream:
            cls.config = tomllib.load(stream)
        setup = cls.config["setup"]
        cls.polar_cap = {
            key.removeprefix("polar_cap."): value
            for key, value in setup.items()
            if key.startswith("polar_cap.")
        }
        cls.qed = {
            key.removeprefix("qed."): value
            for key, value in cls.polar_cap.items()
            if key.startswith("qed.")
        }
        cls.initial_injection = {
            key.removeprefix("initial_injection."): value
            for key, value in cls.polar_cap.items()
            if key.startswith("initial_injection.")
        }

    def test_spectrum_table_contract(self) -> None:
        self.assertGreaterEqual(self.x.size, 2)
        self.assertTrue(np.all(self.x > 0.0))
        self.assertTrue(np.all(np.diff(self.x) > 0.0))
        self.assertTrue(np.all((self.ccdf > 0.0) & (self.ccdf <= 1.0)))
        self.assertTrue(np.all(np.diff(self.ccdf) < 0.0))

    def test_qed_on_scaled_parameter_card(self) -> None:
        with (CASE_DIR / "qed_on_degj5dx_j1p5_test.toml").open("rb") as stream:
            config = tomllib.load(stream)
        setup = config["setup"]
        qed = {
            key.removeprefix("polar_cap.qed."): value
            for key, value in setup.items()
            if key.startswith("polar_cap.qed.")
        }
        species = config["particles"]["species"]

        self.assertTrue(qed["enable"])
        self.assertTrue(qed["curvature_drag"])
        self.assertTrue(qed["curvature_emission"])
        self.assertTrue(qed["magnetic_pair_creation"])
        self.assertFalse(setup["polar_cap.radiation_reaction.enable"])
        self.assertEqual(len(species), 3)
        self.assertEqual(
            [item["label"] for item in species], ["e-", "e+", "gamma"]
        )
        self.assertEqual(species[0]["emission"], "custom")
        self.assertEqual(species[1]["emission"], "custom")
        self.assertEqual(species[2]["pusher"], "Photon")
        self.assertEqual(species[2]["n_payloads_real"], 3)
        self.assertEqual(
            config["output"]["fields"]["quantities"],
            ["N_1", "N_2", "N_3", "E", "B", "J"],
        )

    def test_qed_on_scaled_energy_hierarchy(self) -> None:
        with (CASE_DIR / "qed_on_degj5dx_j1p5_test.toml").open("rb") as stream:
            config = tomllib.load(stream)
        setup = config["setup"]
        gamma_emit = setup["polar_cap.qed.gamma_emit"]
        gamma_rad = setup["polar_cap.qed.gamma_rad"]
        extent = config["grid"]["extent"][0]
        skin_depth = config["scales"]["skindepth0"]
        gamma_pc = 0.5 * ((extent[1] - extent[0]) / skin_depth) ** 2

        self.assertEqual(gamma_rad / gamma_emit, 10.0)
        self.assertEqual(gamma_pc / gamma_rad, 50.0)
        self.assertEqual(gamma_pc, 8.0e6)

    def test_qed_drag_uses_gamma_rad_not_macro_particle_charge(self) -> None:
        source = (CASE_DIR / "pgen.hpp").read_text()
        drag_block = source.split(", drag_step_coefficient {", 1)[1].split(
            ", opacity_prefactor {", 1
        )[0]
        self.assertIn("setup.polar_cap.qed.gamma_rad", source)
        self.assertIn("setup.polar_cap.qed.reference_electric_field", source)
        self.assertIn("scales.omegaB0", drag_block)
        self.assertNotIn("scales.q0", drag_block)
        self.assertRegex(drag_block, re.compile(r"SQR\(SQR\(gamma_rad\)\)"))

    def test_qed_drag_step_coefficient_matches_parameter_card(self) -> None:
        with (CASE_DIR / "qed_on_degj5dx_j1p5_test.toml").open("rb") as stream:
            config = tomllib.load(stream)
        setup = config["setup"]
        extent = config["grid"]["extent"][0]
        dx = (extent[1] - extent[0]) / config["grid"]["resolution"][0]
        dt = config["algorithms"]["timestep"]["CFL"] * dx
        omega_b = 1.0 / config["scales"]["larmor0"]
        coefficient = (
            dt
            * omega_b
            * setup["polar_cap.qed.reference_electric_field"]
            / setup["polar_cap.qed.gamma_rad"] ** 4
        )
        self.assertAlmostEqual(coefficient, 3.0517578125e-19, places=30)

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
            math.log(self.ccdf[-1]) - math.log(self.ccdf[-2])
        ) / (self.x[-1] - self.x[-2])
        self.assertLess(tail_slope, 0.0)

        probability = self.ccdf[-1] * 1.0e-6
        inverse = self.x[-1] + (
            math.log(probability) - math.log(self.ccdf[-1])
        ) / tail_slope
        self.assertGreater(inverse, self.x[-1])
        reconstructed = math.exp(
            math.log(self.ccdf[-1]) + tail_slope * (inverse - self.x[-1])
        )
        self.assertAlmostEqual(reconstructed / probability, 1.0, places=12)

    def test_default_spectrum_never_indexes_empty_device_views(self) -> None:
        spectrum_source = (
            CASE_DIR / "qed" / "curvature_spectrum.hpp"
        ).read_text()
        self.assertIn("CurvatureSpectrum() = default;", spectrum_source)
        self.assertRegex(
            spectrum_source,
            re.compile(
                r"ccdf\(real_t value\).*?m_size == 0.*?return ONE;"
                r".*?m_x\(0\)",
                re.DOTALL,
            ),
        )
        self.assertRegex(
            spectrum_source,
            re.compile(
                r"inverse_ccdf\(real_t probability\).*?m_size == 0"
                r".*?return ZERO;.*?m_ccdf\(0\)",
                re.DOTALL,
            ),
        )

    def test_truncated_spectrum_respects_parent_kinetic_energy(self) -> None:
        qed = self.qed
        gamma = 1.0e6
        energy_scale = (gamma / qed["gamma_emit"]) ** 3 / qed["rho_c"]
        energy_min = qed["photon_energy_min"]
        energy_max = gamma - 1.0
        x_min = energy_min / energy_scale
        x_max = energy_max / energy_scale
        self.assertGreater(x_max, x_min)
        self.assertGreaterEqual(x_min, self.x[0])
        self.assertLessEqual(x_max, self.x[-1])

        probability_at_min = math.exp(
            np.interp(math.log(x_min), np.log(self.x), np.log(self.ccdf))
        )
        probability_at_max = math.exp(
            np.interp(math.log(x_max), np.log(self.x), np.log(self.ccdf))
        )
        probabilities = np.linspace(
            probability_at_max, probability_at_min, 20_000
        )
        normalized_energy = np.exp(
            np.interp(
                np.log(probabilities),
                np.log(self.ccdf[::-1]),
                np.log(self.x[::-1]),
            )
        )
        photon_energy = normalized_energy * energy_scale
        photon_energy = np.clip(photon_energy, energy_min, energy_max)
        self.assertGreaterEqual(float(np.min(photon_energy)), energy_min)
        self.assertLessEqual(float(np.max(photon_energy)), energy_max)

    def test_opacity_is_nonnegative_and_angle_symmetric(self) -> None:
        qed = self.qed
        b_over_bq = qed["b_over_bq"]
        exponent = 8.0 / (3.0 * b_over_bq)
        energy = 10.0

        def attenuation(theta: float) -> float:
            sine = abs(math.sin(theta))
            if energy * sine < 2.0:
                return 0.0
            return sine * math.exp(-exponent / (energy * sine))

        for theta in np.linspace(0.0, 1.2, 100):
            self.assertGreaterEqual(attenuation(float(theta)), 0.0)
            self.assertAlmostEqual(
                attenuation(float(theta)), attenuation(float(-theta)), places=15
            )

    def test_photon_initial_angle_includes_transverse_momentum(self) -> None:
        for direction in (
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0),
            np.array([-0.5, 0.5, math.sqrt(0.5)]),
        ):
            theta = math.atan2(
                math.hypot(float(direction[1]), float(direction[2])),
                abs(float(direction[0])),
            )
            self.assertGreaterEqual(theta, 0.0)
            self.assertLessEqual(theta, 0.5 * math.pi)
            self.assertAlmostEqual(abs(math.sin(theta)), math.hypot(
                float(direction[1]), float(direction[2])
            ))

    def test_photon_initial_angle_avoids_two_argument_norm_macro(self) -> None:
        emission_source = (
            CASE_DIR / "qed" / "curvature_emission.hpp"
        ).read_text()
        self.assertNotRegex(
            emission_source,
            re.compile(r"NORM\s*\(\s*direction\[1\]\s*,\s*direction\[2\]\s*\)"),
        )
        self.assertRegex(
            emission_source,
            re.compile(
                r"transverse_direction\s*=\s*math::sqrt\s*\(\s*"
                r"SQR\(direction\[1\]\)\s*\+\s*SQR\(direction\[2\]\)"
            ),
        )

    def test_device_clamp_does_not_pass_zero_to_kokkos_max(self) -> None:
        emission_source = (
            CASE_DIR / "qed" / "curvature_emission.hpp"
        ).read_text()
        self.assertNotRegex(
            emission_source,
            re.compile(r"math::max\s*\([^;]*ccdf_at_max_energy[^;]*ZERO"),
        )
        self.assertIn(
            "ccdf_interval > static_cast<real_t>(0.0)",
            emission_source,
        )

    def test_opacity_uses_trajectory_length_but_signed_x1_angle(self) -> None:
        dt = 0.2
        rho_c = 1.5
        trajectory_length = dt
        self.assertEqual(trajectory_length, dt)

        angle_increments = []
        for direction_cosine in (0.5, -0.5):
            field_line_distance = dt * direction_cosine
            angle_increment = field_line_distance / rho_c
            angle_increments.append(angle_increment)
            self.assertAlmostEqual(
                angle_increment,
                dt * direction_cosine / rho_c,
            )
            self.assertGreater(trajectory_length, abs(field_line_distance))

        self.assertAlmostEqual(angle_increments[0], -angle_increments[1])

        opacity_source = (
            CASE_DIR / "qed" / "photon_opacity.hpp"
        ).read_text()
        self.assertRegex(
            opacity_source,
            re.compile(
                r"field_line_distance\s*=\s*context\.dt\s*\*\s*"
                r"particles\.ux1\(p\)\s*/\s*u_norm"
            ),
        )
        self.assertNotRegex(
            opacity_source,
            re.compile(
                r"field_line_distance\s*=\s*context\.dt[^;]*"
                r"abs\s*\(\s*particles\.ux1\(p\)"
            ),
        )

    def test_pair_energy_closure(self) -> None:
        for photon_energy in (2.0, 3.0, 10.0, 1.0e4):
            gamma_pair = 0.5 * photon_energy
            momentum = math.sqrt(gamma_pair * gamma_pair - 1.0)
            reconstructed = 2.0 * math.sqrt(1.0 + momentum * momentum)
            self.assertAlmostEqual(reconstructed, photon_energy, places=11)

    def test_pair_threshold_roundoff_is_clamped_before_sqrt(self) -> None:
        pair_source = (
            CASE_DIR / "qed" / "magnetic_pair_creation.hpp"
        ).read_text()
        self.assertRegex(
            pair_source,
            re.compile(
                r"u_magnitude\s*=\s*math::sqrt\s*\(\s*"
                r"math::max\s*\(\s*ZERO\s*,\s*"
                r"SQR\(gamma_pair\)\s*-\s*ONE\s*\)"
            ),
        )

    def test_composite_simpson_opacity_against_dense_reference(self) -> None:
        qed = self.qed
        energy = 10.0
        theta0 = 0.35
        path_length = 0.2
        rho_c = qed["rho_c"]
        substeps = qed["opacity_substeps"]
        exponent = 8.0 / (3.0 * qed["b_over_bq"])

        def attenuation(theta: np.ndarray | float) -> np.ndarray:
            sine = np.abs(np.sin(theta))
            result = np.zeros_like(sine, dtype=float)
            active = energy * sine >= 2.0
            result[active] = sine[active] * np.exp(
                -exponent / (energy * sine[active])
            )
            return result

        theta1 = theta0 + path_length / rho_c
        sample = np.linspace(theta0, theta1, substeps + 1)
        values = attenuation(sample)
        ds = path_length / substeps
        simpson = ds / 3.0 * (
            values[0]
            + values[-1]
            + 4.0 * np.sum(values[1:-1:2])
            + 2.0 * np.sum(values[2:-1:2])
        )

        dense_theta = np.linspace(theta0, theta1, 200_001)
        dense_ds = path_length / (dense_theta.size - 1)
        reference = dense_ds * (
            0.5 * attenuation(dense_theta[0])
            + np.sum(attenuation(dense_theta[1:-1]))
            + 0.5 * attenuation(dense_theta[-1])
        )
        self.assertAlmostEqual(simpson / reference, 1.0, places=5)

    def test_stochastic_rounding_is_unbiased_above_ppc_floor(self) -> None:
        expected = 2.25
        floor = math.floor(expected)
        samples = floor + (
            np.random.default_rng(20260723).random(200_000) < expected - floor
        )
        self.assertAlmostEqual(float(np.mean(samples)), expected, delta=3.0e-3)

    def test_low_density_injection_uses_fixed_ppc_and_weight(self) -> None:
        injection = self.initial_injection
        reference_ppc = self.config["particles"]["ppc0"]
        minimum_ppc = injection["minimum_ppc"]
        minimum_density = injection["minimum_density"]
        self.assertGreater(minimum_ppc, 0)
        self.assertGreaterEqual(minimum_density, 0.0)

        for target_density in (
            minimum_density,
            2.0 * minimum_density,
            0.5 / reference_ppc,
        ):
            target_ppc = reference_ppc * target_density
            self.assertLess(target_ppc, minimum_ppc)
            count = minimum_ppc
            weight = target_ppc / count
            self.assertAlmostEqual(count * weight, target_ppc)

        below_cutoff = 0.5 * minimum_density
        represented_ppc = (
            0.0 if below_cutoff < minimum_density else reference_ppc * below_cutoff
        )
        self.assertEqual(represented_ppc, 0.0)

    def test_cartesian_atmosphere_uses_global_weight_contract(self) -> None:
        self.assertTrue(self.config["particles"]["use_weights"])

        atmosphere_source = (
            SOURCE_ROOT / "src" / "engines" / "srpic" / "particles_bcs.h"
        ).read_text()
        self.assertRegex(
            atmosphere_source,
            re.compile(
                r"const auto use_weights\s*=\s*"
                r'params\.template get<bool>\("particles\.use_weights"\);'
            ),
        )

        pgen_source = (CASE_DIR / "pgen.hpp").read_text()
        neutral_start = pgen_source.index("arch::InjectNonUniform")
        neutral_end = pgen_source.index("const auto extra_charge", neutral_start)
        neutral_injection = pgen_source[neutral_start:neutral_end]
        self.assertRegex(
            neutral_injection,
            re.compile(r"atmosphere,\s*ONE,\s*true\);"),
        )

    def test_initial_field_gauss_closure_with_prescribed_background(self) -> None:
        polar_cap = self.polar_cap
        coefficient = polar_cap["initial_e_coefficient"]
        positron_peak = polar_cap["extra_positron_density"]
        self.assertEqual(positron_peak, coefficient)

        width = self.config["grid"]["boundaries"]["atmosphere"]["ds"]
        transition = 0.03 * width
        x_surface = 0.0
        x = np.linspace(x_surface, x_surface + 2.0 * width, 10_001)
        atmosphere_edge = x_surface + width
        s = (x - atmosphere_edge) / transition
        positron_profile = np.ones_like(x)
        transition_cells = (x > atmosphere_edge) & (s < 11.0)
        positron_profile[transition_cells] = (
            1.01 / (1.0 + 0.01 * np.exp(s[transition_cells]))
        )
        positron_profile[s >= 11.0] = 0.0

        self.assertEqual(positron_profile[x <= atmosphere_edge].min(), 1.0)
        self.assertEqual(positron_profile[s >= 11.0].max(), 0.0)
        particle_charge = positron_peak * positron_profile
        prescribed_background = -coefficient
        gauss_rhs = particle_charge + prescribed_background
        d_e_dx = coefficient * (positron_profile - 1.0)
        np.testing.assert_allclose(gauss_rhs, d_e_dx, atol=2.0e-15)

    def test_external_current_is_ppc_and_resolution_independent(self) -> None:
        """Reproduce Entity's 1D Minkowski Ampere normalization."""
        current_tetrad = self.polar_cap["external_current"]
        skindepth0 = self.config["scales"]["skindepth0"]
        larmor0 = self.config["scales"]["larmor0"]
        expected_field_rate = (
            -larmor0 / skindepth0**2 * current_tetrad
        )

        for dx in (1.0e-4, 2.0e-4):
            volume0 = dx
            current_contravariant = current_tetrad / dx
            for ppc0 in (5.0, 50.0, 500.0):
                q0 = volume0 / (ppc0 * skindepth0**2)
                b0 = 1.0 / larmor0
                ampere_coefficient = -q0 / (b0 * volume0)
                raw_external_current = ppc0 * current_contravariant

                output_current_tetrad = (
                    dx * raw_external_current / ppc0
                )
                output_field_rate_tetrad = (
                    dx * raw_external_current * ampere_coefficient
                )
                self.assertAlmostEqual(
                    output_current_tetrad, current_tetrad
                )
                self.assertAlmostEqual(
                    output_field_rate_tetrad, expected_field_rate
                )

        pgen_source = (CASE_DIR / "pgen.hpp").read_text()
        self.assertRegex(
            pgen_source,
            re.compile(
                r"current_x1_contravariant\s*\{\s*"
                r"current_x1_tetrad\s*/\s*dx\s*\}"
            ),
        )
        self.assertIn('"scales.dx0"', pgen_source)

    def test_qed_off_toml_has_no_photon_allocation(self) -> None:
        species = self.config["particles"]["species"]
        self.assertFalse(self.qed["enable"])
        self.assertEqual([item["label"] for item in species], ["e-", "e+"])
        self.assertEqual(species[0]["emission"], "none")
        self.assertEqual(species[1]["emission"], "none")
        self.assertNotIn("radiative_drag", species[0])
        self.assertNotIn("radiative_drag", species[1])
        self.assertNotIn(
            "N_3", self.config["output"]["fields"]["quantities"]
        )

    def test_pgen_species_contract_is_conditional_on_qed(self) -> None:
        pgen_source = (CASE_DIR / "pgen.hpp").read_text()
        self.assertIn(
            "QED-enabled polar-cap requires exactly three species",
            pgen_source,
        )
        self.assertIn(
            "QED-off polar-cap requires exactly two species and no photon allocation",
            pgen_source,
        )
        self.assertIn(
            "QED-off electron and positron species must use emission=none",
            pgen_source,
        )
        self.assertRegex(
            pgen_source,
            re.compile(
                r"curvature_spectrum\s*\{\s*"
                r".*?qed_enabled\s*\?\s*polar_cap::CurvatureSpectrum",
                re.DOTALL,
            ),
        )

    def test_qed_modes_and_table_path_contract(self) -> None:
        qed = self.qed
        self.assertFalse(qed["enable"])
        for key in (
            "enable",
            "curvature_drag",
            "curvature_emission",
            "magnetic_pair_creation",
        ):
            self.assertIn(key, qed)
            self.assertIsInstance(qed[key], bool)

        table_path = Path(qed["spectrum_table"])
        self.assertFalse(table_path.is_absolute())
        self.assertTrue((CASE_DIR / table_path).is_file())
        self.assertGreater(qed["max_photons_per_particle_step"], 0)
        self.assertGreater(qed["opacity_substeps"], 0)
        self.assertEqual(qed["opacity_substeps"] % 2, 0)

    def test_setup_uses_entity_flat_parameter_contract(self) -> None:
        setup = self.config["setup"]
        self.assertTrue(setup)
        self.assertTrue(all(not isinstance(value, dict) for value in setup.values()))
        self.assertIn("polar_cap.temperature", setup)
        self.assertIn("polar_cap.initial_injection.minimum_ppc", setup)
        self.assertIn("polar_cap.qed.enable", setup)

    def test_pgen_requires_b0_and_validates_host_coefficients(self) -> None:
        pgen_source = (CASE_DIR / "pgen.hpp").read_text()
        self.assertIn(
            'params.template get<real_t>("setup.polar_cap.B0")',
            pgen_source,
        )
        self.assertNotRegex(
            pgen_source,
            re.compile(r'get<real_t>\("setup\.polar_cap\.B0"\s*,'),
        )
        self.assertIn("temperature must be non-negative", pgen_source)
        for coefficient in (
            "emission_step_coefficient",
            "drag_step_coefficient",
            "opacity_prefactor",
        ):
            self.assertIn(
                f"not std::isfinite({coefficient})",
                pgen_source,
            )

    def test_output_smoothing_matches_third_order_particle_shape(self) -> None:
        smoothing = self.config["output"]["fields"]["smoothing"]
        self.assertEqual(smoothing["method"], "spline")
        self.assertEqual(smoothing["order"], 3)


class RadiationReactionBalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (CASE_DIR / "radiation_reaction_balance.toml").open("rb") as stream:
            cls.config = tomllib.load(stream)
        setup = cls.config["setup"]
        cls.rr = {
            key.removeprefix("polar_cap.radiation_reaction."): value
            for key, value in setup.items()
            if key.startswith("polar_cap.radiation_reaction.")
        }

    def test_qed_is_off_and_no_photon_species_exists(self) -> None:
        setup = self.config["setup"]
        self.assertFalse(setup["polar_cap.qed.enable"])
        self.assertNotIn("polar_cap.qed.b_over_bq", setup)
        species = self.config["particles"]["species"]
        self.assertEqual([item["label"] for item in species], ["e-", "e+"])
        self.assertTrue(all(item["emission"] == "custom" for item in species))

    def test_qed_off_rr_constructor_does_not_require_qed_parameters(self) -> None:
        pgen_source = (CASE_DIR / "pgen.hpp").read_text()
        for parameter, fallback in (
            ("rho_c", "ONE"),
            ("gamma_emit", r"static_cast<real_t>\(2\.0\)"),
            ("photon_energy_min", "ONE"),
            ("b_over_bq", "ONE"),
            ("max_photons_per_particle_step", "1"),
        ):
            self.assertRegex(
                pgen_source,
                re.compile(
                    rf"qed_enabled.*?get(?:<[^>]+>)?\s*\(\s*"
                    rf'"setup\.polar_cap\.qed\.{parameter}"\)'
                    rf".*?:\s*{fallback}",
                    re.DOTALL,
                ),
            )

    def test_super_gj_current_is_1p5(self) -> None:
        self.assertEqual(self.config["setup"]["polar_cap.external_current"], 1.5)

    def test_degj_and_gamma_pc_scaling(self) -> None:
        resolution = self.config["grid"]["resolution"][0]
        xmin, xmax = self.config["grid"]["extent"][0]
        dx = (xmax - xmin) / resolution
        d_e_gj = self.config["scales"]["skindepth0"]
        self.assertAlmostEqual(d_e_gj / dx, 5.0)
        gamma_pc = 0.5 * ((xmax - xmin) / d_e_gj) ** 2
        self.assertAlmostEqual(gamma_pc, 8.0e6)

    def test_explicit_drag_balances_at_gamma_rad(self) -> None:
        resolution = self.config["grid"]["resolution"][0]
        xmin, xmax = self.config["grid"]["extent"][0]
        dx = (xmax - xmin) / resolution
        dt = self.config["algorithms"]["timestep"]["CFL"] * dx
        omega_b0 = 1.0 / self.config["scales"]["larmor0"]
        gamma_rad = self.rr["gamma_rad"]
        e_ref = self.rr["reference_electric_field"]
        drag_step_coefficient = dt * omega_b0 * e_ref / gamma_rad**4
        acceleration_per_step = dt * omega_b0 * e_ref
        drag_per_step_at_balance = drag_step_coefficient * gamma_rad**4
        self.assertAlmostEqual(
            drag_per_step_at_balance / acceleration_per_step,
            1.0,
            places=14,
        )
        self.assertAlmostEqual(drag_step_coefficient, 3.0517578125e-19)

    def test_explicit_drag_source_is_ppc_independent(self) -> None:
        source = (CASE_DIR / "pgen.hpp").read_text()
        drag_block = source.split(", drag_step_coefficient {", 1)[1].split(
            ", opacity_prefactor {", 1
        )[0]
        self.assertIn("radiation_reaction_enabled", drag_block)
        self.assertIn("scales.omegaB0", drag_block)
        self.assertIn("reference_electric_field", drag_block)
        self.assertIn("gamma_rad", drag_block)
        self.assertNotIn("scales.q0", drag_block)


if __name__ == "__main__":
    unittest.main()
