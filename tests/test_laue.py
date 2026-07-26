import unittest
import numpy as np

from mantid.simpleapi import CreatePeaksWorkspace, SetUB
from mantid.kernel import V3D
from mantid.api import AlgorithmManager

import garnet.reduction.laue  # noqa: F401


def rotation_matrix_from_axis_angle(axis, angle_rad):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c

    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def direct_basis_from_lattice(a, b, c, alpha, beta, gamma):
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sg = np.sin(gamma)
    volume_factor = 1.0 + 2.0 * ca * cb * cg - ca**2 - cb**2 - cg**2
    return np.array(
        [
            [a, b * cg, c * cb],
            [0.0, b * sg, c * (ca - cb * cg) / sg],
            [0.0, 0.0, c * np.sqrt(volume_factor) / sg],
        ],
        dtype=float,
    )


def reciprocal_basis_from_lattice(a, b, c, alpha, beta, gamma):
    A = direct_basis_from_lattice(a, b, c, alpha, beta, gamma)
    return np.linalg.inv(A).T


class FindUBFromLauePeaksTest(unittest.TestCase):
    def _make_synthetic_laue_workspace(self, UB, hkls, wavelengths):
        ws = CreatePeaksWorkspace(
            NumberOfPeaks=0, OutputType="LeanElasticPeak"
        )
        SetUB(Workspace=ws, UB=UB)

        for (h, k, l), wavelength in zip(hkls, wavelengths):
            peak = ws.createPeakHKL(V3D(float(h), float(k), float(l)))
            peak.setWavelength(float(wavelength))
            ws.addPeak(peak)

        return ws

    def _run_case(
        self,
        a,
        b,
        c,
        alpha_deg,
        beta_deg,
        gamma_deg,
        axis,
        angle_deg,
        seed,
        max_hkl=6,
        n_reflections=300,
        wl_min=2.0,
        wl_max=3.9,
        min_frac_indexed=0.6,
        max_wavelength_error=0.05,
    ):
        alpha = np.deg2rad(alpha_deg)
        beta = np.deg2rad(beta_deg)
        gamma = np.deg2rad(gamma_deg)

        B = reciprocal_basis_from_lattice(a, b, c, alpha, beta, gamma)
        U = rotation_matrix_from_axis_angle(axis, np.deg2rad(angle_deg))
        UB_true = U @ B

        rng = np.random.default_rng(seed)
        hkls = rng.integers(-max_hkl, max_hkl + 1, size=(n_reflections, 3))
        hkls = hkls[np.any(hkls != 0, axis=1)]
        # keep only primitive indices: a composite (h, k, l) is just a
        # higher harmonic of some primitive direction, and whether that
        # harmonic or another is "the" true one for a given wavelength
        # band is a genuine physical ambiguity this algorithm can't
        # resolve without extra information (e.g. intensity) -- not
        # something a synthetic test should be graded on.
        primitive = np.gcd.reduce(np.abs(hkls), axis=1) == 1
        hkls = hkls[primitive]
        wavelengths = rng.uniform(wl_min, wl_max, size=len(hkls))

        ws = self._make_synthetic_laue_workspace(UB_true, hkls, wavelengths)

        alg = AlgorithmManager.create("FindUBFromLauePeaks")
        alg.initialize()
        alg.setProperty("PeaksWorkspace", ws)
        alg.setProperty("a", a)
        alg.setProperty("b", b)
        alg.setProperty("c", c)
        alg.setProperty("alpha", alpha_deg)
        alg.setProperty("beta", beta_deg)
        alg.setProperty("gamma", gamma_deg)
        alg.setProperty("Centering", "P")
        alg.setProperty("WavelengthMin", wl_min)
        alg.setProperty("WavelengthMax", wl_max)
        alg.setProperty("MaxZoneIndex", max_hkl)
        alg.setProperty("MaxHklIndex", max_hkl)
        alg.execute()

        ws_out = alg.getProperty("PeaksWorkspace").value
        ol = ws_out.sample().getOrientedLattice()

        self.assertAlmostEqual(ol.a(), a, delta=0.01 * a)
        self.assertAlmostEqual(ol.b(), b, delta=0.01 * b)
        self.assertAlmostEqual(ol.c(), c, delta=0.01 * c)
        self.assertAlmostEqual(ol.alpha(), alpha_deg, delta=1.0)
        self.assertAlmostEqual(ol.beta(), beta_deg, delta=1.0)
        self.assertAlmostEqual(ol.gamma(), gamma_deg, delta=1.0)

        n_total = ws_out.getNumberPeaks()
        n_indexed = 0
        wavelength_errors = []
        for i in range(n_total):
            pk = ws_out.getPeak(i)
            hkl = np.array([pk.getH(), pk.getK(), pk.getL()])
            if np.any(hkl != 0):
                n_indexed += 1
                # wavelength is symmetry-agnostic: whatever Miller index
                # label the recovered (symmetry-equivalent) UB assigns,
                # the implied wavelength for the SAME physical peak must
                # still match the true wavelength it was created with.
                wavelength_errors.append(
                    abs(pk.getWavelength() - wavelengths[i])
                )

        frac_indexed = n_indexed / n_total
        self.assertGreater(frac_indexed, min_frac_indexed)
        self.assertLess(max(wavelength_errors), max_wavelength_error)

    def test_cubic(self):
        self._run_case(
            8.0, 8.0, 8.0, 90.0, 90.0, 90.0, [1, 2, 3], 37.0, seed=1
        )

    def test_orthorhombic(self):
        self._run_case(
            6.0, 7.5, 9.2, 90.0, 90.0, 90.0, [0.3, 1, -2], 61.0, seed=2
        )

    def test_monoclinic(self):
        self._run_case(
            6.0, 10.0, 6.0, 90.0, 108.0, 90.0, [1, -1, 2], 15.0, seed=3
        )

    def test_triclinic(self):
        self._run_case(
            5.5, 6.3, 7.1, 80.0, 95.0, 100.0, [2, -1, 1], 44.0, seed=4
        )


if __name__ == "__main__":
    unittest.main()
