import itertools

import numpy as np

from mantid.simpleapi import (
    CloneWorkspace,
    CreatePeaksWorkspace,
    SaveIsawUB,
    SetUB,
    mtd,
)
from mantid.kernel import V3D

from garnet.utilities.reflections import Peaks


def test_remove_powder_shells(tmp_path):
    """A ring raising the background of (220) equivalents removes those it changed."""
    rng = np.random.default_rng(1)

    CreatePeaksWorkspace(
        NumberOfPeaks=0, OutputType="LeanElasticPeak", OutputWorkspace="ps"
    )
    SetUB(Workspace="ps", a=5.431, b=5.431, c=5.431)
    ws = mtd["ps"]

    hkls = [
        hkl
        for hkl in itertools.product(range(-4, 5), repeat=3)
        if any(hkl) and np.linalg.norm(hkl) <= 4
    ]
    for i, hkl in enumerate(hkls):
        peak = ws.createPeakHKL([float(v) for v in hkl])
        peak.setIntHKL(V3D(*hkl))
        peak.setRunNumber(1 + i % 2)
        d = 5.431 / np.linalg.norm(hkl)
        peak.setWavelength(rng.uniform(0.5, 1.6) * d)
        peak.setIntensity(100.0)
        peak.setSigmaIntensity(5.0)
        ws.addPeak(peak)

    # a smooth background in wavelength and angle, and a ring at d(220) in the regions of six
    # of its twelve equivalents, three of whose intensities it changed
    ring = [
        (2, 2, 0),
        (-2, 0, 2),
        (0, 2, -2),
        (2, -2, 0),
        (0, -2, -2),
        (2, 0, 2),
    ]
    changed = ring[:3]

    logs = {key: [] for key in ["run", "h", "k", "l", "m", "n", "p", "bkg"]}
    for peak in ws:
        hkl = tuple(int(v) for v in peak.getIntHKL())
        two_sin = 2 * np.sin(peak.getScattering() / 2)
        bkg = np.sqrt(peak.getWavelength()) / two_sin
        bkg *= np.exp(0.1 * rng.standard_normal())
        if hkl in ring:
            bkg *= np.e
        if hkl in changed:
            peak.setIntensity(300.0)
        for key, val in zip(logs, [peak.getRunNumber(), *hkl, 0, 0, 0, bkg]):
            logs[key].append(val)
    for key, val in logs.items():
        ws.run()["peaks_" + key] = val

    peaks = Peaks("ps", str(tmp_path / "ps.nxs"), point_group="m-3m")
    peaks.remove_powder_shells()

    removed = sorted(
        tuple(int(v) for v in peak.getIntHKL())
        for peak in ws
        if peak.getSigmaIntensity() == float("-inf")
    )
    assert removed == sorted(changed)
    assert (tmp_path / "ps_powder.pdf").exists()


def test_load_consistent_UB_after_transform(tmp_path):
    """A saved UB is adopted only if it indexes the peaks in their current setting."""
    CreatePeaksWorkspace(
        NumberOfPeaks=0, OutputType="LeanElasticPeak", OutputWorkspace="ub"
    )
    SetUB(Workspace="ub", a=6.9, b=6.9, c=19.8, gamma=120)
    ws = mtd["ub"]
    for hkl in [(1, 0, 1), (0, 1, 2), (1, 1, 3), (2, -1, 1), (0, 0, 3), (1, -2, 4)]:
        peak = ws.createPeakHKL([float(v) for v in hkl])
        peak.setIntHKL(V3D(*hkl))
        ws.addPeak(peak)

    original = tmp_path / "original.mat"
    SaveIsawUB(InputWorkspace="ub", Filename=str(original))

    T = [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
    peaks = Peaks("ub", str(tmp_path / "ub.nxs"), HKLTransform=T)
    peaks.transform_hkl()
    UB_transformed = ws.sample().getOrientedLattice().getUB().copy()

    peaks.load_consistent_UB(str(original))
    assert np.allclose(ws.sample().getOrientedLattice().getUB(), UB_transformed)

    CloneWorkspace(InputWorkspace="ub", OutputWorkspace="ub_refined")
    SetUB(Workspace="ub_refined", UB=(1.002 * UB_transformed).flatten().tolist())
    refined = tmp_path / "refined.mat"
    SaveIsawUB(InputWorkspace="ub_refined", Filename=str(refined))

    peaks.load_consistent_UB(str(refined))
    assert np.allclose(
        ws.sample().getOrientedLattice().getUB(), 1.002 * UB_transformed
    )


def test_radial_comparison_from_logs(tmp_path):
    """The radial fit logged with each peak is tabulated with its count."""
    from garnet.reduction.peaks import PeakModel

    CreatePeaksWorkspace(
        NumberOfPeaks=0, OutputType="LeanElasticPeak", OutputWorkspace="rc"
    )
    SetUB(Workspace="rc", a=5.431, b=5.431, c=5.431)
    ws = mtd["rc"]
    hkls = [[1, 1, 1], [2, 2, 0], [3, 1, 1]]
    for hkl in hkls:
        peak = ws.createPeakHKL([float(v) for v in hkl])
        peak.setIntHKL(V3D(*hkl))
        peak.setRunNumber(7)
        ws.addPeak(peak)

    I_counts = [100.0, 50.0, 25.0]
    I_rads = [110.0, np.nan, 20.0]
    model = PeakModel("rc")
    for i, (I, I_rad) in enumerate(zip(I_counts, I_rads)):
        info = {
            "d3x": 0.0,
            "n_vox": 0,
            "vol_frac": 1.0,
            "intens_raw": I,
            "sig_raw": np.sqrt(I),
            "pk_data": I,
            "pk_norm": 1.0,
            "bkg_data": 0.0,
            "bkg_norm": 1.0,
            "ratio": 1.0,
            "cntrt": 1.0,
            "I_rad": I_rad,
            "s_rad": 2.0,
            "rad_cov": 0.95,
            "rad_off": 0.0,
        }
        model.add_diagnostic_info(i, info)

    # the logs are keyed in the setting the peaks are transformed to
    T = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
    peaks = Peaks("rc", str(tmp_path / "rc.nxs"), HKLTransform=T)
    peaks.info_dict, peaks.norm_dict, _ = peaks.peak_info()

    assert (7, 1, 1, -1, 0, 0, 0) in peaks.info_dict
    assert (7, 2, 2, 0, 0, 0, 0) in peaks.info_dict
    assert peaks.info_dict[(7, 1, 3, -1, 0, 0, 0)]["I_rad"] == 20.0

    table = tmp_path / "rc_radial.txt"
    peaks.write_radial_comparison(str(table))

    lines = table.read_text().splitlines()
    # header and the two peaks with a fit, the failed one left out
    assert len(lines) == 3
    rows = [line.split() for line in lines[1:]]
    ratios = {tuple(map(int, row[1:4])): float(row[11]) for row in rows}
    assert ratios == {(1, 1, -1): 1.1, (1, 3, -1): 0.8}
