import numpy as np
import pytest

from garnet.reduction.plan import ReductionPlan
from garnet.reduction.integration import Integration


def test_get_file():
    file = "/tmp/test.nxs"

    rp = ReductionPlan()
    rp.generate_plan("TOPAZ")
    rp.plan["OutputName"] = "test"

    data = Integration(rp.plan).get_file(file, ws="")

    base = "/tmp/test"
    app = "_Triclinic_P_d(min)=1.75_r(max)=0.20"
    ext = ".nxs"

    assert data == base + app + ext

    data = Integration(rp.plan).get_file(file, ws="data")

    assert data == base + app + "_data" + ext


def test_get_file_modulated():
    file = "/tmp/test.nxs"

    rp = ReductionPlan()
    rp.generate_plan("TOPAZ")
    rp.plan["OutputName"] = "test"
    rp.plan["Integration"]["Cell"] = "Cubic"
    rp.plan["Integration"]["Centering"] = "I"
    rp.plan["Integration"]["MinD"] = 0.5
    rp.plan["Integration"]["Radius"] = 0.3
    rp.plan["Integration"]["MaxOrder"] = 1
    rp.plan["Integration"]["ModVec1"] = [0, 0, 0.5]
    rp.plan["Integration"]["CrossTerms"] = True

    data = Integration(rp.plan).get_file(file, ws="")

    base = "/tmp/test"
    app = "_Cubic_I_(0,0,0.5)_mix_d(min)=0.50_r(max)=0.30"
    ext = ".nxs"

    assert data == base + app + ext


def test_quick_integration_accepts_satellites():
    rp = ReductionPlan()
    rp.generate_plan("TOPAZ")
    rp.plan["OutputName"] = "test"
    rp.plan["Integration"]["ProfileFit"] = False
    rp.plan["Integration"]["MaxOrder"] = 1
    rp.plan["Integration"]["ModVec1"] = [0, 0, 0.5]

    Integration(rp.plan)


def test_radial_fit_is_not_an_option():
    rp = ReductionPlan()
    rp.generate_plan("TOPAZ")
    assert "RadialFit" not in rp.plan["Integration"]

    # a plan written before it was removed still loads, either way
    rp.plan["OutputName"] = "test"
    for profile_fit in [False, True]:
        rp.plan["Integration"]["ProfileFit"] = profile_fit
        rp.plan["Integration"]["RadialFit"] = True
        Integration(rp.plan)


def _peaks_with_hkls(name, hkls, mnps=None):
    from mantid.simpleapi import CreatePeaksWorkspace, SetUB, mtd
    from mantid.kernel import V3D

    CreatePeaksWorkspace(
        NumberOfPeaks=0, OutputType="LeanElasticPeak", OutputWorkspace=name
    )
    SetUB(Workspace=name, a=5.431, b=5.431, c=5.431)
    ws = mtd[name]
    for i, hkl in enumerate(hkls):
        peak = ws.createPeakHKL([float(v) for v in hkl])
        peak.setIntHKL(V3D(*hkl))
        if mnps is not None:
            peak.setIntMNP(V3D(*mnps[i]))
        ws.addPeak(peak)
    return ws


def _silicon_plan(twin_laws=None):
    rp = ReductionPlan()
    rp.generate_plan("TOPAZ")
    rp.plan["OutputName"] = "test"
    rp.plan["Integration"]["ProfileFit"] = False
    rp.plan["Material"] = {
        "ChemicalFormula": "Si1",
        "ZParameter": 2,
        "SpaceGroup": "Fd-3m",
        "Sites": [["Si1", 0.0, 0.0, 0.0, 1.0, 0.005]],
    }
    if twin_laws is not None:
        rp.plan["Material"]["TwinLaws"] = twin_laws
    return rp.plan


def test_expected_structure_factors():
    hkls = [[1, 1, 1], [2, 0, 0], [2, 2, 0], [1, 1, 1], [0, 0, 0]]
    mnps = [[0, 0, 0]] * 3 + [[1, 0, 0], [0, 0, 0]]
    _peaks_with_hkls("sf_peaks", hkls, mnps)

    F2 = Integration(_silicon_plan()).expected_structure_factors("sf_peaks")
    # 111 and 220 allowed, 200 forbidden by the diamond glide; the
    # satellite and the unindexed peak are taken as absent
    assert F2[0] > 100 and F2[2] > 100
    assert F2[1] == pytest.approx(0, abs=1e-6)
    assert F2[3] == 0 and F2[4] == 0

    # a second domain putting 222, also forbidden, at each 111 peak halves it
    twinned = Integration(
        _silicon_plan([[[2, 0, 0], [0, 2, 0], [0, 0, 2]]])
    ).expected_structure_factors("sf_peaks")
    assert twinned[0] == pytest.approx(F2[0] / 2, rel=1e-6)

    rp = ReductionPlan()
    rp.generate_plan("TOPAZ")
    rp.plan["OutputName"] = "test"
    assert Integration(rp.plan).expected_structure_factors("sf_peaks") is None


def test_structure_factor_prior_scales():
    import numpy as np

    from garnet.reduction.integration import _MIN_PRIOR_SCALE

    hkls = [[1, 1, 1], [2, 2, 0], [3, 1, 1], [4, 0, 0], [2, 0, 0], [1, 1, 1]]
    mnps = [[0, 0, 0]] * 5 + [[1, 0, 0]]
    _peaks_with_hkls("ps_peaks", hkls, mnps)

    integration = Integration(_silicon_plan())
    F2 = integration.expected_structure_factors("ps_peaks")
    scales = integration.structure_factor_prior_scales("ps_peaks")

    # the forbidden 200 and the satellite are held tightest
    assert scales[4] == _MIN_PRIOR_SCALE and scales[5] == _MIN_PRIOR_SCALE
    # stronger reflections are held less tightly, the strongest not at all
    allowed = np.arange(4)
    order = np.argsort(F2[allowed])
    assert np.all(np.diff(scales[allowed][order]) >= 0)
    assert scales[allowed].max() == 1

    rp = ReductionPlan()
    rp.generate_plan("TOPAZ")
    rp.plan["OutputName"] = "test"
    assert (
        Integration(rp.plan).structure_factor_prior_scales("ps_peaks") is None
    )


def _radial_result(I, sigma):
    return {"intensity": I, "sigma": sigma, "coverage": 0.9, "offset": 0.01}


def test_radial_info():
    info = Integration.radial_info(_radial_result(10.0, 2.0))
    assert info == {
        "I_rad": 10.0,
        "s_rad": 2.0,
        "rad_cov": 0.9,
        "rad_off": 0.01,
    }

    # no fit (no resolution model) or a failed one log NaN, not nothing
    for result in [None, _radial_result(np.nan, 1.0), _radial_result(1, 0)]:
        info = Integration.radial_info(result)
        assert sorted(info) == ["I_rad", "rad_cov", "rad_off", "s_rad"]
        assert np.all(np.isnan(list(info.values())))


def _count_info(I):
    return {
        "d3x": 0.0,
        "bkg": 0.0,
        "bkg_err": 0.0,
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
    }


def _log_radial_run(name, run, hkls, I_counts, results):
    from garnet.reduction.peaks import PeakModel

    ws = _peaks_with_hkls(name, hkls)
    for i in range(ws.getNumberPeaks()):
        ws.getPeak(i).setRunNumber(run)

    peak = PeakModel(name)
    for i, (I, result) in enumerate(zip(I_counts, results)):
        info = _count_info(I)
        info.update(Integration.radial_info(result))
        peak.add_diagnostic_info(i, info)


def test_radial_logs_stay_aligned_over_runs():
    from mantid.simpleapi import mtd
    from garnet.reduction.peaks import PeaksModel

    hkls = [[1, 1, 1], [2, 2, 0], [3, 1, 1]]
    # a run with radial fits, one of them failed, and a run without a
    # resolution model, whose peaks have no fits
    _log_radial_run(
        "rad_a",
        1,
        hkls,
        [100.0, 50.0, 25.0],
        [_radial_result(110, 5), _radial_result(np.nan, 1), None],
    )
    _log_radial_run("rad_b", 2, hkls[:2], [80.0, 40.0], [None, None])

    model = PeaksModel()
    model.combine_peaks("rad_a", "rad_merge")
    model.combine_peaks("rad_b", "rad_merge")

    run_info = mtd["rad_merge"].run()
    run = run_info.getLogData("peaks_run").value
    h = run_info.getLogData("peaks_h").value
    I = run_info.getLogData("peaks_intens_raw").value
    I_rad = run_info.getLogData("peaks_I_rad").value
    s_rad = run_info.getLogData("peaks_s_rad").value

    assert len(I_rad) == len(s_rad) == len(I) == len(run) == 5
    assert list(run) == [1, 1, 1, 2, 2]
    assert list(h) == [1, 2, 3, 1, 2]
    assert list(I) == [100, 50, 25, 80, 40]
    assert I_rad[0] == 110 and s_rad[0] == 5
    assert np.all(np.isnan(I_rad[1:]))


def test_profile_fit_logs_radial_fits():
    from mantid.simpleapi import mtd

    hkls = [[1, 1, 1], [2, 2, 0], [3, 1, 1]]
    _peaks_with_hkls("prof_rad", hkls)

    rp = ReductionPlan()
    rp.generate_plan("TOPAZ")
    rp.plan["OutputName"] = "test"
    integration = Integration(rp.plan)
    integration.cntrt = 1.0

    shape = [0.0, 0.0, 0.0, 0.1, 0.1, 0.1, [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    results = {
        0: (100.0, 10.0, shape, {"I_fit": 100.0}, hkls[0]),
        1: None,
        2: (25.0, 5.0, shape, {"I_fit": 25.0}, hkls[2]),
    }
    radial = [_radial_result(110, 5), _radial_result(60, 6), None]
    integration.update_peak_info("prof_rad", results, radial=radial)

    run_info = mtd["prof_rad"].run()
    I_fit = run_info.getLogData("peaks_I_fit").value
    I_rad = run_info.getLogData("peaks_I_rad").value

    # the unfitted peak logs nothing, so the radial fits stay aligned
    assert list(I_fit) == [100, 25]
    assert I_rad[0] == 110 and np.isnan(I_rad[1])
