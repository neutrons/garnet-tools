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


def test_radial_fit_requires_quick_integration():
    rp = ReductionPlan()
    rp.generate_plan("TOPAZ")
    rp.plan["OutputName"] = "test"
    rp.plan["Integration"]["ProfileFit"] = False
    rp.plan["Integration"]["RadialFit"] = True

    Integration(rp.plan)

    rp.plan["Integration"]["ProfileFit"] = True
    with pytest.raises(Exception, match="RadialFit"):
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
