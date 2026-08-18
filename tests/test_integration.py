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
