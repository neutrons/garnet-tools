from garnet.reduction.plan import ReductionPlan
from garnet.reduction.parametrization import Parametrization


def test_get_file():
    file = "/tmp/test.nxs"

    rp = ReductionPlan()
    rp.generate_plan("TOPAZ")
    rp.plan["OutputName"] = "test"

    data = Parametrization(rp.plan).get_file(file, ws="")

    base = "/tmp/test"
    app = "_(h,k,0)_[0,0,l]_[-10,10]_[-10,10]_[-10,10]_201x201x201"
    log = "_temperature_[5,100]x21"
    ext = ".nxs"

    assert data == base + app + log + ext

    data = Parametrization(rp.plan).get_file(file, ws="data")

    assert data == base + app + log + "_data" + ext


def test_get_file_miller_index_no_log_elastic():
    file = "/tmp/test.nxs"

    rp = ReductionPlan()
    rp.generate_plan("TOPAZ")
    rp.plan["OutputName"] = "test"
    rp.plan["Elastic"] = True
    rp.plan["Parametrization"]["MillerIndex"] = [1, 2, 3]
    rp.plan["Parametrization"]["LogBins"] = 0

    data = Parametrization(rp.plan).get_file(file, ws="")

    base = "/tmp/test"
    app = "_(1,2,3)_[-10,10]_[-10,10]_[-10,10]_201x201x201"
    cc = "_cc"
    ext = ".nxs"

    assert data == base + app + cc + ext
