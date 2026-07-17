import os

# import pytest
# import tempfile
import shutil

# import subprocess
import cProfile

from garnet.config.instruments import beamlines
from garnet.reduction.plan import ReductionPlan
from garnet.reduction.peaks import PeaksModel
from garnet.reduction.data import DataModel
from garnet.reduction.integration import Integration

# benchmark = 'shared/benchmark'

config_file = "/SNS/CORELLI/shared/benchmark/test/CORELLI_plan.yaml"

# rp = ReductionPlan()
# rp.load_plan(config_file)

# data_ws = '/SNS/CORELLI/shared/benchmark/test/CORELLI_data.nxs'
# peaks_ws = '/SNS/CORELLI/shared/benchmark/test/CORELLI_peaks.nxs'

# plots = '/SNS/CORELLI/shared/benchmark/test/CORELLI_plan_integration/CORELLI_plan_Hexagonal_P_d(min)=0.70_r(max)=0.20_plots/'

# if os.path.exists(plots):
#     shutil.rmtree(plots)
# os.mkdir(plots)

# data = DataModel(beamlines['CORELLI'])
# data.load_histograms(data_ws, 'md')

# peaks = PeaksModel()
# peaks.load_peaks(peaks_ws, 'peaks')

# params = [0.1, 0]

# integrate = Integration(rp.plan)
# integrate.data = data
# integrate.peaks = peaks
# integrate.run = 0
# integrate.runs = 1
# peak_dict = integrate.extract_peak_info('peaks', params)
# cProfile.run("integrate.integrate_peaks(peak_dict)", 'profile.stats')


# @pytest.mark.skipif(not os.path.exists('/SNS/CORELLI/'), reason='file mount')
# def test_corelli():

#     config_file = 'corelli_reduction_plan.yaml'
#     reduction_plan = os.path.abspath(os.path.join('./tests/data', config_file))
#     script = os.path.abspath('./src/garnet/workflow.py')
#     command = ['python', script, config_file, 'int', '16']

#     with tempfile.TemporaryDirectory() as tmpdir:

#         os.chdir(tmpdir)

#         rp = ReductionPlan()
#         rp.load_plan(reduction_plan)
#         rp.save_plan(os.path.join(tmpdir, config_file))

#         instrument_config = beamlines[rp.plan['Instrument']]
#         facility = instrument_config['Facility']
#         name = instrument_config['Name']
#         baseline_path = os.path.join('/', facility, name, benchmark)

#         subprocess.run(command)

#         if os.path.exists(baseline_path):
#             shutil.rmtree(baseline_path)

#         shutil.copytree(tmpdir, baseline_path)

# @pytest.mark.skipif(not os.path.exists('/HFIR/HB2C/'), reason='file mount')
# def test_wand2():

#     config_file = 'wand2_reduction_plan.yaml'
#     reduction_plan = os.path.abspath(os.path.join('./tests/data', config_file))
#     script = os.path.abspath('./src/garnet/workflow.py')
#     command = ['python', script, config_file, 'int', '4']

#     with tempfile.TemporaryDirectory() as tmpdir:

#         os.chdir(tmpdir)

#         rp = ReductionPlan()
#         rp.load_plan(reduction_plan)
#         rp.save_plan(os.path.join(tmpdir, config_file))

#         instrument_config = beamlines[rp.plan['Instrument']]
#         facility = instrument_config['Facility']
#         name = instrument_config['Name']
#         baseline_path = os.path.join('/', facility, name, benchmark)

#         subprocess.run(command)

#         if os.path.exists(baseline_path):
#             shutil.rmtree(baseline_path)

#         shutil.copytree(tmpdir, baseline_path)

# @pytest.mark.skipif(not os.path.exists('/HFIR/HB3A/'), reason='file mount')
# def test_demand():

#     config_file = 'demand_reduction_plan.yaml'
#     reduction_plan = os.path.abspath(os.path.join('./tests/data', config_file))
#     script = os.path.abspath('./src/garnet/workflow.py')
#     command = ['python', script, config_file, 'int', '4']

#     with tempfile.TemporaryDirectory() as tmpdir:

#         os.chdir(tmpdir)

#         rp = ReductionPlan()
#         rp.load_plan(reduction_plan)
#         rp.save_plan(os.path.join(tmpdir, config_file))

#         instrument_config = beamlines[rp.plan['Instrument']]
#         facility = instrument_config['Facility']
#         name = instrument_config['Name']
#         baseline_path = os.path.join('/', facility, name, benchmark)

#         subprocess.run(command)

#         if os.path.exists(baseline_path):
#             shutil.rmtree(baseline_path)

#         shutil.copytree(tmpdir, baseline_path)

# def test_sphere():

#     r_cut = 0.25

#     A = 1.2
#     s = 0.1

#     r = np.linspace(0, r_cut, 51)

#     I = A*np.tanh((r/s)**3)

#     sphere = PeakSphere(r_cut)

#     radius = sphere.fit(r, I)

#     assert np.tanh((radius/s)**3) > 0.95
#     assert radius < r_cut
