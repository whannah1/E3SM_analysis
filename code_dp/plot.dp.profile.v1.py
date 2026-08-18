import os, glob, copy, xarray as xr, numpy as np, numba
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import hapy; host = hapy.get_host()
#---------------------------------------------------------------------------------------------------
case,case_opts_list = [],[]
def add_case(case_in,**kwargs):
    case.append(case_in)
    case_opts = {}
    for k, val in kwargs.items(): case_opts[k] = val
    if 'n' not in case_opts: case_opts['n'] = case_in
    case_opts_list.append(case_opts)
#-------------------------------------------------------------------------------
var,var_opts_list = [],[]
def add_var(var_name,**kwargs):
    var.append(var_name)
    var_opts = {}
    for k, val in kwargs.items(): var_opts[k] = val
    if 'str' not in var_opts: var_opts['str'] = var_name
    var_opts_list.append(var_opts)
#-------------------------------------------------------------------------------
P0    = 101325       # Pa        surface pressure
Rd    = 287.058      # J/kg/K    gas constant for dry air
cp    = 1004.64      # J/kg/K    specific heat for dry air
Lv    = 2.5104e6     # J / kg    latent heat of vaporization / evaporation
g     = 9.80665      # m/s       global average of gravity at MSLP
#-------------------------------------------------------------------------------
host = None
if os.path.exists('/global/cfs/cdirs'): host = 'nersc'
if os.path.exists('/lustre/orion'):     host = 'olcf'
#-------------------------------------------------------------------------------
if host=='nersc':
    scratch_cpu = '/pscratch/sd/w/whannah/scream_scratch/pm-cpu'
    scratch_gpu = '/pscratch/sd/w/whannah/scream_scratch/pm-gpu'

    # add_case('DP.2026-RCE-01.NN_01.ne_022.lx_200.dt_60.L128_cntrl',ne=22,lx=200,c='black',d=1,n='RCE cntrl lx=200 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_01.ne_022.lx_200.dt_60.L128_pbias',ne=22,lx=200,c='red',  d=1,n='RCE pbias lx=200 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_01.ne_022.lx_200.dt_60.L128_tbias',ne=22,lx=200,c='blue', d=1,n='RCE tbias lx=200 dx=3km', p=scratch_gpu,s='run')

    # add_case('DP.2026-RCE-01.NN_04.ne_067.lx_600.dt_60.L128_cntrl',ne=67,lx=600,c='black',d=0,n='RCE cntrl lx=600 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_04.ne_067.lx_600.dt_60.L128_pbias',ne=67,lx=600,c='red',  d=0,n='RCE pbias lx=600 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_04.ne_067.lx_600.dt_60.L128_tbias',ne=67,lx=600,c='blue', d=0,n='RCE tbias lx=600 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_04.ne_067.lx_600.dt_60.L128_cntrl.qi2qc_1',ne=67,lx=600,c='magenta', d=0,n='RCE qi2qc lx=600 dx=3km', p=scratch_gpu,s='run')

    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1',        ne=67,lx=600,c='black', d=0,n='GATE control 600/3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_pbias',  ne=67,lx=600,c='blue',  d=0,n='GATE p-bias 600/3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha2', ne=67,lx=600,c='green', d=0,n='GATE alpha2 600/3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha3', ne=67,lx=600,c='green', d=0,n='GATE alpha3 600/3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1.qi2qc_1',ne=67,lx=600,c='green', d=0,n='GATE qi2qc 600/3km', p=scratch_gpu,s='run')


    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_300', n='a=1 pm=300', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_400', n='a=1 pm=400', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_500', n='a=1 pm=500', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_600', n='a=1 pm=600', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_700', n='a=1 pm=700', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_800', n='a=1 pm=800', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_900', n='a=1 pm=900', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_990', n='a=1 pm=990', p=scratch_gpu, s='run')

    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_300',   n='a=1 pm=300',   c='black',   d=0, p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_500',   n='a=1 pm=500',   c='red',   d=0, p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_700',   n='a=1 pm=700',   c='green', d=0, p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_900',   n='a=1 pm=900',   c='blue',  d=0, p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1.5_pm_300', n='a=1.5 pm=300', c='red',   d=1, p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_2.0_pm_300', n='a=2.0 pm=300', c='green', d=1, p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_2.5_pm_300', n='a=2.5 pm=300', c='blue',  d=1, p=scratch_gpu, s='run')

    # htype = 'output.scream.1D.1hr.AVERAGE.nhours_x1'
    # first_file,num_files = -10,None


    # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_0',        n='dx=12 dt=300 no-zm', clr='red',  ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.mvgr_3', n='dx=12 dt=300 w/ zm',clr='blue', ls='solid', p=scratch_gpu,s='run')
    # htype = 'output.1D.1hr.AVERAGE.nhours_x1'
    # first_file,num_files = 10,None

    # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_04.ne67.lx_600km.dt_75.enable_zm_0',         n='dx= 3 dt= 75 no-zm',clr='red',    ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_02.ne34.lx_600km.dt_150.enable_zm_0',        n='dx= 6 dt=150 no-zm',clr='orange', ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_0',        n='dx=12 dt=300 no-zm',clr='green',  ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_0',         n='dx=22 dt=600 no-zm',clr='blue',   ls='solid', p=scratch_gpu,s='run')
    # # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_01.ne5.lx_600km.dt_1200.enable_zm_0',        n='dx=40 dt=1200 no-zm',clr='purple', ls='solid', p=scratch_gpu,s='run')

    # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_04.ne67.lx_600km.dt_75.enable_zm_1.mvgr_3',  n='dx= 3 dt= 75 mvgr=3',clr='red',    ls='dashed', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_02.ne34.lx_600km.dt_150.enable_zm_1.mvgr_3', n='dx= 6 dt=150 mvgr=3',clr='orange', ls='dashed', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.mvgr_3', n='dx=12 dt=300 mvgr=3',clr='green',  ls='dashed', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_3',  n='dx=22 dt=600 mvgr=3',clr='blue',   ls='dashed', p=scratch_gpu,s='run')
    
    # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_02.ne34.lx_600km.dt_75.enable_zm_1.mvgr_3',  n='dx= 6 dt= 75 mvgr=3',clr='orange', ls='dashdot', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_01.ne17.lx_600km.dt_75.enable_zm_1.mvgr_3',  n='dx=12 dt= 75 mvgr=3',clr='green',  ls='dashdot', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_01.ne9.lx_600km.dt_75.enable_zm_1.mvgr_3',   n='dx=22 dt= 75 mvgr=3',clr='blue',   ls='dashed', p=scratch_gpu,s='run')

    
    
    # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_3',  n='dx=22 dt=600 mvgr=3',clr='red',    ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_01.ne9.lx_600km.dt_75.enable_zm_1.mvgr_3',   n='dx=22 dt= 75 mvgr=3',clr='blue',   ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-00.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_2.mamsc_8',  n='dx=22 dt=600 mvgr=3 mamsc=8',clr='green',   ls='dashdot', p=scratch_gpu,s='run')
    # add_case('arch_GPU.DP.2026-GATE-ZM-01.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_3',  n='dx=22 dt=600 mvgr=3',clr='magenta',ls='dashed', p=scratch_gpu,s='run')
    # add_case('arch_GPU.DP.2026-GATE-ZM-01.NN_01.ne9.lx_600km.dt_75.enable_zm_1.mvgr_3',   n='dx=22 dt= 75 mvgr=3',clr='cyan',   ls='dashed', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-02.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_3',  n='dx=22 dt=600 mvgr=3',clr='magenta',ls='dashed', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-02.arch_GPU.NN_01.ne9.lx_600km.dt_75.enable_zm_1.mvgr_3',   n='dx=22 dt= 75 mvgr=3',clr='cyan',   ls='dashed', p=scratch_gpu,s='run')

    # add_case('DP.2026-GATE-ZM-02.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_4',  n='dx=22 dt=600 mvgr=3',clr='orange', ls='dashed', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-02.arch_GPU.NN_01.ne9.lx_600km.dt_75.enable_zm_1.mvgr_4',   n='dx=22 dt= 75 mvgr=3',clr='purple', ls='dashed', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-02.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_5',  n='dx=22 dt=600 mvgr=3',clr='yellow', ls='dashed', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-02.arch_GPU.NN_01.ne9.lx_600km.dt_75.enable_zm_1.mvgr_5',   n='dx=22 dt= 75 mvgr=3',clr='pink',   ls='dashed', p=scratch_gpu,s='run')

    # add_case('DP.2026-GATE-ZM-03.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_3',  n='dx=22 dt=600 mvgr=3',clr='magenta',ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-03.arch_GPU.NN_01.ne9.lx_600km.dt_75.enable_zm_1.mvgr_3',   n='dx=22 dt= 75 mvgr=3',clr='cyan',   ls='solid', p=scratch_gpu,s='run')

    # add_case('DP.2026-GATE-ZM-04.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_3',  n='dx=22 dt=600 mvgr=3',clr='magenta',ls='dashed', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-04.arch_GPU.NN_01.ne9.lx_600km.dt_75.enable_zm_1.mvgr_3',   n='dx=22 dt= 75 mvgr=3',clr='cyan',   ls='dashed', p=scratch_gpu,s='run')

    # add_case('DP.2026-GATE-ZM-05.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_3',  n='dx=22 dt=600 mvgr=3',clr='magenta',ls='dotted', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-05.arch_GPU.NN_01.ne9.lx_600km.dt_75.enable_zm_1.mvgr_3',   n='dx=22 dt= 75 mvgr=3',clr='cyan',   ls='dotted', p=scratch_gpu,s='run')

    # add_case('DP.2026-GATE-ZM-06.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_3',  n='dx=22 dt=600 mvgr=3'  ,clr='red',      ls='solid', p=scratch_gpu,s='run')
    # # add_case('DP.2026-GATE-ZM-06.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_10', n='dx=22 dt=600 mvgr=10' ,clr='orange',   ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-06.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_100',n='dx=22 dt=600 mvgr=100',clr='magenta',  ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-06.arch_GPU.NN_01.ne9.lx_600km.dt_75.enable_zm_1.mvgr_3',   n='dx=22 dt= 75 mvgr=3'  ,clr='blue',     ls='solid', p=scratch_gpu,s='run')
    # # add_case('DP.2026-GATE-ZM-06.arch_GPU.NN_01.ne9.lx_600km.dt_75.enable_zm_1.mvgr_10',  n='dx=22 dt= 75 mvgr=10' ,clr='purple',   ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-ZM-06.arch_GPU.NN_01.ne9.lx_600km.dt_75.enable_zm_1.mvgr_100', n='dx=22 dt= 75 mvgr=100',clr='cyan',     ls='solid', p=scratch_gpu,s='run')

    # add_case('DP.2026-GATE-ZM-06a.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_3',  n='dx=22 dt=600 mvgr=3'  ,clr='red',      ls='solid', p=scratch_gpu,s='run')

    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_04.ne67.lx_600km.dt_75.enable_zm_0',        n='dx= 3 km no-zm', clr='red',    ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_02.ne34.lx_600km.dt_150.enable_zm_0',       n='dx= 6 km no-zm', clr='green',  ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_0',       n='dx=12 km no-zm', clr='blue',   ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_0',        n='dx=22 km no-zm', clr='purple', ls='solid', p=scratch_gpu,s='run')
    
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_04.ne67.lx_600km.dt_75.enable_zm_1',        n='dx= 3 km zm-ctl',clr='red',    ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_02.ne34.lx_600km.dt_150.enable_zm_1',       n='dx= 6 km zm-ctl',clr='green',  ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1',       n='dx=12 km zm-ctl',clr='blue',   ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1',        n='dx=22 km zm-ctl',clr='purple', ls='solid', p=scratch_gpu,s='run')
    
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_04.ne67.lx_600km.dt_75.enable_zm_1.mcta_3', n='dx= 3 km mcta=3',clr='red',    ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_02.ne34.lx_600km.dt_150.enable_zm_1.mcta_3',n='dx= 6 km mcta=3',clr='green',  ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.mcta_3',n='dx=12 km mcta=3',clr='blue',   ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mcta_3', n='dx=22 km mcta=3',clr='purple', ls='solid', p=scratch_gpu,s='run')

    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_04.ne67.lx_600km.dt_75.enable_zm_1.mcta_2', n='dx= 3 km mcta=2',clr='red',    ls='dotted', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_02.ne34.lx_600km.dt_150.enable_zm_1.mcta_2',n='dx= 6 km mcta=2',clr='green',  ls='dotted', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.mcta_2',n='dx=12 km mcta=2',clr='blue',   ls='dotted', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mcta_2', n='dx=22 km mcta=2',clr='purple', ls='dotted', p=scratch_gpu,s='run')

    ### DCAPE / CAPE threshold sensitivity runs
    
    # # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_0',                n='dx=12km no-zm',             clr='black',   ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_0.dct_0',    n='dx=12km cape=0 dcape=0.0',  clr='red',     ls='solid', p=scratch_gpu,s='run')
    # # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_0.dct_0.001',n='dx=12km cape=0 dcape=0.001',clr='orange',  ls='solid', p=scratch_gpu,s='run')
    # # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_0.dct_0.005',n='dx=12km cape=0 dcape=0.005',clr='green',   ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_0.dct_0.01', n='dx=12km cape=0 dcape=0.01', clr='orange',    ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_0.dct_0.05', n='dx=12km cape=0 dcape=0.05', clr='green',    ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_0.dct_0.1',  n='dx=12km cape=0 dcape=0.1',  clr='blue',  ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_0.dct_0.5',  n='dx=12km cape=0 dcape=0.5',  clr='purple',    ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_0.dct_1',    n='dx=12km cape=0 dcape=1.0',  clr='magenta', ls='solid', p=scratch_gpu,s='run')

    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_0.dct_0',    n='dx=12km cape=0',    clr='black',     ls='solid', p=scratch_gpu,s='run')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_10.dct_0',   n='dx=12km cape=10',   clr='darkred',   ls='solid', p=scratch_gpu,s='run')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_50.dct_0',   n='dx=12km cape=50',   clr='red',       ls='solid', p=scratch_gpu,s='run')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_100.dct_0',  n='dx=12km cape=100',  clr='orange',    ls='solid', p=scratch_gpu,s='run')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_200.dct_0',  n='dx=12km cape=200',  clr='yellow',    ls='solid', p=scratch_gpu,s='run')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_300.dct_0',  n='dx=12km cape=300',  clr='lightgreen',ls='solid', p=scratch_gpu,s='run')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_400.dct_0',  n='dx=12km cape=400',  clr='green',     ls='solid', p=scratch_gpu,s='run')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_500.dct_0',  n='dx=12km cape=500',  clr='cyan',      ls='solid', p=scratch_gpu,s='run')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_600.dct_0',  n='dx=12km cape=600',  clr='blue',      ls='solid', p=scratch_gpu,s='run')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_700.dct_0',  n='dx=12km cape=700',  clr='purple',    ls='solid', p=scratch_gpu,s='run')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_800.dct_0',  n='dx=12km cape=800',  clr='pink',      ls='solid', p=scratch_gpu,s='run')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.cpt_900.dct_0',  n='dx=12km cape=900',  clr='magenta',   ls='solid', p=scratch_gpu,s='run')

    

    htype = 'output.1D.1hr.AVERAGE.nhours_x1'
    first_file,num_files = 20,20



    # # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_0',           n='RCE 5m control',     c='black',  d=0,  p=scratch_gpu, s='run')
    # # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_100',  n='RCE 5m zm mvgr:100', c='red',    d=0, p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_10',   n='RCE 5m zm mvgr: 10', c='red',    d=0, p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_6',    n='RCE 5m zm mvgr:  6', c='orange', d=0, p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_4',    n='RCE 5m zm mvgr:  4', c='green',  d=0, p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_2',    n='RCE 5m zm mvgr:  2', c='blue',   d=0, p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_1',    n='RCE 5m zm mvgr:  1', c='purple', d=0, p=scratch_gpu, s='run')

    # # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_0',            n='RCE 1m control',     c='red',  d=1,  p=scratch_gpu, s='run')
    # # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_100',   n='RCE 1m zm mvgr:100', c='red',    d=1, p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_10',    n='RCE 1m zm mvgr: 10', c='red',    d=1, p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_6',     n='RCE 1m zm mvgr:  6', c='orange', d=1, p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_4',     n='RCE 1m zm mvgr:  4', c='green',  d=1, p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_2',     n='RCE 1m zm mvgr:  2', c='blue',   d=1, p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_1',     n='RCE 1m zm mvgr:  1', c='purple', d=1, p=scratch_gpu, s='run')


    # htype = 'output.scream.1D.1hr.AVERAGE.nhours_x1'
    # first_file,num_files = 20,None
    # # first_file,num_files = 1,4

    # ### prototype v4 vertical grid testing
    # add_case('DP.2026-GATE-IDEAL-04.NN_04.ne67.lx_600km.dt_60',           n='GATE control', clr='black',  p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-04.NN_04.ne67.lx_600km.dt_60.L128v3.4',  n='GATE L128v3.4',clr='magenta',p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-04.NN_04.ne67.lx_600km.dt_60.L128v3.5',  n='GATE L128v3.5',clr='green',  p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-04.NN_04.ne67.lx_600km.dt_60.L128v3.6',  n='GATE L128v3.6',clr='cyan',   p=scratch_gpu,s='run')
    # htype = 'output.scream.1D.1hr.AVERAGE.nhours_x1'
    # first_file,num_files = 20,20

    # ### prototype v4 vertical grid testing
    # add_case('DP.2026-DYCOMS-SRF1-00.NN_04.ne10.lx_100km.dt_100',         n='DYCOMS control', clr='black',  p=scratch_gpu,s='run')
    # add_case('DP.2026-DYCOMS-SRF1-00.NN_04.ne10.lx_100km.dt_100.L128v3.4',n='DYCOMS L128v3.4',clr='magenta',p=scratch_gpu,s='run')
    # add_case('DP.2026-DYCOMS-SRF1-00.NN_04.ne10.lx_100km.dt_100.L128v3.5',n='DYCOMS L128v3.5',clr='green',  p=scratch_gpu,s='run')
    # add_case('DP.2026-DYCOMS-SRF1-00.NN_04.ne10.lx_100km.dt_100.L128v3.6',n='DYCOMS L128v3.6',clr='cyan',   p=scratch_gpu,s='run')
    # htype = 'output.scream.1D.1hr.AVERAGE.nhours_x1'
    # height_interp_mx_km, height_interp_dz_km = 2,10e-3
    # pressure_min = 800.
    # # add_var('sgs_buoy_flux')
    # # add_var('pbl_height')
    # # add_var('inv_qc_relvar')
    # # add_var('eddy_diff_heat')
    # # add_var('w_variance')
    # # add_var('cldfrac_liq_prev')
    # # add_var('ustar')
    # # add_var('obklen')
    # # add_var('tau_est')
    # # add_var('shoc_mix')
    # add_var('shoc_cond')
    # add_var('shoc_evap')
    # # add_var('wthl_sec')
    # # add_var('wqw_sec')
    # # add_var('qw_sec')
    # # add_var('uw_sec')
    # # add_var('vw_sec')
    # # add_var('w3')
    # # add_var('vapor_flux')
    # # add_var('water_flux')
    # # add_var('heat_flux')

    # ### prototype v4 vertical grid testing
    # add_case('DP.2026-MAGIC-00.NN_04.ne10.lx_100km.dt_100',         n='MAGIC control', clr='black',  p=scratch_gpu,s='run')
    # add_case('DP.2026-MAGIC-00.NN_04.ne10.lx_100km.dt_100.L128v3.4',n='MAGIC L128v3.4',clr='magenta',p=scratch_gpu,s='run')
    # add_case('DP.2026-MAGIC-00.NN_04.ne10.lx_100km.dt_100.L128v3.5',n='MAGIC L128v3.5',clr='green',  p=scratch_gpu,s='run')
    # add_case('DP.2026-MAGIC-00.NN_04.ne10.lx_100km.dt_100.L128v3.6',n='MAGIC L128v3.6',clr='cyan',   p=scratch_gpu,s='run')
    # htype = 'output.1D.1hr.AVERAGE.nhours_x1'
    # # first_file,num_files = 0,1; title = 'MAGIC day 1'
    # # first_file,num_files = 1,1; title = 'MAGIC day 2'
    # first_file,num_files = 2,1; title = 'MAGIC day 3'
    

#-------------------------------------------------------------------------------

# add_var('dz_mid',             str='dz')
# add_var('S')
# add_var('MSE')
# add_var('T_mid')
# add_var('z_mid')
# add_var('omega')
# add_var('dS/dp',              str='-dS/dp')
# add_var('RelativeHumidity')
# add_var('qv')
# add_var('qc')
# add_var('qr')
# add_var('qi')

# add_var('cldfrac_liq')

# add_var('total_T_mid_tend')
# add_var('total_qv_tend')

add_var('zm_T_mid_tend')
add_var('shoc_T_mid_tend')
add_var('p3_T_mid_tend')

# add_var('zm_qv_tend')
# add_var('shoc_qv_tend')
# add_var('p3_qv_tend')

# add_var('zm_qv_tend')
# add_var('zm_dlf')


# add_var('T_mid',              str='T_mid variance', method='variance')
# add_var('T_mid',              str='T_mid variance in space', method='variance_space')
# add_var('T_mid',              str='T_mid variance in time', method='variance_time')

# add_var('T_mid',              str='T_mid variance', method='variance')
# add_var('qv',                 str='qv variance',    method='variance')
# add_var('qc',                 str='qc variance',    method='variance')
# add_var('qi',                 str='qi variance',    method='variance')
# add_var('qr',                 str='qr variance',    method='variance')

#-------------------------------------------------------------------------------
fig_file = 'figs_dp/dp.profile.v1.png'

print_stat = False
# skip_first_twelve_hours = False
use_height_coord = False
if 'height_interp_mx_km' not in locals():
    height_interp_mx_km,height_interp_dz_km = 18, 100e-3
    # height_interp_mx_km, height_interp_dz_km = 5,10e-3
    # height_interp_mx_km, height_interp_dz_km = 2,10e-3
if 'pressure_min' not in locals():
    pressure_min = 100.


num_plot_col = len(var)
# num_plot_col = 2

#---------------------------------------------------------------------------------------------------
num_case, num_var = len(case), len(var)

if 'first_file' not in globals(): first_file = None
if 'num_files'  not in globals(): num_files = None

clr = [None]*num_case
for c in range(num_case): clr[c] = case_opts_list[c].get('clr', None)
clr = hapy.fill_color_list(clr)
#---------------------------------------------------------------------------------------------------
def get_lev_dim(da):
    # check for lev dimension
    valid_lev_dims = ['lev','ilev']
    if not any([d in da.dims for d in valid_lev_dims]):
        print(); print(da); print()
        raise ValueError(f'variable {tvar} does not appear to have a valid vertical dimension?')
    else:
        lev_dim_name = None
        for d in valid_lev_dims:
            if d in da.dims:
                lev_dim_name = d
    return lev_dim_name
#---------------------------------------------------------------------------------------------------
def get_data(ds,var):
    tvar = var
    if var=='horiz_winds_u': tvar = 'horiz_winds'
    if var=='horiz_winds_v': tvar = 'horiz_winds'
    if var=='qcri'         : tvar = 'qc'
    if var=='ncri'         : tvar = 'nc'
    if var=='qci'          : tvar = 'qc'
    if var=='nci'          : tvar = 'nc'
    if var=='S'            : tvar = 'T_mid'
    if var=='dS/dp'        : tvar = 'T_mid'
    if var=='MSE'          : tvar = 'T_mid'
    if var=='dz_mid'       : tvar = 'z_mid'
    if var=='total_T_mid_tend': tvar = 'shoc_T_mid_tend'
    if var=='total_qv_tend'   : tvar = 'shoc_qv_tend'
    #---------------------------------------------------------------------------
    data = ds[tvar]
    lev_dim_name = get_lev_dim(data)
    #---------------------------------------------------------------------------
    if var=='horiz_winds_u': data = data.isel(dim2=0)
    if var=='horiz_winds_v': data = data.isel(dim2=1)
    if var=='qcri'         : data = data + ds['qr'] + ds['qi']
    if var=='ncri'         : data = data + ds['nr'] + ds['ni']
    if var=='qci'          : data = data + ds['qi']
    if var=='nci'          : data = data + ds['ni']
    if var=='S'            : data = data + ds['z_mid']*g/cp
    if var=='dS/dp'        : data = data + ds['z_mid']*g/cp; data = data.differentiate(coord=lev_dim_name)*-1
    if var=='MSE'          : data = data + ds['z_mid']*g/cp + ds['qv']*Lv/cp
    if var=='dz_mid'       : data = data.diff(dim=lev_dim_name)*-1
    if var=='total_T_mid_tend':
        data = data + ds['p3_T_mid_tend']
        data = data + ds['rrtmgp_T_mid_tend']
        data = data + ds['homme_T_mid_tend']
        if 'zm_T_mid_tend' in ds: data = data + ds['zm_T_mid_tend']
    if var=='total_qv_tend':
        data = data + ds['p3_qv_tend']
        data = data + ds['homme_qv_tend']
        if 'zm_T_mid_tend' in ds: data = data + ds['zm_qv_tend']
    #---------------------------------------------------------------------------
    # unit conversions
    if 'precip' in var: data = data*86400.*1e3
    if var in ['qv','qc','qr','qi','qcri','qci']: data = data*1e3
    if var=='RelativeHumidity': data = data*100
    return data
#---------------------------------------------------------------------------------------------------
@numba.njit()
def get_coords(ncol,lx=None):
    nx = int(np.sqrt(ncol))
    ne = int(nx/2)
    uxi = np.linspace(0,lx,nx+1)
    uxc = uxi[:-1] + np.diff(uxi)/2
    xc = np.full(ncol,np.nan)
    yc = np.full(ncol,np.nan)
    for ny in range(ne):
        for nx in range(ne):
            for nyj in range(2):
                for nxi in range(2):
                    g = ny*4*ne + nx*4 + nyj*2 + nxi
                    i = nx*2+nxi
                    j = ny*2+nyj
                    xc[g] = uxc[i]
                    yc[g] = uxc[j]
    return xc,yc
#---------------------------------------------------------------------------------------------------
@numba.njit()
def reorg_data(data,xc,yc,ncol,nlev):
    nx = int(np.sqrt(ncol))
    ne = int(nx/2)
    xx = np.zeros((nx,nx))
    yy = np.zeros((nx,nx))
    data_out = np.zeros((nx,nx,nlev))
    for ny in range(ne):
        for nx in range(ne):
            for nyj in range(2):
                for nxi in range(2):
                    g = ny*4*ne + nx*4 + nyj*2 + nxi
                    i = nx*2+nxi
                    j = ny*2+nyj
                    xx[i,j] = xc[g]
                    yy[i,j] = yc[g]
                    data_out[i,j,:] = data[g,:]
    return data_out,xx,yy
#---------------------------------------------------------------------------------------------------

# Build figure
nrows = int(np.ceil(num_var / float(num_plot_col)))
ncols = num_plot_col
fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5*ncols, 5*nrows),
                                 squeeze=False)
axes_flat = axes.flatten()

#---------------------------------------------------------------------------------------------------
for v in range(num_var):
    var_opts = var_opts_list[v]
    var_method = var_opts.get('method', 'mean')
    print(' '*2+f'var: {hapy.tclr.GREEN}{var[v]}{hapy.tclr.END}')
    data_list = []
    hght_list = []

    for c in range(num_case):
        case_opts = case_opts_list[c]
        print(' '*4+f'case: {hapy.tclr.CYAN}{case[c]}{hapy.tclr.END}')

        case_dir = case_opts['p']
        case_sub = case_opts['s']
        htype_loc = var_opts.get('htype', None)
        if htype_loc is None:
            if 'htype' in globals():
                htype_loc = htype
            else:
                raise ValueError('ERROR: no valid htype identified')

        file_path = f'{case_dir}/{case[c]}/{case_sub}/*{htype_loc}*'
        file_list = sorted(glob.glob(file_path))

        if first_file is not None: file_list = file_list[first_file:]
        if num_files  is not None: file_list = file_list[:num_files]
        #-------------------------------------------------------------------------
        # print()
        # for f in file_list: print(f'  {f}')
        # print()
        #-------------------------------------------------------------------------
        ds = xr.open_mfdataset(file_list)
        #-------------------------------------------------------------------------
        # if skip_first_twelve_hours:
        #     print(' '*4+f'{hapy.tclr.RED}WARNING - skipping first 12 hours{hapy.tclr.END}')
        #     ds = ds.isel(time=slice(12,))
        #-------------------------------------------------------------------------
        if 'area' in ds.variables:
          area = ds['area'].isel(time=0, missing_dims='ignore')
        elif 'area_PG2' in ds.variables:
          area = ds['area_PG2'].isel(time=0, missing_dims='ignore')
        else:
          raise ValueError('area variable not found in dataset?')
        #-------------------------------------------------------------------------
        data = get_data(ds, var[v])
        if print_stat: hapy.print_stat(data, compact=True)
        #-------------------------------------------------------------------------
        # data = data.isel(lev=slice(4,-1))
        #-------------------------------------------------------------------------
        if var_method == 'mean':
            data = data.mean(dim='time')
            data = ( (data*area).sum(dim='ncol') / area.sum(dim='ncol') )
        if var_method == 'variance':
            data = data.var(dim=['time','ncol'])
        if var_method == 'variance_space':
            if len(data.ncol)==1: raise ValueError('no valid ncol dimension - variance_space is not a valid operator')
            data = data.var(dim='ncol').mean(dim='time')
        if var_method == 'variance_time':
            if len(data.time)==1: raise ValueError('no valid time dimension - variance_time is not a valid operator')
            data = ( (data*area).sum(dim='ncol') / area.sum(dim='ncol') ).var(dim='time')
        #-------------------------------------------------------------------------
        lev_dim_name = get_lev_dim(data)
        #-------------------------------------------------------------------------
        if use_height_coord:
            hght = None
            if lev_dim_name=='lev' : hght = get_data(ds,'z_mid')
            if lev_dim_name=='ilev': hght = get_data(ds,'z_int')
            if 'time' in hght.dims: hght = hght.mean(dim='time')
            if 'ncol' in hght.dims and len(hght.ncol)==1: hght = hght.isel(ncol=0,drop=True)
            if var[v] == 'dz_mid': hght = hght[1:]
            data[lev_dim_name] = hght/1e3
            dz = height_interp_dz_km
            data = data.interp(lev=np.arange(dz,height_interp_mx_km+dz,dz))
        #-------------------------------------------------------------------------
        data_list.append(data.values)
        hght_list.append(data[lev_dim_name].values)

    #----------------------------------------------------------------------------
    ax = axes_flat[v]

    data_min = np.min([np.nanmin(d) for d in data_list])
    data_max = np.max([np.nanmax(d) for d in data_list])

    # Per-variable x-axis overrides
    # if var[v] == 'dS/dp': data_max = 0.1
    if var[v] == 'MSE':   data_max = 330.0

    for c in range(num_case):
        case_opts = case_opts_list[c]
        ls = case_opts.get('ls', None)
        ax.plot(data_list[c], hght_list[c],
                  color = clr[c],
                  linestyle = ls,
                  linewidth = 1.5,
                  label = case_opts['n'])

    # Vertical zero line
    ylims = (hght_list[0].min(), hght_list[0].max())
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='solid')

    ax.set_xlim(data_min, data_max)
    ax.set_ylim(ylims)

    fontsize = 12

    if use_height_coord:
        ax.set_ylabel('Height [km]', fontsize=fontsize)
    else:
        ax.set_ylabel('Pressure [hPa]', fontsize=fontsize)
        ax.invert_yaxis()
        if 'pressure_min' in globals(): ax.set_ylim(top=pressure_min)

    ax.set_title(var_opts.get('str'), loc='right', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.xaxis.get_major_formatter().set_useOffset(False)

    if num_case > 1:
        fontsize = 8
        ax.legend(fontsize=fontsize, loc='best', framealpha=0.6)

# Hide unused axes
for i in range(num_var, len(axes_flat)):
    axes_flat[i].set_visible(False)

#---------------------------------------------------------------------------------------------------
# Separate legend figure
#---------------------------------------------------------------------------------------------------
# if num_case > 1:
#    legend_file = fig_file + '.legend.png'
#    fig_lgd, ax_lgd = plt.subplots(figsize=(4, 0.4*num_case))
#    ax_lgd.axis('off')
#    handles = [mlines.Line2D([], [], color=clr[c], linestyle=dash_to_ls(dsh[c]),
#                              linewidth=2, label='    '+case_name[c])
#               for c in range(num_case)]
#    ax_lgd.legend(handles=handles, loc='center', fontsize=9, frameon=False)
#    os.makedirs(os.path.dirname(legend_file) or '.', exist_ok=True)
#    fig_lgd.savefig(legend_file, dpi=150, bbox_inches='tight')
#    plt.close(fig_lgd)
#    print(f'Saved legend: {legend_file}')
#---------------------------------------------------------------------------------------------------
if 'title' in locals(): fig.suptitle(title)
#---------------------------------------------------------------------------------------------------
plt.tight_layout(w_pad=2, h_pad=3)
os.makedirs(os.path.dirname(fig_file) or '.', exist_ok=True)
fig.savefig(fig_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved figure: {fig_file}')
#---------------------------------------------------------------------------------------------------