#---------------------------------------------------------------------------------------------------
# Wheeler-Kiladis space-time spectra of equatorial variability
# v2 => data on the native unstructured grid (ncol) can be used directly without remapping
#   the zonal FFT is replaced by an area-weighted projection onto zonal Fourier modes within
#   latitude bins that are symmetric about the equator - the rest of the calculation is the
#   same as wf.spacetime_power() since detrending/tapering/symmetry decomposition are linear
#   lat/lon data (i.e. obs or remapped model data) still uses wf.spacetime_power()
#   see hapy_waves.py for the native grid methods
#---------------------------------------------------------------------------------------------------
import os, glob, xarray as xr, numpy as np, cmocean
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import wavenumber_frequency_functions as wf
import hapy; host = hapy.get_host()
#---------------------------------------------------------------------------------------------------
import logging
logging.getLogger().setLevel(logging.WARNING)
#---------------------------------------------------------------------------------------------------
case,case_opts_list = [],[]
def add_case(case_in,**kwargs):
    case.append(case_in)
    case_opts = dict(kwargs)
    if 'n' not in case_opts: case_opts['n'] = case_in
    case_opts_list.append(case_opts)
#---------------------------------------------------------------------------------------------------
var,var_opts_list = [],[]
def add_var(var_name,**kwargs):
    var.append(var_name)
    var_opts = dict(kwargs)
    if 'str' not in var_opts: var_opts['str'] = var_name
    var_opts_list.append(var_opts)
#---------------------------------------------------------------------------------------------------
if host=='nersc':

    # 2025 scidac multi-fidelity tests
    tmp_path,tmp_sub = '/pscratch/sd/w/whannah/e3sm_scratch/pm-cpu','remap_73x144'
    # tmp_path,tmp_sub = '/pscratch/sd/w/whannah/e3sm_scratch/pm-cpu','remap_90x180'
    # add_case('NOAA',n='NOAA')
    add_case('E3SM.2025-MF-test-00.ne30pg2.F20TR.NN_32',n='ne30',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2025-MF-test-00.ne26pg2.F20TR.NN_24',n='ne26',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2025-MF-test-00.ne22pg2.F20TR.NN_18',n='ne22',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2025-MF-test-00.ne18pg2.F20TR.NN_12',n='ne18',p=tmp_path,s=tmp_sub)
    htype = 'h1'
    yr1,yr2 = 1985,1994

    # add_var('U',lev=850)
    # add_var('PRECT')
    # add_var('FLNT')
    add_var('FLUT')
    # add_var('U850')
    # add_var('U200')

#-------------------------------------------------------------------------------
if host=='lcrc':

    # add_case('NOAA',n='NOAA',yr1=1990,yr2=2019, recalculate=False)

    tmp_grid_file = '/lcrc/group/e3sm/ac.whannah/scratch/chrys/files_grid/scrip_ne30pg2.nc'
    tmp_scratch = '/lcrc/group/e3sm/ac.whannah/scratch/chrys'
    prefix = 'E3SM.2026-MCSP-SS-00.F2010xx-ZM-CICE.NN_4.ne30pg2'
    add_case(f'{prefix}.ufs_1.usm_0.mtc_0.3',               n='ctrl',                 p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    add_case(f'{prefix}.ufs_1.usm_1.son_20.sox_60.mtc_1',   n='usm=20-60+mtc=1',      p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    add_case(f'{prefix}.ufs_1.usm_1.son_20.sox_60.mtc_1.2', n='usm=20-60+mtc=1.2',      p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # add_case(f'{prefix}.ufs_1.use_1.evn_2.5e-06.evx_3.5e-06',n='use=2.5-3.5(E-06)',   p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # add_case(f'{prefix}.ufs_1.use_1.evn_2.5e-06.evx_4.5e-06',n='use=2.5-4.5(E-06)',   p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    htype,first_file,num_files = 'output.1da.AVERAGE.ndays_x1',0,None
    # fig_title = 'Normalized Spectra'
    add_var('LW_flux_up_at_model_top',    str='OLR',    normalize=False)
    add_var('precip_total_surf_mass_flux',str='precip', normalize=False)
    # add_var('U_at_850hPa',                str='U850',   normalize=False)
    # add_var('LW_flux_up_at_model_top',    str='OLR',   normalize=True)
    # add_var('precip_total_surf_mass_flux',str='precip',normalize=True)
    recalculate = True

#---------------------------------------------------------------------------------------------------
if host=='alcf':

    add_case('NOAA',n='NOAA',yr1=1990,yr2=2019, recalculate=True)
    # add_case('IMERG',n='IMERG',yr1=2011,yr2=2021, recalculate=False)

    ### E3SMv4 ZM tuning ensemble (bundled runs) - native ne30pg2 output
    tmp_path,tmp_sub = '/lus/flare/projects/E3SM_Dec/whannah/scratch/bundles/E3SM.2026-v4-ZM-TUNE-00.ne30/cases','run'
    add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE',                    n='default',  p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cape_100',           n='cape_100', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cape_200',           n='cape_200', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cape_300',           n='cape_300', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cape_400',           n='cape_400', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.dcape_0.005',        n='dcape_0.005', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.dcape_0.01',         n='dcape_0.01', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.dcape_0.05',         n='dcape_0.05', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.dcape_0.1',          n='dcape_0.1', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.mtc_0.5.mqc_0',      n='mtc_0.5.mqc_0', p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.mtc_0.3.mqc_0.3',    n='mtc_0.3.mqc_0.3', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.mtc_0.5.mqc_0.3',    n='mtc_0.5.mqc_0.3', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.mtc_0.5.mqc_0.5',    n='mtc_0.5.mqc_0.5', p=tmp_path,s=tmp_sub)

    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ke_5e-6',            n='ke_5e-6', p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ke_7.5e-6',          n='ke_7.5e-6', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ke_1e-5',            n='ke_1e-5', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.alfa_0.1',           n='alfa_0.1', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.alfa_0.2',           n='alfa_0.2', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.alfa_0.3',           n='alfa_0.3', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ncin_2',             n='ncin_2', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ncin_4',             n='ncin_4', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tau_1200',           n='tau_1200', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tau_1800',           n='tau_1800', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tau_2400',           n='tau_2400', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tpf_1',              n='tpf_1', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tpf_3',              n='tpf_3', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tka_0.8',            n='tka_0.8', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tka_1.6',            n='tka_1.6', p=tmp_path,s=tmp_sub)
    
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.c0l_0.001.c0o_0.002',n='c0l_0.001.c0o_0.002', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.c0l_0.003.c0o_0.002',n='c0l_0.003.c0o_0.002', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.c0l_0.002.c0o_0.003',n='c0l_0.002.c0o_0.003', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.c0l_0.002.c0o_0.001',n='c0l_0.002.c0o_0.001', p=tmp_path,s=tmp_sub)

    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1',              n='cda_1', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.cape_100',     n='cda_1.cape_100', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.cape_200',     n='cda_1.cape_200', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.cape_300',     n='cda_1.cape_300', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.cape_400',     n='cda_1.cape_400', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.dcape_0.005',  n='cda_1.dcape_0.005', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.dcape_0.01',   n='cda_1.dcape_0.01', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.dcape_0.05',   n='cda_1.dcape_0.05', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.dcape_0.1',    n='cda_1.dcape_0.1', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.ke_5e-6',      n='cda_1.ke_5e-6', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.ke_7.5e-6',    n='cda_1.ke_7.5e-6', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.ke_1e-5',      n='cda_1.ke_1e-5', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.alfa_0.1',     n='cda_1.alfa_0.1', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.alfa_0.2',     n='cda_1.alfa_0.2', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.alfa_0.3',     n='cda_1.alfa_0.3', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.tau_1800',     n='cda_1.tau_1800', p=tmp_path,s=tmp_sub)
    htype,first_file,num_files = 'output.1da.AVERAGE.ndays_x',0,None

    # fig_title = 'Normalized Spectra'
    add_var('LW_flux_up_at_model_top',    str='OLR',   normalize=False)
    # add_var('precip_total_surf_mass_flux',str='precip',normalize=False)

    # plot_diff,add_diff = True,False

#---------------------------------------------------------------------------------------------------
if host=='olcf':

    add_case('NOAA', n='NOAA', yr1=1990,yr2=2019, recalculate=False)

    # # ZM in-cloud water test
    # tmp_path,tmp_sub = '/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch','remap_90x180'
    # scrip_file = os.getenv('HOME')+f'/E3SM/data_grid/ne30pg2_scrip.nc'
    # # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_0',          n='no zm_rad', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_0',              n='ctrl',      p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_1',              n='w/ zm_rad', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_0.c0_0.001',     n='c0=1.0e-3', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_0.c0_0.0001',    n='c0=0.1e-3', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_0.c0_0',         n='c0=0.0',    p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_0.alt_zm_call_1',n='alt-ZM-order',p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # htype,first_file,num_files = 'output.daily.AVERAGE.nhours_x24',0,None#73*5

    # ZM baseline simulations
    tmp_path,tmp_sub = '/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch','remap_90x180'
    scrip_file = os.getenv('HOME')+f'/E3SM/data_grid/ne30pg2_scrip.nc'
    # add_case('E3SM.2026-ZM-BASE-00.ne256pg2.NN_128.F2010-SCREAMv1',          n='ne256 noZM', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-ZM-BASE-00.ne256pg2.NN_128.F2010xx-ZM-CICE',         n='ne256 w/ZM', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-ZM-BASE-00.ne256pg2.NN_128.F2010xx-ZM-CICE.L128v3.6',n='ne256 w/ZM L128v3.6', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    prefix = 'E3SM.2026-ZM-BASE-01.ne256pg2.NN_256.F2010xx-ZM-CICE.COSP'
    add_case(f'{prefix}',                            n='ne256+ZM', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    add_case(f'{prefix}.L128v3.6',                   n='ne256+ZM+L128v4', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    add_case(f'{prefix}.L128v3.6.phys_order_swap_1', n='ne256+ZM+L128v4+swap', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # htype,first_file,num_files = 'output.ne30pg2.1da.AVERAGE.nhours_x24',32,73*5 # 5 years - skip year 1
    htype,first_file,num_files = 'output.ne30pg2.1da.AVERAGE.nhours_x24',0,73*5 # 5 years - make sure year 1 files are not present

    # add_var('LW_flux_up_at_model_top')
    # add_var('precip_total_surf_mass_flux',s='precip')
    
    # fig_title = 'Un-normalized Spectra'
    # add_var('LW_flux_up_at_model_top',    str='OLR',   normalize=False)
    # add_var('precip_total_surf_mass_flux',str='precip',normalize=False)

    fig_title = 'Normalized Spectra'
    add_var('LW_flux_up_at_model_top',    str='OLR',   normalize=True)
    # add_var('precip_total_surf_mass_flux',str='precip',normalize=True)


    # add_var('LW_flux_up_at_model_top',    str='raw OLR',   normalize=False)
    # add_var('LW_flux_up_at_model_top',    str='norm OLR',  normalize=True)
    

#---------------------------------------------------------------------------------------------------
lat1,lat2 = -15,15

# native grid options
#   dlat must be larger than the grid spacing to avoid empty bins (ne30pg2 => 2.5 ; ne256pg2 => 0.5)
#   kmax must be larger than the max wavenumber of interest
dlat_native = 2.5
kmax_native = 30

segsize,noverlap = 96,60 # e3sm_diags
# segsize,noverlap = 192,120
# segsize,noverlap = 96,32
# segsize,noverlap = 180,30

#---------------------------------------------------------------------------------------------------
fig_file      = 'figs_wk/wk.v2.png'
tmp_file_head = 'data_temp/wk.v2'

if 'recalculate' not in locals(): recalculate = False

use_daily   = True
normalize   = True
var_x_case  = True

if 'plot_diff' not in locals():
    plot_diff,add_diff = False,False
    # plot_diff,add_diff = True,False   # diff panels replace the full field for non-baseline cases
    # plot_diff,add_diff = True,True    # diff panels are added alongside the full fields

#---------------------------------------------------------------------------------------------------
if case==[]: raise ValueError('ERROR - case list is empty!')
num_var,num_case = len(var),len(case)

diff_base = 0

# region of satellite aliasing artifact in the obs spectra
awn1,awn2 = 13,16
afq1,afq2 = 0.095,0.125

FONT_SMALL = 7
FONT_MED   = 8

#---------------------------------------------------------------------------------------------------
if host=='nersc': obs_root = '/global/cfs/cdirs/e3sm/diagnostics/observations/Atm/time-series'
if host=='lcrc' : obs_root = '/lcrc/soft/climate/e3sm_diags_data/obs_for_e3sm_diags/time-series'
if host=='olcf' : obs_root = '/ccs/proj/cli115/e3sm_diags_data/obs_for_e3sm_diags/time-series'
if host=='alcf' : obs_root = '/lus/flare/projects/E3SMinput/diagnostics/observations/Atm/time-series'
#---------------------------------------------------------------------------------------------------
def get_tmp_file_name(case_in,var_in,case_opts):
    tmp_file = f'{tmp_file_head}.{case_in}.{var_in}'
    if case_opts.get('yr1',None) is not None: tmp_file += f'.{case_opts["yr1"]}-{case_opts["yr2"]}'
    tmp_file += f'.ss_{segsize}.ol_{noverlap}.lat_{lat1}_{lat2}'
    if use_daily: tmp_file += '.daily'
    if case_in!='NOAA': tmp_file += f'.dlat_{dlat_native}.kmax_{kmax_native}'
    return tmp_file + '.nc'
#---------------------------------------------------------------------------------------------------
def nice_cntr_levels(data_min, data_max, max_steps=11, aboutZero=False):
    """Return (cmin, cmax, cint) with 'nice' round numbers, optionally symmetric."""
    if aboutZero:
        mag = max(abs(data_min), abs(data_max))
        raw_int = 2 * mag / (max_steps - 1)
    else:
        raw_int = (data_max - data_min) / (max_steps - 1)

    # Round to a nice number
    exp   = np.floor(np.log10(raw_int)) if raw_int > 0 else 0
    frac  = raw_int / 10**exp
    nice  = np.array([1, 2, 2.5, 5, 10])
    cint  = nice[np.argmin(np.abs(nice - frac))] * 10**exp

    if aboutZero:
        n    = int(np.ceil(mag / cint))
        cmin = -n * cint
        cmax =  n * cint
    else:
        cmin = np.floor(data_min / cint) * cint
        cmax = np.ceil(data_max  / cint) * cint

    return cmin, cmax, cint
#---------------------------------------------------------------------------------------------------
# get data for theoretical dispersion curves
swfq, swwn = wf.genDispersionCurves() # shape => (6, 3, 50)
swf = np.ma.masked_invalid( np.where(swfq==1e20, np.nan, swfq) )
swk = np.ma.masked_invalid( np.where(swwn==1e20, np.nan, swwn) )

def add_dispersion_curves(ax):
    """ overlay dispersion curves for symmetric waves + MJO frequency bounds """
    ax.axvline(0, color='black', linewidth=0.5)
    for ii in range(3,6):                 # symmetric waves => Kelvin, ER, IG
        for ee in range(swk.shape[1]):    # equivalent depths
            ax.plot(swk[ii,ee,:], swf[ii,ee,:], color='black', linewidth=0.5)
    for period in [30.,90.]:
        ax.axhline(1./period, color='black', linewidth=0.5, linestyle='dashed')
#---------------------------------------------------------------------------------------------------
def hide_aliasing_artifact(ax):
    """ put a white rectangle over the satellite aliasing signal in the obs spectra """
    ax.add_patch( Rectangle( (awn1,afq1), awn2-awn1, afq2-afq1,
                             facecolor='white', edgecolor='white', zorder=3) )
#---------------------------------------------------------------------------------------------------
def mask_aliasing_artifact(spec,wvnm,freq):
    """ zero out values over region with satellite aliasing - helps to set difference colormap """
    wn_mask = (wvnm>=awn1) & (wvnm<=awn2)
    fq_mask = (freq>=afq1) & (freq<=afq2)
    spec[np.ix_(fq_mask,wn_mask)] = 0
    return spec
#---------------------------------------------------------------------------------------------------
# build the list of panels along the case dimension
#    each panel is a tuple of (kind, case_index) where kind is 'full' or 'diff'
case_panels = []
if plot_diff and add_diff:
    # retain all full-field panels and append difference panels
    for c in range(num_case): case_panels.append(('full',c))
    for c in range(num_case):
        if c!=diff_base: case_panels.append(('diff',c))
elif plot_diff:
    # replace non-baseline full fields with differences (no added panels)
    for c in range(num_case):
        case_panels.append( ('full',c) if c==diff_base else ('diff',c) )
else:
    for c in range(num_case): case_panels.append(('full',c))
num_panel = len(case_panels)

#---------------------------------------------------------------------------------------------------
# set up figure objects
(d1,d2) = (num_var,num_panel) if var_x_case else (num_panel,num_var)
fdx,fdy = 4,3
figsize = (fdx*d2,fdy*d1)
fig, axs = plt.subplots(d1,d2, figsize=figsize, squeeze=False)
fig.subplots_adjust(hspace=0.4, wspace=0.35)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
    var_opts = var_opts_list[v]
    hapy.print_line()
    print(' '*2+f'var: {hapy.tclr.MAGENTA}{var[v]}{hapy.tclr.END}')
    spec_list, freq_list, wvnm_list = [],[],[]
    #-----------------------------------------------------------------------
    if var_opts.get('normalize',None) is not None:
        normalize_var = var_opts['normalize']
    else:
        normalize_var = normalize
    #-----------------------------------------------------------------------
    for c in range(num_case):
        case_opts = case_opts_list[c]
        print(' '*4+f'case: {hapy.tclr.GREEN}{case[c]}{hapy.tclr.END}')
        tmp_file = get_tmp_file_name(case[c],var[v],case_opts)
        #-----------------------------------------------------------------------
        recalculate_loc = recalculate
        if case_opts.get('recalculate') is not None:
            recalculate_loc = case_opts.get('recalculate')
        if not os.path.isfile(tmp_file): recalculate_loc = True
        #-----------------------------------------------------------------------
        if recalculate_loc:
            #-------------------------------------------------------------------
            print(' '*6+'loading data...')
            if case[c]=='NOAA':
                # obs_file = os.getenv('HOME')+'/Data/Obs/OLR/olr.daily.1974-2021.nc'; var_name = 'olr'
                obs_file = f'{obs_root}/NOAA-OLR_Daily/FLUT_198601_202112.nc'; var_name = 'FLUT'
                ds = xr.open_dataset(obs_file)
            elif case[c]=='IMERG':
                obs_file = f'{obs_root}/IMERG_Daily/PRECT_200101_202012.nc'; var_name = 'PRECT'
                ds = xr.open_dataset(obs_file)
            else:
                file_path = f'{case_opts["p"]}/{case[c]}/{case_opts["s"]}/*{htype}*'
                file_list = sorted(glob.glob(file_path))
                #---------------------------------------------------------------
                # print()
                # for f in file_list: print(f)
                # print()
                #---------------------------------------------------------------
                if 'first_file' in globals() and first_file is not None: file_list = file_list[first_file:]
                if 'num_files'  in globals() and num_files  is not None: file_list = file_list[:num_files]
                if file_list==[]: raise ValueError(f'No files found for path: {file_path}')
                print(' '*6+f'num files: {len(file_list)}')
                ds = xr.open_mfdataset( file_list, data_vars='minimal' )
                var_name = var[v]

            if case_opts.get('yr1',None) is not None: ds = ds.where( ds['time.year']>=case_opts['yr1'], drop=True)
            if case_opts.get('yr2',None) is not None: ds = ds.where( ds['time.year']<=case_opts['yr2'], drop=True)
            data = ds[var_name]
            #-------------------------------------------------------------------
            # reduce to equatorial subset
            native = 'ncol' in data.dims
            if native:
                if lat1!=-lat2: raise ValueError('native grid method requires lat1 = -lat2')
                lat_tmp = ds['lat'].values
                band = np.where( np.abs(lat_tmp) <= lat2+dlat_native )[0]
                if 'area' in ds:
                    area_tmp = ds['area'].values
                elif 'scrip_file' in case_opts:
                    area_tmp = xr.open_dataset(case_opts['scrip_file'])['grid_area'].values
                else:
                    raise ValueError('no area variable in data - specify scrip_file for this case')
                data = data.isel(ncol=band)
                lat_band,lon_band,area_band = lat_tmp[band], ds['lon'].values[band], area_tmp[band]
            else:
                lat_tmp = data['lat'].load()
                data = data.where( (lat_tmp>=lat1) & (lat_tmp<=lat2), drop=True)

            # Convert to daily mean
            if use_daily: data = data.resample(time='D').mean(dim='time')

            # calculate samples per day for FFT
            dtime = ( data['time'][1]-data['time'][0] ).values / np.timedelta64(1, 's')
            spd = int( 86400./dtime )
            #-------------------------------------------------------------------
            hapy.print_stat(data,name=var[v],compact=True,indent=' '*6)

            data.load()
            num_days = len(data['time'].values)
            #-------------------------------------------------------------------
            print(' '*6+'calculating WK spectra...')

            if native:
                spec = hapy.spacetime_power_native( data, lat_band, lon_band, area_band,
                                                    lat2=lat2, dlat=dlat_native, kmax=kmax_native,
                                                    segsize=segsize,
                                                    noverlap=noverlap,
                                                    spd=spd )
            else:
                spec = wf.spacetime_power( data,
                                           segsize=segsize,
                                           noverlap=noverlap,
                                           spd=spd,
                                           dosymmetries=True )

            # calculate background from average of symmetric & antisymmetric components
            smooth_kwargs = dict(kern=wf.simple_smooth_kernel(), nsmooth=40, freq_name='frequency')
            bkgd_all = wf.smooth_wavefreq( spec.mean(dim='component'), **smooth_kwargs )
            bkgd_sym = wf.smooth_wavefreq( spec[0,:,:],                **smooth_kwargs )
            bkgd_asm = wf.smooth_wavefreq( spec[1,:,:],                **smooth_kwargs )

            # separate components and throw away negative frequencies
            nf = int( len(spec['frequency'])/2 )
            rspec_sym = spec[0,:,nf:]
            rspec_asy = spec[1,:,nf:]
            bkgd_all  = bkgd_all[:,nf:]
            bkgd_sym  = bkgd_sym[:,nf:]
            bkgd_asm  = bkgd_asm[:,nf:]

            # collect everything into a dataset
            wk_ds = xr.Dataset()
            wk_ds['num_days']  = num_days
            wk_ds['segsize']   = segsize
            wk_ds['noverlap']  = noverlap
            wk_ds['rspec_sym'] = rspec_sym
            wk_ds['rspec_asy'] = rspec_asy
            wk_ds['bkgd_all']  = bkgd_all
            wk_ds['bkgd_sym']  = bkgd_sym
            wk_ds['bkgd_asm']  = bkgd_asm
            #-------------------------------------------------------------------
            print(' '*6+f'writing to file: {tmp_file}')
            os.makedirs('data_temp', exist_ok=True)
            wk_ds.to_netcdf(path=tmp_file,mode='w')
        else:
            print(' '*6+f'reading from file: {tmp_file}')
            wk_ds = xr.open_dataset( tmp_file )
        #-----------------------------------------------------------------------
        num_days  = wk_ds['num_days'].values
        rspec_sym = wk_ds['rspec_sym']
        rspec_asy = wk_ds['rspec_asy']
        bkgd_sym  = wk_ds['bkgd_sym']
        bkgd_asm  = wk_ds['bkgd_asm']

        print(' '*6+f'num_days: {num_days}')
        #-----------------------------------------------------------------------
        # normalize by the background
        if normalize_var:
            spec = rspec_sym / bkgd_sym
            # spec = rspec_asy / bkgd_asm
        else:
            spec = rspec_sym
            # spec = rspec_asy

        spec_list.append( spec.transpose().values )
        freq_list.append( spec['frequency'].values )
        wvnm_list.append( spec['wavenumber'].values )
        #-----------------------------------------------------------------------
        if case[c]=='NOAA':
            spec_list[c] = mask_aliasing_artifact(spec_list[c],wvnm_list[c],freq_list[c])
    #---------------------------------------------------------------------------
    # calculate data limits for common color bar
    data_min = np.min([np.nanmin(d) for d in spec_list])
    data_max = np.max([np.nanmax(d) for d in spec_list])
    print()
    print(' '*6+f'data_min: {data_min:.20f}')
    print(' '*6+f'data_max: {data_max:.20f}')

    if plot_diff:
        diff_list = [ spec_list[c] - spec_list[diff_base] for c in range(num_case) ]
        diff_min  = np.min([np.nanmin(d) for d in diff_list])
        diff_max  = np.max([np.nanmax(d) for d in diff_list])
        print()
        print(' '*6+f'diff_min: {diff_min:.20f}')
        print(' '*6+f'diff_max: {diff_max:.20f}')
    print()
    #---------------------------------------------------------------------------
    # contour levels for the full spectra panels
    if normalize_var:
        # clev = np.arange(4,44+4,4)/1e1
        # clev = np.arange(1.0,3.0,0.2)
        clev = np.array([0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.7,1.9,2.1,2.4,2.7,3.0]) # e3sm_diags
    else:
        cmin,cmax,_ = nice_cntr_levels(data_min*1.1, data_max*0.9, max_steps=21)
        clev = np.linspace(cmin,cmax,num=21)
        # if var[v]=='LW_flux_up_at_model_top': clev = np.arange(2,34,2)
    #---------------------------------------------------------------------------
    # contour levels for the difference panels
    if plot_diff:
        dlev = None
        # if var[v]=='FLUT': mx=25 ; dx=int(mx*2/6) ; dlev = np.arange(-mx,mx+dx,dx)
        if var[v]=='FLUT': dx,nsub = 4,6 ; dlev = np.arange(-(dx*nsub-dx/2),dx*nsub-dx/2+dx,dx)
        if dlev is None:
            cmin,cmax,_ = nice_cntr_levels(diff_min*1.2, diff_max*0.8, max_steps=21, aboutZero=True)
            dlev = np.linspace(cmin,cmax,num=21)
    #----------------------------------------------------------------------------
    # set color maps
    cmap_full = mcolors.LinearSegmentedColormap.from_list( 'WhiteBlueGreenYellowRed',
                ['white','#3050c8','#00a0a0','#20a020','#f0f000','#e05000','#a00000'] )
    # cmap_full = 'viridis'
    # cmap_full = cmocean.cm.rain
    # cmap_full = cmocean.cm.matter
    # cmap_full = cmocean.cm.amp
    # cmap_full = cmocean.cm.thermal
    cmap_diff = cmocean.cm.balance
    #---------------------------------------------------------------------------
    # Create plot
    #---------------------------------------------------------------------------
    for p in range(num_panel):
        kind,c = case_panels[p]
        #-----------------------------------------------------------------------
        if kind=='diff':
            spec_data   = diff_list[c]
            levels,cmap = dlev,cmap_diff
            title_left  = f'{case_opts_list[c]["n"]} - {case_opts_list[diff_base]["n"]}'
            # title_right = f'{var_opts["str"]} (diff)'
        else:
            spec_data   = spec_list[c]
            levels,cmap = clev,cmap_full
            title_left  = case_opts_list[c]['n']
            # title_right = var_opts['str']
        title_right = var_opts['str']
        if normalize_var: title_right += ' (norm)'
        if kind=='diff' : title_right += ' (diff)'
        #-----------------------------------------------------------------------
        ax = axs[v,p] if var_x_case else axs[p,v]

        cf = ax.contourf( wvnm_list[c], freq_list[c], np.ma.masked_invalid(spec_data),
                          levels=levels, cmap=cmap,
                          norm=mcolors.BoundaryNorm(levels, ncolors=256, clip=True),
                          extend='both' )

        add_dispersion_curves(ax)
        if case[c]=='NOAA' or (kind=='diff' and case[diff_base]=='NOAA'):
            hide_aliasing_artifact(ax)
        #-----------------------------------------------------------------------
        # label the frequency axis by period - must be done before setting the limits
        tm_period_values = np.array([2,5,10,30,90])
        ax.set_yticks( 1./tm_period_values )
        ax.set_yticklabels( tm_period_values, fontsize=FONT_SMALL )
        ax.tick_params(axis='x', labelsize=FONT_SMALL)
        #-----------------------------------------------------------------------
        # axis limits
        # ax.set_xlim(-10,10) ; ax.set_ylim(0,0.2)
        ax.set_xlim(-12,12) ; ax.set_ylim(0,0.3)
        # ax.set_xlim(-30,30) ; ax.set_ylim(0,0.5)
        # ax.set_xlim(-15,15) ; ax.set_ylim(0,0.5)
        # ax.set_xlim(-20,20) ; ax.set_ylim(0,0.3)
        # ax.set_xlim(-8,8)   ; ax.set_ylim(0,0.1)
        # ax.set_xlim(-5,5)   ; ax.set_ylim(0,0.05)
        #-----------------------------------------------------------------------
        ax.set_xlabel('Zonal Wavenumber', fontsize=FONT_SMALL)
        ax.set_ylabel('Period [days]',    fontsize=FONT_SMALL)
        #-----------------------------------------------------------------------
        ax.set_title(title_left,  loc='left',  fontsize=FONT_MED, pad=3)
        ax.set_title(title_right, loc='right', fontsize=FONT_MED, pad=3)
        #-----------------------------------------------------------------------
        cb = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.04, aspect=30, shrink=0.8)
        cb.ax.tick_params(labelsize=FONT_SMALL)
        cb.set_ticks(levels[::max(1, len(levels)//10)])

#---------------------------------------------------------------------------------------------------
if 'fig_title' in locals():
    fig.suptitle(fig_title, fontsize=16, fontweight='bold', y=0.98)
#---------------------------------------------------------------------------------------------------
# Finalize plot
os.makedirs(os.path.dirname(fig_file), exist_ok=True)
fig.savefig(fig_file, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'\n{fig_file}\n')
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
