from time import perf_counter
timer_start = perf_counter()
import os, subprocess as sp, numpy as np, xarray as xr, dask, copy, string, cmocean, glob
import cartopy.crs as ccrs, matplotlib.pyplot as plt, matplotlib.colors as mcolors#, uxarray as ux
import hapy
host = hapy.get_host()
xr.set_options(use_new_combine_kwarg_defaults=True)
#---------------------------------------------------------------------------------------------------
name,case,case_dir,case_sub = [],[],[],[]
scrip_file_list = []
def add_case(case_in,n=None,p=None,s=None,g=None,c=None,scrip_file=None):
    global name,case,case_dir,case_sub
    tmp_name = case_in if n is None else n
    case.append(case_in); name.append(tmp_name); 
    case_dir.append(p); case_sub.append(s);
    if scrip_file is None: raise ValueError('scrip_file cannot be None')
    scrip_file_list.append(scrip_file)
#---------------------------------------------------------------------------------------------------
var,lev_list,var_str = [],[],[]
var_opts_list = []
# lev can be positive [mb], negative [mlev], or "sum"
def add_var(var_name,lev=None,s=None,**kwargs):
    var.append(var_name); lev_list.append(lev); 
    var_str.append(var_name if s is None else s)
    var_opts = {}
    for k, val in kwargs.items(): var_opts[k] = val
    var_opts_list.append(var_opts)
#---------------------------------------------------------------------------------------------------
# fig_file = os.getenv('HOME')+'/E3SM_analysis/figs_clim/clim.map.v1.png'
fig_file = 'figs_clim/clim.map.v1.png'
#---------------------------------------------------------------------------------------------------
if host=='alcf':

    # add_case('NOAA',n='NOAA',yr1=1990,yr2=2019, recalculate=False)
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
    # add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ke_7.5e-6',          n='ke_7.5e-6', p=tmp_path,s=tmp_sub)
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
    
    # htype,first_file,num_files = 'output.1da.AVERAGE.ndays_x',0,None
    htype,first_file,num_files = 'output.1ma.AVERAGE.nmonths_x1',0,None

    # fig_title = 'Normalized Spectra'
    add_var('precip_total_surf_mass_flux')
    add_var('VapWaterPath')
    add_var('LiqWaterPath')
    add_var('IceWaterPath')

    # plot_diff,add_diff = True,False

#---------------------------------------------------------------------------------------------------
if host=='olcf':

    # check for grids here:
    # /lustre/orion/cli115/proj-shared/hannah6/files_grid/*pg2_scrip.nc

    # ### ZM in-cloud water test
    # tmp_path,tmp_sub = '/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch','run'
    # # scrip_file = os.getenv('HOME')+f'/E3SM/data_grid/ne4pg2_scrip.nc'
    # # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne4pg2.NN_1.zm_rad_0', n='no zm_rad', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne4pg2.NN_1.zm_rad_1', n='w/ zm_rad', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # scrip_file = os.getenv('HOME')+f'/E3SM/data_grid/ne30pg2_scrip.nc'
    # # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_0', n='no zm_rad', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_1', n='w/ zm_rad', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_0',          n='ctrl',      p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_1',          n='w/ zm_rad', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_0.c0_0.001', n='c0=1.0e-3', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_0.c0_0.0001',n='c0=0.1e-3', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_0.c0_0',     n='c0=0.0',    p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_0.alt_zm_call_1',n='alt-ZM-order',p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # htype,first_file,num_files = 'output.daily.AVERAGE.nhours_x24',0,None
    # htype,first_file,num_files = 'output.monthly.AVERAGE.nmonths_x1',0,5*12
    # # use_snapshot,ss_t = True,4
    # # htype,first_file,num_files = 'output.2D.INSTANT.nsteps_x1',0,1
    # # add_var('precip_total_surf_mass_flux',s='precip')
    # # add_var('LW_flux_up_at_model_top',s='OLR')
    # # add_var('VapWaterPath',s='VWP')
    # # add_var('LiqWaterPath',s='LWP')
    # # add_var('IceWaterPath',s='IWP')
    # # add_var('zm_activity')
    # # add_var('zm_prec')
    # add_var('RESTOM',s='RESTOM')

    ### ZM baseline simulations
    # tmp_path,tmp_sub = '/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch','archive/atm/hist_ne30'
    # # scrip_file = '/lustre/orion/cli115/proj-shared/hannah6/files_grid/ne256pg2_scrip.nc'
    # # htype,first_file,num_files = 'output.1ma.AVERAGE.nmonths_x1',0,6#12*5
    # scrip_file = '/lustre/orion/cli115/proj-shared/hannah6/files_grid/ne30pg2_scrip.nc'
    # htype,first_file,num_files = 'output.ne30pg2.1ma.AVERAGE.nmonths_x1',0,12*5
    # # scrip_file = '/lustre/orion/cli115/proj-shared/hannah6/files_grid/ne256pg2_scrip.nc'
    # # htype,first_file,num_files,tmp_sub = 'output.1ma.AVERAGE.nmonths_x1',12,1,'archive/atm/hist'
    # # add_case('E3SM.2026-ZM-BASE-00.ne256pg2.NN_128.F2010-SCREAMv1',          n='ne256 noZM',        p=tmp_path,s='archive/atm/hist_ne30',scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-BASE-00.ne256pg2.NN_128.F2010xx-ZM-CICE',         n='ne256_ZM',          p=tmp_path,s='archive/atm/hist_ne30',scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-BASE-00.ne256pg2.NN_128.F2010xx-ZM-CICE.L128v3.6',n='ne256_ZM_L128v3.6', p=tmp_path,s='archive/atm/hist_ne30',scrip_file=scrip_file)
    # prefix = 'E3SM.2026-ZM-BASE-01.ne256pg2.NN_256.F2010xx-ZM-CICE.COSP'
    # add_case(f'{prefix}',                            n='ne256+ZM', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case(f'{prefix}.L128v3.6',                   n='ne256+ZM+L128v4', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case(f'{prefix}.L128v3.6.phys_order_swap_1', n='ne256+ZM+L128v4+swap', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_var('precip_total_surf_mass_flux',s='precip')
    # # add_var('RESTOM',s='RESTOM')
    # add_var('VapWaterPath',s='VWP')
    # # add_var('LiqWaterPath',s='LWP')
    # # add_var('IceWaterPath',s='IWP')
    # # add_var('ShortwaveCloudForcing',s='SWCF')
    # # add_var('LongwaveCloudForcing', s='LWCF')
    # # add_var('U', s='850mb U', lev=850e2)
    # # add_var('U', s='200mb U', lev=200e2)
    # # add_var('zm_activity')
    # # add_var('zm_prec')


    # ### INITE 2026 CONUS
    # # salloc --time 4:00:00 --account=cli115 --nodes=1 -p gpu
    # tmp_path,tmp_sub = '/lustre/orion/cli115/proj-shared/brhillman/e3sm_scratch','run'
    # scrip_file = '/lustre/orion/cli115/world-shared/e3sm/2026-INCITE-CONUS-RRM/files_grid/2025-scream-conus-1024x2-pg2_scrip.cdf5.nc'
    # add_case('conus1024x2v1pg2_RRSwISC6to18E3r5.F2010-SCREAMv1.20260803c', n='CONUS x2 20260803c', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_var('precip_total_surf_mass_flux',htype='eamxx.precip_total_surf_mass_flux.INSTANT.nmins_x5')
    # # add_var('precip_ice_surf_mass_flux',htype='eamxx.precip_ice_surf_mass_flux.INSTANT.nmins_x5')
    # # add_var('precip_liq_surf_mass_flux',htype='eamxx.precip_liq_surf_mass_flux.INSTANT.nmins_x5')
    # # add_var('VapWaterPath',htype='eamxx.VapWaterPath.INSTANT.nmins_x5')
    # # add_var('LiqWaterPath',htype='eamxx.LiqWaterPath.INSTANT.nmins_x5')
    # # add_var('RainWaterPath',htype='eamxx.RainWaterPath.INSTANT.nmins_x5')
    # # add_var('IceWaterPath',htype='eamxx.IceWaterPath.INSTANT.nmins_x5')
    # # add_var('T_2m',htype='eamxx.T_2m.INSTANT.nmins_x5')
    # # add_var('wind_speed_10m',htype='eamxx.wind_speed_10m.INSTANT.nmins_x5')
    # # add_var('LW_flux_dn_at_model_bot',htype='eamxx.LW_flux_dn_at_model_bot.INSTANT.nmins_x5')
    # # add_var('LW_flux_up_at_model_bot',htype='eamxx.LW_flux_up_at_model_bot.INSTANT.nmins_x5')
    # # add_var('LW_flux_up_at_model_top',htype='eamxx.LW_flux_up_at_model_top.INSTANT.nmins_x5')
    # # add_var('SW_flux_dn_at_model_bot',htype='eamxx.SW_flux_dn_at_model_bot.INSTANT.nmins_x5')
    # # add_var('SW_flux_dn_at_model_top',htype='eamxx.SW_flux_dn_at_model_top.INSTANT.nmins_x5')
    # # add_var('SW_flux_up_at_model_bot',htype='eamxx.SW_flux_up_at_model_bot.INSTANT.nmins_x5')
    # # add_var('SW_flux_up_at_model_top',htype='eamxx.SW_flux_up_at_model_top.INSTANT.nmins_x5')
    # # add_var('surface_upward_latent_heat_flux',htype='eamxx.surface_upward_latent_heat_flux.INSTANT.nmins_x5')
    # # add_var('surf_evap',htype='eamxx.surf_evap.INSTANT.nmins_x5')
    # # add_var('surf_mom_flux',htype='eamxx.surf_mom_flux.INSTANT.nmins_x5')
    # add_var('surf_sens_flux',htype='eamxx.surf_sens_flux.INSTANT.nmins_x5')
    # # add_var('SeaLevelPressure',htype='eamxx.SeaLevelPressure.INSTANT.nmins_x5')
    # # add_var('ps',htype='eamxx.ps.INSTANT.nmins_x30')
    # # first_file,num_files = None,None

    ### ZM baseline simulations
    # tmp_path,tmp_sub = '/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch','run'
    # scrip_file = '/lustre/orion/cli115/proj-shared/hannah6/files_grid/ne32pg2_scrip.nc'
    # add_case(f'E3SM.2026-GW-DEV-01.F2010xx-ZM-CICE.ne32pg2_ne32pg2.NN_4.ic_old', n='ne32 Old IC', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case(f'E3SM.2026-GW-DEV-01.F2010xx-ZM-CICE.ne32pg2_ne32pg2.NN_4.ic_new', n='ne32 New IC', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # htype,first_file,num_files = 'output.debug.INSTANT.nhours_x1',0,1
    # use_snapshot,ss_t = True,11
    # add_var('SeaLevelPressure',s='SLP')

    ### TMS parameter sensitivity tests
    tmp_path,tmp_sub = '/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch','run'
    scrip_file = '/lustre/orion/cli115/proj-shared/hannah6/files_grid/ne30pg2_scrip.nc'
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_0',                   n='noTMS', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1',                   n='w/TMS', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)

    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.5',           n='w/TMS+ohf=0.5', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.rlf_0.035',         n='w/TMS+rlf=0.035', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.5',           n='w/TMS+dsf=0.5', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_1.5.dsf_0.5',   n='w/TMS+ohf=1.5+dsf=0.5', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohn_0.5',           n='w/TMS+ohn=0.5', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohn_2',             n='w/TMS+ohn=2', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)

    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25',            n='w/TMS+ohf=0.25',       p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.rlf_0.01',            n='w/TMS+rlf=0.01',       p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.25',            n='w/TMS+dsf=0.25',       p=tmp_path,s=tmp_sub,scrip_file=scrip_file)

    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.4.rlf_0.07.dsf_0.4',  n='w/TMS ohf=0.4+rlf=0.07+dsf=0.4',   p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.75.rlf_0.04.dsf_0.25',n='w/TMS ohf=0.75+rlf=0.04+dsf=0.25', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25.rlf_0.04.dsf_0.75',n='w/TMS ohf=0.25+rlf=0.04+dsf=0.75', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.5.rlf_0.04.dsf_0.5',  n='w/TMS ohf=0.5+rlf=0.04+dsf=0.5',   p=tmp_path,s=tmp_sub,scrip_file=scrip_file)

    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.5.wsn_4',   n='w/TMS+dsf=0.50+wsn=4', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)

    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.05.wsn_1',    n='w/TMS+dsf=0.05+wsn=1', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.1.wsn_1',     n='w/TMS+dsf=0.1+wsn=1',  p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.2.wsn_1',     n='w/TMS+dsf=0.2+wsn=1',  p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.4.wsn_1',     n='w/TMS+dsf=0.4+wsn=1',  p=tmp_path,s=tmp_sub,scrip_file=scrip_file)

    htype,first_file,num_files = 'output.1ma.AVERAGE.nmonths_x1',None,60
    # plot_diff,add_diff = True,False
    num_plot_col = 2
    add_var('T_2m')
    # add_var('wind_speed_10m')
    # add_var('surf_radiative_T')
    # add_var('ps')
    # add_var('VapWaterPath')
    # add_var('LiqWaterPath')
    # add_var('precip_total_surf_mass_flux',s='precip_total')


    # ### COSP porting tests
    # tmp_path,tmp_sub = '/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch','run'
    # scrip_file = '/lustre/orion/cli115/proj-shared/hannah6/files_grid/ne30pg2_scrip.nc'
    # add_case('E3SM.2026-COSP-TEST-00.ne30pg2.cosp_impl_cxx',n='cosp_impl=cxx',p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-COSP-TEST-00.ne30pg2.cosp_impl_f90',n='cosp_impl=f90',p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # htype,first_file,num_files = 'output.1ma.AVERAGE.nmonths_x1',None,None
    # plot_diff,add_diff = True,True
    # add_var('isccp_cldtot')
    

#-------------------------------------------------------------------------------
# if host=='nersc':
    
    ### 2025 scidac multi-fidelity tests
    # add_case('E3SM.2025-MF-test-00.ne18pg2.F20TR.NN_12',n='ne18',p='/pscratch/sd/w/whannah/e3sm_scratch/pm-cpu',s='run',scrip_file='/global/cfs/cdirs/m4310/whannah/files_grid/ne18pg2_scrip.nc')

    # ### Wandi's frontogensis gradient correction
    # tmp_path,tmp_sub = '/pscratch/sd/w/wandiyu/e3sm_scratch/pm-cpu','run'
    # scrip_file = os.getenv('HOME')+f'/E3SM/data_grid/ne30pg2_scrip.nc'
    # # add_case('v3.LR.GW.fronto.correction.nofix.Jan.20251104',n='control',   p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('v3.LR.GW.fronto.correction.pfix.Jan.20251104', n='p-grad fix', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('v3.LR.GW.fronto.correction.zfix.Jan.20251104', n='z-grad fix', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # htype,first_file,num_files = 'eam.h0',0,0

    # add_case('v3.LR.GW.fronto.correction.nofix.Jan.20251104',n='control',   p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # htype,first_file,num_files = 'eam.h1',-1,1
    # use_snapshot,ss_t = True,-1

    ### ERA5 remapped surface pressure for 2026 CONUS-RRM
    # tmp_path,tmp_sub = '/pscratch/sd/m/meng/hiccup','tmp'
    # scrip_file = '/global/cfs/cdirs/e3sm/2026-INCITE-CONUS-RRM/files_grid/2026-incite-conus-1024x2-np4_scrip.nc'
    # # add_case('v3.LR.GW.fronto.correction.nofix.Jan.20251104',n='control',   p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('incite_conus', n='', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # htype,first_file,num_files = 'eam.h0',0,1


    # tmp_path,tmp_sub = '/pscratch/sd/w/whannah/scream_scratch/pm-gpu','run'
    # # scrip_file = os.getenv('HOME')+f'/E3SM/data_grid/ne4pg2_scrip.nc'
    # # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne4pg2.NN_1.zm_cxx.debug', n='zm_cxx',  p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne4pg2.NN_1.zm_f90.debug', n='zm_f90',  p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # scrip_file = os.getenv('HOME')+f'/E3SM/data_grid/ne30pg2_scrip.nc'
    # # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_cxx',   p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_f90',   p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_f90.zm_detr_0',n='zm_f90.zm_detr_0',p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_f90.zm_detr_1',n='ZM-F90',p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_cxx.zm_detr_0',n='zm_cxx.zm_detr_0',p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_cxx.zm_detr_1',n='ZM-C++.',p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_cxx.zm_detr_1.mcsp_mod_1',n='zm_cxx.zm_detr_1.mcsp_mod_1',p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-ZM-DEV-01.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_f90',n='zm_f90',  p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # add_case('E3SM.2026-ZM-DEV-01.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_cxx',n='zm_cxx',  p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
    # use_snapshot,ss_t = True,4
    # htype,first_file,num_files = 'output.2D.INSTANT.nsteps_x1',0,1
    # htype,first_file,num_files = 'output.1ma.AVERAGE.nmonths_x1',0,1
    # htype,first_file,num_files = 'output.monthly.AVERAGE.nmonths_x1',0,12
    # htype,first_file,num_files = 'output.daily.AVERAGE.nhours_x24',0,0

    ### implicit flux tests
    # tmp_scratch = '/pscratch/sd/w/whannah/scream_scratch/pm-gpu'
    # tmp_grid_file = '/pscratch/sd/w/whannah/files_grid/scrip_ne256pg2.nc'
    # # add_case('E3SM.2026-impflx-test-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_128.iflx_0.gust_0',n='ne256 ctrl - old', p=tmp_scratch,s='run',scrip_file=tmp_grid_file)
    # # add_case('E3SM.2026-impflx-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_128.iflx_0.gust_0',     n='ne256 ctrl - new', p=tmp_scratch,s='archive/atm/hist',scrip_file=tmp_grid_file)
    # add_case('E3SM.2026-impflx-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_128.iflx_0.gust_0',n='ne256 control',          p=tmp_scratch,s='archive/atm/hist',scrip_file=tmp_grid_file)
    # add_case('E3SM.2026-impflx-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_128.iflx_1.gust_1',n='ne256 w/ implicit flux', p=tmp_scratch,s='archive/atm/hist',scrip_file=tmp_grid_file)
    # htype='output.1ma.AVERAGE.nmonths_x1'
    # first_file,num_files = None,12
    # add_var('RESTOM',s='RESTOM')
    # add_var('ShortwaveCloudForcing',s='SWCF')
    # add_var('LongwaveCloudForcing', s='LWCF')


    ### new time step recipe tests
    # tmp_scratch = '/pscratch/sd/w/whannah/scream_scratch/pm-gpu'
    # tmp_grid_file = '/pscratch/sd/w/whannah/files_grid/scrip_ne256pg2.nc'
    # add_case('E3SM.2026-L128-TEST-02.F2010-SCREAMv1.GPU.NN_64.ne256pg2_ne256pg2',n='ne256 control',          p=tmp_scratch,s='archive/atm/hist',scrip_file=tmp_grid_file)
    # htype='output.1da.AVERAGE'
    # first_file,num_files = None,12
    # add_var('???',s='???')

    

#-------------------------------------------------------------------------------
if host=='lcrc':

    # tmp_grid_file = '/lcrc/group/e3sm/ac.whannah/scratch/chrys/files_grid/ne256pg2_scrip.nc'

    # tmp_grid_file = '/lcrc/group/e3sm/ac.whannah/scratch/chrys/files_grid/scrip_ne30pg2.nc'
    # tmp_scratch = '/lcrc/group/e3sm/ac.whannah/scratch/chrys'
    # add_case('E3SM.2026-ZM-DEV-01.F2010xx-ZM-CICE.ne30pg2.NN_4.mscp_new.mcsp_mom_coeff_0',     n='mom_coeff=0',     p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # add_case('E3SM.2026-ZM-DEV-01.F2010xx-ZM-CICE.ne30pg2.NN_4.mscp_new.mcsp_mom_coeff_-0.1',  n='mom_coeff=-0.1',  p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # # add_case('E3SM.2026-ZM-DEV-01.F2010xx-ZM-CICE.ne30pg2.NN_4.mscp_new.mcsp_mom_coeff_-0.01', n='mom_coeff=-0.01', p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # # add_case('E3SM.2026-ZM-DEV-01.F2010xx-ZM-CICE.ne30pg2.NN_4.mscp_new.mcsp_mom_coeff_0.01',  n='mom_coeff=0.01',  p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # add_case('E3SM.2026-ZM-DEV-01.F2010xx-ZM-CICE.ne30pg2.NN_4.mscp_new.mcsp_mom_coeff_0.1',   n='mom_coeff=0.1',   p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # htype,first_file,num_files = 'output.1ma.AVERAGE.nmonths_x1',0,12
    # add_var('precip_total_surf_mass_flux',s='precip')
    # # add_var('wind_speed_10m',s='wspeed_10m')
    # # add_var('T_2m',s='T_2m')
    # add_var('U',s='U850',lev=850)
    # # add_var('U',s='U200',lev=200)
    # plot_diff,add_diff = True,False

    # tmp_grid_file = '/lcrc/group/e3sm/ac.whannah/scratch/chrys/files_grid/scrip_ne30pg2.nc'
    # tmp_scratch = '/lcrc/group/e3sm/ac.whannah/scratch/chrys'
    # add_case('E3SM.2026-ZM-TURB-00.F2010xx-ZM-CICE.ne30pg2.NN_8.ftm_0',      n='control',    p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # add_case('E3SM.2026-ZM-TURB-00.F2010xx-ZM-CICE.ne30pg2.NN_8.ftm_0.0001', n='ftm=0.1e-3', p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # add_case('E3SM.2026-ZM-TURB-00.F2010xx-ZM-CICE.ne30pg2.NN_8.ftm_0.0005', n='ftm=0.5e-3', p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # add_case('E3SM.2026-ZM-TURB-00.F2010xx-ZM-CICE.ne30pg2.NN_8.ftm_0.001',  n='ftm=1.0e-3',  p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # htype,first_file,num_files = 'output.1ma.AVERAGE.nmonths_x1',0,0
    # # add_var('precip_total_surf_mass_flux',s='precip')
    # # add_var('VapWaterPath')
    # # add_var('LiqWaterPath')
    # # add_var('IceWaterPath')
    # # add_var('U',s='U850',lev=850)
    # add_var('zm_activity')
    # add_var('zm_prec')
    # plot_diff,add_diff = True,False


    tmp_grid_file = '/lcrc/group/e3sm/ac.whannah/scratch/chrys/files_grid/scrip_ne30pg2.nc'
    tmp_scratch = '/lcrc/group/e3sm/ac.whannah/scratch/chrys'
    prefix = 'E3SM.2026-MCSP-SS-00.F2010xx-ZM-CICE.NN_4.ne30pg2'
    add_case(f'{prefix}.ufs_1.usm_0.mtc_0.3',               n='ctrl',       p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # add_case(f'{prefix}.ufs_1.usm_0.mtc_0.6',               n='mtc=0.6',    p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # add_case(f'{prefix}.ufs_1.usm_1.son_20.sox_60.mtc_0.3', n='usm=20-60+mtc=0.3',    p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # add_case(f'{prefix}.ufs_1.usm_1.son_20.sox_60.mtc_0.6', n='usm=20-60+mtc=0.6',    p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # add_case(f'{prefix}.ufs_1.usm_1.son_20.sox_60.mtc_0.8', n='usm=20-60+mtc=0.8',    p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # add_case(f'{prefix}.ufs_1.usm_1.son_20.sox_60.mtc_1',   n='usm=20-60+mtc=1',      p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # add_case(f'{prefix}.ufs_1.usm_1.son_20.sox_60.mtc_1.2', n='usm=20-60+mtc=1',      p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # add_case(f'{prefix}.ufs_1.use_1.evn_2e-06.evx_4e-06',   n='use=2.0-4.0(E-06)',    p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # add_case(f'{prefix}.ufs_1.use_1.evn_1e-07.evx_6e-06',   n='use=0.1-6.0(E-06)',    p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    add_case(f'{prefix}.ufs_1.use_1.evn_2.5e-06.evx_3.5e-06',n='use=2.5-3.5(E-06)',   p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    # add_case(f'{prefix}.ufs_1.use_1.evn_2.5e-06.evx_4.5e-06',n='use=2.5-4.5(E-06)',   p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
    htype,first_file,num_files = 'output.1ma.AVERAGE.nmonths_x1',0,5*12
    # htype,first_file,num_files = 'output.1da.AVERAGE.ndays_x1',0,0
    add_var('precip_total_surf_mass_flux',s='precip')
    # add_var('VapWaterPath')
    add_var('LiqWaterPath')
    add_var('IceWaterPath')
    # add_var('U',s='U850',lev=850)
    # add_var('zm_activity')
    add_var('zm_prec')
    # add_var('mcsp_freq')
    # add_var('mcsp_shear')
    # add_var('mcsp_ds_out',lev='sum_abs')
    # add_var('mcsp_dq_out',lev='sum')
    # add_var('mcsp_du_out',lev='sum')
    # add_var('evap_ds_out',lev='sum')
    # add_var('evap_dq_out',lev='sum')
    plot_diff,add_diff = True,False
    # plot_diff,add_diff = True,True
    

#-------------------------------------------------------------------------------
if host=='llnl':
    scrip_file_path = '/g/g19/hannah6/files_grid/ne30pg2_scrip.nc'

    ### STRONG tests
    add_case('v3.2026-STRONG-ENS-TEST-00.start_2018-07-04.seed_113355',n='00 seed_113355',p='/p/vast1/strong/hannah6',s='run',scrip_file=scrip_file_path)
    # add_case('v3.2026-STRONG-ENS-TEST-00.start_2018-07-04.seed_224466',n='00 seed_224466',p='/p/vast1/strong/hannah6',s='run',scrip_file=scrip_file_path)
    # add_case('v3.2026-STRONG-ENS-TEST-01.start_2018-07-04.seed_113355',n='01 seed_113355',p='/p/vast1/strong/hannah6',s='run',scrip_file=scrip_file_path)
    # add_case('v3.2026-STRONG-ENS-TEST-01.start_2018-07-04.seed_224466',n='01 seed_224466',p='/p/vast1/strong/hannah6',s='run',scrip_file=scrip_file_path)
    htype,first_file,num_files = 'eam.h1',0,0

#-------------------------------------------------------------------------------
if host=='alcf':

    tmp_grid_file = '/lus/flare/projects/E3SM_Dec/whannah/E3SM_grid_support/files_grid/ne256pg2_scrip.nc'
    tmp_scratch = '/lus/flare/projects/E3SM_Dec/whannah/scratch'
    add_case('E3SM.2026-osc-test-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_256',         n='ne256 dt_phys=10min',p=tmp_scratch,s='run',scrip_file=tmp_grid_file)
    # add_case('E3SM.2026-osc-test-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_256.NCPL_288',n='ne256 dt_phys=5min', p=tmp_scratch,s='run',scrip_file=tmp_grid_file)
    first_file,num_files = 5,0

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

# add_var('sp',s='ncremap',htype='erasfc.tr.sp.20250630_00.nc')
# add_var('sp',s='moab',   htype='erasfc.mb.sp.20250630_00.nc')

# add_var('ps')

# add_var('precip_total_surf_mass_flux',s='precip')
# add_var('LW_flux_up_at_model_top')

# add_var('VapWaterPath')
# add_var('LiqWaterPath')
# add_var('IceWaterPath')
# add_var('RainWaterPath')

# add_var('cldfrac_liq',              lev='sum')
# add_var('cldfrac_ice_for_analysis', lev='sum')
# add_var('cldfrac_tot_for_analysis', lev='sum')

# add_var('T_2m')
# add_var('SeaLevelPressure',s='PSL')

# add_var('zm_activity')
# add_var('zm_cape')
# add_var('zm_prec')
# add_var('zm_prec_frac')
# add_var('zm_dcape')
# add_var('zm_detr_liq',lev='sum')
# add_var('zm_detr_ice',lev='sum')

# add_var('TS')
# add_var('PRECT',s='Precipitation')
# add_var('TGCLDLWP',s='Liq Water Path')
# add_var('TGCLDIWP',s='Ice Water Path')
# add_var('P-E')
# add_var('TMQ')
# add_var('LHFLX')
# add_var('FSNT'); add_var('FLNT')
# add_var('LWCF'); add_var('SWCF')
# add_var('U10')

# add_var('T_mid', s='T_mid mlev=95 (~500mb)',  lev=-95)
# add_var('T_mid', s='T_mid plev=500', lev=500)

# add_var('z_mid', s='z_mid mlev=95 (~500mb)',  lev=-95)
# add_var('z_mid', s='z_mid plev=500', lev=500)

# add_var('U', s='100mb U', lev=100e2)
# add_var('U', s='500mb U', lev=500e2)
# add_var('U', s='850mb U', lev=850e2)

# add_var('T',     lev=500)

# add_var('CLDLIQ',lev=500)
# add_var('CLDICE',lev=500)
# add_var('Q',     lev=900)
# add_var('OMEGA',lev=500)

# add_var('FRONTGF',lev=-57,s='FRONTGF [K^2/km^2/day]')

#-------------------------------------------------------------------------------
# lat1,lat2 = -30,30
# lat1,lat2,lon1,lon2 = -10,70,180,340             # N. America
# lat1,lat2,lon1,lon2 = 0,40,360-90,360-10         # Atlantic/carribean
# lat1,lat2,lon1,lon2 = 25, 50, 360-125, 360-75    # CONUS
# lat1,lat2,lon1,lon2 = -40,40,90,240              # MC + West Pac
# lat1,lat2,lon1,lon2 = 0,60,50,120                # India
# lat1,lat2,lon1,lon2 = -40,30,360-120,360-40      # S. America
# lat1,lat2,lon1,lon2 = -20,15,360-90,360-60       # S. America - zoom in on Andes
#-------------------------------------------------------------------------------
if 'plot_diff' not in locals():
    plot_diff,add_diff = False,False
    # plot_diff,add_diff = True,False
    # plot_diff,add_diff = True,True

print_stats          = True
var_x_case           = False

# num_plot_col = None => rows/cols set by var_x_case (var x panel grid)
if 'num_plot_col' not in locals():
    num_plot_col     = None
    # num_plot_col     = len(case)
    # num_plot_col     = 1#len(var)

use_common_label_bar = False

if 'use_snapshot' not in locals(): use_snapshot,ss_t = False,-1

#---------------------------------------------------------------------------------------------------
# Set up plot resources
if case==[]: raise ValueError('ERROR - case list is empty!')
if var==[]: raise ValueError('ERROR - var list is empty!')
num_var,num_case = len(var),len(case)

diff_base = 0

if 'first_file' not in locals(): first_file = None
if 'num_files'  not in locals(): num_files  = None

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
        if c==diff_base: case_panels.append(('full',c))
        else:            case_panels.append(('diff',c))
else:
    for c in range(num_case): case_panels.append(('full',c))
num_panel = len(case_panels)

#---------------------------------------------------------------------------------------------------
# set up figure objects
subplot_kwargs = {}
# subplot_kwargs['projection'] = ccrs.Orthographic(central_latitude=-85); lat_min, lat_max = -90, -80
subplot_kwargs['projection'] = ccrs.PlateCarree(central_longitude=180)
# subplot_kwargs['projection'] = ccrs.PlateCarree()
num_plots = num_var*num_panel
if num_plot_col is None:
    (d1,d2) = (num_var,num_panel) if var_x_case else (num_panel,num_var)
else:
    d1 = int(np.ceil(num_plots / float(num_plot_col)))
    d2 = min(num_plot_col, num_plots)
fdx,fdy=10,5;figsize = (fdx*d2,fdy*d1)
title_fontsize,lable_fontsize = 20,18
# fdx,fdy=20,10;figsize = (fdx*num_panel,fdy*num_var) if var_x_case else (fdx*num_var,fdy*num_panel)
# title_fontsize,lable_fontsize = 30,25

fig, axs = plt.subplots(d1,d2, subplot_kw=subplot_kwargs, figsize=figsize, squeeze=False )
axs_flat = axs.flatten()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
    var_opts = var_opts_list[v]
    hapy.print_line()
    print(' '*2+'var: '+hapy.tclr.MAGENTA+var[v]+hapy.tclr.END)
    data_list = []
    glb_avg_list = []
    grid_ds_list = []
    for c in range(num_case):
        print(' '*4+'case: '+hapy.tclr.GREEN+case[c]+hapy.tclr.END)
        #-------------------------------------------------------------------------
        htype_tmp = None
        if 'htype' in globals(): htype_tmp = htype
        if 'htype' in var_opts: htype_tmp = var_opts['htype']
        #-------------------------------------------------------------------------
        # file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/{case[c]}*{htype}*'
        file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/*{htype_tmp}*'
        file_list = sorted(glob.glob(file_path))
        if first_file is not None and first_file!=0: file_list = file_list[first_file:]
        if num_files  is not None and num_files !=0: file_list = file_list[:num_files]
        #-------------------------------------------------------------------------
        if file_list==[]: print('ERROR: Empty file list:'); print(); print(file_path); exit()
        #-------------------------------------------------------------------------
        # print();print(' '*6+'file_list:')
        # for f in file_list: print(' '*6+f'{f}')
        # print()
        #-------------------------------------------------------------------------
        ds = xr.open_mfdataset( file_list, data_vars='minimal' )
        # ds = ux.open_mfdataset(scrip_file_list[c], file_list, data_vars='minimal')

        # if 'valid_time' in ds.dims:
        #    if len(ds['valid_time'])==1:
        #       ds = ds.isel(valid_time=0)

        # hapy.check_invalid_values(ds, variables=[var[v]], check_range=None)
        # exit()

        if var[v]=='zm_prec_frac':
            data = ds['zm_prec'] / ds['precip_total_surf_mass_flux']
        if var[v]=='RESTOM':
            data = ds['SW_flux_dn_at_model_top'] \
                  -ds['SW_flux_up_at_model_top'] \
                  -ds['LW_flux_up_at_model_top']
        else:
            data = ds[var[v]]


        # # mask out -999 fill values
        # if var[v] in ['mcsp_shear']:
        #     data = data.where(data<100)
        #     # data = data.where(data != -999)

        # # lon_min, lon_max = -180, 180
        # mask = ((ds['lat'] >= lat_min) & (ds['lat'] <= lat_max))
        # # mask.load()
        # data = data.where(mask, drop=True)

        # if print_stats: hapy.print_stat(data,name=var[v],stat='naxsh',indent='    ',compact=True)

        # if 'time' in data.dims:
        #    data = data.mean(dim='time')
        #    # data = data.min(dim='time')

        grid_ds = xr.open_dataset(scrip_file_list[c])
        # grid_ds = grid_ds.where(mask, drop=True)
        grid_ds_list.append(grid_ds)

        #-----------------------------------------------------------------------
        if 'lev' in data.dims:
            if lev_list[v] is None:
                raise ValueError('lev_list[v] is None - not sure what to do...')
            if isinstance(lev_list[v], str):
                if lev_list[v]=='sum':
                    data = data.sum(dim='lev')
                if lev_list[v]=='sum_abs':
                    data = np.abs(data).sum(dim='lev')
            elif isinstance(lev_list[v], (int, float)):
                if lev_list[v]>0:
                    ps_var = next((v for v in ['ps', 'PS'] if v in ds), None)
                    if ps_var is None: raise ValueError('No sfc pressure variable found in dataset?')
                    # lev is specified in mb, but vinth2p_simple expects Pa
                    data = hapy.vinth2p_simple(data, ds['hyam'], ds['hybm'], lev_list[v]*1e2, ds[ps_var],
                                                        interp_type='log', extrapolate=False)
                    # drop the singleton output level dimension
                    if 'plev' in data.dims: data = data.isel(plev=0,drop=True)
                    if 'lev'  in data.dims: data = data.isel(lev=0,drop=True)
                if lev_list[v]<0:
                    # if requested lev<0 use model levels without interpolation
                    klev_tmp = np.absolute(lev_list[v])
                    plev_tmp = data['lev'].values[klev_tmp]
                    print(' '*4+f'{hapy.tclr.YELLOW}selecting model level {klev_tmp} (plev~{plev_tmp}mb){hapy.tclr.END}')
                    data = data.isel(lev=klev_tmp,drop=True)
        #-----------------------------------------------------------------------
        # # integrate over levels
        # if 'lev' in data.dims:
        #    print(' '*4+f'{hapy.tclr.RED}WARNING - summing over lev dimension{hapy.tclr.END}')
        #    data = data.sum(dim='lev')
        #-----------------------------------------------------------------------
        # adjust units
        if var[v]=='FRONTGF': data = data*86400e6 # K^2/M^2/S > K^2/KM^2/day
        if 'prec' in var[v] and var[v]!='zm_prec_frac': data = data*86400*1e3 # m/s > mm/day
        #-------------------------------------------------------------------------
        if use_snapshot and 'time' in data.dims:
            # data = data.isel(time=ss_t,drop=True)
            # print(' '*4+f'{hapy.tclr.RED}WARNING - snapshot mode enabled{hapy.tclr.END}')

            data = data.isel(time=1,drop=True) - data.isel(time=0,drop=True)
            print(' '*4+f'{hapy.tclr.RED}WARNING - special time difference mode enabled{hapy.tclr.END}')
        #-----------------------------------------------------------------------
        if 'dim2' in data.dims: data = data.isel(dim2=0)
        #-----------------------------------------------------------------------
        # print stats before time averaging
        if print_stats: hapy.print_stat(data,name=var[v],stat='naxh',indent='    ',compact=True)
        #-----------------------------------------------------------------------
        # average over time dimension
        if 'time' in data.dims : 
            # hapy.print_time_length(data.time,indent=' '*6,print_span=True, print_length=False)
            # if use_snapshot:
            #    data = data.isel(time=ss_t,drop=True)
            #    print(' '*4+f'{hapy.tclr.RED}WARNING - snapshot mode enabled{hapy.tclr.END}')
            if not use_snapshot:
                data = data.mean(dim='time')
        #-----------------------------------------------------------------------
        # load into memory once - otherwise the min/max, the difference, and
        # to_raster each trigger a separate re-read of the input files
        data.load()
        #-------------------------------------------------------------------------
        # # Calculate area weighted global mean
        # if 'area' in locals() :
        #    gbl_mean = ( (data*area).sum() / area.sum() ).values 
        #    print(hapy.tcolor.CYAN+f'      Area Weighted Global Mean : {gbl_mean:6.4}'+hapy.tcolor.END)
        # else:
        #    glb_avg_list.append(None)
        #-------------------------------------------------------------------------
        # # print stats after time averaging
        # if print_stats: hapy.print_stat(data,name=var[v],stat='naxsh',indent='    ',compact=True)
        #-------------------------------------------------------------------------
        # append to data lists
        data_list.append( data )
        #-------------------------------------------------------------------------
        # save baseline for diff map
        if plot_diff :
            if c==diff_base:
                data_baseline = data.copy()
    #----------------------------------------------------------------------------
    # calculate common limits for consistent contour levels
    data_min = np.min([np.nanmin(d) for d in data_list])
    data_max = np.max([np.nanmax(d) for d in data_list])
    #----------------------------------------------------------------------------
    # compute differences relative to the baseline case (data_list keeps full fields)
    if plot_diff:
        diff_list = [ data_list[c] - data_list[diff_base] for c in range(num_case) ]
        diff_data_max = np.max([np.nanmax(np.absolute(d)) for d in diff_list])
        diff_data_min = -1*diff_data_max
    #----------------------------------------------------------------------------
    # set color bar levels
    clev = None
    if var[v]=='FRONTGF': clev = np.logspace( -5, -1, num=40)
    if 'prec' in var[v]: clev = clev = np.arange(0,20+1,1)
    # if 'prec' in var[v]: clev = np.logspace( np.log10(0.1), np.log10(50), num=40)
    if var[v]=='zm_prec_frac': clev = None
    # if var[v]=='zm_prec_frac': clev = np.linspace( 0, 1, num=20)
    # if var[v]=='zm_prec_frac': clev = np.logspace( np.log10(0.1), np.log10(1.0), num=40)
    # if var[v] in ['PS','sp']: clev = np.linspace( 800e2, 1020e2, num=40)
    if var[v] in ['PS','sp']: clev = np.arange(600e2,1040e2+2e2,10e2)
    if var[v]=='PCONVT': clev = np.arange(100e2,600e2+5e2,5e2)

    # if var[v]=='PCONVT': clev = np.logspace( np.log10(50e2), np.log10(500e2), num=40)
    #----------------------------------------------------------------------------
    # set color map
    cmap = 'viridis'
    # cmap = cmocean.cm.rain
    # cmap = cmocean.cm.amp
    # cmap = cmocean.cm.balance
    if var[v]=='PCONVT': cmap = cmocean.cm.rain_r
    if var[v] in ['U','V']:
        cmap = cmocean.cm.curl
    #----------------------------------------------------------------------------
    for p in range(num_panel):
        kind,c = case_panels[p]
        #-------------------------------------------------------------------------
        img_kwargs = {}
        img_kwargs['origin'] = 'lower'
        # img_kwargs['extent'] = [-180, 180, -90, 90]
        img_kwargs['cmap']   = cmap

        if kind=='diff':
            plot_data = diff_list[c]
            img_kwargs['cmap']   = cmocean.cm.balance
            img_kwargs['vmin']   = diff_data_min
            img_kwargs['vmax']   = diff_data_max
        else:
            plot_data = data_list[c]
            if clev is None:
                img_kwargs['vmin'],img_kwargs['vmax'] = data_min, data_max
            else:
                img_kwargs['norm'] = mcolors.BoundaryNorm(clev, ncolors=256)


        # print()
        # print(img_kwargs)
        # print()

        if num_plot_col is None:
            ax = axs[v,p] if var_x_case else axs[p,v]
        else:
            ip = v*num_panel+p if var_x_case else p*num_var+v
            ax = axs_flat[ip]
        ax.coastlines(linewidth=0.2,edgecolor='white')
        if kind=='diff':
            ax.set_title(f'{name[c]} - {name[diff_base]}',fontsize=title_fontsize, loc='left')
        else:
            ax.set_title(name[c],fontsize=title_fontsize, loc='left')
        ax.set_title(var_str[v],fontsize=title_fontsize, loc='right')
        
        # add global mean value
        # print(grid_ds_list[c].grid_area)
        area = grid_ds_list[c].grid_area.values#rename('grid_size':'ncol')
        if kind=='diff':
            gbl_mean = ( (diff_list[c]*area).sum() / area.sum() ).values 
        else:
            gbl_mean = ( (data_list[c]*area).sum() / area.sum() ).values 
        ax.set_title(f'mean: {gbl_mean:8.2f}', fontsize=int(title_fontsize*0.5), \
                     loc='center', y=0.9, backgroundcolor='white')

        ax.set_global()

        rst = hapy.to_raster(plot_data, grid_ds_list[c], ax)
        # rst = plot_data.to_raster(ax=ax)

        img = ax.imshow(rst, extent=ax.get_xlim() + ax.get_ylim(), **img_kwargs)

        # orientation = 'vertical' if var_x_case else 'horizontal'
        # if c==num_case-1: fig.colorbar(img, ax=ax, fraction=0.02, orientation=orientation)

        cbar = fig.colorbar(img, ax=ax, fraction=0.02, orientation='vertical')
        cbar.ax.tick_params(labelsize=lable_fontsize)

        # if var_x_case:
        #    if c==num_case-1: fig.colorbar(img, ax=ax, fraction=0.02, orientation='vertical')
        # else:
        #    fig.colorbar(img, ax=ax, fraction=0.05, orientation='horizontal')
        

#---------------------------------------------------------------------------------------------------
# hide unused panels
for i in range(num_plots, len(axs_flat)):
    axs_flat[i].set_visible(False)
#---------------------------------------------------------------------------------------------------
# Finalize plot
fig.savefig(fig_file, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f'\n{fig_file}')

#---------------------------------------------------------------------------------------------------
etime = perf_counter()-timer_start
time_str = f'{etime:10.1f} sec'
if etime>60       : time_str += f' ({(etime/60):4.1f} min)'
if etime>(2*3600) : time_str += f' ({(etime/3600):.1f} hr)'
print(f'\n{hapy.tclr.YELLOW}elapsed time: {time_str} {hapy.tclr.END}')
print()
#---------------------------------------------------------------------------------------------------
