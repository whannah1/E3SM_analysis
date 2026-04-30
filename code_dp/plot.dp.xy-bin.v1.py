import os, glob, xarray as xr, numpy as np, cmocean
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import hapy
host = hapy.get_host()
#-------------------------------------------------------------------------------
class Stop(Exception): pass
#-------------------------------------------------------------------------------
case,case_opts_list = [],[]
def add_case(case_in,**kwargs):
    case.append(case_in)
    case_opts = {}
    for k, val in kwargs.items(): case_opts[k] = val
    case_opts_list.append(case_opts)
#---------------------------------------------------------------------------------------------------
var,var_opts_list = [],[]
def add_var(**kwargs):
    # var.append(var_name)
    var_opts = {}
    for k, val in kwargs.items(): var_opts[k] = val
    # if 'str' not in var_opts: var_opts['str'] = var_name
    var_opts_list.append(var_opts)
#-------------------------------------------------------------------------------
if host=='nersc':
    scratch_cpu = '/pscratch/sd/w/whannah/scream_scratch/pm-cpu'
    scratch_gpu = '/pscratch/sd/w/whannah/scream_scratch/pm-gpu'

    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_300', n='a=1.0 pm=300', clr='black', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_400', n='a=1.0 pm=400', p=scratch_gpu, s='run', clr='C1')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_500', n='a=1.0 pm=500', p=scratch_gpu, s='run', clr='C2')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_600', n='a=1.0 pm=600', p=scratch_gpu, s='run', clr='C3')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_700', n='a=1.0 pm=700', p=scratch_gpu, s='run', clr='C4')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_800', n='a=1.0 pm=800', p=scratch_gpu, s='run', clr='C5')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_900', n='a=1.0 pm=900', p=scratch_gpu, s='run', clr='C6')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_990', n='a=1.0 pm=990', p=scratch_gpu, s='run', clr='red')

    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_500',   n='a=1.0 pm=500',   clr='red',   ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_700',   n='a=1.0 pm=700',   clr='green', ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_900',   n='a=1.0 pm=900',   clr='blue',  ls='solid', p=scratch_gpu, s='run')

    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1.5_pm_300', n='a=1.5 pm=300', clr='red',   ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_2.0_pm_300', n='a=2.0 pm=300', clr='green', ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_2.5_pm_300', n='a=2.5 pm=300', clr='blue',  ls='dashed', p=scratch_gpu, s='run')

    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_1.0_pm_100', n='a=1.0 pm=100',clr='red',    ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_1.5_pm_100', n='a=1.5 pm=100',clr='orange', ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_2.0_pm_100', n='a=2.0 pm=100',clr='green',  ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_2.5_pm_100', n='a=2.5 pm=100',clr='blue',   ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_3.0_pm_100', n='a=3.0 pm=100',clr='purple', ls='solid', p=scratch_gpu,s='run')

    # add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_1.0_pm_100',  n='a=1.0 pm=100',clr='red',    ls='dashed', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_1.5_pm_100',  n='a=1.5 pm=100',clr='orange', ls='dashed', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_2.0_pm_100',  n='a=2.0 pm=100',clr='green',  ls='dashed', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_2.5_pm_100',  n='a=2.5 pm=100',clr='blue',   ls='dashed', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_3.0_pm_100',  n='a=3.0 pm=100',clr='purple', ls='dashed', p=scratch_gpu,s='run')

    htype='output.scream.2D.1hr.INSTANT.nhours_x1'

    # first_file,num_files = -20,None
    first_file,num_files = 2,None




    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_0',           n='RCE 5m control',     clr='black', ls='solid',  p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_100',  n='RCE 5m zm mvgr:100', clr='red',   ls='solid',  p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_10',   n='RCE 5m zm mvgr: 10', clr='green', ls='solid',  p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_1',    n='RCE 5m zm mvgr:  1', clr='blue',  ls='dashed', p=scratch_gpu, s='run')

    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_0',            n='RCE 1m control',     clr='black', ls='dashed',  p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_100',   n='RCE 1m zm mvgr:100', clr='red',   ls='solid',  p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_10',    n='RCE 1m zm mvgr: 10', clr='green', ls='solid',  p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_1',     n='RCE 1m zm mvgr:  1', clr='blue',  ls='dashed', p=scratch_gpu, s='run')

    # # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_0',           n='RCE 5m control',     clr='black', ls='solid',  p=scratch_gpu, s='run')
    # # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_100',  n='RCE 5m zm mvgr:100', clr='red',    ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_10',   n='RCE 5m zm mvgr: 10', clr='red',    ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_6',    n='RCE 5m zm mvgr:  6', clr='orange', ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_4',    n='RCE 5m zm mvgr:  4', clr='green',  ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_2',    n='RCE 5m zm mvgr:  2', clr='blue',   ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_1',    n='RCE 5m zm mvgr:  1', clr='purple', ls='solid', p=scratch_gpu, s='run')

    # # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_0',            n='RCE 1m control',     clr='black', ls='dashed',  p=scratch_gpu, s='run')
    # # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_100',   n='RCE 1m zm mvgr:100', clr='red',    ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_10',    n='RCE 1m zm mvgr: 10', clr='red',    ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_6',     n='RCE 1m zm mvgr:  6', clr='orange', ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_4',     n='RCE 1m zm mvgr:  4', clr='green',  ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_2',     n='RCE 1m zm mvgr:  2', clr='blue',   ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_1',     n='RCE 1m zm mvgr:  1', clr='purple', ls='dashed', p=scratch_gpu, s='run')

    # htype = 'output.scream.2D.1hr.AVERAGE.nhours_x1'
    # # first_file,num_files = 2,3
    # first_file,num_files = 10,None

    # add_var(x_var='precip_total_surf_mass_flux', y_var='zm_activity', str='zm_activity vs precip', \
    #     htype=htype, xbin_min=0.1,xbin_max=1000,use_log=True)

    # add_var(x_var='precip_total_surf_mass_flux', y_var='zm_prec', str='zm_prec vs precip', \
    #     htype=htype, xbin_min=0.1,xbin_max=1000,use_log=True)

    # add_var(x_var='precip_total_surf_mass_flux', y_var='LiqWaterPath', str='LWP vs precip', \
    #     htype=htype, xbin_min=0.1,xbin_max=1000,use_log=True)

    # add_var(x_var='precip_total_surf_mass_flux', y_var='IceWaterPath', str='IWP vs precip', \
    #     htype=htype, xbin_min=0.1,xbin_max=1000,use_log=True)

    # first_file,num_files = -20,None
    # first_file,num_files = 2,None

#---------------------------------------------------------------------------------------------------

# add_var(x_var='precip_total_surf_mass_flux', y_var='LW_flux_up_at_model_bot', str='OLR vs precip', \
#         htype=htype, xbin_min=0.1,xbin_max=1000,use_log=True)

add_var(x_var='precip_total_surf_mass_flux', y_var='wind_speed_10m', str='wspd_10m vs precip', \
        htype=htype, xbin_min=0.1,xbin_max=1000,use_log=True)

# add_var(x_var='precip_total_surf_mass_flux', y_var='ps', str='psfc vs precip', \
#         htype=htype, xbin_min=0.1,xbin_max=1000,use_log=True)

# add_var(x_var='precip_total_surf_mass_flux', y_var='surf_sens_flux', str='SHF vs precip', \
#         htype=htype, xbin_min=0.1,xbin_max=1000,use_log=True)

add_var(x_var='precip_total_surf_mass_flux', y_var='LiqWaterPath', str='LWP vs precip', \
        htype=htype, xbin_min=0.1,xbin_max=1000,use_log=True)

# add_var(x_var='precip_total_surf_mass_flux', y_var='RainWaterPath', str='RWP vs precip', \
#         htype=htype, xbin_min=0.1,xbin_max=1000,use_log=True)

# add_var(x_var='precip_total_surf_mass_flux', y_var='IceWaterPath', str='IWP vs precip', \
#         htype=htype, xbin_min=0.1,xbin_max=1000,use_log=True)


add_var(x_var='VapWaterPath', y_var='precip_total_surf_mass_flux', str='precip vs VWP', \
        htype=htype, xbin_min=20,xbin_max=64,use_log=False)

# add_var(x_var='LiqWaterPath', y_var='wind_speed_10m', str='wspd_10m vs LWP', \
#         htype=htype, xbin_min=0.01,xbin_max=8,use_log=True)
# add_var(x_var='IceWaterPath', y_var='wind_speed_10m', str='wspd_10m vs IWP', \
#         htype=htype, xbin_min=0.01,xbin_max=8,use_log=True)

# add_var(x_var='ps', y_var='VapWaterPath', str='VWP vs ps', \
#         htype=htype, xbin_min=1007e2,xbin_max=1013e2,use_log=False)
# add_var(x_var='ps', y_var='LiqWaterPath', str='LWP vs ps', \
#         htype=htype, xbin_min=1007e2,xbin_max=1013e2,use_log=False)
# add_var(x_var='ps', y_var='RainWaterPath', str='RWP vs ps', \
#         htype=htype, xbin_min=1007e2,xbin_max=1013e2,use_log=False)
# add_var(x_var='ps', y_var='IceWaterPath', str='IWP vs ps', \
#         htype=htype, xbin_min=1007e2,xbin_max=1013e2,use_log=False)
# add_var(x_var='ps', y_var='omega', str='omega vs ps', \
#         htype=htype, xbin_min=1007e2,xbin_max=1013e2,use_log=False)

# add_var(x_var='omega', y_var='ps', str='ps vs min(omega)', \
#         htype=htype, xbin_min=-40,xbin_max=4,use_log=False)

# add_var(x_var='precip_total_surf_mass_flux', y_var='omega', str='omega vs precip', \
#         htype=htype, xbin_min=0.1,xbin_max=1000,use_log=True)

#---------------------------------------------------------------------------------------------------

fig_file = 'figs_dp/dp.xy-bin.v1.png'
tmp_file_head = 'data_temp/dp.xy-bin.v2'

recalculate = True

plot_diff   = False
print_stats = True
num_plot_col = min(2,len(var_opts_list))

# ncol_chunk_size = int(100e3)

#---------------------------------------------------------------------------------------------------
num_var,num_case = len(var_opts_list),len(case)
if 'first_file' not in globals(): first_file = None
if 'num_files'  not in globals(): num_files = None
#---------------------------------------------------------------------------------------------------
# set up figure objects - one panel per var pair, all cases overlaid
fdx,fdy=20,10
title_fontsize,lable_fontsize = 25,25
if 'num_plot_col' in locals():
    num_plot_row = int(np.ceil(num_var/float(num_plot_col)))
    figsize = (fdx*num_plot_col, fdy*num_plot_row)
    fig, axs = plt.subplots( num_plot_row, num_plot_col, figsize=figsize, squeeze=False )
else:
    figsize = (fdx, fdy*num_var)
    fig, axs = plt.subplots( num_var, 1, figsize=figsize, squeeze=False )
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
data_list_list = []
for v in range(num_var):
    var_opts = var_opts_list[v]
    x_var_name = var_opts['x_var']
    y_var_name = var_opts['y_var']
    var_str = var_opts['str'] if 'str' in var_opts else f'{y_var_name} vs {x_var_name}'
    hapy.print_line()
    print(' '*2+f'var: {hapy.tclr.GREEN}{x_var_name}{hapy.tclr.END} / {hapy.tclr.GREEN}{y_var_name}{hapy.tclr.END}')
    #----------------------------------------------------------------------------
    data_list = []
    bins_list = []
    #----------------------------------------------------------------------------
    for c in range(num_case):
        case_opts = case_opts_list[c]
        print('\n'+' '*4+f'case: {hapy.tclr.CYAN}{case[c]}{hapy.tclr.END}')
        # tmp_file = f'{tmp_file_head}.{case[c]}.{var[v]}.f0_{first_file}.nf_{num_files}.nc'
        # print(' '*4+f'case: {hapy.tclr.CYAN}{case[c]}{hapy.tclr.END}  =>  {tmp_file}')
        #-------------------------------------------------------------------------
        if recalculate:
        # if True:
            #-------------------------------------------------------------------
            case_dir = case_opts['p']
            case_sub = case_opts['s']
            htype    = var_opts['htype']
            #-------------------------------------------------------------------
            def get_file_list(file_path):
                global first_file,num_files
                file_list = sorted(glob.glob(file_path))
                if first_file is not None: file_list = file_list[first_file:]
                if num_files  is not None: file_list = file_list[:num_files]
                if file_list==[]:
                    print('ERROR: file list is empty?')
                    print(f'file_path: {file_path}')
                    print(f'file_list: {file_list}')
                    exit()
                return file_list
            #-------------------------------------------------------------------
            file_list = get_file_list(f'{case_dir}/{case[c]}/{case_sub}/*{htype}*')
            #-------------------------------------------------------------------
            ds = xr.open_mfdataset( file_list )
            #-------------------------------------------------------------------
            # x_da = ds[ var_opts['x_var'] ]
            # y_da = ds[ var_opts['y_var'] ]

            if var_opts['x_var']=='omega':
                htype_3D = 'output.scream.3D.1hr.INSTANT.nhours_x1'
                file_list = get_file_list(f'{case_dir}/{case[c]}/{case_sub}/*{htype_3D}*')
                ds_3D = xr.open_mfdataset( file_list )
                # x_da = ds_3D[ var_opts['x_var'] ].min(dim='lev')
                x_da = ds_3D[ var_opts['x_var'] ].mean(dim='lev')
            else:
                x_da = ds[ var_opts['x_var'] ]

            if var_opts['y_var']=='omega':
                htype_3D = 'output.scream.3D.1hr.INSTANT.nhours_x1'
                file_list = get_file_list(f'{case_dir}/{case[c]}/{case_sub}/*{htype_3D}*')
                ds_3D = xr.open_mfdataset( file_list )
                y_da = ds_3D[ var_opts['y_var'] ].min(dim='lev')
            elif var_opts['y_var']=='zm_prec':
                # convert ZM precip into fraction of total
                y_da = ds[ var_opts['y_var'] ] / ds['precip_total_surf_mass_flux']
            else:
                y_da = ds[ var_opts['y_var'] ]
            #-------------------------------------------------------------------
            if 'prec' in var_opts['x_var']: x_da = x_da *86400.*1e3
            #-------------------------------------------------------------------
            def avg_wgt(da,wgt,dim='ncol'):
                return ( (da*wgt).sum(dim=dim,drop=True) / wgt.sum(dim=dim,drop=True) )
            #-------------------------------------------------------------------
            x_da.load()
            y_da.load()
            #-------------------------------------------------------------------
            if print_stats:
                hapy.print_stat(x_da,name=f'{x_var_name:30}',indent=' '*4)
                hapy.print_stat(y_da,name=f'{y_var_name:30}',indent=' '*4)
            #-------------------------------------------------------------------
            if 'ncol' in x_da.dims and 'ncol' in y_da.dims:
                x_da = x_da.stack(sample=('time','ncol'))
                y_da = y_da.stack(sample=('time','ncol'))
                mean_dim = 'sample'
            else:
                mean_dim = 'time'
            #-------------------------------------------------------------------
            if 'xbin_min' in var_opts:
                xbin_min = var_opts['xbin_min']
            else:
                xbin_min = x_da.where(x_da > 0).min().values

            if 'xbin_max' in var_opts:
                xbin_max = var_opts['xbin_max']
            else:
                xbin_max = x_da.max().values

            if 'use_log' in var_opts and var_opts['use_log']==True:
                bins = np.logspace(np.log10(xbin_min), np.log10(xbin_max), 40)
            else:
                bins = np.linspace(xbin_min, xbin_max, 20)

            y_da_binned = y_da.groupby_bins(x_da, bins=bins).mean(dim=mean_dim)
            # y_da_binned = x_da.groupby_bins(x_da, bins=bins).count() / x_da.count() # just for a test

            y_da_binned = y_da_binned.rename({f'{x_var_name}_bins':'bins'})

            xbin_coord = y_da_binned['bins']
            xlft_edges = np.array([b.left  for b in xbin_coord.values])
            xrgt_edges = np.array([b.right for b in xbin_coord.values])
            xbin_edges = np.append(xlft_edges, xrgt_edges[-1])
            #-------------------------------------------------------------------
        #    # Write to file
        #    if os.path.isfile(tmp_file) : os.remove(tmp_file)
        #    tmp_ds = xr.Dataset()
        #    tmp_ds[var[v]] = data
        #    tmp_ds.to_netcdf(path=tmp_file,mode='w')
        # else:
        #    tmp_ds = xr.open_dataset( tmp_file )
        #    data = tmp_ds[var[v]]
        #-----------------------------------------------------------------------
        # # adjust units
        # if 'unit_fac' in var_opts:
        #     if var_opts['unit_fac']:
        #         y_da_binned = y_da_binned*var_opts['unit_fac']
        #-----------------------------------------------------------------------
        # if print_stats: hapy.print_stat(y_da_binned,compact=True)
        #-----------------------------------------------------------------------
        data_list.append( y_da_binned.values )
        bins_list.append( xbin_edges )
    #----------------------------------------------------------------------------
    # Create plot - single panel, all cases overlaid

    ip = v

    #----------------------------------------------------------------------------
    for c in range(num_case):
        data_list[c] = np.ma.masked_where(data_list[c]==0, data_list[c])
    #----------------------------------------------------------------------------
    if plot_diff:
        for c in range(1,num_case): data_list[c] = data_list[c] - data_list[0]
    #----------------------------------------------------------------------------
    if 'num_plot_col' in locals():
        ax = axs[ ip//num_plot_col, ip%num_plot_col ]
    else:
        ax = axs[v, 0]

    ax.set_title(var_str,             fontsize=title_fontsize, loc='right')
    ax.set_xlabel(var_opts['x_var'],  fontsize=title_fontsize)
    ax.set_ylabel(var_opts['y_var'],  fontsize=title_fontsize)
    ax.tick_params(axis='both',       labelsize=title_fontsize)

    ymin = np.nanmin([np.nanmin(d) for d in data_list])
    ymax = np.nanmax([np.nanmax(d) for d in data_list])
    ymrg = (ymax - ymin) * 0.05
    ax.set_ylim(ymin - ymrg, ymax + ymrg)

    if 'use_log' in var_opts and var_opts['use_log']==True: ax.set_xscale('log')

    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')

    for c in range(num_case):
        linestyle = case_opts_list[c].get('ls', None)
        data = data_list[c]
        bins = bins_list[c]
        bin_centers = 0.5*(bins[:-1]+bins[1:])
        clr   = case_opts_list[c].get('clr', None)
        label = case_opts_list[c]['n'] if (not plot_diff or c==0) else f"{case_opts_list[c]['n']} (diff)"
        ax.plot( bin_centers, data, color=clr, label=label, linestyle=linestyle )

    ax.legend(fontsize=lable_fontsize)

#-------------------------------------------------------------------------------
# Add legend

#---------------------------------------------------------------------------------------------------
# Finalize plot
fig.savefig(fig_file, dpi=100, bbox_inches='tight')
plt.close(fig)
print(f'\n{fig_file}\n')
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------