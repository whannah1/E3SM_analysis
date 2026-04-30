import os, glob, copy, xarray as xr, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import hapy; host = hapy.get_host()
#-------------------------------------------------------------------------------
# case_name,case,case_dir,case_sub,case_grid,clr,dsh,mrk = [],[],[],[],[],[],[],[]
# lx_list,ne_list = [],[]
# def add_case(case_in,n=None,p=None,s=None,g=None,d=0,c=None,m=0,lx=None,ne=None):
#      global name,case,case_dir,case_sub,clr,dsh,mrk
#      tmp_name = '' if n is None else n
#      case.append(case_in); case_name.append(tmp_name)
#      case_dir.append(p); case_sub.append(s); case_grid.append(g)
#      dsh.append(d) ; clr.append(c) ; mrk.append(m)
#      lx_list.append(lx); ne_list.append(ne)
#-------------------------------------------------------------------------------
case,case_opts_list = [],[]
def add_case(case_in,**kwargs):
    case.append(case_in)
    case_opts = {}
    for k, val in kwargs.items(): case_opts[k] = val
    if 'n' not in case_opts: case_opts['n'] = case_in
    case_opts_list.append(case_opts)
#-------------------------------------------------------------------------------
# var, var_str, file_type_list, method_list = [], [], [], []
# def add_var(var_name,file_type,n=None,method='avg'):
#     var.append(var_name)
#     file_type_list.append(file_type)
#     var_str.append(var_name if n is None else n)
#     method_list.append(method)
#-------------------------------------------------------------------------------
var,var_opts_list = [],[]
def add_var(var_name,**kwargs):
    var.append(var_name)
    var_opts = {}
    for k, val in kwargs.items(): var_opts[k] = val
    if 'str' not in var_opts: var_opts['str'] = var_name
    var_opts_list.append(var_opts)
#-------------------------------------------------------------------------------
if host=='nersc':
    scratch_cpu = '/pscratch/sd/w/whannah/scream_scratch/pm-cpu'
    scratch_gpu = '/pscratch/sd/w/whannah/scream_scratch/pm-gpu'

    # add_case('DP.2026-RCE-01.NN_01.ne_022.lx_200.dt_60.L128_cntrl',ne=22,lx=200,c='red',  d=0,n='RCE cntrl lx=200 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_01.ne_022.lx_200.dt_60.L128_pbias',ne=22,lx=200,c='green',d=0,n='RCE pbias lx=200 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_01.ne_022.lx_200.dt_60.L128_tbias',ne=22,lx=200,c='blue', d=0,n='RCE tbias lx=200 dx=3km', p=scratch_gpu,s='run')

    # add_case('DP.2026-RCE-01.NN_04.ne_067.lx_600.dt_60.L128_cntrl',ne=67,lx=600,c='black',d=0,n='RCE cntrl lx=600 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_04.ne_067.lx_600.dt_60.L128_pbias',ne=67,lx=600,c='red',  d=0,n='RCE pbias lx=600 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_04.ne_067.lx_600.dt_60.L128_tbias',ne=67,lx=600,c='blue', d=0,n='RCE tbias lx=600 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_04.ne_067.lx_600.dt_60.L128_cntrl.qi2qc_1',ne=67,lx=600,c='magenta', d=0,n='RCE qi2qc lx=600 dx=3km', p=scratch_gpu,s='run')

    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1',        ne=67,lx=600,c='black', d=0,n='GATE control 600/3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_pbias',  ne=67,lx=600,c='blue',  d=0,n='GATE p-bias 600/3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha2', ne=67,lx=600,c='green', d=0,n='GATE alpha2 600/3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha3', ne=67,lx=600,c='green', d=0,n='GATE alpha3 600/3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1.qi2qc_1',ne=67,lx=600,c='magenta', d=0,n='GATE qi2qc 600/3km', p=scratch_gpu,s='run')

    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_300', n='a=1 pm=300', clr='black', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_400', n='a=1 pm=400', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_500', n='a=1 pm=500', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_600', n='a=1 pm=600', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_700', n='a=1 pm=700', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_800', n='a=1 pm=800', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_900', n='a=1 pm=900', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_990', n='a=1 pm=990', p=scratch_gpu, s='run')

    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_500',   n='a=1 pm=500',   clr='red',   ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_700',   n='a=1 pm=700',   clr='green', ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_900',   n='a=1 pm=900',   clr='blue',  ls='solid', p=scratch_gpu, s='run')

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

    htype = 'output.scream.2D.1hr.AVERAGE.nhours_x1'
    # htype = 'output.scream.1D.1hr.AVERAGE.nhours_x1'
    # first_file,num_files = -10,None

    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_0',           n='RCE 5m control',     clr='black', ls='solid',  p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_100',  n='RCE 5m zm mvgr:100', clr='red',    ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_10',   n='RCE 5m zm mvgr: 10', clr='orange', ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_6',    n='RCE 5m zm mvgr:  6', clr='green',  ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_4',    n='RCE 5m zm mvgr:  4', clr='cyan',   ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_2',    n='RCE 5m zm mvgr:  2', clr='blue',   ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_1',    n='RCE 5m zm mvgr:  1', clr='purple', ls='solid', p=scratch_gpu, s='run')

    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_0',            n='RCE 1m control',     clr='black', ls='dashed',  p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_100',   n='RCE 1m zm mvgr:100', clr='red',    ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_10',    n='RCE 1m zm mvgr: 10', clr='orange', ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_6',     n='RCE 1m zm mvgr:  6', clr='green',  ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_4',     n='RCE 1m zm mvgr:  4', clr='cyan',   ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_2',     n='RCE 1m zm mvgr:  2', clr='blue',   ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_1',     n='RCE 1m zm mvgr:  1', clr='purple', ls='dashed', p=scratch_gpu, s='run')

    # htype = 'output.scream.1D.1hr.AVERAGE.nhours_x1'
    # htype = 'output.scream.2D.1hr.AVERAGE.nhours_x1'
    # first_file,num_files = 0,5
    # add_var('zm_activity')
    # add_var('zm_prec')

#-------------------------------------------------------------------------------

add_var('precip_total_surf_mass_flux',method='std')

# add_var('precip_total_surf_mass_flux')#, htype=htype, n=None)
# add_var('VapWaterPath')#,                htype=htype, n=None)
# add_var('LiqWaterPath')#,                htype=htype, n='LiqWP')
# add_var('IceWaterPath')#,                htype=htype, n='IceWP')
# add_var('RainWaterPath')#,               htype=htype, n='RainWP')

# add_var('wind_speed_10m',              htype)

# add_var('surf_sens_flux',              htype, n=None)
# add_var('surf_evap',                   htype, n=None)
# add_var('surf_evap',                   htype, n='sfc evap',method='std')


# tmp_var='VapWaterPath'; add_var(tmp_var, method='avg'); add_var(tmp_var, method='std')
# tmp_var='LiqWaterPath'; add_var(tmp_var, method='avg'); add_var(tmp_var, method='std')
# tmp_var='IceWaterPath'; add_var(tmp_var, method='avg'); add_var(tmp_var, method='std')
# tmp_var='RainWaterPath';add_var(tmp_var, method='avg'); add_var(tmp_var, method='std')


# add_var('pme',                         htype, n='P-E')

# add_var('RelativeHumidity_at_700hPa', htype, n='RH 700hPa')
# add_var('omega_at_500hPa',            htype, n='Omega 500hPa')
# add_var('surf_evap',                  htype, n='surf_evap')

# add_var('SW_flux_dn_at_model_bot',htype)
# add_var('SW_flux_up_at_model_bot',htype)
# add_var('LW_flux_dn_at_model_bot',htype)
# add_var('LW_flux_up_at_model_bot',htype)
# add_var('SW_flux_up_at_model_top',htype,n='SW_flux_up_at_model_top')
# add_var('SW_flux_dn_at_model_top',htype,n='SW_flux_dn_at_model_top')
# add_var('LW_flux_up_at_model_top',htype,n='LW_flux_up_at_model_top')

#-------------------------------------------------------------------------------
fig_file = 'figs_dp/dp.timeseries.v1.png'

write_file    = False
print_stats   = True
overlay_cases = True

convert_to_daily_mean = False
add_trend = False
num_plot_col = 2

#---------------------------------------------------------------------------------------------------
num_var  = len(var_opts_list)
num_case = len(case)

if num_case == 1: overlay_cases = False

if 'lev' not in vars(): lev = np.array([0])

if 'first_file' not in globals(): first_file = None
if 'num_files'  not in globals(): num_files = None
#---------------------------------------------------------------------------------------------------
# Build figure layout
#---------------------------------------------------------------------------------------------------
num_plots = num_var if overlay_cases else num_var * num_case
nrows = int(np.ceil(num_plots / float(num_plot_col)))
ncols = num_plot_col

fig, axes = plt.subplots(nrows, ncols,
                                                 figsize=(7*ncols, 3*nrows),
                                                 squeeze=False)
axes_flat = axes.flatten()

clr = [None]*num_case
for c in range(num_case): clr[c] = case_opts_list[c].get('clr', None)
clr = hapy.fill_color_list(clr)
# dsh = hapy.dash_to_ls(dsh)
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
    var_opts = var_opts_list[v]
    var_method = var_opts.get('method', 'avg')
    print(' '*2+'var: '+hapy.tclr.GREEN+var[v]+hapy.tclr.END)
    time_list, data_list = [], []
    #-----------------------------------------------------------------------------
    for c in range(num_case):
        case_opts = case_opts_list[c]
        print(' '*4+'case: '+hapy.tclr.CYAN+case[c]+hapy.tclr.END)

        if True:
            #-------------------------------------------------------------------
            case_dir = case_opts['p']
            case_sub = case_opts['s']
            #-------------------------------------------------------------------
            htype_loc = None
            if var_opts.get('htype',None) is None:
                if 'htype' in globals():
                    htype_loc = htype
                else:
                    raise ValueError('ERROR: no valid htype identified')
            else:
                htype_loc = var_opts.get('htype')
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
            file_list = get_file_list(f'{case_dir}/{case[c]}/{case_sub}/*{htype_loc}*')
            #-------------------------------------------------------------------
            # case_dir = case_opts['p']
            # case_sub = case_opts['s']
            # file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/*{file_type_list[v]}*'
            # file_list = sorted(glob.glob(file_path))
            # if file_list == []: raise ValueError(f'No files found for path: {file_path}')
            # if 'first_file' in globals() and first_file is not None: file_list = file_list[first_file:]
            # if 'num_files'  in globals() and num_files  is not None: file_list = file_list[:num_files]
            #-------------------------------------------------------------------
            decode_times = True
            # if 'horiz_avg' in file_type_list[v]: decode_times = False
            ds = xr.open_mfdataset(file_list, decode_times=decode_times)

            if 'area' in locals(): del area
            if 'area' in ds.variables:
                area = ds['area'].isel(time=0, missing_dims='ignore')
            #-------------------------------------------------------------------------
            if var[v] == 'pme':
                data = ds['precip_total_surf_mass_flux'] - ds['surf_evap']/1e3
                data = data * 86400 * 1e3
            elif var[v] == 'precip_total_surf_mass_flux' and 'precip_total_surf_mass_flux' not in ds.variables:
                data = ds['precip_liq_surf_mass_flux'] + ds['precip_ice_surf_mass_flux']
            else:
                data = ds[var[v]]
            #-------------------------------------------------------------------------
            # unit conversions
            if 'precip' in var[v]: data = data * 86400. * 1e3
            if var[v] == 'surf_evap': data = data / 1e3 * 86400. * 1e3
            #-------------------------------------------------------------------------
            if convert_to_daily_mean: data = data.resample(time='D').mean(dim='time')
            #-------------------------------------------------------------------------
            if 'area' in locals():
                if var_method == 'avg':
                    data = (data * area).sum(dim='ncol', skipna=True) / area.sum(dim='ncol')
                elif var_method == 'std':
                    data = data.std(dim='ncol', skipna=True)
            else:
                if var_method == 'std': raise ValueError('Cannot use std method with pre-averaged data!')
                data = data.isel(ncol=0)
            #-------------------------------------------------------------------------
            time_mean = data.mean(dim='time', skipna=True).values
            print('      Time Mean : '+hapy.tclr.GREEN+f'{time_mean:10.6f}'+hapy.tclr.END)
            #-------------------------------------------------------------------------
            # Make time start at zero (days)
            def fix_time(time): return (time - time[0]).astype('float') / 86400e6
            if 'units' in data['time'].attrs:
                if 'days' not in data['time'].units:
                    data['time'] = fix_time(data['time'])
            else:
                data['time'] = fix_time(data['time'])
            #-------------------------------------------------------------------------
            data_list.append(data.values)
            time_list.append(data['time'].values)

    #----------------------------------------------------------------------------
    # Determine axis index
    #----------------------------------------------------------------------------
    ip = v  # overlay_cases: one panel per var
    ax = axes_flat[ip]

    ymin = np.min([np.nanmin(d) for d in data_list])
    ymax = np.max([np.nanmax(d) for d in data_list])
    xmin = np.min([np.nanmin(t) for t in time_list])
    xmax = np.max([np.nanmax(t) for t in time_list])

    for c in range(num_case):
        case_opts = case_opts_list[c]
        ls  = case_opts_list[c].get('ls', None)
        # clr = case_opts_list[c].get('clr', None)
        ax.plot(time_list[c], np.where(np.isfinite(data_list[c]), data_list[c], np.nan),
                        color=clr[c],
                        linestyle=ls,
                        linewidth=1.5,
                        label=case_opts['n'])

        # linear trend overlay
        if add_trend:
            px = time_list[c]
            py = data_list[c]
            a = np.cov(px.flatten(), py.flatten())[1,0] / np.var(px)
            b = np.mean(py) - a * np.mean(px)
            print(' '*4+f'linear regression a: {a}    b: {b}')
            px_range = np.abs(np.max(px) - np.min(px))
            lx = np.array([-1e2*px_range, 1e2*px_range])
            ax.plot(lx, lx*a + b, color=clr[c], linestyle='solid', linewidth=0.8)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Time [days]', fontsize=9)
    var_str = var_opts.get('str')
    ax.set_title(f'{var_str} ({var_method})', loc='right', fontsize=8)
    ax.tick_params(labelsize=8)

    # Inline legend per panel when overlaying cases
    if overlay_cases and num_case > 1:
        ax.legend(fontsize=7, loc='best', framealpha=0.6)

# Hide unused axes
for i in range(num_plots, len(axes_flat)):
    axes_flat[i].set_visible(False)

print()
#---------------------------------------------------------------------------------------------------
# # Separate legend figure (mirrors original legend_file behaviour)
# if num_case > 1:
#   legend_file = fig_file + '.legend.png'
#   fig_lgd, ax_lgd = plt.subplots(figsize=(4, 0.4 * num_case))
#   ax_lgd.axis('off')
#   handles = [mlines.Line2D([], [], color=clr[c], linestyle=dash_to_ls(dsh[c]),
#                             linewidth=2, label='    '+case_name[c])
#              for c in range(num_case)]
#   ax_lgd.legend(handles=handles, loc='center', fontsize=9, frameon=False)
#   os.makedirs(os.path.dirname(legend_file) or '.', exist_ok=True)
#   # fig_lgd.savefig(legend_file, dpi=150, bbox_inches='tight')
#   # plt.close(fig_lgd)
#   # print(f'Saved legend: {legend_file}')
#---------------------------------------------------------------------------------------------------
plt.tight_layout(w_pad=2, h_pad=3)
os.makedirs(os.path.dirname(fig_file) or '.', exist_ok=True)
fig.savefig(fig_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved figure: {fig_file}')
#---------------------------------------------------------------------------------------------------