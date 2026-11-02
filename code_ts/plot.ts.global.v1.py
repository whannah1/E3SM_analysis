# --------------------------------------------------------------------------------------------------
# Plot time series of global statistics (avg/max/min/std) of the specified variables
# --------------------------------------------------------------------------------------------------
import os, glob, xarray as xr, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hapy; host = hapy.get_host()
# --------------------------------------------------------------------------------------------------
# case options:  n=name  p=case root  s=sub-dir  clr=color  ls=linestyle  lw=linewidth
case,case_opts_list = [],[]
def add_case(case_in,**kwargs):
    case.append(case_in)
    case_opts = {}
    for k, val in kwargs.items(): case_opts[k] = val
    if 'n' not in case_opts: case_opts['n'] = case_in
    case_opts_list.append(case_opts)
# --------------------------------------------------------------------------------------------------
# var options:  str=label  method=avg/absavg/max/min/absmax/std  htype=file type  lev=level index
var,var_opts_list = [],[]
def add_var(var_name,**kwargs):
    var.append(var_name)
    var_opts = {}
    for k, val in kwargs.items(): var_opts[k] = val
    if 'str' not in var_opts: var_opts['str'] = var_name
    var_opts_list.append(var_opts)
# --------------------------------------------------------------------------------------------------
if host=='nersc':
    scratch_cpu = '/pscratch/sd/w/whannah/e3sm_scratch/pm-cpu'
    scratch_gpu = '/pscratch/sd/w/whannah/e3sm_scratch/pm-gpu'

    ### Rayleigh friction tests
    htype = 'output.1si.INSTANT.nsteps_x1'
    first_file,num_files = None,None
    add_case('E3SM.2026-RAYFRIC-TEST-00.ne30pg2.draytau_0.praytau_0',n='dyn_tau=0 phy_tau=0',clr='black',ls='solid',p=scratch_gpu,s='run')
    add_case('E3SM.2026-RAYFRIC-TEST-00.ne30pg2.draytau_5.praytau_0',n='dyn_tau=5 phy_tau=0',clr='red',  ls='solid',p=scratch_gpu,s='run')
    add_case('E3SM.2026-RAYFRIC-TEST-00.ne30pg2.draytau_0.praytau_5',n='dyn_tau=0 phy_tau=5',clr='blue', ls='solid',p=scratch_gpu,s='run')

# --------------------------------------------------------------------------------------------------

# add_var('U',method='max')
# add_var('U',method='max',lev=0, str='U (model top)')
# add_var('U',method='max',lev=1, str='U (model top)')
# add_var('U',method='max',lev=2, str='U (model top)')

# add_var('U',method='avg',lev=0, str='U (model top)')
# add_var('U',method='avg',lev=1, str='U (model top)')

# add_var('U',method='min',lev=0, str='U (model top)')
# add_var('U',method='min',lev=1, str='U (model top)')

# add_var('U',method='min')
# add_var('U',method='absmax')

add_var('U',method='absavg',lev=0,str='U absavg k=0')
add_var('U',method='absavg',lev=1,str='U absavg k=1')
add_var('U',method='absavg',lev=2,str='U absavg k=2')
add_var('U',method='absavg',lev=3,str='U absavg k=3')


# --------------------------------------------------------------------------------------------------
fig_file = 'figs_ts/ts.global.v1.png'

print_stats   = True
overlay_cases = True    # True: one panel per var w/ all cases / False: one panel per var+case
var_x_case    = False   # only used when overlay_cases=False - swap row/col ordering of panels
add_legend    = True

convert_to_daily_mean = False
add_trend = False

num_plot_col = 1

# --------------------------------------------------------------------------------------------------
num_var  = len(var)
num_case = len(case)

if num_case == 1: overlay_cases = False

if 'first_file' not in globals(): first_file = None
if 'num_files'  not in globals(): num_files  = None

clr = [None]*num_case
for c in range(num_case): clr[c] = case_opts_list[c].get('clr', None)
clr = hapy.fill_color_list(clr)
# --------------------------------------------------------------------------------------------------
def get_file_list(file_path):
    file_list = sorted(glob.glob(file_path))
    if first_file is not None: file_list = file_list[first_file:]
    if num_files  is not None: file_list = file_list[:num_files]
    if file_list==[]:
        print('ERROR: file list is empty?')
        print(f'  first_file  : {first_file}')
        print(f'  num_files   : {num_files}')
        print(f'  file_path   : {file_path}')
        exit()
    return file_list
# --------------------------------------------------------------------------------------------------
# Load data and compute global statistic
# --------------------------------------------------------------------------------------------------
time_list = [[None]*num_case for v in range(num_var)]
data_list = [[None]*num_case for v in range(num_var)]
for v in range(num_var):
    var_opts = var_opts_list[v]
    var_method = var_opts.get('method', 'avg')
    print(' '*2+'var: '+hapy.tclr.GREEN+var[v]+hapy.tclr.END+f'  ({var_method})')
    for c in range(num_case):
        case_opts = case_opts_list[c]
        print(' '*4+'case: '+hapy.tclr.CYAN+case[c]+hapy.tclr.END)
        # ----------------------------------------------------------------------
        htype_loc = var_opts.get('htype', None)
        if htype_loc is None:
            if 'htype' in globals():
                htype_loc = htype
            else:
                raise ValueError('ERROR: no valid htype identified')
        file_list = get_file_list(f'{case_opts["p"]}/{case[c]}/{case_opts["s"]}/*{htype_loc}*')
        # ----------------------------------------------------------------------
        ds = xr.open_mfdataset(file_list, decode_times=False)
        data = ds[var[v]]
        if 'lev' in var_opts and 'lev' in data.dims: data = data.isel(lev=var_opts['lev'])
        # ----------------------------------------------------------------------
        # reduce over all non-time dimensions
        rdims = [d for d in data.dims if d!='time']
        if var_method in ['avg','absavg']:
            if 'ncol' not in data.dims: raise ValueError(f'{var_method} method requires ncol dimension')
            if var_method == 'absavg': data = np.abs(data)
            area = ds['area'].isel(time=0, missing_dims='ignore')
            data = (data*area).sum(dim='ncol') / area.sum(dim='ncol')
            rdims = [d for d in data.dims if d!='time']
            if rdims != []: data = data.mean(dim=rdims)
        elif var_method == 'max':    data = data.max(dim=rdims)
        elif var_method == 'min':    data = data.min(dim=rdims)
        elif var_method == 'absmax': data = np.abs(data).max(dim=rdims)
        elif var_method == 'std':    data = data.std(dim=rdims)
        else: raise ValueError(f'unknown method: {var_method}')
        data = data.compute()
        # ----------------------------------------------------------------------
        # convert time to days since first time step
        time = data['time'].values
        if 'units' in ds['time'].attrs and 'days' not in ds['time'].units:
            raise ValueError(f'unexpected time units: {ds["time"].units}')
        time = time - time[0]
        data['time'] = time
        if convert_to_daily_mean: data = data.groupby(np.floor(data['time'])).mean()
        # ----------------------------------------------------------------------
        if print_stats:
            msg = ' '*6+f'min: {np.nanmin(data.values):12.4f}'
            msg += f'   max: {np.nanmax(data.values):12.4f}'
            msg += f'   mean: '+hapy.tclr.GREEN+f'{np.nanmean(data.values):12.4f}'+hapy.tclr.END
            print(msg)
        # ----------------------------------------------------------------------
        data_list[v][c] = data.values
        time_list[v][c] = data[data.dims[0]].values
# --------------------------------------------------------------------------------------------------
# Build figure layout
# --------------------------------------------------------------------------------------------------
num_plots = num_var if overlay_cases else num_var * num_case
nrows = int(np.ceil(num_plots / float(num_plot_col)))
ncols = num_plot_col

fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 3*nrows), squeeze=False)
axes_flat = axes.flatten()
# --------------------------------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------------------------------
for v in range(num_var):
    var_opts = var_opts_list[v]
    var_method = var_opts.get('method', 'avg')
    for c in range(num_case):
        # ----------------------------------------------------------------------
        # determine panel index
        if overlay_cases:
            ip = v
        else:
            ip = v*num_case+c if var_x_case else c*num_var+v
        ax = axes_flat[ip]
        # ----------------------------------------------------------------------
        case_opts = case_opts_list[c]
        px, py = time_list[v][c], data_list[v][c]
        ax.plot(px, py, color=clr[c],
                linestyle=case_opts.get('ls','solid'),
                linewidth=case_opts.get('lw',1.5),
                label=case_opts['n'])
        if add_trend:
            a, b = np.polyfit(px, py, 1)
            print(' '*4+f'linear regression a: {a}    b: {b}')
            ax.plot(px, px*a + b, color=clr[c], linestyle='dashed', linewidth=0.8)
        # ----------------------------------------------------------------------
        ax.set_xlabel('Time [days]', fontsize=9)
        ax.set_title(f'{var_opts["str"]} ({var_method})', loc='right', fontsize=8)
        if not overlay_cases: ax.set_title(case_opts['n'], loc='left', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_xlim(np.nanmin(px), np.nanmax(px))

    if overlay_cases and add_legend:
        axes_flat[v].legend(fontsize=7, loc='best', framealpha=0.6)

# hide unused axes
for i in range(num_plots, len(axes_flat)): axes_flat[i].set_visible(False)
# --------------------------------------------------------------------------------------------------
plt.tight_layout(w_pad=2, h_pad=3)
os.makedirs(os.path.dirname(fig_file) or '.', exist_ok=True)
fig.savefig(fig_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n  {fig_file}\n')
# --------------------------------------------------------------------------------------------------
