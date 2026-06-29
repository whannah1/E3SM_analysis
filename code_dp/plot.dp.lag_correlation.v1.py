import os, glob, copy, xarray as xr, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import hapy; host = hapy.get_host()
#-------------------------------------------------------------------------------
# Lead/lag correlation between two variables (var_x and var_y).
#
# For a given lag tau the correlation is defined as:
#     corr( var_x(t) , var_y(t+tau) )
# so a positive lag means var_y lags var_x (i.e. var_x leads). When
# var_x == var_y this reduces to the lagged auto-correlation, which is
# exactly 1 at lag zero.
#
# Lags are specified in hours (lag_min / lag_max). The lag axis is drawn
# in hours, or converted to days automatically when the lag range is large.
#-------------------------------------------------------------------------------
case,case_opts_list = [],[]
def add_case(case_in,**kwargs):
    case.append(case_in)
    case_opts = {}
    for k, val in kwargs.items(): case_opts[k] = val
    if 'n' not in case_opts: case_opts['n'] = case_in
    case_opts_list.append(case_opts)
#-------------------------------------------------------------------------------
# add_var() now takes a pair of variables. The first is var_x and the second
# is var_y. If only one variable is given, the pair is treated as an auto-
# correlation (var_y = var_x).
#-------------------------------------------------------------------------------
var_x,var_y,var_opts_list = [],[],[]
def add_var(var_x_name,var_y_name=None,**kwargs):
    if var_y_name is None: var_y_name = var_x_name
    var_x.append(var_x_name)
    var_y.append(var_y_name)
    var_opts = {}
    for k, val in kwargs.items(): var_opts[k] = val
    if 'str' not in var_opts:
        if var_x_name == var_y_name:
            var_opts['str'] = f'{var_x_name} (auto)'
        else:
            var_opts['str'] = f'corr( {var_x_name} , {var_y_name} )'
    var_opts_list.append(var_opts)
#-------------------------------------------------------------------------------
if host=='nersc':
    scratch_cpu = '/pscratch/sd/w/whannah/scream_scratch/pm-cpu'
    scratch_gpu = '/pscratch/sd/w/whannah/scream_scratch/pm-gpu'

    add_case('DP.2026-GATE-ZM-06.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_3',  n='dx=22 dt=600 mvgr=3'  ,clr='red',      ls='solid', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-ZM-06.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_10', n='dx=22 dt=600 mvgr=10' ,clr='orange',   ls='solid', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-ZM-06.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mvgr_100',n='dx=22 dt=600 mvgr=100',clr='magenta',  ls='solid', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-ZM-06.arch_GPU.NN_01.ne9.lx_600km.dt_75.enable_zm_1.mvgr_3',   n='dx=22 dt= 75 mvgr=3'  ,clr='blue',     ls='solid', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-ZM-06.arch_GPU.NN_01.ne9.lx_600km.dt_75.enable_zm_1.mvgr_10',  n='dx=22 dt= 75 mvgr=10' ,clr='purple',   ls='solid', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-ZM-06.arch_GPU.NN_01.ne9.lx_600km.dt_75.enable_zm_1.mvgr_100', n='dx=22 dt= 75 mvgr=100',clr='cyan',     ls='solid', p=scratch_gpu,s='run')

    htype = 'output.2D.1hr.INSTANT.nhours_x1'
    first_file,num_files = 10,10

#-------------------------------------------------------------------------------
# Variable pairs. Each call adds one panel.
#   add_var(var_x, var_y)                  -> cross-correlation of two variables
#   add_var(var_x)                         -> lagged auto-correlation
# Optional per-panel kwargs:
#   lag_min / lag_max   (hours)  -> override the global lag window
#   str                          -> override the panel title
#-------------------------------------------------------------------------------

add_var('zm_cape','zm_prec')
add_var('zm_prec','zm_activity')
add_var('zm_prec')

#-------------------------------------------------------------------------------
fig_file = 'figs_dp/dp.lag_correlation.v1.png'

write_file    = False
print_stats   = True
overlay_cases = True

# Toggle whether the domain mean is computed before correlating. When True the
# correlation is computed on the area-weighted domain mean time series. When
# False the correlation is computed separately at each column and the resulting
# correlation curves are area-weighted averaged together. Data that is already
# a domain mean (no 'ncol' dimension) is handled automatically either way.
compute_domain_mean = True

convert_to_daily_mean = False
num_plot_col = 1

# Global default lag window in hours (used when not set per-panel)
lag_min = -48
lag_max = +96

#---------------------------------------------------------------------------------------------------
num_var  = len(var_opts_list)
num_case = len(case)

if num_case == 1: overlay_cases = False

if 'first_file' not in globals(): first_file = None
if 'num_files'  not in globals(): num_files = None
#---------------------------------------------------------------------------------------------------
# Helper routines
#---------------------------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------
# Load a single variable as a DataArray, applying the same special cases and
# unit conversions used by the timeseries script.
#-------------------------------------------------------------------------------
def get_var_data(ds, vname):
    if vname == 'pme':
        data = ds['precip_total_surf_mass_flux'] - ds['surf_evap']/1e3
        data = data * 86400 * 1e3
    elif vname == 'precip_total_surf_mass_flux' and 'precip_total_surf_mass_flux' not in ds.variables:
        data = ds['precip_liq_surf_mass_flux'] + ds['precip_ice_surf_mass_flux']
    elif vname in ['shoc_T_mid_tend']:
        data = ds[vname].sum(dim='lev')
    else:
        data = ds[vname]
    #---------------------------------------------------------------------------
    # unit conversions
    if 'precip' in vname:  data = data * 86400. * 1e3
    if vname == 'surf_evap': data = data / 1e3 * 86400. * 1e3
    #---------------------------------------------------------------------------
    if convert_to_daily_mean: data = data.resample(time='D').mean(dim='time')
    return data
#-------------------------------------------------------------------------------
# Pearson correlation along the time axis (axis 0). Works for 1D time series
# (returns a scalar) or 2D (time,ncol) arrays (returns one value per column).
#-------------------------------------------------------------------------------
def pearson_axis0(a, b):
    am = a - np.nanmean(a, axis=0)
    bm = b - np.nanmean(b, axis=0)
    num = np.nansum(am*bm, axis=0)
    den = np.sqrt(np.nansum(am**2, axis=0) * np.nansum(bm**2, axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(den>0, num/den, np.nan)
    return r
#-------------------------------------------------------------------------------
# Lead/lag correlation curve. x and y have time on axis 0 (and optionally an
# ncol axis). For each lag the overlapping segments of x(t) and y(t+lag) are
# correlated. When an ncol axis is present the per-column correlations are
# combined into a single curve using an area-weighted average.
#-------------------------------------------------------------------------------
def lag_correlation(x, y, lag_indices, weights=None):
    nt = x.shape[0]
    corr = np.full(len(lag_indices), np.nan)
    for i, lag in enumerate(lag_indices):
        if lag >= 0:
            xa = x[:nt-lag]
            ya = y[lag:nt]
        else:
            xa = x[-lag:nt]
            ya = y[:nt+lag]
        if xa.shape[0] < 3: continue
        r = pearson_axis0(xa, ya)
        if np.ndim(r) == 0:
            corr[i] = r
        else:
            if weights is not None:
                msk = np.isfinite(r)
                if np.any(msk):
                    corr[i] = np.sum(r[msk]*weights[msk]) / np.sum(weights[msk])
            else:
                corr[i] = np.nanmean(r)
    return corr
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
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
    var_opts = var_opts_list[v]
    vx, vy = var_x[v], var_y[v]
    lag_min_loc = var_opts.get('lag_min', lag_min)
    lag_max_loc = var_opts.get('lag_max', lag_max)
    print(' '*2+'var pair: '+hapy.tclr.GREEN+f'{vx} , {vy}'+hapy.tclr.END)
    lag_list, corr_list = [], []
    #-----------------------------------------------------------------------------
    for c in range(num_case):
        case_opts = case_opts_list[c]
        print(' '*4+'case: '+hapy.tclr.CYAN+case[c]+hapy.tclr.END)
        #-----------------------------------------------------------------------
        case_dir = case_opts['p']
        case_sub = case_opts['s']
        #-----------------------------------------------------------------------
        htype_loc = None
        if var_opts.get('htype',None) is None:
            if 'htype' in globals():
                htype_loc = htype
            else:
                raise ValueError('ERROR: no valid htype identified')
        else:
            htype_loc = var_opts.get('htype')
        #-----------------------------------------------------------------------
        file_list = get_file_list(f'{case_dir}/{case[c]}/{case_sub}/*{htype_loc}*')
        #-----------------------------------------------------------------------
        ds = xr.open_mfdataset(file_list, decode_times=True)

        area = None
        if 'area' in ds.variables:
            area = ds['area'].isel(time=0, missing_dims='ignore')
        #-----------------------------------------------------------------------
        data_x = get_var_data(ds, vx)
        data_y = get_var_data(ds, vy)
        #-----------------------------------------------------------------------
        # Domain mean handling. If 'ncol' is absent the data is already a
        # domain mean (or single column) and is used as-is.
        #-----------------------------------------------------------------------
        col_weights = None
        if 'ncol' in data_x.dims:
            if compute_domain_mean:
                if area is not None:
                    data_x = (data_x*area).sum(dim='ncol',skipna=True) / area.sum(dim='ncol')
                    data_y = (data_y*area).sum(dim='ncol',skipna=True) / area.sum(dim='ncol')
                else:
                    data_x = data_x.mean(dim='ncol',skipna=True)
                    data_y = data_y.mean(dim='ncol',skipna=True)
            else:
                # keep per-column data; remember area weights for averaging
                if area is not None: col_weights = area.values
        #-----------------------------------------------------------------------
        # Transpose so time is the leading axis
        #-----------------------------------------------------------------------
        data_x = data_x.transpose('time', ...)
        data_y = data_y.transpose('time', ...)
        #-----------------------------------------------------------------------
        # Determine the sampling interval (hours) from the time coordinate
        #-----------------------------------------------------------------------
        time = data_x['time']
        def fix_time(t): return (t - t[0]).astype('float') / 86400e6
        if 'units' in time.attrs:
            if 'days' not in time.units:
                time_days = fix_time(time.values)
            else:
                time_days = (time.values - time.values[0])
        else:
            time_days = fix_time(time.values)
        dt_hours = np.median(np.diff(time_days)) * 24.
        #-----------------------------------------------------------------------
        # Build the integer lag index window from the requested hour window
        #-----------------------------------------------------------------------
        lag_lo = int(np.round(lag_min_loc / dt_hours))
        lag_hi = int(np.round(lag_max_loc / dt_hours))
        lag_indices = np.arange(lag_lo, lag_hi+1)
        lag_hours   = lag_indices * dt_hours
        #-----------------------------------------------------------------------
        xv = np.asarray(data_x.values, dtype=float)
        yv = np.asarray(data_y.values, dtype=float)
        corr = lag_correlation(xv, yv, lag_indices, weights=col_weights)
        #-----------------------------------------------------------------------
        if print_stats:
            ipk = np.nanargmax(np.abs(corr))
            print('      peak corr : '+hapy.tclr.GREEN+f'{corr[ipk]:+.4f}'+hapy.tclr.END
                  +f'  at lag {lag_hours[ipk]:+.1f} hr ({lag_hours[ipk]/24.:+.2f} days)'
                  +f'   [dt={dt_hours:.3f} hr]')
        #-----------------------------------------------------------------------
        lag_list.append(lag_hours)
        corr_list.append(corr)

    #----------------------------------------------------------------------------
    # Determine axis index and lag-axis unit (hours vs days)
    #----------------------------------------------------------------------------
    ip = v
    ax = axes_flat[ip]

    max_abs_lag = np.max([np.nanmax(np.abs(l)) for l in lag_list])
    use_days = max_abs_lag > 48.
    xscale = (1./24.) if use_days else 1.
    xunit  = 'days'   if use_days else 'hours'

    for c in range(num_case):
        case_opts = case_opts_list[c]
        ls = case_opts.get('ls', None)
        ax.plot(lag_list[c]*xscale,
                np.where(np.isfinite(corr_list[c]), corr_list[c], np.nan),
                color=clr[c],
                linestyle=ls,
                linewidth=1.5,
                label=case_opts['n'])

    xmin = np.min([np.nanmin(l) for l in lag_list]) * xscale
    xmax = np.max([np.nanmax(l) for l in lag_list]) * xscale
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0., color='gray', linewidth=0.6)
    ax.axvline(0., color='gray', linewidth=0.6)
    ax.set_xlabel(f'Lag [{xunit}]', fontsize=9)
    ax.set_ylabel('Correlation', fontsize=9)
    var_str = var_opts.get('str')
    method_str = 'domain mean' if compute_domain_mean else 'per-point'
    ax.set_title(f'{var_str} ({method_str})', loc='right', fontsize=8)
    ax.tick_params(labelsize=8)

    # Inline legend per panel when overlaying cases
    if overlay_cases and num_case > 1:
        ax.legend(fontsize=7, loc='best', framealpha=0.6)

# Hide unused axes
for i in range(num_plots, len(axes_flat)):
    axes_flat[i].set_visible(False)

print()
#---------------------------------------------------------------------------------------------------
plt.tight_layout(w_pad=2, h_pad=3)
os.makedirs(os.path.dirname(fig_file) or '.', exist_ok=True)
fig.savefig(fig_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved figure: {fig_file}')
#---------------------------------------------------------------------------------------------------
