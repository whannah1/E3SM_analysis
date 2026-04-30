#---------------------------------------------------------------------------------------------------
# Plot the zonal mean line plot of 2D (surface/column) variables
# Supports E3SM unstructured pg2 grid (ncol) and equiangular lat-lon grids
#---------------------------------------------------------------------------------------------------
import os, glob, warnings, xarray as xr, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hapy; host = hapy.get_host()
#---------------------------------------------------------------------------------------------------
case, case_opts_list = [], []
def add_case(case_in, **kwargs):
    case.append(case_in)
    case_opts = dict(kwargs)
    if 'n' not in case_opts: case_opts['n'] = case_in
    case_opts_list.append(case_opts)
#---------------------------------------------------------------------------------------------------
var, var_opts_list = [], []
def add_var(var_name, **kwargs):
    var.append(var_name)
    var_opts = dict(kwargs)
    if 'str' not in var_opts: var_opts['str'] = var_name
    var_opts_list.append(var_opts)
#---------------------------------------------------------------------------------------------------
if host == 'nersc':

    tmp_scratch = f'/pscratch/sd/w/whannah/e3sm_scratch/pm-cpu'
    add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8',           n='E3SM control',   clr='black',   p=tmp_scratch, s='run')
    add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_55', n='E3SM top_km_55', clr='red',     p=tmp_scratch, s='run')
    add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_50', n='E3SM top_km_50', clr='orange',  p=tmp_scratch, s='run')
    add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_45', n='E3SM top_km_45', clr='yellow',  p=tmp_scratch, s='run')
    add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_40', n='E3SM top_km_40', clr='green',   p=tmp_scratch, s='run')
    add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_35', n='E3SM top_km_35', clr='blue',    p=tmp_scratch, s='run')
    add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_30', n='E3SM top_km_30', clr='purple',  p=tmp_scratch, s='run')
    # add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_25', n='E3SM top_km_25', p=tmp_scratch, s='run')
    htype = 'eam.h0'
    yr1, yr2 = 1995, 1999
    month_list = [1] # 1 / 7
    add_var('U',str='U 10mb [m/s]',plev=10.0)

    # tmp_scratch   = '/pscratch/sd/w/whannah/scream_scratch/pm-gpu'
    # tmp_grid_file = '/pscratch/sd/w/whannah/files_grid/scrip_ne256pg2.nc'

    # add_case('E3SM.2026-osc-test-01.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_32',            n='ne256 control', p=tmp_scratch, s='run')
    # add_case('E3SM.2026-osc-test-03.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_32.imp_flux_1', n='ne256 imp_flux_1', p=tmp_scratch, s='run')
    # first_file,num_files = 2,None
    # htype = 'output.scream.2D.AVERAGE.ndays_x1.'

    # # add_var('precip_total_surf_mass_flux', unit_fac=86400*1e3, str='Precip [mm/day]')
    # add_var('LiqWaterPath')
    # add_var('IceWaterPath')
    # add_var('surf_sens_flux')
    # add_var('surface_upward_latent_heat_flux')
    

#---------------------------------------------------------------------------------------------------
# yr1, yr2 = 1995, 1999

lat1, lat2, dlat = -90., 90., 2.

fig_file      = 'figs_clim/clim.zonal-mean-2D.v1.png'
tmp_file_head = 'clim.zonal-mean-2D.v1'

recalculate   = True
overlay_cases = True
plot_diff     = False
num_plot_col  = 2

if 'month_list' not in locals(): month_list = None
# month_list = [12, 1, 2]  # DJF
# month_list = [6, 7, 8]   # JJA

#---------------------------------------------------------------------------------------------------
if 'first_file' not in globals(): first_file = None
if 'num_files'  not in globals(): num_files  = None

num_var  = len(var)
num_case = len(case)

if num_case == 1: overlay_cases = False

#---------------------------------------------------------------------------------------------------
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
month_str = ('(' + ','.join(month_names[m] for m in month_list) + ')'
             if month_list is not None else '')

def get_tmp_file_name(case_in, var_in):
    tmp_file_name = f'data_temp/{tmp_file_head}.{case_in}.{var_in}'
    if 'yr1' in globals(): tmp_file_name += f'.yr1_{yr1}'
    if 'yr2' in globals(): tmp_file_name += f'.yr2_{yr2}'
    if month_list is not None:
        tmp_file_name += '.m' + '-'.join(str(m) for m in month_list)
    return tmp_file_name + '.nc'
#---------------------------------------------------------------------------------------------------
all_data = {}  # keyed by ('list', v) -> list of 1D arrays
all_lat  = {}  # keyed by ('list', v) -> list of lat arrays

for v in range(num_var):
    print()
    print(f'  var: {hapy.tclr.MAGENTA}{var[v]}{hapy.tclr.END}')
    data_list = []
    lat_list  = []

    for c in range(num_case):
        case_opts = case_opts_list[c]
        print('    case: ' + hapy.tclr.GREEN + case[c] + hapy.tclr.END)

        if recalculate:
            case_dir = case_opts['p']
            case_sub = case_opts['s']

            # resolve htype: var-level override > global
            htype_loc = var_opts_list[v].get('htype', None)
            if htype_loc is None:
                if 'htype' in globals(): htype_loc = htype
                else: raise ValueError('ERROR: no valid htype identified')

            if 'eam.h' in htype_loc:
                file_path = f'{case_dir}/{case[c]}/{case_sub}/{case[c]}*{htype_loc}*'
            else:
                file_path = f'{case_dir}/{case[c]}/{case_sub}/*{htype_loc}*'

            file_list = sorted(glob.glob(file_path))
            if first_file is not None: file_list = file_list[first_file:]
            if num_files  is not None: file_list = file_list[:num_files]
            if file_list == []: raise ValueError(f'No files found for path: {file_path}')

            ds = xr.open_mfdataset(file_list)
            if 'yr1' in globals(): ds = ds.where(ds['time.year'] >= yr1, drop=True)
            if 'yr2' in globals(): ds = ds.where(ds['time.year'] <= yr2, drop=True)

            if month_list is not None:
                ds = ds.sel(time=ds['time.month'].isin(month_list))

            

            #-------------------------------------------------------------------
            if var_opts_list[v].get('plev'):
                target_plev_Pa = [float(var_opts_list[v].get('plev'))*1e2]
                if 'lev' in ds.dims:
                    area,lat = ds['area'],ds['lat']
                    ps_var = None
                    if 'ps' in ds: ps_var = 'ps'
                    if 'PS' in ds: ps_var = 'PS'
                    data = hapy.vinth2p_simple( ds[var[v]], ds['hyam'], ds['hybm'], 
                                                target_plev_Pa, ds[ps_var],
                                                interp_type='linear', extrapolate=False)
                elif 'plev' in ds.dims:
                    data = data.sel(plev=target_plev_Pa)
            else:
                data = ds[var[v]]

            #-------------------------------------------------------------------
            # adjust units
            if 'unit_fac' in var_opts_list[v]:
                data = data * var_opts_list[v]['unit_fac']
            # average in time
            data_tmean = data.mean(dim='time')
            #-------------------------------------------------------------------
            # unstructured pg2 grid
            if 'ncol' in ds.dims:
                area = ds['area']
                lat  = ds['lat']
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    bin_ds = hapy.bin_YbyX(data_tmean, lat,
                                           bin_min=lat1, bin_max=lat2, bin_spc=dlat,
                                           wgt=area, verbose=False)
                lat_vals   = bin_ds['bin'].values
                data_zonal = bin_ds['bin_val'].values

            #-------------------------------------------------------------------
            # structured lat-lon grid
            else:
                lat = ds['lat']
                # find lon dimension name
                lon_dim = next((d for d in data_tmean.dims if d not in ('lat', 'latitude')), None)
                if lon_dim is not None:
                    data_zonal_xr = data_tmean.mean(dim=lon_dim)
                else:
                    data_zonal_xr = data_tmean
                # ensure sorted by lat
                data_zonal_xr = data_zonal_xr.sortby('lat')
                lat_vals   = data_zonal_xr['lat'].values
                data_zonal = data_zonal_xr.values

            # ---- save to tmp file ---------------------------------------------
            tmp_file = get_tmp_file_name(case[c], var[v])
            print(' '*6 + f'writing to file: {tmp_file}')
            os.makedirs('data_temp', exist_ok=True)
            ds_tmp = xr.Dataset({'zonal_mean': xr.DataArray(data_zonal, coords=[lat_vals], dims=['lat'])})
            ds_tmp.to_netcdf(path=tmp_file, mode='w')

        else:
            tmp_file = get_tmp_file_name(case[c], var[v])
            print(' '*6 + f'reading from file: {tmp_file}')
            ds_tmp     = xr.open_dataset(tmp_file)
            lat_vals   = ds_tmp['lat'].values
            data_zonal = ds_tmp['zonal_mean'].values

        data_list.append(data_zonal)
        lat_list.append(lat_vals)

    all_data[('list', v)] = data_list
    all_lat[('list', v)]  = lat_list

#---------------------------------------------------------------------------------------------------
# Build figure layout
#---------------------------------------------------------------------------------------------------
lat_ticks  = np.array([-90, -60, -30, 0, 30, 60, 90])
FONT_SMALL = 7
FONT_MED   = 8

num_plots = num_var if overlay_cases else num_var * num_case
nrows = int(np.ceil(num_plots / float(num_plot_col)))
ncols = min(num_plot_col, num_plots)

fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3*nrows), squeeze=False)
axes_flat = axes.flatten()
fig.subplots_adjust(hspace=0.4, wspace=0.35)

clr = [case_opts_list[c].get('clr', None) for c in range(num_case)]
clr = hapy.fill_color_list(clr)

#---------------------------------------------------------------------------------------------------
# Plotting loop
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
    var_opts  = var_opts_list[v]
    data_list = all_data[('list', v)]
    lat_list  = all_lat[('list', v)]

    for c in range(num_case):
        ip  = v if overlay_cases else v * num_case + c
        ax  = axes_flat[ip]
        ls  = case_opts_list[c].get('ls', 'solid')
        lw  = case_opts_list[c].get('lw', 1.5)
        lbl = case_opts_list[c].get('n', case[c])

        ydata = (data_list[c] - data_list[0]) if plot_diff and c > 0 else data_list[c]

        ax.plot(lat_list[c], ydata, color=clr[c], linestyle=ls, linewidth=lw, label=lbl)

        if not overlay_cases:
            ax.set_title(lbl, loc='left', fontsize=FONT_MED, pad=3)

    # format panel (once per panel)
    panels = [v] if overlay_cases else [v * num_case + c for c in range(num_case)]
    for ip in panels:
        ax = axes_flat[ip]
        ax.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lat1, lat2)
        ax.set_xticks(lat_ticks)
        ax.set_xticklabels([f'{t}°' for t in lat_ticks], fontsize=FONT_SMALL)
        ax.tick_params(labelsize=FONT_SMALL)
        ax.set_xlabel('Latitude', fontsize=FONT_SMALL)
        title_right = var_opts.get('str', var[v]) + (' ' + month_str if month_str else '')
        if plot_diff and num_case > 1: title_right += ' (diff)' if not overlay_cases else ''
        ax.set_title(title_right, loc='right', fontsize=FONT_MED, pad=3)
        if overlay_cases and num_case > 1:
            ax.legend(fontsize=FONT_SMALL, loc='best', framealpha=0.6)

# hide unused axes
for i in range(num_plots, len(axes_flat)):
    axes_flat[i].set_visible(False)

#---------------------------------------------------------------------------------------------------
os.makedirs(os.path.dirname(fig_file) or '.', exist_ok=True)
fig.savefig(fig_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n{fig_file}')
#---------------------------------------------------------------------------------------------------
