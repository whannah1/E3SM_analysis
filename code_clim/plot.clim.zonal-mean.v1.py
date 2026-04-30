#---------------------------------------------------------------------------------------------------
# Plot the zonal mean of the specified variables
# Rewritten from PyNGL to matplotlib
#---------------------------------------------------------------------------------------------------
import copy, cftime, warnings, os, xarray as xr, numpy as np, cmocean, glob, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import hapy; host = hapy.get_host()
#---------------------------------------------------------------------------------------------------
name, case, case_dir, case_sub, case_grid = [], [], [], [], []
def add_case(case_in, n=None, p=None, s=None, g=None, d=None, c=None, **kwargs):
    global name, case, case_dir, case_sub
    tmp_name = '' if n is None else n
    case.append(case_in); name.append(tmp_name); case_dir.append(p)
    case_sub.append(s); case_grid.append(g)
#---------------------------------------------------------------------------------------------------
var, var_str, lev_list_var, tem_list = [], [], [], []
def add_var(var_name, lev=-1, tem=False, s=None):
    if lev == -1: lev = np.array([0])
    var.append(var_name); lev_list_var.append(lev)
    tem_list.append(tem)
    var_str.append(var_name if s is None else s)
#---------------------------------------------------------------------------------------------------
if host == 'nersc':

    tmp_scratch = f'/pscratch/sd/w/whannah/e3sm_scratch/pm-cpu'
    add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8',           n='E3SM control',   p=tmp_scratch, s='run')
    add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_55', n='E3SM top_km_55', p=tmp_scratch, s='run')
    add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_50', n='E3SM top_km_50', p=tmp_scratch, s='run')
    add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_45', n='E3SM top_km_45', p=tmp_scratch, s='run')
    add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_40', n='E3SM top_km_40', p=tmp_scratch, s='run')
    add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_35', n='E3SM top_km_35', p=tmp_scratch, s='run')
    add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_30', n='E3SM top_km_30', p=tmp_scratch, s='run')
    add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_25', n='E3SM top_km_25', p=tmp_scratch, s='run')
    htype = 'eam.h0'

    ### oscillation diagnostic tests
    # tmp_scratch = '/pscratch/sd/w/whannah/e3sm_scratch/pm-gpu'
    # tmp_scratch = '/pscratch/sd/w/whannah/scream_scratch/pm-gpu'
    # tmp_grid_file = '/pscratch/sd/w/whannah/files_grid/scrip_ne256pg2.nc'   
    # add_case('E3SM.2026-osc-test-01.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_32',           n='ne256 control',   p=tmp_scratch,s='run',scrip_file=tmp_grid_file)
    # add_case('E3SM.2026-osc-test-03.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_32.imp_flux_1',n='ne256 imp_flux_1',p=tmp_scratch,s='run',scrip_file=tmp_grid_file)
    # first_file,num_files = 2,None
    # htype = 'output.scream.2D.AVERAGE.ndays_x1.'
    # add_var('precip_total_surf_mass_flux')
    # add_var('LiqWaterPath')
    # add_var('IceWaterPath')
    # add_var('surf_sens_flux')
    # add_var('surface_upward_latent_heat_flux')

#---------------------------------------------------------------------------------------------------
# target_plev_mb = np.array([   1.,    2.,    3.,    5.,    7.,   10.,   20.,   30.,   50.,   70.,
#                         100.,  125.,  150.,  175.,  200.,  225.,  250.,  300.,  350.,  400.,
#                         450.,  500.,  550.,  600.,  650.,  700.,  750.,  775.,  800.,  825.,
#                         850.,  875.,  900.,  925.,  950.,  975., 1000.])
# target_plev_mb = np.array([   1.,    2.,    3.,    5.,    7.,   10.,   20.,   30.,   50.,   70.,  100.,  125.,  150.,])
target_plev_mb = np.array([0.1, 0.2, 0.5, 1., 2., 3., 5., 7., 10., 20., 30., 50., 70., 100., 125., 150.])

add_var('U',s='Zonal Wind [m/s]')
# add_var('V',s='Meridional Wind [m/s]')
# add_var('OMEGA',s='Omega [Pa/s]')
# add_var('T',s='Temperature [K]')

month_list = None
# month_list = [1]
month_list = [7]
# month_list = [6, 7, 8]   # JJA

#---------------------------------------------------------------------------------------------------
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
month_str = ('(' + ','.join(month_names[m] for m in month_list) + ')'
             if month_list is not None else '')
#---------------------------------------------------------------------------------------------------
recalculate = True

plot_diff            = False
var_x_case           = False
use_common_label_bar = False

num_plot_col = 2

# yr1, yr2 = 1995, 1995
yr1, yr2 = 1995, 1999

fig_file       = 'figs_clim/clim.zonal-mean.v1.png'
tmp_file_head  = 'clim.zonal-mean.v1'

lat1, lat2, dlat = -88., 88., 4

#---------------------------------------------------------------------------------------------------
# Utility: mimic ngl.nice_cntr_levels
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
#---------------------------------------------------------------------------------------------------
def get_tmp_file_name(case_in, var_in):
    tmp_file = f'data_temp/{tmp_file_head}.{case_in}.{var_in}.{yr1}-{yr2}'
    if month_list is not None:
        tmp_file += '.m' + '-'.join(str(m) for m in month_list)
    return tmp_file + '.nc'

#---------------------------------------------------------------------------------------------------
# Main data loading / calculation loop (unchanged logic)
#---------------------------------------------------------------------------------------------------
num_var  = len(var)
num_case = len(case)

all_data = {}   # keyed by (v, c) -> data_zonal.values
all_lat  = {}
all_lev  = {}

for v in range(num_var):
    print()
    print(f'  var: {hapy.tclr.MAGENTA}{var[v]}{hapy.tclr.END}')
    data_list = []
    lat_list  = []
    lev_list  = []

    for c in range(num_case):
        print('    case: ' + hapy.tclr.GREEN + case[c] + hapy.tclr.END)
        #-----------------------------------------------------------------------
        if recalculate:

            if case[c] == 'ERA5':
                lat_name, lon_name = 'lat', 'lon'
                xy_dims = (lon_name, lat_name)
                obs_root = '/global/cfs/cdirs/e3sm/diagnostics/observations/Atm/time-series/ERA5'
                if var[v] == 'U' : obs_var, input_file_name = 'ua',   f'{obs_root}/ua_197901_201912.nc'
                if var[v] == 'V' : obs_var, input_file_name = 'va',   f'{obs_root}/va_197901_201912.nc'
                if var[v] == 'T' : obs_var, input_file_name = 'ta',   f'{obs_root}/ta_197901_201912.nc'
                if var[v] == 'Q' : obs_var, input_file_name = 'hus',  f'{obs_root}/hus_197901_201912.nc'
                if var[v] == 'O3': obs_var, input_file_name = 'tro3', f'{obs_root}/tro3_197901_201912.nc'

                ds   = xr.open_dataset(input_file_name)
                ds = ds.where(ds['time.year'] >= yr1, drop=True)
                ds = ds.where(ds['time.year'] <= yr2, drop=True)
                #---------------------------------------------------------------
                if month_list is not None:
                    ds = ds.sel(time=ds['time.month'].isin(month_list))
                #---------------------------------------------------------------
                lat  = ds['lat']
                area = hapy.calculate_area_from_latlon(ds['lon'].values, ds['lat'].values,
                                                       ds['lon_bnds'].values, ds['lat_bnds'].values)
                area = xr.DataArray(area, coords=[ds['lat'], ds['lon']])
                data = ds[obs_var]
                data = data.rename({'plev': 'lev'})
                data['lev'] = data['lev'] / 1e2
                data = data.sel(lev=lev)

            else:
                htype_tmp = None
                if 'htype' in globals(): htype_tmp = htype
                tmp_sub = case_sub[c]
                if tem_list[v]: htype,tmp_sub = 'tem','h0-tem'
                #---------------------------------------------------------------
                if 'eam.h' in htype:
                    file_path = f'{case_dir[c]}/{case[c]}/{tmp_sub}/{case[c]}*{htype}*'
                else:
                    file_path = f'{case_dir[c]}/{case[c]}/{tmp_sub}/*{htype}*'
                #---------------------------------------------------------------
                file_list = sorted(glob.glob(file_path))
                # file_list_all = file_list; file_list = []
                # for f in range(len(file_list_all)):
                #     yr = int(file_list_all[f][-10:-10+4])
                #     if yr >= yr1 and yr <= yr2:
                #         file_list.append(file_list_all[f])
                if 'first_file' in locals(): file_list = file_list[first_file:]
                if 'num_files'  in locals(): file_list = file_list[:num_files]
                if file_list == []: raise ValueError(f'No files found for path: {file_path}')
                #---------------------------------------------------------------
                ds = xr.open_mfdataset(file_list)
                ds = ds.where(ds['time.year'] >= yr1, drop=True)
                ds = ds.where(ds['time.year'] <= yr2, drop=True)
                #---------------------------------------------------------------
                if month_list is not None:
                    ds = ds.sel(time=ds['time.month'].isin(month_list))
                #---------------------------------------------------------------
                if 'ncol' in ds.dims:
                    area = ds['area']
                    lat  = ds['lat']
                    ps_var = None
                    if 'ps' in ds: ps_var = 'ps'
                    if 'PS' in ds: ps_var = 'PS'

                    data = hapy.vinth2p_simple( ds[var[v]], ds['hyam'], ds['hybm'], 
                                                target_plev_mb*1e2, ds[ps_var],
                                                interp_type='linear', extrapolate=False)
                else:
                    lat  = ds['lat']
                    data = ds[var[v]].mean(dim='time')
                    data = data.sel(lev=target_plev_mb)
                #---------------------------------------------------------------
                # adjust units

                # if 'unit_fac' in var_opts: data = data*var_opts['unit_fac']
                # if var[v] == 'O3': data = data * 1e6
                # if var[v] in ['PUTEND', 'UTGWSPEC', 'FUTGWSPEC', 'BUTGWSPEC', 'UTGWORO']:
                #     data = data * 86400.
                # if tem_list[v]: data = data * 86400.
            #-------------------------------------------------------------------
            if 'ncol' in ds.dims:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    bin_ds = hapy.bin_YbyX(data, lat,
                                         bin_min=lat1, bin_max=lat2, bin_spc=dlat,
                                         wgt=area, keep_lev=True, verbose=False)
                # lat_bins   = bin_ds['bin'].values
                data_zonal = bin_ds['bin_val'].transpose()
            #-------------------------------------------------------------------
            tmp_file = get_tmp_file_name(case[c], var[v])
            print(' '*6 + f'writing to file: {tmp_file}')
            ds_tmp = xr.Dataset()
            ds_tmp[var[v]] = data_zonal
            ds_tmp.to_netcdf(path=tmp_file, mode='w')
        else:
            tmp_file = get_tmp_file_name(case[c], var[v])
            print(' '*6 + f'reading from file: {tmp_file}')
            ds_tmp      = xr.open_dataset(tmp_file)
            data_zonal = ds_tmp[var[v]]
        #-----------------------------------------------------------------------
        data_list.append(data_zonal)
        # lat_list.append(data_zonal['lat'].values)
        # lev_list.append(data_zonal['lev'].values)
    #---------------------------------------------------------------------------
    for c in range(num_case):
        all_data[(v, c)] = data_list[c]
        # all_lat[(v, c)]  = lat_list[c]
        # all_lev[(v, c)]  = lev_list[c]

    #---------------------------------------------------------------------------
    # Stash per-variable data_list for diff calcs below
    all_data[('list', v)] = data_list
    # all_lat[('list', v)]  = lat_list
    # all_lev[('list', v)]  = lev_list


#---------------------------------------------------------------------------------------------------
# Build figure layout
#---------------------------------------------------------------------------------------------------
lat_ticks  = np.array([-90, -60, -30, 0, 30, 60, 90])
FONT_SMALL = 7
FONT_MED   = 8

if plot_diff:
    num_cols = num_case * 2 - 1
else:
    num_cols = int(np.ceil((num_var * num_case) / float(num_plot_col)))

num_cols = num_case
num_rows = num_var

fig_width  = 3.5 * num_cols
fig_height = 3.0 * num_rows
fig, axes  = plt.subplots(num_rows, num_cols,
                           figsize=(fig_width, fig_height),
                           squeeze=False)
fig.subplots_adjust(hspace=0.4, wspace=0.35)

#---------------------------------------------------------------------------------------------------
# Plotting loop
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
    data_list = all_data[('list', v)]
    # lat_list  = all_lat[('list', v)]
    # lev_list  = all_lev[('list', v)]

    # ------------------------------------------------------------------
    # Pre-compute diff range if needed
    # ------------------------------------------------------------------
    if plot_diff:
        tmp_data = [data_list[c] - data_list[0] for c in range(num_case)]
        diff_data_min = np.nanmin([np.nanmin(np.ma.masked_invalid(d)) for d in tmp_data])
        diff_data_max = np.nanmax([np.nanmax(np.ma.masked_invalid(d)) for d in tmp_data])

        if diff_data_min < (-1e3 * diff_data_max): diff_data_min = -diff_data_max
        if diff_data_max > (-1e3 * diff_data_min): diff_data_max = -diff_data_min

    # ------------------------------------------------------------------
    # Build contour levels for the absolute panels
    # ------------------------------------------------------------------
    if var[v] == 'OMEGA':
        abs_levels = np.linspace(-1, 1, 21) * 0.002
    elif var[v] == 'U':
        abs_levels = np.linspace(-50, 50, 41)
    elif tem_list[v]:
        abs_levels = np.linspace(-4, 4, 21)
    else:
        data_min = np.nanmin([np.nanmin(d) for d in data_list])
        data_max = np.nanmax([np.nanmax(d) for d in data_list])
        cmin, cmax, _ = nice_cntr_levels(data_min, data_max, max_steps=11, aboutZero=False)
        abs_levels = np.linspace(cmin, cmax, 21)

    # Override for GW tendency variables (log-style levels)
    if var[v] in ['PUTEND', 'UTGWSPEC', 'BUTGWSPEC', 'UTGWORO']:
        cn_lev2  = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])
        cn_lev1  = cn_lev2[::-1] * -1
        abs_levels = np.concatenate([cn_lev1, [0], cn_lev2])

    #---------------------------------------------------------------------------
    # Build contour levels for difference panels

    if plot_diff:
        if var[v] in ['PUTEND', 'UTGWSPEC', 'BUTGWSPEC', 'UTGWORO']:
            cn_lev2  = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])
            cn_lev1  = cn_lev2[::-1] * -1
            diff_levels = np.concatenate([cn_lev1, [0], cn_lev2])
        else:
            cmin, cmax, _ = nice_cntr_levels(diff_data_min, diff_data_max,
                                              max_steps=11, aboutZero=True)
            diff_levels = np.linspace(cmin, cmax, 21)

    #---------------------------------------------------------------------------
    # Colormaps

    # cmap_abs  = plt.cm.RdYlBu_r      # same feel as NGL default for filled contours
    cmap_abs  = 'viridis'
    cmap_diff = cmocean.cm.balance

    if var[v]=='OMEGA': cmap_abs = cmocean.cm.balance
    
    # Norm helpers
    def make_norm(levels):
        return mcolors.BoundaryNorm(levels, ncolors=256, clip=True)
    #---------------------------------------------------------------------------
    # Panel loop
    
    num_grp = 2 if plot_diff else 1

    for i in range(num_grp):          # i=0 → absolute,  i=1 → difference
        for c in range(num_case):
            if plot_diff and i > 0 and c == 0:
                continue              # no diff panel for the reference case

            # Map (i,c) to column index to match original panel ordering:
            #   [ctrl | diff1 | diff2 | ... | diffN] for each var row
            if plot_diff:
                col = c if i == 0 else num_case + (c - 1)
            else:
                col = c if var_x_case else c
            row = v

            ax = axes[row, col]

            LAT = data_list[c]['bin'].values
            LEV = data_list[c]['plev'].values/1e2

            # LAT = lat_list[c]
            # LEV = lev_list[c]

            #-------------------------------------------------------------------
            # mask where interpolated data is exactly zero
            data_list[c] = np.ma.masked_where(data_list[c]==0, data_list[c])
            #-------------------------------------------------------------------
            if plot_diff and i > 0:
                Z      = np.ma.masked_invalid(data_list[c] - data_list[0])
                levels = diff_levels
                cmap   = cmap_diff
                title_left  = f'{name[c]} - {name[0]}'
                title_right = f'{var[v]} (diff)' + (' ' + month_str if month_str else '')
            else:
                Z      = np.ma.masked_invalid(data_list[c])
                levels = abs_levels
                cmap   = cmap_abs
                title_left  = name[c]
                title_right = var[v] + (' ' + month_str if month_str else '')

            norm = make_norm(levels)
            cf   = ax.contourf(LAT, LEV, Z, levels=levels,
                               cmap=cmap, norm=norm, extend='both')
            ax.contour(LAT, LEV, Z, levels=levels, colors='k',
                       linewidths=0.2, alpha=0.4)

            # Axes formatting
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            ax.set_ylim(max(LEV), min(LEV))   # pressure decreasing upward
            ax.set_xlim(lat1, lat2)
            ax.set_xticks(lat_ticks)
            ax.set_xticklabels([f'{t}°' for t in lat_ticks], fontsize=FONT_SMALL)
            ax.tick_params(axis='y', labelsize=FONT_SMALL)
            ax.set_xlabel('Latitude', fontsize=FONT_SMALL)
            if col == 0:
                ax.set_ylabel('Pressure [hPa]', fontsize=FONT_SMALL)

            # Subtitles: left / right above panel (mimics hs.set_subtitles)
            ax.set_title(title_left,  loc='left',  fontsize=FONT_MED, pad=3)
            ax.set_title(title_right, loc='right', fontsize=FONT_MED, pad=3)

            # Colorbar (per-panel, unless using a common bar)
            if not use_common_label_bar:
                cb = fig.colorbar(cf, ax=ax, pad=0.18, fraction=0.04, aspect=30)
                cb.ax.tick_params(labelsize=FONT_SMALL)
                cb.set_ticks(levels[::max(1, len(levels)//10)])

    # ------------------------------------------------------------------
    # Hide any unused axes in this row
    # ------------------------------------------------------------------
    for col in range(num_cols):
        if not axes[row, col].collections and not axes[row, col].has_data():
            axes[row, col].set_visible(False)

# ------------------------------------------------------------------
# Common label bar (optional)
# ------------------------------------------------------------------
if use_common_label_bar:
    # Attach a single colorbar to the right of the whole figure
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap_abs, norm=make_norm(abs_levels))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

#---------------------------------------------------------------------------------------------------
# Save
#---------------------------------------------------------------------------------------------------
os.makedirs(os.path.dirname(fig_file), exist_ok=True)
fig.savefig(fig_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nFigure saved to: {fig_file}')
#---------------------------------------------------------------------------------------------------