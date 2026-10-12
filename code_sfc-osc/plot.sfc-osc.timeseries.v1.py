import os, copy, glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import hapy

#---------------------------------------------------------------------------------------------------
case_name, case, case_dir, case_sub, case_grid = [], [], [], [], []
clr, dsh, mrk, remap_flag = [], [], [], []

def add_case(case_in, n=None, p=None, s=None, g=None, d=0, c='black', m=0, r=False):
    case.append(case_in);  case_name.append(n or '')
    case_dir.append(p);    case_sub.append(s);  case_grid.append(g)
    dsh.append(d);         clr.append(c);       mrk.append(m)
    remap_flag.append(r)

#---------------------------------------------------------------------------------------------------
var, vclr, vdsh = [], [], []

def add_var(var_name, clr='black', dsh=0):
    var.append(var_name); vclr.append(clr); vdsh.append(dsh)

#---------------------------------------------------------------------------------------------------
topo_file     = '/lus/flare/projects/E3SMinput/data/atm/cam/topo/USGS-gtopo30_ne256np4pg2_x6t-SGH.c20210614.nc'
scrip_file_path = '/lus/flare/projects/E3SM_Dec/whannah/files_grid/ne30pg2_scrip.nc'
tmp_scratch   = '/lus/flare/projects/E3SM_Dec/whannah/scratch'

# add_case('E3SM.2026-time-step-test-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_64.NCPL_288',n='NCPL_288', d=0, c='blue', p=tmp_scratch, s='run')
add_case('E3SM.2026-time-step-test-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_64.NCPL_120',n='NCPL_120', d=0, c='red',  p=tmp_scratch, s='run')

first_file,num_files = 2,0

htype = 'output.scream.2D.INSTANT.nsteps_x1'

# ds_topo  = xr.open_dataset(topo_file)
# landfrac = ds_topo['LANDFRAC']

ivar = 'surf_mom_flux'

add_var('T_2m')
add_var('surf_sens_flux')
add_var('wind_speed_10m')

num_plot_col = 1
fig_file     =  'figs_sfc-osc/sfc-osc.timeseries.v1.png'
print_stats  = True

#---------------------------------------------------------------------------------------------------
num_var  = len(var)
num_case = len(case)
lw       = 2.0      # line width (was xyLineThicknessF / ~4 px in NGL)

if 'first_file' not in locals(): first_file = 0
if 'num_files'  not in locals(): num_files = 0
#---------------------------------------------------------------------------------------------------
num_plot_row = int(np.ceil(num_var / num_plot_col))
fig, axes = plt.subplots(
    num_plot_row, num_plot_col,
    figsize=(10, 3 * num_plot_row),
    squeeze=False,
)
fig.subplots_adjust(top=0.93, hspace=0.45)

#---------------------------------------------------------------------------------------------------
icol_list     = []
icol_msg_list = []

for v in range(num_var):
    hapy.print_line()
    print('  ' + f'{hapy.tclr.GREEN}var: {var[v]}{hapy.tclr.END}')

    time_list = []
    data_list = []

    for c in range(num_case):
        print('    ' + f'{hapy.tclr.CYAN}case: {case[c]}{hapy.tclr.END}')

        # if 'NCPL_288' in case[c]: first_file = int(288 / 10)
        # if 'NCPL_120' in case[c]: first_file = int(120 / 10)

        file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/*{htype}*'
        file_list = sorted(glob.glob(file_path))
        if first_file!=0: file_list = file_list[first_file:]
        if num_files !=0: file_list = file_list[:num_files]

        ds = xr.open_mfdataset(file_list, data_vars='all')

        #-----------------------------------------------------------------------
        lat1, lat2 = -30, 30
        mask = xr.DataArray(np.ones(len(ds['lat']), dtype=bool), coords=ds['lat'].coords)
        if 'lat1' in locals(): mask = mask & (ds['lat'] >= lat1) & (ds['lat'] <= lat2)
        if 'lon1' in locals(): mask = mask & (ds['lon'] >= lon1) & (ds['lon'] <= lon2)
        mask.load()
        ds       = ds.where(mask, drop=True)
        # landfrac = landfrac.where(mask, drop=True)

        #-----------------------------------------------------------------------
        print(f'{hapy.tclr.RED}WARNING - special time coord handling in place{hapy.tclr.END}')
        time_day = (ds.time.dt.day
                    + ds.time.dt.hour   / 24
                    + ds.time.dt.minute / (24 * 60)
                    + ds.time.dt.second / (24 * 60 * 60))

        t_beg = np.where(time_day > 1.5)[0][0]
        t_end = np.where(time_day > 2.5)[0][0]
        ds = ds.isel(time=slice(t_beg, t_end))

        #-----------------------------------------------------------------------
        if v == 0:
            data = ds[ivar]
            if ivar == 'surf_mom_flux':
                data = np.sqrt(np.square(data.isel(dim2=0, drop=True))
                             + np.square(data.isel(dim2=1, drop=True)))
            lat = ds['lat'].values
            lon = ds['lon'].values

            tmp_idx         = data.argmax().values
            (itime, icol)   = np.unravel_index(tmp_idx, data.shape)

            msg  = f'   icol: {icol}'
            msg += f'   lat: {lat[icol]:6.2f}'
            msg += f'   lon: {lon[icol]:6.2f}'
            # msg += f'   landfrac: {landfrac[icol].values:6.2f}'
            msg += f'   itime: {itime}'
            msg += f'   min({ivar}): {data[itime, icol].values}'
            print(f'\n{msg}\n')
            print(f' ({lat[icol]:6.2f}, {lon[icol]:6.2f})')
            icol_msg_list.append(msg)
            icol_list.append(icol)

        #-----------------------------------------------------------------------
        data = ds[var[v]].isel(ncol=icol_list[c])
        if print_stats:
            hapy.print_stat(data, name='      \n' + var[v], indent='      ', compact=True)

        # elapsed time in days from first sample
        time_vals = (data.time - data.time[0]).astype('float').values / 86400e9
        time_list.append(time_vals)
        data_list.append(data.values)

    #---------------------------------------------------------------------------
    row = v // num_plot_col
    col = v %  num_plot_col
    ax  = axes[row, col]

    for c in range(num_case):
        ax.plot(time_list[c], data_list[c],
                color=clr[c],
                linestyle=['-', '--', ':', '-.'][dsh[c] % 4],
                linewidth=lw,
                label=case_name[c])

    ax.set_xlabel('Time [days]')
    ax.set_title(var[v], fontsize=10)
    ax.set_xlim(np.min([t.min() for t in time_list]),
                np.max([t.max() for t in time_list]))
    ax.set_ylim(np.min([np.nanmin(d) for d in data_list]),
                np.max([np.nanmax(d) for d in data_list]))
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)

    if num_case > 1:
        ax.legend(fontsize=8, loc='best', framealpha=0.7)

#-------------------------------------------------------------------------------
for v in range(num_var, num_plot_row * num_plot_col):
    axes[v // num_plot_col, v % num_plot_col].set_visible(False)

#-------------------------------------------------------------------------------
os.makedirs(os.path.dirname(fig_file), exist_ok=True)
fig.savefig(fig_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {fig_file}')

#-------------------------------------------------------------------------------
for c in range(num_case):
    print()
    print('    ' + f'{hapy.tclr.CYAN}case: {case[c]}{hapy.tclr.END}')
    print('    ' + icol_msg_list[c])