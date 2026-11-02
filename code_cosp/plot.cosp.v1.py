#---------------------------------------------------------------------------------------------------
# Compare COSP joint histograms (cloud top pressure/height vs optical depth)
# Each row is a variable, columns are the cases followed by the difference (case[1:] - case[0])
#---------------------------------------------------------------------------------------------------
import os, glob, xarray as xr, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hapy; host = hapy.get_host()
#---------------------------------------------------------------------------------------------------
name, case, case_dir, case_sub = [], [], [], []
def add_case(case_in, n=None, p=None, s=None):
    case.append(case_in); name.append(case_in if n is None else n)
    case_dir.append(p); case_sub.append(s)
#---------------------------------------------------------------------------------------------------
var, var_str = [], []
def add_var(var_name, s=None):
    var.append(var_name); var_str.append(var_name if s is None else s)
#---------------------------------------------------------------------------------------------------
if host == 'olcf':
    tmp_path,tmp_sub = '/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch','run'
    add_case('E3SM.2026-COSP-TEST-00.ne30pg2.cosp_impl_f90', n='COSP f90', p=tmp_path, s=tmp_sub)
    add_case('E3SM.2026-COSP-TEST-00.ne30pg2.cosp_impl_cxx', n='COSP cxx', p=tmp_path, s=tmp_sub)
    htype,first_file,num_files = 'output.1ma.AVERAGE.nmonths_x1',0,None
#---------------------------------------------------------------------------------------------------
add_var('isccp_ctptau', s='ISCCP CTP-tau')
add_var('modis_ctptau', s='MODIS CTP-tau')
add_var('misr_cthtau',  s='MISR CTH-tau')

# latitude bands to average over - one figure per band
lat_bands = []
lat_bands.append(( -90, 90,'global'))
# lat_bands.append(( -30, 30,'tropics'))
# lat_bands.append((  30, 60,'NH-midlat'))
# lat_bands.append(( -60,-30,'SH-midlat'))

fig_file_root = 'figs_cosp/cosp.v1'

print_stats = True
#---------------------------------------------------------------------------------------------------
num_case, num_var = len(case), len(var)
#---------------------------------------------------------------------------------------------------
def get_file_list(c):
    file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/{htype}*.nc'
    file_list = sorted(glob.glob(file_path))
    if file_list == []: raise ValueError(f'No files found for: {file_path}')
    last_file = None if num_files is None else first_file+num_files
    return file_list[first_file:last_file]
#---------------------------------------------------------------------------------------------------
def get_bin_labels(ds, dim):
    # use bin edges for tick labels since the COSP bins are not evenly spaced
    bnds = ds[f'{dim}_bnds'].values
    edges = np.append(bnds[:,0], bnds[-1,1])
    if dim == 'cosp_prs': edges = edges/1e2   # Pa => hPa
    if dim == 'cosp_cth': edges = edges/1e3   # m => km
    labels = [f'{e:g}' for e in edges]
    # the outer COSP bins are open-ended (e.g. tau > 60, cth < 0)
    if bnds[-1,1] >= 1e5 or (dim=='cosp_cth' and edges[-1] > 50): labels[-1] = ''
    if dim == 'cosp_cth' and edges[0] < 0: labels[0] = ''
    return labels
dim_str = {'cosp_tau':'optical depth', 'cosp_prs':'cloud top pressure [hPa]', 'cosp_cth':'cloud top height [km]'}
#---------------------------------------------------------------------------------------------------
# load data and compute time and area-weighted column averages
#---------------------------------------------------------------------------------------------------
hist_list = [[[None]*num_case for v in range(num_var)] for b in range(len(lat_bands))]
labels = [None]*num_var
for c in range(num_case):
    print(f'  case: {case[c]}')
    ds = xr.open_mfdataset(get_file_list(c), combine='by_coords', data_vars='minimal', coords='minimal', compat='override')
    lat, area = ds['lat'], ds['area']
    if 'time' in lat.dims:  lat  = lat.isel(time=0)
    if 'time' in area.dims: area = area.isel(time=0)
    for v in range(num_var):
        # columns that never saw sunlight during the averaging window are filled (NaN)
        data = ds[var[v]].mean(dim='time', skipna=True).load()
        ydim = [d for d in data.dims if d not in ['ncol','cosp_tau']][0]
        data = data.transpose('ncol','cosp_tau',ydim)
        if labels[v] is None: labels[v] = (ydim, get_bin_labels(ds,'cosp_tau'), get_bin_labels(ds,ydim))
        for b,(lat1,lat2,band_name) in enumerate(lat_bands):
            mask = (lat>=lat1) & (lat<=lat2)
            tmp_data = data.where(mask, drop=True)
            tmp_area = area.where(mask, drop=True)
            # only weight columns with valid data
            wgt = tmp_area.where(tmp_data.notnull())
            hist_list[b][v][c] = ( (tmp_data*wgt).sum(dim='ncol') / wgt.sum(dim='ncol') ).values
#---------------------------------------------------------------------------------------------------
# print summary stats
#---------------------------------------------------------------------------------------------------
if print_stats:
    for b,(lat1,lat2,band_name) in enumerate(lat_bands):
        print(f'\n  {band_name} ({lat1} to {lat2})')
        for v in range(num_var):
            print(f'    {var[v]}')
            for c in range(num_case):
                hist = hist_list[b][v][c]
                msg = f'      {name[c]:12}  total: {np.sum(hist):10.4f} %'
                if c > 0:
                    diff = hist - hist_list[b][v][0]
                    msg += f'  max abs diff: {np.max(np.abs(diff)):10.3e}  rms diff: {np.sqrt(np.mean(diff**2)):10.3e}'
                print(msg)
#---------------------------------------------------------------------------------------------------
# plot
#---------------------------------------------------------------------------------------------------
num_col = num_case + (num_case-1)
for b,(lat1,lat2,band_name) in enumerate(lat_bands):
    fig, axs = plt.subplots(num_var, num_col, figsize=(4.5*num_col, 3.8*num_var), squeeze=False)
    for v in range(num_var):
        ydim, tau_labels, y_labels = labels[v]
        ny, nt = len(y_labels)-1, len(tau_labels)-1
        hmax = np.nanmax([np.nanmax(h) for h in hist_list[b][v]])
        for k in range(num_col):
            ax = axs[v,k]
            if k < num_case:
                hist = hist_list[b][v][k]
                im = ax.pcolormesh(np.arange(nt+1), np.arange(ny+1), hist.T, cmap='Blues', vmin=0, vmax=hmax)
                title, cbar_str = name[k], '%'
                # annotate bin values
                for i in range(nt):
                    for j in range(ny):
                        ax.text(i+0.5, j+0.5, f'{hist[i,j]:.1f}', ha='center', va='center', fontsize=6,
                                color='white' if hist[i,j] > 0.6*hmax else 'black')
            else:
                c = k - num_case + 1
                diff = hist_list[b][v][c] - hist_list[b][v][0]
                dmax = np.nanmax(np.abs(diff))
                if dmax == 0:
                    dmax = 1
                    ax.text(nt/2, ny/2, 'identical', ha='center', va='center', fontsize=12, color='gray')
                im = ax.pcolormesh(np.arange(nt+1), np.arange(ny+1), diff.T, cmap='RdBu_r', vmin=-dmax, vmax=dmax)
                title, cbar_str = f'{name[c]} - {name[0]}', '% diff'
            ax.set_xticks(np.arange(nt+1)); ax.set_xticklabels(tau_labels, fontsize=7)
            ax.set_yticks(np.arange(ny+1)); ax.set_yticklabels(y_labels, fontsize=7)
            ax.set_xlabel(dim_str['cosp_tau'], fontsize=8)
            if k == 0: ax.set_ylabel(f'{var_str[v]}\n{dim_str[ydim]}', fontsize=8)
            ax.set_title(title, fontsize=9, loc='left')
            ax.set_title(band_name, fontsize=9, loc='right')
            cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
            cb.set_label(cbar_str, fontsize=7); cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig_file = f'{fig_file_root}.{band_name}.png'
    os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    fig.savefig(fig_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\n{fig_file}\n')
#---------------------------------------------------------------------------------------------------
