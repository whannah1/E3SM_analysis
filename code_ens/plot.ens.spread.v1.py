# --------------------------------------------------------------------------------------------------
# Plot regional-mean time series for groups of ensemble members (top panel) and the
# spread across members within each ensemble at each time (bottom panel)
# - region is defined by cells that touch a polygon (e.g. a US state) and exceed a
#   minimum land fraction - fixed-SST ocean points would systematically damp the spread
# - each member is a standalone case with its own case directory
# --------------------------------------------------------------------------------------------------
import os, glob, numpy as np, xarray as xr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
import hapy
host = hapy.get_host()
# --------------------------------------------------------------------------------------------------
# case options:  n=name  p=case root  s=sub-dir  e=ensemble group name (None for control)
case,case_opts_list = [],[]
def add_case(case_in,**kwargs):
    case.append(case_in)
    case_opts = {}
    for k, val in kwargs.items(): case_opts[k] = val
    if 'n' not in case_opts: case_opts['n'] = case_in
    if 'e' not in case_opts: case_opts['e'] = None
    case_opts_list.append(case_opts)
# --------------------------------------------------------------------------------------------------
# var options:  str=label  htype=history file type  units=label for y-axis
#               daily=max/min/mean - per-column daily (UTC) statistic before regional mean
var,var_opts_list = [],[]
def add_var(var_name,**kwargs):
    var.append(var_name)
    var_opts = {}
    for k, val in kwargs.items(): var_opts[k] = val
    if 'str' not in var_opts: var_opts['str'] = var_name
    var_opts_list.append(var_opts)
# --------------------------------------------------------------------------------------------------
# ensemble options:  clr=color for all members of the group
ens_clr = {}
# --------------------------------------------------------------------------------------------------
if host=='nersc':

    ### STRONG LDRD - 2022 California heat wave - white vs correlated noise IC perturbations
    ens_name = '2026-STRONG-HW'
    tmp_scratch = '/pscratch/sd/w/whannah/e3sm_scratch/pm-cpu/2026-STRONG'
    tmp_prefix = 'E3SM.2026-STRONG-HW.ne30pg2'
    num_mem = 10
    add_case(f'{tmp_prefix}.CTL', n='CTL', p=tmp_scratch, s='run')
    for ens,clr in [('WN','red'),('CN500','green'),('CN2000','blue')]:
        ens_clr[ens] = clr
        for m in range(1,num_mem+1):
            add_case(f'{tmp_prefix}.{ens}.m{m:02d}', n=f'{ens}.m{m:02d}', e=ens, p=tmp_scratch, s='run')

    # cell corners and land fraction - column order must match the history files
    domain_file = '/global/cfs/cdirs/e3sm/inputdata/share/domains/domain.lnd.ne30pg2_IcoswISC30E3r5.231121.nc'

    htype = 'eam.h1'  # hourly
    first_file,num_files = None,None

    # date of the target event (marked with a vertical line)
    event_date = '2022-09-06'

# --------------------------------------------------------------------------------------------------

add_var('TREFHT', str='TREFHT daily max', units='K', daily='max')
# add_var('TREFHT', str='TREFHT', units='K')
# add_var('TS',     str='TS',     units='K')
# add_var('QREFHT', str='QREFHT', units='kg/kg')
# add_var('T850',   str='T850',   units='K')
# add_var('Z500',   str='Z500',   units='m')

# --------------------------------------------------------------------------------------------------

# region definition - Natural Earth admin-1 name and minimum land fraction
region_name   = 'California'
min_land_frac = 0.5

spread_log_scale = True   # log y-axis for spread panel - spread grows by orders of magnitude
plot_region_map  = True   # separate map of the selected cells for verification
print_stats      = True

fig_file = f'figs_ens/ens.spread.v1.{ens_name}.{region_name}.png'
map_file = f'figs_ens/ens.spread.v1.region-map.{region_name}.png'

# --------------------------------------------------------------------------------------------------
if case==[]: raise ValueError('ERROR - case list is empty!')
if var==[]:  raise ValueError('ERROR - var list is empty!')
num_var,num_case = len(var),len(case)

if 'first_file' not in globals(): first_file = None
if 'num_files'  not in globals(): num_files  = None
if 'event_date' not in globals(): event_date = None

ens_list = list(ens_clr.keys())
# --------------------------------------------------------------------------------------------------
def get_file_list(c,htype_loc):
    case_opts = case_opts_list[c]
    file_path = f'{case_opts["p"]}/{case[c]}/{case_opts["s"]}/*.{htype_loc}.*.nc'
    file_list = sorted(glob.glob(file_path))
    if first_file is not None: file_list = file_list[first_file:]
    if num_files  is not None: file_list = file_list[:num_files]
    if file_list==[]: raise ValueError(f'ERROR: file list is empty - {file_path}')
    return file_list
# --------------------------------------------------------------------------------------------------
def get_region_polygon(name):
    shp_file = shpreader.natural_earth(resolution='10m', category='cultural',
                                       name='admin_1_states_provinces_lakes')
    for rec in shpreader.Reader(shp_file).records():
        if rec.attributes['name']==name and rec.attributes['admin']=='United States of America':
            return rec.geometry
    raise ValueError(f'ERROR: region not found in Natural Earth admin-1 file: {name}')
# --------------------------------------------------------------------------------------------------
def get_region_mask():
    # select cells whose polygon intersects the region and whose land fraction exceeds threshold
    ds = xr.open_dataset(domain_file)
    xv = ds['xv'].values.squeeze()
    yv = ds['yv'].values.squeeze()
    frac = ds['frac'].values.squeeze()
    xv = np.where(xv>180, xv-360, xv)
    region = get_region_polygon(region_name)
    # cheap bounding box pre-filter before exact polygon test
    rx1,ry1,rx2,ry2 = region.bounds
    cand = np.where( (xv.max(axis=1)>=rx1) & (xv.min(axis=1)<=rx2) & \
                     (yv.max(axis=1)>=ry1) & (yv.min(axis=1)<=ry2) )[0]
    touch = np.zeros(frac.shape, dtype=bool)
    for i in cand:
        cell = sgeom.Polygon(np.column_stack([xv[i],yv[i]]))
        touch[i] = cell.intersects(region)
    mask = touch & (frac>min_land_frac)
    print(' '*2+f'region: {region_name}  cells touching: {touch.sum()}'
          +f'  with land frac > {min_land_frac}: '+hapy.tclr.GREEN+f'{mask.sum()}'+hapy.tclr.END)
    return mask, touch, xv, yv, frac, region
# --------------------------------------------------------------------------------------------------
# Build region mask
# --------------------------------------------------------------------------------------------------
mask, touch, xv, yv, frac, region = get_region_mask()
col_idx = np.where(mask)[0]
# --------------------------------------------------------------------------------------------------
# Load data and compute area-weighted regional mean
# --------------------------------------------------------------------------------------------------
time_list = [[None]*num_case for v in range(num_var)]
data_list = [[None]*num_case for v in range(num_var)]
for v in range(num_var):
    var_opts = var_opts_list[v]
    htype_loc = var_opts.get('htype', htype)
    hapy.print_line()
    print(' '*2+'var: '+hapy.tclr.GREEN+var[v]+hapy.tclr.END)
    for c in range(num_case):
        print(' '*4+'case: '+hapy.tclr.CYAN+case[c]+hapy.tclr.END)
        file_list = get_file_list(c,htype_loc)
        ds = xr.open_mfdataset(file_list, combine='by_coords', data_vars='minimal',
                               coords='minimal', compat='override')
        area = ds['area']
        if 'time' in area.dims: area = area.isel(time=0)
        area = area.isel(ncol=col_idx)
        data = ds[var[v]].isel(ncol=col_idx)
        if c==0: start_time = ds['time'].values[0]
        daily = var_opts.get('daily', None)
        if daily is None:
            # time stamps mark the end of the averaging window - convert to days since start
            data = ( (data*area).sum(dim='ncol') / area.sum(dim='ncol') ).compute()
            time = data['time'].values
            time = np.array([ (t-start_time).total_seconds()/86400. for t in time ])
        else:
            # assign each record to a UTC day using the start of its averaging window, and
            # drop the zero-length initial record - daily stat is per column, then area mean
            tbnd = ds['time_bnds'].values
            dt = np.array([ (b[1]-b[0]).total_seconds() for b in tbnd ])
            day = np.array([ np.floor((b[0]-start_time).total_seconds()/86400.) for b in tbnd ])
            keep = dt>0
            data = data.isel(time=keep).assign_coords(day=('time',day[keep]))
            if   daily=='max':  data = data.groupby('day').max()
            elif daily=='min':  data = data.groupby('day').min()
            elif daily=='mean': data = data.groupby('day').mean()
            else: raise ValueError(f'unknown daily option: {daily}')
            data = ( (data*area).sum(dim='ncol') / area.sum(dim='ncol') ).compute()
            time = data['day'].values
        # ----------------------------------------------------------------------
        if print_stats:
            msg = ' '*6+f'min: {np.nanmin(data.values):10.4f}'
            msg += f'   max: {np.nanmax(data.values):10.4f}'
            msg += f'   mean: '+hapy.tclr.GREEN+f'{np.nanmean(data.values):10.4f}'+hapy.tclr.END
            print(msg)
        data_list[v][c] = data.values
        time_list[v][c] = time
# --------------------------------------------------------------------------------------------------
# lead time of target event
# --------------------------------------------------------------------------------------------------
event_lead = None
if event_date is not None:
    yr,mn,dy = [int(x) for x in event_date.split('-')]
    event_lead = (start_time.replace(year=yr,month=mn,day=dy,hour=0) - start_time).total_seconds()/86400.
# --------------------------------------------------------------------------------------------------
# Plot - one row per variable: raw time series (left) and ensemble spread (right)
# --------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(num_var, 2, figsize=(14, 3.5*num_var), squeeze=False)
for v in range(num_var):
    var_opts = var_opts_list[v]
    units = var_opts.get('units','')
    ax_ts, ax_sp = axes[v,0], axes[v,1]
    # --------------------------------------------------------------------------
    # raw time series - members first so the control is drawn on top
    for c in range(num_case):
        ens = case_opts_list[c]['e']
        if ens is None: continue
        ax_ts.plot(time_list[v][c], data_list[v][c], color=ens_clr[ens], linewidth=0.6, alpha=0.5)
    for c in range(num_case):
        if case_opts_list[c]['e'] is not None: continue
        ax_ts.plot(time_list[v][c], data_list[v][c], color='black', linewidth=1.5,
                   label=case_opts_list[c]['n'])
    for ens in ens_list:
        ax_ts.plot([], [], color=ens_clr[ens], linewidth=1.5, label=ens)
    # --------------------------------------------------------------------------
    # spread - standard deviation across members of each ensemble at each time
    for ens in ens_list:
        mem_idx = [c for c in range(num_case) if case_opts_list[c]['e']==ens]
        ntime = min( len(data_list[v][c]) for c in mem_idx )
        ens_data = np.stack([ data_list[v][c][:ntime] for c in mem_idx ])
        spread = np.std(ens_data, axis=0, ddof=1)
        ax_sp.plot(time_list[v][mem_idx[0]][:ntime], spread, color=ens_clr[ens], linewidth=1.5,
                   label=f'{ens} (n={len(mem_idx)})')
        if print_stats:
            msg = ' '*2+f'{ens:8s} spread'
            for lead in [1,3,5,10,15]:
                it = np.argmin(np.abs(time_list[v][mem_idx[0]][:ntime]-lead))
                msg += f'   day {lead:2d}: {spread[it]:8.4f}'
            print(msg)
    if spread_log_scale: ax_sp.set_yscale('log')
    # --------------------------------------------------------------------------
    for ax in [ax_ts, ax_sp]:
        if event_lead is not None:
            ax.axvline(event_lead, color='gray', linestyle='dashed', linewidth=0.8)
        ax.set_xlabel(f'Days since {str(start_time)[:13]}Z', fontsize=9)
        ax.set_xlim(0, max( np.nanmax(t) for t in time_list[v] ))
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc='best', framealpha=0.6)
    ax_ts.set_ylabel(f'[{units}]', fontsize=9)
    ax_sp.set_ylabel(f'std dev [{units}]', fontsize=9)
    ax_ts.set_title(f'{var_opts["str"]}', loc='left', fontsize=10)
    ax_sp.set_title(f'{var_opts["str"]} ensemble spread', loc='left', fontsize=10)
    for ax in [ax_ts, ax_sp]:
        ax.set_title(f'{region_name} (land frac > {min_land_frac}, {len(col_idx)} cols)',
                     loc='right', fontsize=8)
# --------------------------------------------------------------------------------------------------
plt.tight_layout(w_pad=2, h_pad=3)
os.makedirs(os.path.dirname(fig_file) or '.', exist_ok=True)
fig.savefig(fig_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n  {fig_file}\n')
# --------------------------------------------------------------------------------------------------
# Optional map of selected cells for verification
# --------------------------------------------------------------------------------------------------
if plot_region_map:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    from matplotlib.collections import PolyCollection
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
    rx1,ry1,rx2,ry2 = region.bounds
    ax.set_extent([rx1-3, rx2+3, ry1-3, ry2+3], crs=ccrs.PlateCarree())
    near = np.where( (xv.mean(axis=1)>rx1-4) & (xv.mean(axis=1)<rx2+4) & \
                     (yv.mean(axis=1)>ry1-4) & (yv.mean(axis=1)<ry2+4) )[0]
    verts = [ np.column_stack([xv[i],yv[i]]) for i in near ]
    fclr = [ 'tab:orange' if mask[i] else ('tab:gray' if touch[i] else 'none') for i in near ]
    pc = PolyCollection(verts, facecolors=fclr, edgecolors='k', linewidths=0.3, alpha=0.6,
                        transform=ccrs.PlateCarree())
    ax.add_collection(pc)
    for i in near:
        ax.text(xv[i].mean(), yv[i].mean(), f'{frac[i]:.1f}', fontsize=4, ha='center', va='center',
                transform=ccrs.PlateCarree(), clip_on=True)
    ax.add_geometries([region], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red',
                      linewidth=1)
    ax.coastlines(resolution='50m', linewidth=0.5)
    ax.set_title(f'{region_name}: orange = selected ({mask.sum()})  '
                 +f'gray = land frac <= {min_land_frac}', fontsize=8)
    fig.savefig(map_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  {map_file}\n')
# --------------------------------------------------------------------------------------------------
