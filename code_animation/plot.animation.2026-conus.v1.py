#!/usr/bin/env python
"""
Animation of native (unstructured) grid output from the 2026 CONUS RRM SCREAM case.

This is a matplotlib+cartopy replacement for the old PyNGL based script
(Research/E3SM/code_scream/plot.scream.animation.map.v1.py). Rasterization of the
native grid is done with either of two back ends, selected with --method:

    hapy     hapy.to_raster() - great-circle nearest neighbor from a KDTree built
             on the cell *centers*. Fast to build, works with any SCRIP file,
             ignores cell shape (cells appear as Voronoi-ish blobs).

    uxarray  UxDataArray.to_raster() - point-in-face search using the actual cell
             *boundaries*, so cell edges are exact. Grid construction and the face
             search are much more expensive, but the resulting pixel mapping is
             cached to disk here and reused by later runs.

Both back ends produce a (ny,nx) raster that is drawn with ax.imshow(), so the cost
per frame after the first is just "read one time slice + gather + save png".

--------------------------------------------------------------------------------
PERFORMANCE NOTES (this grid is ~31.6M columns, ~126 MB per float32 snapshot)
--------------------------------------------------------------------------------
 1. The pixel->cell mapping is built ONCE and reused for every frame. That means
    the figure and axes are created once, outside the frame loop, and only the
    image data is swapped (img.set_data). Creating a new figure per frame would
    rebuild the KDTree / face search every single time.
 2. hapy.to_raster caches its mapping internally, but ONLY when the data contains
    no NaN (NaNs change which columns enter the tree). EAMxx writes a _FillValue
    of 3.4e33 which xarray decodes to NaN, so NaNs are replaced with a sentinel
    before rasterizing and masked back out afterwards - see rasterize_hapy().
 3. Zooming in is handled by subsetting in ncol space FIRST (region_index), which
    shrinks the tree/face search from 31.6M cells to whatever is in the padded
    view. This is the single biggest win for zoomed animations.
 4. The ncol subset is applied in numpy after reading the full time slice, not as
    a lazy fancy-index. Fancy indexing through netCDF/dask turns one contiguous
    126 MB read into thousands of small strided reads on Lustre.
 5. Frame reads are prefetched on a background thread so the next slice comes off
    disk while the current frame is being rendered.
 6. bbox_inches='tight' is deliberately NOT used - it makes the png dimensions
    vary from frame to frame, which breaks video encoders. The layout is fixed
    with subplots_adjust instead.
 7. For long animations, launch several processes over disjoint frame ranges,
    e.g.   --frames 0:48 &  --frames 48:96 &
    Each pays the mapping build once. With --method uxarray the mapping is read
    from the on-disk cache, so extra processes are nearly free.

--------------------------------------------------------------------------------
EXAMPLES
--------------------------------------------------------------------------------
    # default: precip over CONUS, hapy rasterization
    python plot.animation.2026-conus.v1.py

    # zoomed into the southern plains, exact cell edges, high resolution
    python plot.animation.2026-conus.v1.py --region plains --method uxarray \
           --pixel-ratio 2 --dpi 200

    # just the first few frames while tuning color levels
    python plot.animation.2026-conus.v1.py --frames 0:4 --no-video

    # stack the variable panels in a 2-column grid instead of a single wide row
    python plot.animation.2026-conus.v1.py --num-plot-col 2
"""
from time import perf_counter
timer_start = perf_counter()
import os, sys, glob, shutil, argparse, subprocess as sp, numpy as np, xarray as xr, matplotlib, cmocean
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import cartopy.crs as ccrs, cartopy.feature as cfeature
from concurrent.futures import ThreadPoolExecutor
import hapy
host = hapy.get_host()
xr.set_options(use_new_combine_kwarg_defaults=True)

#---------------------------------------------------------------------------------------------------
# Case / data configuration
#---------------------------------------------------------------------------------------------------
if host=='olcf':
    name      = '2026 CONUS RRM (1024x2)'
    case      = 'conus1024x2v1pg2_RRSwISC6to18E3r5.F2010-SCREAMv1.20260803c'
    case_root = '/lustre/orion/cli115/proj-shared/brhillman/e3sm_scratch'
    case_sub  = 'run'
    grid_root = '/lustre/orion/cli115/world-shared/e3sm/2026-INCITE-CONUS-RRM/files_grid'
    scrip_file = f'{grid_root}/2025-scream-conus-1024x2-pg2_scrip.cdf5.nc'
    ux_grid_file = scrip_file
    # t1,t2,dt = 0,10,1
    t1,t2,dt = 0,144,2 # first 12 hours
    # t1,t2,dt = 288,288+5,1
    # the simulation starts at 00Z or 6pm central standard time (CST)
    # so let's start the animation 12 hours later at 6am CST
    # t1,t2,dt = 144,288,1
else:
    raise ValueError(f'ERROR - no case configuration for host: {host}')

data_dir = f'{case_root}/{case}/{case_sub}'

#---------------------------------------------------------------------------------------------------
# Variable configuration
#   htype        - EAMxx output stream name (one variable per stream in this case)
#   s            - string used for plot labels
#   scale        - multiplicative unit conversion applied after reading
#   cmap / clev / vmin / vmax - color settings, clev takes precedence over vmin/vmax
#---------------------------------------------------------------------------------------------------
var,var_opts_list = [],[]
def add_var(var_name,**kwargs):
    var.append(var_name); var_opts_list.append(dict(kwargs))
#---------------------------------------------------------------------------------------------------
# add_var('precip_total_surf_mass_flux', s='Precipitation [mm/day]', scale=86400*1e3,
#         htype='eamxx.precip_total_surf_mass_flux.INSTANT.nmins_x5',
#         cmap=cmocean.cm.rain, clev=np.logspace(np.log10(0.1),np.log10(200),num=40))

# add_var('LiqWaterPath', s='LWP [kg/m2]', #scale=1e3,
#         htype='eamxx.LiqWaterPath.INSTANT.nmins_x5',
#         cmap=cmocean.cm.rain, clev=np.logspace(np.log10(0.1),np.log10(200),num=40))

# add_var('T_2m', s='2m Temperature [K]',
#         htype='eamxx.T_2m.INSTANT.nmins_x30',
#         cmap=cmocean.cm.thermal, vmin=240, vmax=310)

add_var('LW_flux_up_at_model_top', s='OLR [W/m2]',
        htype='eamxx.LW_flux_up_at_model_top.INSTANT.nmins_x5',
        cmap='gray_r', vmin=80, vmax=320)

# add_var('VapWaterPath', s='VWP [kg/m2]',
#         htype='eamxx.VapWaterPath.INSTANT.nmins_x30',
#         cmap=cmocean.cm.haline, vmin=0, vmax=70)

# add_var('wind_speed_10m', s='10m Wind Speed [m/s]',
#         htype='eamxx.wind_speed_10m.INSTANT.nmins_x5',
#         cmap=cmocean.cm.speed, vmin=0, vmax=40)

# add_var('ps', s='Surface Pressure [hPa]', scale=1e-2,
#         htype='eamxx.ps.INSTANT.nmins_x30',
#         cmap='viridis', vmin=950, vmax=1030)

#---------------------------------------------------------------------------------------------------
# Region presets - lon may be given as 0:360 or -180:180, it is normalized below.
#   the "zoom" regions are the whole point of plotting the native grid, since the
#   refined part of this mesh is finer than anything a global remap would preserve
#---------------------------------------------------------------------------------------------------
region_dict = {
    'global'   : dict(lon1=None, lon2=None, lat1=None, lat2=None, proj='platecarree', clon=180),
    'namerica' : dict(lon1=-170, lon2= -50, lat1= 10, lat2= 75, proj='platecarree'),
    'conus'    : dict(lon1=-127, lon2= -65, lat1= 22, lat2= 52, proj='platecarree'),
    'conus_lcc': dict(lon1=-127, lon2= -65, lat1= 22, lat2= 52, proj='lambert'),
    'plains'   : dict(lon1=-105, lon2= -88, lat1= 28, lat2= 42, proj='platecarree'),
    'gulf'     : dict(lon1= -98, lon2= -80, lat1= 24, lat2= 34, proj='platecarree'),
    'westcoast': dict(lon1=-128, lon2=-114, lat1= 32, lat2= 50, proj='platecarree'),
    'frontrange':dict(lon1=-107, lon2=-102, lat1= 37, lat2= 42, proj='platecarree'),
    'okc'      : dict(lon1=-100, lon2= -95, lat1= 33, lat2= 38, proj='platecarree'),
    ### ~1 degree box - this is where the <1km spacing actually shows up
    'zoom'     : dict(lon1=-98.5, lon2=-97.0, lat1=35.0, lat2=36.2, proj='platecarree'),
}

#---------------------------------------------------------------------------------------------------
# Default run settings (all overridable on the command line)
#---------------------------------------------------------------------------------------------------
default_region     = 'conus'
default_method     = 'hapy'      # 'hapy' or 'uxarray'
default_interp     = 'nearest'   # hapy only: 'nearest' or 'linear'
default_pixel_ratio= 1.0
default_dpi        = 100
default_figsize    = (16,9)      # inches - (16,9)*dpi=150 -> 2400x1350 px
default_fps        = 30

### number of columns of subplot panels - None puts every variable in a single row
default_num_plot_col = 2

### number of frames to keep in flight on the background reader thread
prefetch_depth     = 1

### padding (deg) added around the plot region when subsetting in ncol space, so
### that nearest-neighbor lookups near the edge of the view still find columns
region_pad_frac    = 0.05
region_pad_min     = 0.25

### sentinel used to hide NaNs from hapy.to_raster so its mapping cache stays valid
NAN_SENTINEL       = -9.0e30

### output locations (relative to the repo root, one level up from code_animation)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fig_root   = os.path.join(repo_root,'figs_animation')

# frame_root = os.path.join(fig_root,'frames')
# cache_root = os.path.join(fig_root,'.cache')     # on-disk uxarray pixel mappings

scratch_root = '/lustre/orion/cli115/proj-shared/'
frame_root = os.path.join(f'{scratch_root}/figs_animation','frames')
cache_root = os.path.join(f'{scratch_root}/figs_animation','.cache')     # on-disk uxarray pixel mappings

#---------------------------------------------------------------------------------------------------
# Command line overrides
#---------------------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method',      default=default_method, choices=['hapy','uxarray'],
                   help='rasterization back end (default: %(default)s)')
    p.add_argument('--interp',      default=default_interp, choices=['nearest','linear'],
                   help='hapy interpolation method (default: %(default)s)')
    p.add_argument('--region',      default=default_region, choices=list(region_dict.keys()),
                   help='plot region preset (default: %(default)s)')
    p.add_argument('--extent',      default=None, metavar='lon1,lon2,lat1,lat2',
                   help='explicit region, overrides --region')
    p.add_argument('--var',         default=None,
                   help='comma separated variable list, overrides the add_var() calls')
    p.add_argument('--frames',      default=None, metavar='t1:t2[:dt]',
                   help='time index range to render (python slice semantics)')
    p.add_argument('--pixel-ratio', default=default_pixel_ratio, type=float,
                   help='raster pixels per screen pixel (default: %(default)s)')
    p.add_argument('--dpi',         default=default_dpi, type=int,
                   help='figure dpi (default: %(default)s)')
    p.add_argument('--figsize',     default=None, metavar='W,H',
                   help=f'figure size in inches (default: {default_figsize[0]},{default_figsize[1]})')
    p.add_argument('--num-plot-col',default=default_num_plot_col, type=int, metavar='N',
                   help='number of columns of subplot panels (default: all panels in one row)')
    p.add_argument('--fps',         default=default_fps, type=int,
                   help='frames per second of the output video (default: %(default)s)')
    p.add_argument('--no-subset',   action='store_true',
                   help='skip the ncol region subset (slower, but useful for debugging)')
    p.add_argument('--no-prefetch', action='store_true', help='disable background frame reads')
    p.add_argument('--no-video',    action='store_true', help='write frames but do not encode')
    p.add_argument('--video-only',  action='store_true', help='encode existing frames and exit')
    p.add_argument('--overwrite',   action='store_true', help='re-render frames that already exist')
    p.add_argument('--coast-res',   default=None, choices=['110m','50m','10m'],
                   help='cartopy feature resolution (default: chosen from region size)')
    p.add_argument('--stats',       action='store_true',
                   help='print the min/max of every frame (useful for picking clev)')
    p.add_argument('--dry-run',     action='store_true',
                   help='report the configuration and frame list, then exit')
    return p.parse_args()

args = parse_args()

#---------------------------------------------------------------------------------------------------
# Resolve configuration
#---------------------------------------------------------------------------------------------------
if args.var is not None:
    ### keep only the requested variables, preserving the add_var() options
    keep = [v.strip() for v in args.var.split(',')]
    known = {var[i]:var_opts_list[i] for i in range(len(var))}
    missing = [v for v in keep if v not in known]
    if missing: raise ValueError(f'ERROR - no add_var() entry for: {missing}')
    var = keep; var_opts_list = [known[v] for v in keep]
if var==[]: raise ValueError('ERROR - variable list is empty!')
num_var = len(var)

#---------------------------------------------------------------------------------------------------
# subplot panel layout - one panel per variable, arranged in num_plot_col columns
#---------------------------------------------------------------------------------------------------
num_plot_col = args.num_plot_col
if num_plot_col is None: num_plot_col = num_var          # default - single row
if num_plot_col < 1: raise ValueError('ERROR - num_plot_col must be >= 1')
ncols = min(num_plot_col,num_var)
nrows = int(np.ceil(num_var/float(ncols)))

def norm_lon(x):
    """normalize longitude to [-180,180]"""
    if x is None: return None
    x = float(x)
    while x >  180: x -= 360
    while x < -180: x += 360
    return x

if args.extent is not None:
    lon1,lon2,lat1,lat2 = [float(x) for x in args.extent.split(',')]
    region = dict(lon1=lon1, lon2=lon2, lat1=lat1, lat2=lat2, proj='platecarree')
    region_name = 'custom'
else:
    region = dict(region_dict[args.region]); region_name = args.region
region['lon1'],region['lon2'] = norm_lon(region['lon1']),norm_lon(region['lon2'])
is_global = region['lon1'] is None

### projection
if region['proj']=='lambert':
    clon = 0.5*(region['lon1']+region['lon2'])
    clat = 0.5*(region['lat1']+region['lat2'])
    projection = ccrs.LambertConformal(central_longitude=clon, central_latitude=clat)
else:
    clon = region.get('clon', 0. if is_global else 0.5*(region['lon1']+region['lon2']))
    projection = ccrs.PlateCarree(central_longitude=clon)

figsize = default_figsize if args.figsize is None else tuple(float(x) for x in args.figsize.split(','))

### feature resolution - 10m linework is slow and pointless on a global map
if args.coast_res is not None:
    coast_res = args.coast_res
elif is_global:
    coast_res = '110m'
else:
    span = max(abs(region['lon2']-region['lon1']), abs(region['lat2']-region['lat1']))
    coast_res = '110m' if span>90 else ('50m' if span>15 else '10m')

fig_prefix = f'animation.2026-conus.{args.method}.{region_name}.' + '_'.join(var)

hapy.print_line()
print(' '*2+f'case            : {hapy.tclr.GREEN}{case}{hapy.tclr.END}')
print(' '*2+f'variables       : {hapy.tclr.MAGENTA}{" ".join(var)}{hapy.tclr.END}')
print(' '*2+f'raster method   : {args.method}' + (f' ({args.interp})' if args.method=='hapy' else ''))
print(' '*2+f'region          : {region_name}  {region}')
print(' '*2+f'projection      : {type(projection).__name__} (central_longitude={clon:.1f})')
print(' '*2+f'panel layout    : {nrows} rows x {ncols} cols  ({num_var} panels)')
print(' '*2+f'figsize / dpi   : {figsize} / {args.dpi}   ->  {int(figsize[0]*args.dpi)}x{int(figsize[1]*args.dpi)} px')
print(' '*2+f'pixel_ratio     : {args.pixel_ratio}')
print(' '*2+f'feature res     : {coast_res}')
print(' '*2+f'frame dir       : {frame_root}')
hapy.print_line()

#---------------------------------------------------------------------------------------------------
# Open the data
#---------------------------------------------------------------------------------------------------
def open_var_dataset(v):
    """lazily open all files of the output stream that holds variable v"""
    htype = var_opts_list[v].get('htype', None)
    if htype is None: raise ValueError(f'ERROR - no htype specified for {var[v]}')
    file_path = f'{data_dir}/*{htype}*.nc'
    file_list = sorted(glob.glob(file_path))

    # print()
    # for f,file in enumerate(file_list): print(f'{f:4}  {file}')
    # print()
    # exit()

    if file_list==[]: raise ValueError(f'ERROR - empty file list for:\n{file_path}')
    ### chunks={'time':1} keeps each frame read to a single contiguous slice
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    return xr.open_mfdataset(file_list, chunks={'time':1}, data_vars='minimal',
                             coords='minimal', compat='override', decode_times=time_coder)

ds_list = [open_var_dataset(v) for v in range(num_var)]

### the time axis is taken from the first variable - the streams in this case are
### all written on the same 30 minute cadence
time_coord = ds_list[0]['time']
num_time   = len(time_coord)

# t1,t2,dt = 0,num_time,1
if 't1' not in locals(): t1 = 0
if 't2' not in locals(): t2 = num_time
if 'dt' not in locals(): dt = 0,

if args.frames is not None:
    parts = args.frames.split(':')
    t1 = int(parts[0]) if parts[0] else 0
    t2 = int(parts[1]) if len(parts)>1 and parts[1] else num_time
    dt = int(parts[2]) if len(parts)>2 and parts[2] else 1
time_list = list(range(t1,min(t2,num_time),dt))

print(' '*2+f'time steps      : {num_time} available, rendering {len(time_list)} '
            f'({t1}:{t2}:{dt})')
for v in range(num_var):
    if len(ds_list[v]['time'])!=num_time:
        print(' '*2+f'{hapy.tclr.RED}WARNING - {var[v]} has {len(ds_list[v]["time"])} '
                    f'times, expected {num_time}{hapy.tclr.END}')

#---------------------------------------------------------------------------------------------------
# Grid centers and region subsetting
#---------------------------------------------------------------------------------------------------
def load_grid_centers():
    """cell center lon/lat from the SCRIP file, as float32 to halve the footprint"""
    print(' '*2+'reading grid center coordinates...', end='', flush=True)
    tt = perf_counter()
    grid_ds = xr.open_dataset(scrip_file)
    lon = grid_ds['grid_center_lon'].values.astype(np.float32)
    lat = grid_ds['grid_center_lat'].values.astype(np.float32)
    grid_ds.close()
    lon = np.where(lon>180, lon-360, lon).astype(np.float32)
    print(f' done ({perf_counter()-tt:.1f} s)  ncol={lon.size}')
    return lon,lat

def region_pad():
    """padding (deg) added around the requested view before subsetting"""
    dlon = abs(region['lon2']-region['lon1'])
    dlat = abs(region['lat2']-region['lat1'])
    return max(region_pad_min, region_pad_frac*max(dlon,dlat))

def region_index(lon,lat):
    """
    integer indices of the columns needed to fill the current view

    Subsetting here is what makes zoomed animations tractable - it shrinks both
    the KDTree build and the per-frame gather from 31.6M columns to the handful
    that are actually on screen. A pad is added so nearest-neighbor lookups along
    the edge of the view still have columns to find.
    """
    pad  = region_pad()
    mask = (lat>=region['lat1']-pad) & (lat<=region['lat2']+pad)
    if region['lon1'] <= region['lon2']:
        mask &= (lon>=region['lon1']-pad) & (lon<=region['lon2']+pad)
    else:
        ### view spans the dateline
        mask &= (lon>=region['lon1']-pad) | (lon<=region['lon2']+pad)
    idx = np.flatnonzero(mask)
    if idx.size==0: raise ValueError('ERROR - no grid columns inside the requested region!')
    return idx

do_subset = (not is_global) and (not args.no_subset)

grid_lon,grid_lat = load_grid_centers()
n_grid = grid_lon.size

#---------------------------------------------------------------------------------------------------
# uxarray grid setup
#   this has to happen before the subset indices are settled, because with the
#   uxarray back end the per-frame data must be gathered in the SUBSET GRID's face
#   order - not in the order of an independently computed lat/lon mask
#---------------------------------------------------------------------------------------------------
ux_state = {}

def setup_uxarray_grid():
    import uxarray as ux
    tt = perf_counter()
    print(' '*2+f'building uxarray grid from {os.path.basename(ux_grid_file)} '
                f'(this is the slow part)...', flush=True)
    uxgrid = ux.open_grid(ux_grid_file)
    print(' '*2+f'uxarray grid    : {uxgrid.n_face} faces ({perf_counter()-tt:.1f} s)')
    face_idx = None
    if do_subset:
        tt = perf_counter()
        pad = region_pad()
        uxgrid = uxgrid.subset.bounding_box((region['lon1']-pad,region['lon2']+pad),
                                            (region['lat1']-pad,region['lat2']+pad),
                                            inverse_indices=True)
        ### indices into the full grid, in the subset grid's face order
        face_idx = np.asarray(uxgrid.inverse_indices['face'].values)
        print(' '*2+f'uxarray subset  : {uxgrid.n_face} of {n_grid} faces '
                    f'({100.*uxgrid.n_face/n_grid:.2f}%), pad={pad:.2f} deg '
                    f'({perf_counter()-tt:.1f} s)')
    ux_state['grid'] = uxgrid
    return face_idx

if args.method=='uxarray' and not args.dry_run:
    ncol_idx = setup_uxarray_grid()
elif do_subset:
    ncol_idx = region_index(grid_lon,grid_lat)
    print(' '*2+f'region subset   : {ncol_idx.size} of {n_grid} columns '
                f'({100.*ncol_idx.size/n_grid:.2f}%), pad={region_pad():.2f} deg')
else:
    ncol_idx = None

if ncol_idx is None:
    sub_lon,sub_lat = grid_lon,grid_lat
else:
    sub_lon,sub_lat = grid_lon[ncol_idx],grid_lat[ncol_idx]
    ### the full arrays are no longer needed - drop them before the tree is built
    del grid_lon,grid_lat

if args.dry_run:
    print(' '*2+f'frames          : {time_list}')
    print(' '*2+'dry run - exiting')
    sys.exit(0)

#---------------------------------------------------------------------------------------------------
# Rasterization back ends
#---------------------------------------------------------------------------------------------------
def rasterize_hapy(vals, ax):
    """
    hapy.to_raster() with the NaN sentinel trick.

    hapy caches its pixel->column mapping, but skips the cache whenever the data
    contains NaN, because NaN columns are excluded from the KDTree. EAMxx fields
    carry a _FillValue that decodes to NaN, which would force a full tree rebuild
    on every frame. Replacing NaN with an out-of-range sentinel keeps the cache
    alive; nearest-neighbor gathering passes the sentinel through unchanged so it
    can be masked back to NaN afterwards. Not valid for 'linear', which would
    blend the sentinel into neighboring pixels, so that path takes the slow route.
    """
    has_nan = bool(np.isnan(vals).any())
    use_sentinel = has_nan and args.interp=='nearest'
    if use_sentinel: vals = np.where(np.isnan(vals), NAN_SENTINEL, vals)
    data = xr.DataArray(vals, dims='ncol',
                        coords={'lat':('ncol',sub_lat), 'lon':('ncol',sub_lon)})
    ### grid is unused because the DataArray carries its own lat/lon coordinates
    rst = hapy.to_raster(data, None, ax, pixel_ratio=args.pixel_ratio, method=args.interp)
    if use_sentinel:
        rst = np.where(rst<NAN_SENTINEL/2., np.nan, rst)
    elif has_nan:
        print(' '*4+f'{hapy.tclr.RED}WARNING - NaN present with method=linear, '
                    f'the raster cache is bypassed and every frame rebuilds the '
                    f'KDTree{hapy.tclr.END}')
    return rst

def ux_pixel_mapping_file(ax):
    """path of the on-disk pixel mapping cache for this grid+axes combination"""
    from uxarray.plot.matplotlib import _RasterAxAttrs
    a = _RasterAxAttrs.from_ax(ax, pixel_ratio=args.pixel_ratio)
    key = (os.path.basename(ux_grid_file), region_name,
           0 if ncol_idx is None else int(ncol_idx.size),
           a.projection, a.xlim, a.ylim, a.shape, a.pixel_ratio)
    import hashlib
    tag = hashlib.md5(repr(key).encode()).hexdigest()[:16]
    return os.path.join(cache_root,f'ux_pixel_mapping.{region_name}.{tag}.nc')

def rasterize_uxarray(vals, ax):
    """
    UxDataArray.to_raster() with the pixel mapping cached in memory and on disk.

    The expensive part is Grid.get_faces_containing_point(), which is done once
    for the first frame. The resulting mapping depends only on the grid and the
    axes geometry, so it is written to figs_animation/.cache and picked up by
    later runs (and by parallel workers rendering different frame ranges).

    vals already arrives in the subset grid's face order, because ncol_idx was
    taken from uxgrid.inverse_indices in setup_uxarray_grid().
    """
    import uxarray as ux
    if 'pm' not in ux_state:
        ### look for a cached pixel mapping before paying for the face search
        pm_file = ux_pixel_mapping_file(ax)
        ux_state['pm_file'] = pm_file
        if os.path.exists(pm_file):
            ux_state['pm'] = xr.open_dataarray(pm_file).load()
            print(' '*2+f'pixel mapping   : loaded from {os.path.basename(pm_file)}')
        else:
            ux_state['pm'] = None

    uxda = ux.UxDataArray(vals, dims=('n_face',), uxgrid=ux_state['grid'])

    if ux_state['pm'] is None:
        tt = perf_counter()
        print(' '*2+'searching for the face containing each pixel...', end='', flush=True)
        rst,pm = uxda.to_raster(ax=ax, pixel_ratio=args.pixel_ratio, return_pixel_mapping=True)
        print(f' done ({perf_counter()-tt:.1f} s)')
        ux_state['pm'] = pm
        try:
            os.makedirs(cache_root,exist_ok=True)
            pm.to_netcdf(ux_state['pm_file'])
            print(' '*2+f'pixel mapping   : saved to {os.path.basename(ux_state["pm_file"])}')
        except Exception as e:
            print(' '*2+f'{hapy.tclr.RED}WARNING - could not cache pixel mapping: {e}{hapy.tclr.END}')
        return rst
    return uxda.to_raster(ax=ax, pixel_mapping=ux_state['pm'])

#---------------------------------------------------------------------------------------------------
# Frame reading (with optional background prefetch)
#---------------------------------------------------------------------------------------------------
def read_frame(t):
    """
    read one time slice of every variable

    The full ncol slice is read and then indexed in numpy. Handing the index
    array to xarray instead would push a fancy-index down into the netCDF
    library, which turns one 126 MB sequential read into a very large number of
    small strided reads.
    """
    out = []
    for v in range(num_var):
        vals = ds_list[v][var[v]].isel(time=t).values
        if ncol_idx is not None: vals = vals[ncol_idx]
        scale = var_opts_list[v].get('scale',None)
        vals = vals.astype(np.float32)
        if scale is not None: vals = vals*np.float32(scale)
        out.append(vals)
    return out

class FramePrefetcher:
    """one-deep read-ahead so disk I/O overlaps with rendering"""
    def __init__(self, indices, enabled=True):
        self.indices = list(indices); self.enabled = enabled; self.pos = 0
        self.pool = ThreadPoolExecutor(max_workers=1) if enabled else None
        self.pending = {}
        if enabled:
            for t in self.indices[:prefetch_depth]:
                self.pending[t] = self.pool.submit(read_frame,t)
    def get(self, t):
        if not self.enabled: return read_frame(t)
        fut = self.pending.pop(t, None)
        if fut is None: fut = self.pool.submit(read_frame,t)
        ### queue up the next frame before waiting on this one
        i = self.indices.index(t)
        for j in range(i+1, min(i+1+prefetch_depth, len(self.indices))):
            nt = self.indices[j]
            if nt not in self.pending: self.pending[nt] = self.pool.submit(read_frame,nt)
        return fut.result()
    def close(self):
        if self.pool is not None: self.pool.shutdown(wait=False, cancel_futures=True)

#---------------------------------------------------------------------------------------------------
# Image / color settings
#---------------------------------------------------------------------------------------------------
def image_kwargs(v):
    opts = var_opts_list[v]
    kw = {'origin':'lower', 'interpolation':'nearest'}
    kw['cmap'] = opts.get('cmap','viridis')
    clev = opts.get('clev',None)
    if clev is not None:
        ### extend on the norm rather than the colorbar - matplotlib rejects
        ### colorbar(extend=...) when the mappable already carries a BoundaryNorm
        kw['norm'] = mcolors.BoundaryNorm(clev, ncolors=256, extend='both')
    else:
        if 'vmin' in opts: kw['vmin'] = opts['vmin']
        if 'vmax' in opts: kw['vmax'] = opts['vmax']
    return kw

def time_string(t):
    tv = time_coord.values[t]
    try:
        return tv.strftime('%Y-%m-%d %H:%M UTC')
    except AttributeError:
        return str(tv)[:16]+' UTC'

#---------------------------------------------------------------------------------------------------
# Set up the figure ONCE - see performance note 1
#
# Everything that affects the geometry of the map axes (figure size, margins, and
# the colorbar) is settled here, before the first call to a rasterizer. Both back
# ends key their cached pixel mapping on the axes size and limits, so an axes that
# gets resized later - which is exactly what fig.colorbar(ax=...) does - would
# silently throw the mapping away and rebuild it on the following frame.
#---------------------------------------------------------------------------------------------------
title_fontsize,label_fontsize = 16,12

### inches reserved around the map: left/bottom/top margins, colorbar strip per
### panel, and the gaps between panels (horizontal and vertical)
pad_l,pad_b,pad_t,cbar_in,gap_in = 0.35,0.35,0.55,1.05,0.45
### multi-panel frames carry the time stamp as a suptitle, which needs a band of
### its own above the row of panel titles
if num_var>1: pad_t += 0.40
vgap_in = 0.55                    # inches between panel rows - holds the panel title
panel_w = 11.0                    # inches of map per panel when auto-sizing

fig = plt.figure(figsize=figsize, dpi=args.dpi)
### only num_var axes are created, so a partly filled last row leaves empty space
### rather than a set of blank panels
axs = [fig.add_subplot(nrows,ncols,v+1, projection=projection) for v in range(num_var)]

for ax in axs:
    if is_global:
        ax.set_global()
    else:
        ax.set_extent([region['lon1'],region['lon2'],region['lat1'],region['lat2']],
                      crs=ccrs.PlateCarree())

#---------------------------------------------------------------------------------------------------
# size the canvas to the shape of the map so the frames are not mostly whitespace
if args.figsize is None:
    x1,x2 = axs[0].get_xlim(); y1,y2 = axs[0].get_ylim()
    map_aspect = float(x2-x1)/float(y2-y1)
    panel_h = panel_w/map_aspect
    ### keep very tall or very wide views from producing an absurd canvas
    if panel_h > 2.0*panel_w:
        panel_h = 2.0*panel_w; panel_w = panel_h*map_aspect
    fig_w = pad_l + ncols*(panel_w+cbar_in) + (ncols-1)*gap_in
    fig_h = pad_b + nrows*panel_h + (nrows-1)*vgap_in + pad_t
    fig.set_size_inches(fig_w,fig_h)
    figsize = (fig_w,fig_h)
    print(' '*2+f'auto figsize    : {fig_w:.2f} x {fig_h:.2f} in  '
                f'(map aspect {map_aspect:.2f})  ->  '
                f'{int(fig_w*args.dpi)}x{int(fig_h*args.dpi)} px')
    ### wspace/hspace are fractions of the panel size, so these expressions put
    ### exactly (cbar_in+gap_in) and vgap_in inches between neighboring panels
    fig.subplots_adjust(left=pad_l/fig_w, right=1.-cbar_in/fig_w,
                        bottom=pad_b/fig_h, top=1.-pad_t/fig_h,
                        wspace=(cbar_in+gap_in)/panel_w,
                        hspace=vgap_in/panel_h)
else:
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.05, top=0.93,
                        wspace=0.10, hspace=0.15)

#---------------------------------------------------------------------------------------------------
# map features - the linework is drawn once and reused by every frame
def add_map_features(ax,color):
    """add coastlines/borders/states, backing off to a coarser set if unavailable"""
    for res in [coast_res,'50m','110m']:
        try:
            ax.add_feature(cfeature.COASTLINE.with_scale(res), linewidth=0.6, edgecolor=color)
            ax.add_feature(cfeature.BORDERS.with_scale(res),   linewidth=0.5, edgecolor=color)
            if res in ['50m','10m']:
                ax.add_feature(cfeature.STATES.with_scale(res), linewidth=0.35,
                               edgecolor=color, alpha=0.7)
            return res
        except Exception as e:
            ### cartopy downloads Natural Earth data on demand, which fails on
            ### compute nodes with no outbound network
            print(' '*2+f'{hapy.tclr.RED}WARNING - {res} map features unavailable '
                        f'({type(e).__name__}), trying coarser{hapy.tclr.END}')
    return None

#---------------------------------------------------------------------------------------------------
# create the image artists and colorbars up front - the image starts as a dummy
# array and gets swapped for the real raster with set_data() on every frame
img_list,cbar_list = [None]*num_var,[None]*num_var
for v in range(num_var):
    ax = axs[v]
    opts = var_opts_list[v]
    ax.set_facecolor(opts.get('bgcolor','0.15'))
    add_map_features(ax, opts.get('linecolor','0.4'))
    ax.set_title(opts.get('s',var[v]), fontsize=title_fontsize, loc='left')
    img_list[v] = ax.imshow(np.full((2,2),np.nan), extent=ax.get_xlim()+ax.get_ylim(),
                            **image_kwargs(v))
    ### explicit colorbar axes in the reserved strip, so the map axes is untouched
    pos = ax.get_position()
    cx0 = pos.x1 + 0.15*cbar_in/figsize[0]
    cax = fig.add_axes([cx0, pos.y0, 0.20*cbar_in/figsize[0], pos.height])
    cbar_kw = {} if 'norm' in image_kwargs(v) else {'extend':'both'}
    cbar_list[v] = fig.colorbar(img_list[v], cax=cax, orientation='vertical', **cbar_kw)
    cbar_list[v].ax.tick_params(labelsize=label_fontsize)

### time stamp position for multi-panel frames - pinned to the top of the canvas so
### it clears the panel titles regardless of how many rows there are
suptitle_y = 1.-0.08/figsize[1]

### force a draw so the axes bboxes are final before any rasterization
fig.canvas.draw()

#---------------------------------------------------------------------------------------------------
# Render the frames
#---------------------------------------------------------------------------------------------------
os.makedirs(frame_root,exist_ok=True)
frame_files = []
rasterize = rasterize_hapy if args.method=='hapy' else rasterize_uxarray

if not args.video_only:
    prefetcher = FramePrefetcher(time_list, enabled=(not args.no_prefetch))
    hapy.print_line()
    for i,t in enumerate(time_list):
        frame_file = os.path.join(frame_root,f'{fig_prefix}.{t:05d}.png')
        frame_files.append(frame_file)
        if os.path.exists(frame_file) and not args.overwrite:
            print(f'  frame {i+1:5d}/{len(time_list)}  t={t:5d}  (exists, skipping)')
            continue
        t_frame = perf_counter()
        #-------------------------------------------------------------------------
        data_list = prefetcher.get(t)
        t_read = perf_counter()-t_frame
        #-------------------------------------------------------------------------
        t_rst = 0.
        stat_str = ''
        for v in range(num_var):
            ax = axs[v]
            if args.stats:
                d = data_list[v]
                if np.isnan(d).all():
                    stat_str += '   all NaN'
                else:
                    stat_str += f'   [{np.nanmin(d):.3g},{np.nanmax(d):.3g}]'
            tt = perf_counter()
            rst = rasterize(data_list[v], ax)
            t_rst += perf_counter()-tt
            ### reuse the artist - this is what keeps the per-frame cost flat
            img_list[v].set_data(rst)
        #-------------------------------------------------------------------------
        stamp = f'{name}   {time_string(t)}'
        if num_var==1:
            axs[0].set_title(stamp, fontsize=title_fontsize, loc='right')
        else:
            fig.suptitle(stamp, fontsize=title_fontsize, y=suptitle_y, va='top')
        #-------------------------------------------------------------------------
        tt = perf_counter()
        fig.savefig(frame_file, dpi=args.dpi)   # no bbox_inches - see note 6
        t_save = perf_counter()-tt
        #-------------------------------------------------------------------------
        t_tot = perf_counter()-t_frame
        print(f'  frame {i+1:5d}/{len(time_list)}  t={t:5d}  {time_string(t)}   '
              f'read {t_read:6.2f}s  raster {t_rst:6.2f}s  save {t_save:6.2f}s  '
              f'total {t_tot:6.2f}s{stat_str}')
    prefetcher.close()
else:
    frame_files = sorted(glob.glob(os.path.join(frame_root,f'{fig_prefix}.*.png')))

plt.close(fig)

#---------------------------------------------------------------------------------------------------
# Encode the animation
#---------------------------------------------------------------------------------------------------
def encode_animation():
    frames = [f for f in frame_files if os.path.exists(f)]
    if frames==[]:
        print(f'{hapy.tclr.RED}ERROR - no frames to encode{hapy.tclr.END}'); return
    hapy.print_line()
    ### ffmpeg first - mp4 handles thousands of multi-megapixel frames, gif does not
    if shutil.which('ffmpeg') is not None:
        out_file = os.path.join(fig_root,f'{fig_prefix}.mp4')
        list_file = os.path.join(fig_root,f'.{fig_prefix}.frames.txt')
        with open(list_file,'w') as f:
            for fr in frames: f.write(f"file '{fr}'\n")
        cmd = ['ffmpeg','-y','-r',str(args.fps),'-f','concat','-safe','0','-i',list_file,
               '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-pix_fmt','yuv420p',
               '-crf','18',out_file]
        print('  '+' '.join(cmd))
        res = sp.run(cmd)
        os.remove(list_file)
        if res.returncode!=0:
            print(f'{hapy.tclr.RED}ERROR - ffmpeg failed (exit {res.returncode}){hapy.tclr.END}')
            return
        print(f'\n  {out_file}\n')
    elif shutil.which('magick') is not None:
        out_file = os.path.join(fig_root,f'{fig_prefix}.gif')
        delay = max(1,int(round(100./args.fps)))
        ### only -delay and -loop may precede the input files - ImageMagick 7.0.8
        ### rejects image *operators* like -repage before it has read any image
        cmd = ['magick','-delay',str(delay),'-loop','0']+frames+[out_file]
        print(f'  magick -delay {delay} -loop 0 <{len(frames)} frames> {out_file}')
        if len(frames)>200 or (figsize[0]*args.dpi)>2000:
            print(f'  {hapy.tclr.YELLOW}WARNING - gif encoding of large/long animations is '
                  f'slow and memory hungry, install ffmpeg for mp4 output{hapy.tclr.END}')
        res = sp.run(cmd)
        if res.returncode!=0:
            print(f'{hapy.tclr.RED}ERROR - magick failed (exit {res.returncode}){hapy.tclr.END}')
            return
        print(f'\n  {out_file}\n')
    else:
        print(f'{hapy.tclr.RED}ERROR - neither ffmpeg nor magick found, frames left in '
              f'{frame_root}{hapy.tclr.END}')

if not args.no_video: encode_animation()

#---------------------------------------------------------------------------------------------------
etime = perf_counter()-timer_start
time_str = f'{etime:10.1f} sec'
if etime>60       : time_str += f' ({(etime/60):4.1f} min)'
if etime>(2*3600) : time_str += f' ({(etime/3600):.1f} hr)'
print(f'\n{hapy.tclr.YELLOW}elapsed time: {time_str} {hapy.tclr.END}')
print()
#---------------------------------------------------------------------------------------------------
