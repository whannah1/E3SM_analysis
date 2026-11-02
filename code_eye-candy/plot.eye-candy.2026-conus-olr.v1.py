#!/usr/bin/env python
"""
Stacked still maps of model OLR from the 2026 CONUS RRM SCREAM case (top) and
GOES-13 channel 4 (10.7 um window) brightness temperature from GridSat-CONUS (bottom).

Stripped down from plot.eye-candy.2026-conus.v1.py - same rasterization back ends
(--method hapy|uxarray), projections, background and alpha-ramp options - but with
a single variable, a single snapshot, and an observation panel under the model.

The model OLR is converted to an equivalent window-channel brightness temperature
with hapy.olr_to_tb() (inverse of the Ohring et al. (1984) fit, following Yang and
Slingo (2001)), so both panels share the same colormap and color levels. The
comparison is still qualitative.

NOTE - the first record of an EAMxx INSTANT stream can be all fill value
(~3.4e38 or 3.4e33, not flagged with _FillValue), so huge values are masked and
all-fill records are skipped when choosing the model time.

Model time selection (--time-match):

    nearest  the model output time closest to the observation time
    tod      the model output time closest to the observation's time of day on any
             day, latest day preferred - useful when the obs time falls before the
             model start, or right at it while the model is still spinning up
    --model-time 'YYYY-MM-DD HH:MM' overrides both

--------------------------------------------------------------------------------
NOTE - GOES satellite data was taken from:
https://www.ncei.noaa.gov/data/gridsat-goes/access/conus/2011/05/
--------------------------------------------------------------------------------
EXAMPLES
--------------------------------------------------------------------------------
    python plot.eye-candy.2026-conus-olr.v1.py
    python plot.eye-candy.2026-conus-olr.v1.py --time-match tod --method hapy
    python plot.eye-candy.2026-conus-olr.v1.py --model-time '2011-05-20 12:00'
"""
from time import perf_counter
timer_start = perf_counter()
import os, sys, glob, hashlib, argparse, cftime, numpy as np, xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import cartopy.crs as ccrs, cartopy.feature as cfeature
import hapy
host = hapy.get_host()
xr.set_options(use_new_combine_kwarg_defaults=True)

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#---------------------------------------------------------------------------------------------------
# Case configuration
#---------------------------------------------------------------------------------------------------
if host=='olcf':
    case     = 'conus1024x2v1pg2_RRSwISC6to18E3r5.F2010-SCREAMv1.with-3d-turbulence.20260929'
    name     = '1024x2 (20260929)'
    case_dir = '/lustre/orion/cli115/proj-shared/brhillman/e3sm_scratch'
    case_sub = 'run'
    grid_root  = '/lustre/orion/cli115/world-shared/e3sm/2026-INCITE-CONUS-RRM/files_grid'
    scrip_file = f'{grid_root}/2025-scream-conus-1024x2-pg2_scrip.cdf5.nc'
    ux_grid    = scrip_file
    # obs_file_default = os.path.join(repo_root,'GridSat-CONUS.goes13.2011.05.19.1745.v01.nc')
    obs_file_default = os.path.join(repo_root,'GridSat-CONUS.goes13.2011.05.20.1745.v01.nc')
    ### default Blue Marble location on this machine - see notes_eye-candy.md
    bg_image_default = '/lustre/orion/cli115/proj-shared/hannah6/Blue_Marble/'\
                       'world.topo.bathy.200408.3x21600x10800.jpg'
else:
    raise ValueError(f'ERROR - no case configuration for host: {host}')

#---------------------------------------------------------------------------------------------------
# Variable configuration
#---------------------------------------------------------------------------------------------------
model_var   = 'LW_flux_up_at_model_top'
model_htype = 'eamxx.LW_flux_up_at_model_top.INSTANT.nmins_x5'
obs_var     = 'ch4'
obs_name    = 'GOES-13 ch4'

### gray_r puts cold cloud tops in white, like an IR image - same colormap and
### limits for both panels. alpha_floor: under --alpha-ramp, values above this
### fade out toward vmax (see make_rgba)
cmap        = 'gray_r'
tb_clim     = dict(vmin=190., vmax=310., alpha_floor=260.)   # K

### anything above this is treated as fill value
FILL_THRESH = 1.0e30

#---------------------------------------------------------------------------------------------------
# Region presets - lon may be given as 0:360 or -180:180, it is normalized below
#---------------------------------------------------------------------------------------------------
region_dict = {
    'conus'    : dict(lon1=-125, lon2= -65, lat1= 24, lat2= 50),
    'plains'   : dict(lon1=-105, lon2= -88, lat1= 28, lat2= 42),
    'gulf'     : dict(lon1= -98, lon2= -80, lat1= 24, lat2= 34),
    'westcoast': dict(lon1=-125, lon2=-114, lat1= 32, lat2= 50),
    'frontrange':dict(lon1=-107, lon2=-102, lat1= 37, lat2= 42),
    'okc'      : dict(lon1=-98.5, lon2=-97.0, lat1=35.0, lat2=36.2),
}

#---------------------------------------------------------------------------------------------------
# Default settings (all overridable on the command line)
#---------------------------------------------------------------------------------------------------
default_region      = 'conus'
default_method      = 'uxarray'   # 'hapy' or 'uxarray'
default_interp      = 'nearest'   # hapy only: 'nearest' or 'linear'
default_proj        = 'platecarree'
default_pixel_ratio = 1.0
default_dpi         = 300
panel_w             = 11.0        # inches of map per panel when auto-sizing

print_stats         = True

### padding (deg) added around the view when subsetting in ncol space
region_pad_frac     = 0.05
region_pad_min      = 0.25

### sentinel used to hide NaNs from hapy.to_raster so its mapping cache stays valid
NAN_SENTINEL        = -9.0e30

fig_root   = 'figs_eye-candy'
cache_root = os.path.join(fig_root,'.cache')

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
    p.add_argument('--proj',        default=default_proj, choices=['platecarree','lambert'],
                   help='map projection (default: %(default)s)')
    p.add_argument('--obs-file',    default=obs_file_default,
                   help='GridSat-CONUS file (default: %(default)s)')
    p.add_argument('--time-match',  default='nearest', choices=['nearest','tod'],
                   help='how to pick the model time (default: %(default)s)')
    p.add_argument('--model-time',  default=None, metavar='"YYYY-MM-DD HH:MM"',
                   help='explicit model time, overrides --time-match')
    p.add_argument('--clim',        default=None, metavar='vmin,vmax',
                   help='color limits for both panels (default: from tb_clim)')
    p.add_argument('--background',  default='none', choices=['none','stock','image'],
                   help='what to draw under the data (default: %(default)s)')
    p.add_argument('--bg-image',    default=None,
                   help='global plate carree jpg/png for --background image')
    p.add_argument('--alpha-ramp',  action='store_true',
                   help='fade warm (clear sky) values to transparent so the background shows')
    p.add_argument('--alpha-gamma', default=2.0, type=float,
                   help='shape of the alpha ramp - >1 fades harder (default: %(default)s)')
    p.add_argument('--bare',        action='store_true',
                   help='no titles, colorbar or frame - full-bleed image')
    p.add_argument('--dark',        action='store_true', help='dark figure background')
    p.add_argument('--pixel-ratio', default=default_pixel_ratio, type=float,
                   help='raster pixels per screen pixel (default: %(default)s)')
    p.add_argument('--dpi',         default=default_dpi, type=int,
                   help='figure dpi (default: %(default)s)')
    p.add_argument('--no-subset',   action='store_true',
                   help='skip the ncol region subset (slower, but useful for debugging)')
    p.add_argument('--no-features', action='store_true', help='no coastlines/borders/states')
    p.add_argument('--coast-res',   default=None, choices=['110m','50m','10m'],
                   help='cartopy feature resolution (default: chosen from region size)')
    p.add_argument('--tag',         default=None, help='extra string appended to the figure name')
    p.add_argument('--dry-run',     action='store_true', help='report the configuration and exit')
    return p.parse_args()

args = parse_args()

#---------------------------------------------------------------------------------------------------
# Resolve configuration
#---------------------------------------------------------------------------------------------------
def norm_lon(x):
    """normalize longitude to [-180,180]"""
    x = float(x)
    while x >  180: x -= 360
    while x < -180: x += 360
    return x

if args.extent is not None:
    lon1,lon2,lat1,lat2 = [float(x) for x in args.extent.split(',')]
    region = dict(lon1=lon1, lon2=lon2, lat1=lat1, lat2=lat2); region_name = 'custom'
else:
    region = dict(region_dict[args.region]); region_name = args.region
region['lon1'],region['lon2'] = norm_lon(region['lon1']),norm_lon(region['lon2'])

clon = 0.5*(region['lon1']+region['lon2'])
clat = 0.5*(region['lat1']+region['lat2'])
if args.proj=='lambert': projection = ccrs.LambertConformal(central_longitude=clon,
                                                            central_latitude=clat)
else                   : projection = ccrs.PlateCarree(central_longitude=clon)

if args.coast_res is not None:
    coast_res = args.coast_res
else:
    span = max(abs(region['lon2']-region['lon1']), abs(region['lat2']-region['lat1']))
    coast_res = '110m' if span>90 else ('50m' if span>15 else '10m')

bg_image = args.bg_image if args.bg_image is not None else bg_image_default

def resolve_clim(base, override):
    c = dict(base)
    if override is not None:
        c['vmin'],c['vmax'] = [float(x) for x in override.split(',')]
    return c
clim       = resolve_clim(tb_clim, args.clim)
model_str  = 'OLR-equivalent Tb [K]'
obs_str    = 'ch4 Brightness Temperature [K]'

#---------------------------------------------------------------------------------------------------
# Observation time, and the model time to go with it
#---------------------------------------------------------------------------------------------------
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

obs_ds   = xr.open_dataset(args.obs_file, decode_times=time_coder)
obs_time = obs_ds['time'].values[0]

file_path = f'{case_dir}/{case}/{case_sub}/*{model_htype}*.nc'
file_list = sorted(glob.glob(file_path))
if file_list==[]: raise ValueError(f'ERROR - empty file list for:\n{file_path}')

def rank_model_times():
    """
    candidate (file, time index, model time, offset in s) sorted best first

    Only the time coordinate of each file is read. Model and obs calendars differ
    (noleap vs standard), so the target is rebuilt in the model calendar from its
    date/time components before differencing.
    """
    cands = []
    for f in file_list:
        with xr.open_dataset(f, decode_times=time_coder) as ds:
            for t,tm in enumerate(ds['time'].values): cands.append((f,t,tm))
    cal = cands[0][2].calendar
    if args.model_time is not None:
        ref = cftime.datetime.strptime(args.model_time, '%Y-%m-%d %H:%M', calendar=cal)
    else:
        ref = cftime.datetime(obs_time.year, obs_time.month, obs_time.day,
                              obs_time.hour, obs_time.minute, obs_time.second, calendar=cal)
    t0 = cands[0][2]
    def key(c):
        dt = (c[2]-ref).total_seconds()
        if args.time_match=='tod' and args.model_time is None:
            ### distance in time of day, ties go to the latest (most spun up) day
            return (abs((dt+43200.)%86400.-43200.), -(c[2]-t0).total_seconds())
        return (abs(dt), 0.)
    return [(c[0],c[1],c[2],(c[2]-ref).total_seconds()) for c in sorted(cands, key=key)]

def read_model_slice(f,t):
    """one full time slice of the model field, with fill values set to NaN"""
    with xr.open_dataset(f, decode_times=time_coder) as ds:
        vals = ds[model_var].isel(time=t).values.astype(np.float32)
    vals[~(np.abs(vals)<FILL_THRESH)] = np.nan
    return vals

### the best-ranked time can be an all-fill record (e.g. the first record of the
### stream), so walk down the ranking until one has data. Read only in a real run.
model_ranked = rank_model_times()
model_file,model_tidx,model_time,model_dt = model_ranked[0]
model_full = None
if not args.dry_run:
    for model_file,model_tidx,model_time,model_dt in model_ranked:
        model_full = read_model_slice(model_file,model_tidx)
        if not np.isnan(model_full).all(): break
        print(' '*2+f'{hapy.tclr.RED}WARNING - {model_var} is all fill value at '
                    f'{model_time}, trying the next closest time{hapy.tclr.END}')
    else:
        raise ValueError(f'ERROR - every record of {model_var} is fill value')

def fmt_time(t): return f'{t.year:04d}-{t.month:02d}-{t.day:02d} {t.hour:02d}:{t.minute:02d}Z'
model_time_str = fmt_time(model_time)
obs_time_str   = fmt_time(obs_time)

fig_file = os.path.join(fig_root, f'eye-candy.2026-conus-olr.{args.method}.{region_name}.'
                                  f'{args.proj}.{model_time_str[:-1].replace(" ","_").replace(":","")}'
                                  +('' if args.tag is None else f'.{args.tag}')+'.png')

hapy.print_line()
print(' '*2+f'case            : {hapy.tclr.GREEN}{name}{hapy.tclr.END}')
print(' '*2+f'obs file        : {os.path.basename(args.obs_file)}')
print(' '*2+f'obs time        : {obs_time_str}')
print(' '*2+f'model time      : {model_time_str}  (t={model_tidx}, '
            f'{model_dt/60.:+.0f} min from '+('requested' if args.model_time else 'obs')+')'
            +('  [dry run - fill check skipped]' if args.dry_run else ''))
print(' '*2+f'model file      : {hapy.tclr.YELLOW}{os.path.basename(model_file)}{hapy.tclr.END}')
print(' '*2+f'color limits    : {clim["vmin"]} to {clim["vmax"]} K')
print(' '*2+f'raster method   : {args.method}' + (f' ({args.interp})' if args.method=='hapy' else ''))
print(' '*2+f'region          : {region_name}  {region}')
print(' '*2+f'projection      : {type(projection).__name__} (central_longitude={clon:.1f})')
print(' '*2+f'background      : {args.background}'
            + (f'  {bg_image}' if args.background=='image' else ''))
print(' '*2+f'pixel_ratio/dpi : {args.pixel_ratio} / {args.dpi}')
print(' '*2+f'figure file     : {fig_file}')
hapy.print_line()

#---------------------------------------------------------------------------------------------------
# Set up the figure FIRST - the ncol subset and the cached pixel mapping both depend
# on the final axes geometry, so nothing may resize the map axes after this point.
# Layout: model on top, obs below, each with its own colorbar on the right
#---------------------------------------------------------------------------------------------------
if args.dark: plt.rcParams.update({'text.color':'white','axes.labelcolor':'white',
                                   'xtick.color':'white','ytick.color':'white'})
fig_face = 'black' if args.dark else 'white'
title_fontsize,label_fontsize = 16,12

### inches reserved around the maps
if args.bare: pad_l,pad_b,pad_t,cbar_in,gap_in = 0.,0.,0.,0.,0.
else:         pad_l,pad_b,pad_t,cbar_in,gap_in = 0.35,0.35,0.55,1.05,0.45

num_plot = 2
fig = plt.figure(figsize=(10,6), dpi=args.dpi, facecolor=fig_face)
axs = [fig.add_subplot(num_plot,1,k+1, projection=projection) for k in range(num_plot)]
for ax in axs:
    ax.set_extent([region['lon1'],region['lon2'],region['lat1'],region['lat2']],
                  crs=ccrs.PlateCarree())

### size the canvas to the shape of the map so the figure is not mostly whitespace
x1,x2 = axs[0].get_xlim(); y1,y2 = axs[0].get_ylim()
map_aspect = float(x2-x1)/float(y2-y1)
pw = panel_w; ph = pw/map_aspect
if ph > 2.0*pw: ph = 2.0*pw; pw = ph*map_aspect
fig_w = pad_l + pw + cbar_in
fig_h = pad_b + num_plot*ph + (num_plot-1)*gap_in + pad_t
fig.set_size_inches(fig_w,fig_h); figsize = (fig_w,fig_h)
print(' '*2+f'auto figsize    : {fig_w:.2f} x {fig_h:.2f} in  (map aspect {map_aspect:.2f})'
            f'  ->  {int(fig_w*args.dpi)}x{int(fig_h*args.dpi)} px')
if args.bare:
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
else:
    fig.subplots_adjust(left=pad_l/fig_w, right=1.-cbar_in/fig_w,
                        bottom=pad_b/fig_h, top=1.-pad_t/fig_h, hspace=gap_in/ph)

### settle the axes bboxes before anything asks for their size
fig.canvas.draw()

#---------------------------------------------------------------------------------------------------
# What the axes actually shows, in lon/lat
#---------------------------------------------------------------------------------------------------
def view_bounds():
    """(lon1,lon2,lat1,lat2) covering everything visible in the axes, padded"""
    if args.no_subset: return None
    try:
        e = axs[0].get_extent(crs=ccrs.PlateCarree())
    except Exception as ex:
        print(' '*2+f'{hapy.tclr.RED}WARNING - could not determine the view extent '
                    f'({type(ex).__name__}), skipping the ncol subset{hapy.tclr.END}')
        return None
    lo1,lo2,la1,la2 = float(e[0]),float(e[1]),float(e[2]),float(e[3])
    pad = max(region_pad_min, region_pad_frac*max(abs(lo2-lo1),abs(la2-la1)))
    return (lo1-pad, lo2+pad, max(-90.,la1-pad), min(90.,la2+pad))

view = view_bounds()
if view is None:
    print(' '*2+'view bounds     : global (no ncol subset)')
else:
    print(' '*2+f'view bounds     : lon {view[0]:.2f}:{view[1]:.2f}  lat {view[2]:.2f}:{view[3]:.2f}')

#---------------------------------------------------------------------------------------------------
# Grid centers and region subsetting
#---------------------------------------------------------------------------------------------------
def load_grid_centers(scrip_file):
    """cell center lon/lat from a SCRIP file, as float32 to halve the footprint"""
    print(' '*2+f'reading grid centers from {os.path.basename(scrip_file)}...', end='', flush=True)
    tt = perf_counter()
    grid_ds = xr.open_dataset(scrip_file)
    lon = grid_ds['grid_center_lon'].values.astype(np.float32)
    lat = grid_ds['grid_center_lat'].values.astype(np.float32)
    grid_ds.close()
    lon = np.where(lon>180, lon-360, lon).astype(np.float32)
    print(f' done ({perf_counter()-tt:.1f} s)  ncol={lon.size}')
    return lon,lat

def region_index(lon,lat):
    """integer indices of the columns needed to fill the current view"""
    lo1,lo2,la1,la2 = view
    mask = (lat>=la1) & (lat<=la2) & (lon>=lo1) & (lon<=lo2)
    idx = np.flatnonzero(mask)
    if idx.size==0: raise ValueError('ERROR - no grid columns inside the requested region!')
    return idx

#---------------------------------------------------------------------------------------------------
# uxarray grid setup - with this back end the data must be gathered in the SUBSET
# GRID's face order, so the subset indices come from uxarray
#---------------------------------------------------------------------------------------------------
ux_state = {}

def setup_uxarray_grid():
    import uxarray as ux
    tt = perf_counter()
    print(' '*2+f'building uxarray grid from {os.path.basename(ux_grid)} '
                f'(this is the slow part)...', flush=True)
    uxgrid = ux.open_grid(ux_grid)
    print(' '*2+f'uxarray grid    : {uxgrid.n_face} faces ({perf_counter()-tt:.1f} s)')
    face_idx = None
    if view is not None:
        tt = perf_counter()
        uxgrid = uxgrid.subset.bounding_box((view[0],view[1]),(view[2],view[3]),
                                            inverse_indices=True)
        face_idx = np.asarray(uxgrid.inverse_indices['face'].values)
        print(' '*2+f'uxarray subset  : {uxgrid.n_face} faces ({perf_counter()-tt:.1f} s)')
    ux_state['grid'] = uxgrid
    return face_idx

#---------------------------------------------------------------------------------------------------
# Rasterization back ends
#---------------------------------------------------------------------------------------------------
def rasterize_hapy(vals, lon, lat, ax):
    """hapy.to_raster() with NaN swapped for a sentinel so the mapping cache stays valid"""
    has_nan = bool(np.isnan(vals).any())
    use_sentinel = has_nan and args.interp=='nearest'
    if use_sentinel: vals = np.where(np.isnan(vals), NAN_SENTINEL, vals)
    data = xr.DataArray(vals, dims='ncol', coords={'lat':('ncol',lat), 'lon':('ncol',lon)})
    rst = hapy.to_raster(data, None, ax, pixel_ratio=args.pixel_ratio, method=args.interp)
    if use_sentinel: rst = np.where(rst<NAN_SENTINEL/2., np.nan, rst)
    return rst

def ux_pixel_mapping_file(ax):
    """path of the on-disk pixel mapping cache for this grid+axes combination"""
    from uxarray.plot.matplotlib import _RasterAxAttrs
    a = _RasterAxAttrs.from_ax(ax, pixel_ratio=args.pixel_ratio)
    key = (os.path.basename(ux_grid), region_name, int(ux_state['grid'].n_face),
           a.projection, a.xlim, a.ylim, a.shape, a.pixel_ratio)
    tag = hashlib.md5(repr(key).encode()).hexdigest()[:16]
    return os.path.join(cache_root,f'ux_pixel_mapping.{region_name}.{tag}.nc')

def rasterize_uxarray(vals, ax):
    """UxDataArray.to_raster() with the pixel mapping cached on disk"""
    import uxarray as ux
    pm_file = ux_pixel_mapping_file(ax)
    pm = None
    if os.path.exists(pm_file):
        pm = xr.open_dataarray(pm_file).load()
        print(' '*4+f'pixel mapping   : loaded from {os.path.basename(pm_file)}')
    uxda = ux.UxDataArray(vals, dims=('n_face',), uxgrid=ux_state['grid'])
    if pm is not None: return uxda.to_raster(ax=ax, pixel_mapping=pm)
    tt = perf_counter()
    print(' '*4+'searching for the face containing each pixel...', end='', flush=True)
    rst,pm = uxda.to_raster(ax=ax, pixel_ratio=args.pixel_ratio, return_pixel_mapping=True)
    print(f' done ({perf_counter()-tt:.1f} s)')
    try:
        os.makedirs(cache_root,exist_ok=True)
        pm.to_netcdf(pm_file)
    except Exception as e:
        print(' '*4+f'{hapy.tclr.RED}WARNING - could not cache pixel mapping: {e}{hapy.tclr.END}')
    return rst

#---------------------------------------------------------------------------------------------------
# Read the data
#---------------------------------------------------------------------------------------------------
glon,glat = load_grid_centers(scrip_file)
if args.method=='uxarray' and not args.dry_run:
    ncol_idx = setup_uxarray_grid()
elif view is not None:
    ncol_idx = region_index(glon,glat)
    print(' '*2+f'region subset   : {ncol_idx.size} of {glon.size} columns '
                f'({100.*ncol_idx.size/glon.size:.2f}%)')
else:
    ncol_idx = None
mlon,mlat = (glon,glat) if ncol_idx is None else (glon[ncol_idx],glat[ncol_idx])

if args.dry_run:
    print(' '*2+'dry run - exiting'); sys.exit(0)

hapy.print_line()
### the slice was read in full during time selection - the ncol subset is applied
### in numpy after a contiguous read, see the parent script
olr = model_full if ncol_idx is None else model_full[ncol_idx]
del model_full
model_vals = hapy.olr_to_tb(olr).astype(np.float32)

print(' '*2+'reading obs '+hapy.tclr.MAGENTA+obs_var+hapy.tclr.END)
obs_tb = obs_ds[obs_var].isel(time=0).values.astype(np.float32)
obs_lon_b = obs_ds['lon_bounds'].values; obs_lat_b = obs_ds['lat_bounds'].values
obs_extent = [float(obs_lon_b.min()), float(obs_lon_b.max()),
              float(obs_lat_b.min()), float(obs_lat_b.max())]
### imshow wants rows running south to north with origin='lower'
if obs_ds['lat'].values[0] > obs_ds['lat'].values[-1]: obs_tb = obs_tb[::-1,:]
obs_ds.close()

if print_stats:
    for lbl,d in [('model OLR',olr),('model Tb',model_vals),('obs Tb',obs_tb)]:
        print(' '*4+f'{lbl:10}: min={np.nanmin(d):.4g}  max={np.nanmax(d):.4g}  '
                    f'mean={np.nanmean(d):.4g}  nan={int(np.isnan(d).sum())}')

#---------------------------------------------------------------------------------------------------
# Background and map features - drawn before the data so the field sits on top
#---------------------------------------------------------------------------------------------------
def add_background(ax):
    if args.background=='stock':
        ax.stock_img(); return
    if args.background!='image': return
    if not os.path.exists(bg_image):
        print(' '*2+f'{hapy.tclr.RED}WARNING - background image not found, falling back to the '
                    f'cartopy stock image:{hapy.tclr.END}\n{" "*4}{bg_image}\n'
                    f'{" "*4}see notes_eye-candy.md for where to get Blue Marble data')
        ax.stock_img(); return
    img = plt.imread(bg_image)
    ext = [-180,180,-90,90]
    if view is not None:
        ### crop before handing it to cartopy - the full Blue Marble tile is huge
        ny,nx = img.shape[0],img.shape[1]
        i1 = max(0, int((view[0]+180.)/360.*nx)); i2 = min(nx, int(np.ceil((view[1]+180.)/360.*nx)))
        j1 = max(0, int((90.-view[3])/180.*ny));  j2 = min(ny, int(np.ceil((90.-view[2])/180.*ny)))
        if i2>i1 and j2>j1:
            img = img[j1:j2, i1:i2]
            ext = [i1/nx*360.-180., i2/nx*360.-180., 90.-j2/ny*180., 90.-j1/ny*180.]
    ax.imshow(img, origin='upper', extent=ext, transform=ccrs.PlateCarree(), regrid_shape=2000)

def add_map_features(ax,color):
    """add coastlines/borders/states, backing off to a coarser set if unavailable"""
    if args.no_features: return
    for res in [coast_res,'50m','110m']:
        try:
            ax.add_feature(cfeature.COASTLINE.with_scale(res), linewidth=0.6, edgecolor=color)
            ax.add_feature(cfeature.BORDERS.with_scale(res),   linewidth=0.5, edgecolor=color)
            if res in ['50m','10m']:
                ax.add_feature(cfeature.STATES.with_scale(res), linewidth=0.35,
                               edgecolor=color, alpha=0.7)
            return
        except Exception as e:
            print(' '*2+f'{hapy.tclr.RED}WARNING - {res} map features unavailable '
                        f'({type(e).__name__}), trying coarser{hapy.tclr.END}')

#---------------------------------------------------------------------------------------------------
# Color settings - same colormap and limits for both panels
#---------------------------------------------------------------------------------------------------
def make_rgba(rst, norm, alpha_floor):
    """
    map the raster through the colormap by hand so clear sky can fade out

    With gray_r the interesting (cold, cloudy) end is the LOW end of the scale, so
    unlike the parent script the ramp fades the HIGH end: fully opaque below
    alpha_floor, fading to transparent at vmax.
    """
    cm = plt.get_cmap(cmap)
    masked = np.ma.masked_invalid(rst)
    rgba = cm(norm(masked))
    rgba[...,3] = np.where(np.isfinite(rst), 1., 0.)
    if args.alpha_ramp:
        f = float(np.clip(norm(alpha_floor),1e-3,0.999))
        pos = np.clip(np.ma.filled(norm(masked),1.),0.,1.)
        ramp = np.clip((1.-pos)/(1.-f), 0., 1.)
        rgba[...,3] *= ramp**args.alpha_gamma
    return rgba

#---------------------------------------------------------------------------------------------------
# Draw the panels
#---------------------------------------------------------------------------------------------------
use_rgba = not (args.background=='none' and not args.alpha_ramp)
norm = mcolors.Normalize(vmin=clim['vmin'], vmax=clim['vmax'])
panels = [('model',name,    model_time_str,model_str),
          ('obs',  obs_name,obs_time_str,  obs_str)]
for k,(kind,lbl,tstr,vstr) in enumerate(panels):
    ax = axs[k]
    add_background(ax)
    add_map_features(ax, '0.4' if args.background=='none' else 'white')
    if args.background=='none': ax.set_facecolor('0.15')
    #-------------------------------------------------------------------------------
    if kind=='model':
        tt = perf_counter()
        rst = rasterize_hapy(model_vals,mlon,mlat,ax) if args.method=='hapy' \
              else rasterize_uxarray(model_vals,ax)
        print(' '*2+f'panel {k}: model rasterized in {perf_counter()-tt:.2f} s')
        img_kw = dict(origin='lower', extent=ax.get_xlim()+ax.get_ylim(), interpolation='nearest')
    else:
        ### obs is already on a regular lat/lon grid - let cartopy warp it
        rst = obs_tb
        img_kw = dict(origin='lower', extent=obs_extent, transform=ccrs.PlateCarree(),
                      interpolation='nearest', regrid_shape=2000)
    #-------------------------------------------------------------------------------
    if use_rgba:
        ax.imshow(make_rgba(rst,norm,clim['alpha_floor']), **img_kw)
    else:
        ax.imshow(rst, norm=norm, cmap=cmap, **img_kw)
    ### re-pin the extent in case imshow nudged the limits
    ax.set_extent([region['lon1'],region['lon2'],region['lat1'],region['lat2']],
                  crs=ccrs.PlateCarree())
    #-------------------------------------------------------------------------------
    if args.bare:
        ax.set_frame_on(False)
        for s in ax.spines.values(): s.set_visible(False)
        continue
    #-------------------------------------------------------------------------------
    ax.set_title(vstr, fontsize=title_fontsize, loc='left')
    ax.set_title(f'{lbl}  {tstr}', fontsize=title_fontsize, loc='right')
    #-------------------------------------------------------------------------------
    ### explicit colorbar axes in the reserved strip, so the map axes is untouched
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.15*cbar_in/figsize[0], pos.y0,
                        0.20*cbar_in/figsize[0], pos.height])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                        orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=label_fontsize)

#---------------------------------------------------------------------------------------------------
# Finalize
#---------------------------------------------------------------------------------------------------
os.makedirs(fig_root,exist_ok=True)
fig.savefig(fig_file, dpi=args.dpi, facecolor=fig_face,
            pad_inches=0 if args.bare else 0.05)
plt.close(fig)

hapy.print_line()
print(f'\n{fig_file}\n')

etime = perf_counter()-timer_start
time_str = f'{etime:10.1f} sec'
if etime>60       : time_str += f' ({(etime/60):4.1f} min)'
if etime>(2*3600) : time_str += f' ({(etime/3600):.1f} hr)'
print(f'{hapy.tclr.YELLOW}elapsed time: {time_str} {hapy.tclr.END}')
print()
#---------------------------------------------------------------------------------------------------
