#!/usr/bin/env python
"""
Still maps of native (unstructured) grid output from the 2026 CONUS RRM SCREAM case.

This is a matplotlib+cartopy replacement for the old PyNGL panel script
(Research/E3SM/code_scream/plot.scream.map.v2.py). It keeps that script's shape -
add_case()/add_var(), a case x variable panel layout, optional difference panels,
time averaging or a single snapshot - and swaps ngl.contour_map(CellFill) for a
rasterized native-grid image drawn with ax.imshow().

Rasterization uses one of two back ends, selected with --method:

    hapy     hapy.to_raster() - great-circle nearest neighbor from a KDTree built
             on the cell *centers*. Fast to build, works with any SCRIP file.

    uxarray  UxDataArray.to_raster() - point-in-face search using the actual cell
             *boundaries*, so cell edges are exact. Grid construction and the face
             search are much more expensive; the resulting pixel mapping is cached
             to disk here so repeat runs are cheap.

The "eye-candy" part is everything layered on top of that raster:

    --background {none,stock,image}  put a shaded relief / Blue Marble image under
                                     the data instead of a flat color
    --alpha-ramp                     fade the low end of the color scale to
                                     transparent, so the background shows through
                                     wherever there is nothing interesting
    --bare                           drop every bit of chrome - no titles, no
                                     colorbar, no frame - and write a full-bleed
                                     image at an exact pixel size
    --proj {platecarree,lambert,ortho,nearside}
                                     nearside/ortho give the "view from orbit" look

Blue Marble imagery is not bundled anywhere; see notes_eye-candy.md at the repo
root for where to download it and how to convert it. Point --bg-image at a jpg or
png of the whole globe in plate carree (lon -180:180, lat -90:90) and it is used
as the background, cropped to the view first. --background stock falls back to the
low resolution shaded relief image that ships with cartopy, which works offline.

--------------------------------------------------------------------------------
PERFORMANCE NOTES (this grid is ~31.6M columns, ~126 MB per float32 snapshot)
--------------------------------------------------------------------------------
 1. The figure is built BEFORE the data is read, and the ncol subset is taken from
    the geographic extent of the finished axes rather than from the requested
    lat/lon box. That keeps the subset correct for every projection - including
    the globe views, where a nearside camera at low altitude sees far less than a
    lat/lon box would suggest, and an orthographic view sees far more.
 2. Subsetting in ncol space is the biggest single win: it shrinks the KDTree
    build / face search from 31.6M cells to whatever is actually on screen.
 3. Time averaging streams one time slice at a time and accumulates in numpy,
    rather than handing xarray a lazy mean over the whole stream. A dask mean over
    93 x 126 MB slices builds a huge graph and will happily try to hold far more
    than one slice in memory at a time.
 4. The ncol subset is applied in numpy after reading a full time slice, not as a
    lazy fancy-index - a fancy index pushed into netCDF turns one contiguous
    126 MB read into thousands of small strided reads on Lustre.
 5. All panels are laid out at the same size with the same limits, and the
    colorbars live in their own axes, so nothing resizes a map axes after the
    fact and every panel reuses a single cached pixel mapping.
 6. hapy.to_raster caches its mapping only when the data has no NaN, so NaNs are
    replaced with a sentinel and masked back out afterwards - see rasterize_hapy().

--------------------------------------------------------------------------------
EXAMPLES
--------------------------------------------------------------------------------
    # default: precip over CONUS
    python plot.eye-candy.2026-conus.v1.py

    # poster shot - view from orbit, shaded relief underneath, no chrome
    python plot.eye-candy.2026-conus.v1.py --proj nearside --altitude 2e6 \\
           --background stock --alpha-ramp --bare --dark --dpi 300

    # exact cell edges, zoomed way in
    python plot.eye-candy.2026-conus.v1.py --extent -126,-121.5,45.5,49 \\
           --method uxarray --pixel-ratio 2
"""
from time import perf_counter
timer_start = perf_counter()
import os, sys, glob, hashlib, argparse
import numpy as np, xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import cartopy.crs as ccrs, cartopy.feature as cfeature
import cmocean
import hapy
from concurrent.futures import ThreadPoolExecutor
host = hapy.get_host()
xr.set_options(use_new_combine_kwarg_defaults=True)

#---------------------------------------------------------------------------------------------------
name,case,case_dir,case_sub,scrip_file_list,ux_grid_list = [],[],[],[],[],[]
def add_case(case_in,n=None,p=None,s=None,scrip_file=None,ux_grid=None):
    case.append(case_in); name.append(case_in if n is None else n)
    case_dir.append(p); case_sub.append(s)
    if scrip_file is None: raise ValueError('scrip_file cannot be None')
    scrip_file_list.append(scrip_file)
    ### uxarray can read the SCRIP file directly, but an exodus/ugrid file is
    ### usually faster to convert if one is available
    ux_grid_list.append(scrip_file if ux_grid is None else ux_grid)
#---------------------------------------------------------------------------------------------------
var,var_str,var_opts_list = [],[],[]
def add_var(var_name,s=None,**kwargs):
    var.append(var_name); var_str.append(var_name if s is None else s)
    var_opts_list.append(dict(kwargs))
#---------------------------------------------------------------------------------------------------
# Case configuration
#---------------------------------------------------------------------------------------------------
if host=='olcf':

    tmp_scratch,tmp_sub = '/lustre/orion/cli115/proj-shared/brhillman/e3sm_scratch','run'
    grid_root = '/lustre/orion/cli115/world-shared/e3sm/2026-INCITE-CONUS-RRM/files_grid'
    ### NOTE - the plain "*_scrip.nc" files in grid_root are currently 0 bytes,
    ###        the cdf5 copy is the one with actual data in it
    tmp_grid_file = f'{grid_root}/2025-scream-conus-1024x2-pg2_scrip.cdf5.nc'

    add_case('conus1024x2v1pg2_RRSwISC6to18E3r5.F2010-SCREAMv1.20260727',
             n='2026 CONUS RRM (1024x2)', p=tmp_scratch, s=tmp_sub, scrip_file=tmp_grid_file)

    ### which files of the stream to use - 0/None means "all"
    first_file,num_files = 0,1

    ### default Blue Marble location on this machine - see notes_eye-candy.md
    bg_image_default = '/lustre/orion/cli115/proj-shared/hannah6/Blue_Marble/'\
                       'world.topo.bathy.200408.3x21600x10800.jpg'
else:
    raise ValueError(f'ERROR - no case configuration for host: {host}')

#---------------------------------------------------------------------------------------------------
# Variable configuration
#   htype       - EAMxx output stream (one variable per stream in this case)
#   scale       - unit conversion applied after reading
#   cmap / clev / vmin / vmax - color settings, clev takes precedence
#   alpha_floor - value at which the field becomes fully opaque under --alpha-ramp;
#                 everything below it fades linearly to transparent at the bottom
#                 of the color scale. Set it well up into the interesting range -
#                 too low and a light colormap washes out the whole background.
#---------------------------------------------------------------------------------------------------
add_var('precip_total_surf_mass_flux', s='Precipitation [mm/day]', scale=86400*1e3,
        htype='eamxx.precip_total_surf_mass_flux.INSTANT.nmins_x30',
        cmap=cmocean.cm.rain, clev=np.logspace(np.log10(0.1),np.log10(200),num=40),
        alpha_floor=20.)

# add_var('LW_flux_up_at_model_top', s='OLR [W/m2]',
#         htype='eamxx.LW_flux_up_at_model_top.INSTANT.nmins_x30',
#         cmap='gray_r', vmin=80, vmax=320)

# add_var('VapWaterPath', s='CWV [mm]',
#         htype='eamxx.VapWaterPath.INSTANT.nmins_x30',
#         cmap=cmocean.cm.haline, vmin=0, vmax=70)

# add_var('LiqWaterPath', s='LWP [kg/m2]',
#         htype='eamxx.LiqWaterPath.INSTANT.nmins_x30',
#         cmap='bone', clev=np.logspace(-2,0.5,num=30), alpha_floor=0.02)

# add_var('IceWaterPath', s='IWP [kg/m2]',
#         htype='eamxx.IceWaterPath.INSTANT.nmins_x30',
#         cmap='bone', clev=np.logspace(-2,0.5,num=30), alpha_floor=0.02)

# add_var('T_2m', s='2m Temperature [K]',
#         htype='eamxx.T_2m.INSTANT.nmins_x30',
#         cmap=cmocean.cm.thermal, vmin=240, vmax=310)

# add_var('wind_speed_10m', s='10m Wind Speed [m/s]',
#         htype='eamxx.wind_speed_10m.INSTANT.nmins_x30',
#         cmap=cmocean.cm.speed, vmin=0, vmax=40)

# add_var('ps', s='Surface Pressure [hPa]', scale=1e-2,
#         htype='eamxx.ps.INSTANT.nmins_x30',
#         cmap='viridis', vmin=950, vmax=1030)

#---------------------------------------------------------------------------------------------------
# Region presets - lon may be given as 0:360 or -180:180, it is normalized below
#---------------------------------------------------------------------------------------------------
region_dict = {
    'global'   : dict(lon1=None, lon2=None, lat1=None, lat2=None, clon=180),
    'namerica' : dict(lon1=-170, lon2= -50, lat1= 10, lat2= 75),
    'conus'    : dict(lon1=-127, lon2= -65, lat1= 22, lat2= 52),
    'plains'   : dict(lon1=-105, lon2= -88, lat1= 28, lat2= 42),
    'gulf'     : dict(lon1= -98, lon2= -80, lat1= 24, lat2= 34),
    'westcoast': dict(lon1=-128, lon2=-114, lat1= 32, lat2= 50),
    'frontrange':dict(lon1=-107, lon2=-102, lat1= 37, lat2= 42),
    ### ~1 degree boxes - this is where the <1km spacing actually shows up
    'okc'      : dict(lon1=-98.5, lon2=-97.0, lat1=35.0, lat2=36.2),
    'olympic'  : dict(lon1=-124.9, lon2=-123.0, lat1=47.2, lat2=48.4),
}

#---------------------------------------------------------------------------------------------------
# Default settings (all overridable on the command line)
#---------------------------------------------------------------------------------------------------
default_region      = 'conus'
default_method      = 'hapy'      # 'hapy' or 'uxarray'
default_interp      = 'nearest'   # hapy only: 'nearest' or 'linear'
default_proj        = 'platecarree'
default_pixel_ratio = 1.0
default_dpi         = 150
panel_w             = 11.0        # inches of map per panel when auto-sizing

plot_diff,add_diff  = False,False # difference panels relative to diff_base
diff_base           = 0

print_stats         = True
var_x_case          = True        # panels are var rows x case columns

### padding (deg) added around the view when subsetting in ncol space
region_pad_frac     = 0.05
region_pad_min      = 0.25

### sentinel used to hide NaNs from hapy.to_raster so its mapping cache stays valid
NAN_SENTINEL        = -9.0e30

### number of time slices read concurrently when averaging
read_workers        = 4

repo_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fig_root   = os.path.join(repo_root,'figs_eye-candy')
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
    p.add_argument('--proj',        default=default_proj,
                   choices=['platecarree','lambert','ortho','nearside'],
                   help='map projection (default: %(default)s)')
    p.add_argument('--altitude',    default=2.0e6, type=float,
                   help='camera height (m) for --proj nearside (default: %(default)s)')
    p.add_argument('--var',         default=None,
                   help='comma separated variable list, overrides the add_var() calls')
    p.add_argument('--snapshot',    default=None, type=int, metavar='T',
                   help='plot a single time index instead of the time mean')
    p.add_argument('--background',  default='none', choices=['none','stock','image'],
                   help='what to draw under the data (default: %(default)s)')
    p.add_argument('--bg-image',    default=None,
                   help='global plate carree jpg/png for --background image')
    p.add_argument('--alpha-ramp',  action='store_true',
                   help='fade the color scale to transparent below each variable\'s alpha_floor')
    p.add_argument('--alpha-gamma', default=2.0, type=float,
                   help='shape of the alpha ramp - >1 fades the low end harder (default: %(default)s)')
    p.add_argument('--bare',        action='store_true',
                   help='no titles, colorbar or frame - full-bleed image')
    p.add_argument('--dark',        action='store_true', help='dark figure background')
    p.add_argument('--pixel-ratio', default=default_pixel_ratio, type=float,
                   help='raster pixels per screen pixel (default: %(default)s)')
    p.add_argument('--dpi',         default=default_dpi, type=int,
                   help='figure dpi (default: %(default)s)')
    p.add_argument('--figsize',     default=None, metavar='W,H',
                   help='figure size in inches (default: sized from the map aspect)')
    p.add_argument('--no-subset',   action='store_true',
                   help='skip the ncol region subset (slower, but useful for debugging)')
    p.add_argument('--no-features', action='store_true', help='no coastlines/borders/states')
    p.add_argument('--coast-res',   default=None, choices=['110m','50m','10m'],
                   help='cartopy feature resolution (default: chosen from region size)')
    p.add_argument('--diff',        action='store_true', help='plot differences vs the first case')
    p.add_argument('--add-diff',    action='store_true', help='keep full fields AND add diff panels')
    p.add_argument('--tag',         default=None, help='extra string appended to the figure name')
    p.add_argument('--dry-run',     action='store_true', help='report the configuration and exit')
    return p.parse_args()

args = parse_args()
if args.diff:     plot_diff = True
if args.add_diff: plot_diff,add_diff = True,True

#---------------------------------------------------------------------------------------------------
# Resolve configuration
#---------------------------------------------------------------------------------------------------
if args.var is not None:
    keep = [v.strip() for v in args.var.split(',')]
    known = {var[i]:(var_str[i],var_opts_list[i]) for i in range(len(var))}
    missing = [v for v in keep if v not in known]
    if missing: raise ValueError(f'ERROR - no add_var() entry for: {missing}')
    var           = keep
    var_str       = [known[v][0] for v in keep]
    var_opts_list = [known[v][1] for v in keep]
if case==[]: raise ValueError('ERROR - case list is empty!')
if var ==[]: raise ValueError('ERROR - variable list is empty!')
num_var,num_case = len(var),len(case)

def norm_lon(x):
    """normalize longitude to [-180,180]"""
    if x is None: return None
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
is_global = region['lon1'] is None

clon = region.get('clon', 0. if is_global else 0.5*(region['lon1']+region['lon2']))
clat = 0. if is_global else 0.5*(region['lat1']+region['lat2'])
if   args.proj=='lambert' : projection = ccrs.LambertConformal(central_longitude=clon,
                                                               central_latitude=clat)
elif args.proj=='ortho'   : projection = ccrs.Orthographic(central_longitude=clon,
                                                           central_latitude=clat)
elif args.proj=='nearside': projection = ccrs.NearsidePerspective(central_longitude=clon,
                                                                  central_latitude=clat,
                                                                  satellite_height=args.altitude)
else                      : projection = ccrs.PlateCarree(central_longitude=clon)

### the globe views look wrong clipped to a lat/lon box, so they always show the
### full visible disk - the ncol subset then comes from the axes extent instead
full_disk = args.proj in ['ortho','nearside']

if args.coast_res is not None:
    coast_res = args.coast_res
elif is_global or full_disk:
    coast_res = '110m'
else:
    span = max(abs(region['lon2']-region['lon1']), abs(region['lat2']-region['lat1']))
    coast_res = '110m' if span>90 else ('50m' if span>15 else '10m')

bg_image = args.bg_image if args.bg_image is not None else bg_image_default

fig_file = os.path.join(fig_root, f'eye-candy.2026-conus.{args.method}.{region_name}.'
                                  f'{args.proj}.'+'_'.join(var)
                                  +('' if args.tag is None else f'.{args.tag}')+'.png')

hapy.print_line()
print(' '*2+f'cases           : {hapy.tclr.GREEN}{" | ".join(name)}{hapy.tclr.END}')
print(' '*2+f'variables       : {hapy.tclr.MAGENTA}{" ".join(var)}{hapy.tclr.END}')
print(' '*2+f'raster method   : {args.method}' + (f' ({args.interp})' if args.method=='hapy' else ''))
print(' '*2+f'region          : {region_name}  {region}')
print(' '*2+f'projection      : {type(projection).__name__} (central_longitude={clon:.1f})')
print(' '*2+f'time            : ' + (f'snapshot t={args.snapshot}' if args.snapshot is not None
                                     else 'mean over all times'))
print(' '*2+f'background      : {args.background}'
            + (f'  {bg_image}' if args.background=='image' else ''))
print(' '*2+f'pixel_ratio/dpi : {args.pixel_ratio} / {args.dpi}')
print(' '*2+f'figure file     : {fig_file}')
hapy.print_line()

#---------------------------------------------------------------------------------------------------
# panel list along the case dimension - same convention as the clim map script
#---------------------------------------------------------------------------------------------------
case_panels = []
if plot_diff and add_diff:
    for c in range(num_case): case_panels.append(('full',c))
    for c in range(num_case):
        if c!=diff_base: case_panels.append(('diff',c))
elif plot_diff:
    for c in range(num_case):
        case_panels.append(('full',c) if c==diff_base else ('diff',c))
else:
    for c in range(num_case): case_panels.append(('full',c))
num_panel = len(case_panels)

#---------------------------------------------------------------------------------------------------
# Set up the figure FIRST - see performance note 1
#
# Everything that affects the geometry of the map axes (figure size, margins and
# the colorbar) is settled here, before anything is rasterized. Both back ends key
# their cached pixel mapping on the axes size and limits, so an axes that gets
# resized later - which is what fig.colorbar(ax=...) does - would silently throw
# the mapping away and rebuild it for the next panel.
#---------------------------------------------------------------------------------------------------
if args.dark: plt.rcParams.update({'text.color':'white','axes.labelcolor':'white',
                                   'xtick.color':'white','ytick.color':'white'})
fig_face = 'black' if args.dark else 'white'
title_fontsize,label_fontsize = 16,12

### inches reserved around the maps
if args.bare: pad_l,pad_b,pad_t,cbar_in,gap_in = 0.,0.,0.,0.,0.
else:         pad_l,pad_b,pad_t,cbar_in,gap_in = 0.35,0.35,0.55,1.05,0.45

(d1,d2) = (num_var,num_panel) if var_x_case else (num_panel,num_var)

figsize = (10,6) if args.figsize is None else tuple(float(x) for x in args.figsize.split(','))
fig = plt.figure(figsize=figsize, dpi=args.dpi, facecolor=fig_face)
axs = np.empty((d1,d2), dtype=object)
for i in range(d1):
    for j in range(d2):
        axs[i,j] = fig.add_subplot(d1,d2,i*d2+j+1, projection=projection)
        if is_global or full_disk:
            axs[i,j].set_global()
        else:
            axs[i,j].set_extent([region['lon1'],region['lon2'],region['lat1'],region['lat2']],
                                crs=ccrs.PlateCarree())

#---------------------------------------------------------------------------------------------------
# size the canvas to the shape of the map so the figure is not mostly whitespace
if args.figsize is None:
    x1,x2 = axs[0,0].get_xlim(); y1,y2 = axs[0,0].get_ylim()
    map_aspect = float(x2-x1)/float(y2-y1)
    pw = panel_w; ph = pw/map_aspect
    if ph > 2.0*pw: ph = 2.0*pw; pw = ph*map_aspect
    fig_w = pad_l + d2*(pw+cbar_in) + (d2-1)*gap_in
    fig_h = pad_b + d1*ph + (d1-1)*gap_in + pad_t
    fig.set_size_inches(fig_w,fig_h); figsize = (fig_w,fig_h)
    print(' '*2+f'auto figsize    : {fig_w:.2f} x {fig_h:.2f} in  (map aspect {map_aspect:.2f})'
                f'  ->  {int(fig_w*args.dpi)}x{int(fig_h*args.dpi)} px')
    if args.bare:
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
    else:
        fig.subplots_adjust(left=pad_l/fig_w, right=1.-cbar_in/fig_w,
                            bottom=pad_b/fig_h, top=1.-pad_t/fig_h,
                            wspace=(cbar_in+gap_in)/pw, hspace=gap_in/ph)
elif args.bare:
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
else:
    fig.subplots_adjust(left=0.02,right=0.94,bottom=0.05,top=0.93,wspace=0.10,hspace=0.10)

### settle the axes bboxes before anything asks for their size
fig.canvas.draw()

#---------------------------------------------------------------------------------------------------
# What the axes actually shows, in lon/lat
#---------------------------------------------------------------------------------------------------
def view_bounds():
    """
    (lon1,lon2,lat1,lat2) covering everything visible in the axes, padded

    Taken from the finished axes rather than the requested region, so it is right
    for every projection: a nearside camera at low altitude sees much less than
    its lat/lon box, an orthographic view sees much more, and a Lambert view is
    not a lat/lon box at all. Returns None when the view is effectively global,
    in which case there is nothing to subset.
    """
    if args.no_subset: return None
    try:
        e = axs[0,0].get_extent(crs=ccrs.PlateCarree())
    except Exception as ex:
        print(' '*2+f'{hapy.tclr.RED}WARNING - could not determine the view extent '
                    f'({type(ex).__name__}), skipping the ncol subset{hapy.tclr.END}')
        return None
    lo1,lo2,la1,la2 = float(e[0]),float(e[1]),float(e[2]),float(e[3])
    pad = max(region_pad_min, region_pad_frac*max(abs(lo2-lo1),abs(la2-la1)))
    lo1,lo2 = lo1-pad,lo2+pad
    la1,la2 = max(-90.,la1-pad),min(90.,la2+pad)
    if (lo2-lo1)>=359.5 and (la2-la1)>=179.5: return None
    return (lo1,lo2,la1,la2)

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
    """
    integer indices of the columns needed to fill the current view

    This is what makes zoomed plots tractable - it shrinks the KDTree build /
    face search from 31.6M cells to the handful that are on screen.
    """
    lo1,lo2,la1,la2 = view
    mask = (lat>=la1) & (lat<=la2)
    if lo1<=lo2: mask &= (lon>=lo1) & (lon<=lo2)
    else:        mask &= (lon>=lo1) | (lon<=lo2)     # view spans the dateline
    idx = np.flatnonzero(mask)
    if idx.size==0: raise ValueError('ERROR - no grid columns inside the requested region!')
    return idx

#---------------------------------------------------------------------------------------------------
# uxarray grid setup
#   with the uxarray back end the data must be gathered in the SUBSET GRID's face
#   order, so the subset indices come from uxarray rather than from an
#   independently computed lat/lon mask
#---------------------------------------------------------------------------------------------------
ux_state = {}

def setup_uxarray_grid(c):
    import uxarray as ux
    tt = perf_counter()
    print(' '*2+f'building uxarray grid from {os.path.basename(ux_grid_list[c])} '
                f'(this is the slow part)...', flush=True)
    uxgrid = ux.open_grid(ux_grid_list[c])
    print(' '*2+f'uxarray grid    : {uxgrid.n_face} faces ({perf_counter()-tt:.1f} s)')
    face_idx = None
    if view is not None:
        tt = perf_counter()
        uxgrid = uxgrid.subset.bounding_box((view[0],view[1]),(view[2],view[3]),
                                            inverse_indices=True)
        face_idx = np.asarray(uxgrid.inverse_indices['face'].values)
        print(' '*2+f'uxarray subset  : {uxgrid.n_face} faces ({perf_counter()-tt:.1f} s)')
    ux_state[c] = {'grid':uxgrid}
    return face_idx

#---------------------------------------------------------------------------------------------------
# Rasterization back ends
#---------------------------------------------------------------------------------------------------
def rasterize_hapy(vals, lon, lat, ax):
    """
    hapy.to_raster() with the NaN sentinel trick.

    hapy caches its pixel->column mapping, but skips the cache whenever the data
    contains NaN, because NaN columns are excluded from the KDTree. EAMxx fields
    carry a _FillValue that decodes to NaN, which would force a full tree rebuild
    for every panel. Replacing NaN with an out-of-range sentinel keeps the cache
    alive; nearest-neighbor gathering passes the sentinel through unchanged so it
    can be masked back to NaN afterwards. Not valid for 'linear', which would
    blend the sentinel into neighboring pixels, so that path takes the slow route.
    """
    has_nan = bool(np.isnan(vals).any())
    use_sentinel = has_nan and args.interp=='nearest'
    if use_sentinel: vals = np.where(np.isnan(vals), NAN_SENTINEL, vals)
    data = xr.DataArray(vals, dims='ncol', coords={'lat':('ncol',lat), 'lon':('ncol',lon)})
    ### grid is unused because the DataArray carries its own lat/lon coordinates
    rst = hapy.to_raster(data, None, ax, pixel_ratio=args.pixel_ratio, method=args.interp)
    if use_sentinel: rst = np.where(rst<NAN_SENTINEL/2., np.nan, rst)
    return rst

def ux_pixel_mapping_file(c,ax):
    """path of the on-disk pixel mapping cache for this grid+axes combination"""
    from uxarray.plot.matplotlib import _RasterAxAttrs
    a = _RasterAxAttrs.from_ax(ax, pixel_ratio=args.pixel_ratio)
    key = (os.path.basename(ux_grid_list[c]), region_name, int(ux_state[c]['grid'].n_face),
           a.projection, a.xlim, a.ylim, a.shape, a.pixel_ratio)
    tag = hashlib.md5(repr(key).encode()).hexdigest()[:16]
    return os.path.join(cache_root,f'ux_pixel_mapping.{region_name}.{tag}.nc')

def rasterize_uxarray(vals, c, ax):
    """
    UxDataArray.to_raster() with the pixel mapping cached in memory and on disk.

    The expensive part is Grid.get_faces_containing_point(). It is done once, and
    the result depends only on the grid and the axes geometry, so it is written to
    figs_eye-candy/.cache and picked up by later runs. Every panel of the figure
    has the same size and limits, so they all share one mapping.
    """
    import uxarray as ux
    st = ux_state[c]
    if 'pm' not in st:
        pm_file = ux_pixel_mapping_file(c,ax)
        st['pm_file'] = pm_file
        if os.path.exists(pm_file):
            st['pm'] = xr.open_dataarray(pm_file).load()
            print(' '*4+f'pixel mapping   : loaded from {os.path.basename(pm_file)}')
        else:
            st['pm'] = None
    uxda = ux.UxDataArray(vals, dims=('n_face',), uxgrid=st['grid'])
    if st['pm'] is None:
        tt = perf_counter()
        print(' '*4+'searching for the face containing each pixel...', end='', flush=True)
        rst,pm = uxda.to_raster(ax=ax, pixel_ratio=args.pixel_ratio, return_pixel_mapping=True)
        print(f' done ({perf_counter()-tt:.1f} s)')
        st['pm'] = pm
        try:
            os.makedirs(cache_root,exist_ok=True)
            pm.to_netcdf(st['pm_file'])
        except Exception as e:
            print(' '*4+f'{hapy.tclr.RED}WARNING - could not cache pixel mapping: {e}{hapy.tclr.END}')
        return rst
    return uxda.to_raster(ax=ax, pixel_mapping=st['pm'])

#---------------------------------------------------------------------------------------------------
# Field loading - snapshot or streaming time mean
#---------------------------------------------------------------------------------------------------
def open_dataset(c,v):
    """lazily open the output stream that holds variable v for case c"""
    htype = var_opts_list[v].get('htype',None)
    if htype is None: raise ValueError(f'ERROR - no htype specified for {var[v]}')
    file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/*{htype}*.nc'
    file_list = sorted(glob.glob(file_path))
    if first_file not in [None,0]: file_list = file_list[first_file:]
    if num_files  not in [None,0]: file_list = file_list[:num_files]
    if file_list==[]: raise ValueError(f'ERROR - empty file list for:\n{file_path}')
    for f in file_list: print(' '*4+f'{hapy.tclr.YELLOW}{os.path.basename(f)}{hapy.tclr.END}')
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    return xr.open_mfdataset(file_list, chunks={'time':1}, data_vars='minimal',
                             coords='minimal', compat='override', decode_times=time_coder)

def load_field(ds, vname, idx, scale):
    """
    read a snapshot, or accumulate a time mean one slice at a time

    A lazy ds[var].mean('time') would let dask decide how many 126 MB slices to
    hold at once, and the whole point here is that only one needs to be resident.
    The ncol subset is applied in numpy after the read, not pushed into netCDF as
    a fancy index.
    """
    da = ds[vname]
    def finish(vals): return (vals if idx is None else vals[idx]).astype(np.float32)

    if 'time' not in da.dims:            return finish(da.values)*np.float32(scale)
    if args.snapshot is not None:
        return finish(da.isel(time=args.snapshot).values)*np.float32(scale)

    nt = len(da['time'])
    print(' '*4+f'averaging {nt} time slices...', end='', flush=True)
    tt = perf_counter()
    total,count = None,0
    with ThreadPoolExecutor(max_workers=read_workers) as pool:
        for vals in pool.map(lambda t: finish(da.isel(time=t).values), range(nt)):
            ### EAMxx writes fill values for the first record of a stream
            if np.isnan(vals).all(): continue
            if total is None: total = np.zeros(vals.size, dtype=np.float64)
            total += np.nan_to_num(vals, nan=0.0); count += 1
    if count==0: raise ValueError(f'ERROR - every time slice of {vname} is fill value')
    print(f' done, used {count}/{nt} ({perf_counter()-tt:.1f} s)')
    return (total/count).astype(np.float32)*np.float32(scale)

#---------------------------------------------------------------------------------------------------
# Read the data
#---------------------------------------------------------------------------------------------------
grid_cache,sub_coords,ncol_idx = {},{},{}
data_all = {}     # (v,c) -> 1D float32 array on the subset columns

for c in range(num_case):
    if scrip_file_list[c] not in grid_cache:
        grid_cache[scrip_file_list[c]] = load_grid_centers(scrip_file_list[c])
    glon,glat = grid_cache[scrip_file_list[c]]
    if args.method=='uxarray' and not args.dry_run:
        idx = setup_uxarray_grid(c)
    elif view is not None:
        idx = region_index(glon,glat)
        print(' '*2+f'region subset   : {idx.size} of {glon.size} columns '
                    f'({100.*idx.size/glon.size:.2f}%)')
    else:
        idx = None
    ncol_idx[c]   = idx
    sub_coords[c] = (glon,glat) if idx is None else (glon[idx],glat[idx])

if args.dry_run:
    print(' '*2+'dry run - exiting'); sys.exit(0)

for v in range(num_var):
    hapy.print_line()
    print(' '*2+'var: '+hapy.tclr.MAGENTA+var[v]+hapy.tclr.END)
    scale = var_opts_list[v].get('scale',1.0)
    for c in range(num_case):
        print(' '*2+'case: '+hapy.tclr.GREEN+case[c]+hapy.tclr.END)
        data_all[(v,c)] = load_field(open_dataset(c,v), var[v], ncol_idx[c], scale)
        if print_stats:
            d = data_all[(v,c)]
            print(' '*4+f'{var[v]}: min={np.nanmin(d):.4g}  max={np.nanmax(d):.4g}  '
                        f'mean={np.nanmean(d):.4g}  nan={int(np.isnan(d).sum())}')

#---------------------------------------------------------------------------------------------------
# Background and map features - drawn before the data so the field sits on top
#---------------------------------------------------------------------------------------------------
def add_background(ax):
    if args.background=='stock':
        ### low resolution shaded relief that ships with cartopy - always offline
        ax.stock_img(); return
    if args.background!='image': return
    if not os.path.exists(bg_image):
        print(' '*2+f'{hapy.tclr.RED}WARNING - background image not found, falling back to the '
                    f'cartopy stock image:{hapy.tclr.END}\n{" "*4}{bg_image}\n'
                    f'{" "*4}see notes_eye-candy.md for where to get Blue Marble data')
        ax.stock_img(); return
    img = plt.imread(bg_image)
    ext = [-180,180,-90,90]
    if view is not None and view[0]<=view[1]:
        ### crop before handing it to cartopy - the full Blue Marble tile is
        ### 21600x10800, and warping all of that for a small view is pure waste
        ny,nx = img.shape[0],img.shape[1]
        i1 = max(0, int((view[0]+180.)/360.*nx)); i2 = min(nx, int(np.ceil((view[1]+180.)/360.*nx)))
        ### the image rows run north to south
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
            ### cartopy downloads Natural Earth data on demand, which fails on
            ### compute nodes with no outbound network
            print(' '*2+f'{hapy.tclr.RED}WARNING - {res} map features unavailable '
                        f'({type(e).__name__}), trying coarser{hapy.tclr.END}')

#---------------------------------------------------------------------------------------------------
# Color settings
#---------------------------------------------------------------------------------------------------
def get_norm(v, kind, dmin, dmax):
    opts = var_opts_list[v]
    if kind=='diff':
        lim = max(abs(dmin),abs(dmax))
        return mcolors.Normalize(vmin=-lim, vmax=lim), cmocean.cm.balance
    clev = opts.get('clev',None)
    cmap = opts.get('cmap','viridis')
    if clev is not None: return mcolors.BoundaryNorm(clev, ncolors=256, extend='both'), cmap
    return mcolors.Normalize(vmin=opts.get('vmin',dmin), vmax=opts.get('vmax',dmax)), cmap

def norm_position(norm, x):
    """
    normalized 0-1 position of x in color space

    Normalize already returns 0-1, but BoundaryNorm returns an integer color
    index (and -1 / Ncmap for the extend regions), so it has to be rescaled
    before it can be used for anything but a colormap lookup.
    """
    y = np.ma.filled(np.ma.asarray(norm(x)), 0.).astype(np.float64)
    if isinstance(norm, mcolors.BoundaryNorm):
        y = y/max(float(norm.Ncmap-1), 1.)
    return np.clip(y, 0., 1.)

def make_rgba(rst, norm, cmap, alpha_floor):
    """
    map the raster through the colormap by hand so the low end can fade out

    The colormap lookup uses the norm's own output (integer indices for a
    BoundaryNorm) so under/over/bad colors behave exactly as they would in a
    normal imshow; the alpha ramp uses the rescaled 0-1 position instead.
    """
    cmap = plt.get_cmap(cmap) if isinstance(cmap,str) else cmap
    masked = np.ma.masked_invalid(rst)
    rgba = cmap(norm(masked))
    rgba[...,3] = np.where(np.isfinite(rst), 1., 0.)
    if args.alpha_ramp:
        f = 0.25 if alpha_floor is None else float(norm_position(norm, alpha_floor))
        f = min(max(f,1e-3),0.999)
        ramp = np.clip(norm_position(norm,masked)/f, 0., 1.)
        ### gamma>1 pushes the fade harder at the low end, which matters for a
        ### colormap with a near-white bottom - a linear ramp still leaves a wash
        ### of pale color over everywhere the field is weak but nonzero
        rgba[...,3] *= ramp**args.alpha_gamma
    return rgba

#---------------------------------------------------------------------------------------------------
# Draw the panels
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
    opts = var_opts_list[v]
    #-------------------------------------------------------------------------------
    ### common limits across cases so the panels are comparable
    dmin = float(np.min([np.nanmin(data_all[(v,c)]) for c in range(num_case)]))
    dmax = float(np.max([np.nanmax(data_all[(v,c)]) for c in range(num_case)]))
    if plot_diff:
        dlim = float(np.max([np.nanmax(np.absolute(data_all[(v,c)]-data_all[(v,diff_base)]))
                             for c in range(num_case)]))
    #-------------------------------------------------------------------------------
    for p in range(num_panel):
        kind,c = case_panels[p]
        ax = axs[v,p] if var_x_case else axs[p,v]
        lon,lat = sub_coords[c]
        #---------------------------------------------------------------------------
        if kind=='diff':
            vals = data_all[(v,c)] - data_all[(v,diff_base)]
            norm,cmap = get_norm(v,'diff',-dlim,dlim); alpha_floor = None
        else:
            vals = data_all[(v,c)]
            norm,cmap = get_norm(v,'full',dmin,dmax); alpha_floor = opts.get('alpha_floor',None)
        #---------------------------------------------------------------------------
        add_background(ax)
        add_map_features(ax, opts.get('linecolor','0.4' if args.background=='none' else 'white'))
        if args.background=='none': ax.set_facecolor(opts.get('bgcolor','0.15'))
        #---------------------------------------------------------------------------
        tt = perf_counter()
        rst = rasterize_hapy(vals,lon,lat,ax) if args.method=='hapy' \
              else rasterize_uxarray(vals,c,ax)
        print(' '*2+f'panel {p}: {var[v]} rasterized in {perf_counter()-tt:.2f} s')
        #---------------------------------------------------------------------------
        if args.background=='none' and not args.alpha_ramp:
            img = ax.imshow(rst, origin='lower', extent=ax.get_xlim()+ax.get_ylim(),
                            norm=norm, cmap=cmap, interpolation='nearest')
        else:
            ### hand-built RGBA so the field can fade into the background
            ax.imshow(make_rgba(rst,norm,cmap,alpha_floor), origin='lower',
                      extent=ax.get_xlim()+ax.get_ylim(), interpolation='nearest')
            ### a separate mappable keeps the colorbar tied to the norm, not the RGBA
            img = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        #---------------------------------------------------------------------------
        if args.bare:
            ax.set_frame_on(False)
            for s in ax.spines.values(): s.set_visible(False)
            continue
        #---------------------------------------------------------------------------
        ax.set_title(var_str[v]+(' (diff)' if kind=='diff' else ''),
                     fontsize=title_fontsize, loc='left')
        ax.set_title(f'{name[c]} - {name[diff_base]}' if kind=='diff' else name[c],
                     fontsize=title_fontsize, loc='right')
        #---------------------------------------------------------------------------
        ### explicit colorbar axes in the reserved strip, so the map axes is untouched
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1 + 0.15*cbar_in/figsize[0], pos.y0,
                            0.20*cbar_in/figsize[0], pos.height])
        cbar_kw = {} if isinstance(norm,mcolors.BoundaryNorm) else {'extend':'both'}
        cbar = fig.colorbar(img, cax=cax, orientation='vertical', **cbar_kw)
        cbar.ax.tick_params(labelsize=label_fontsize)

#---------------------------------------------------------------------------------------------------
# Finalize
#---------------------------------------------------------------------------------------------------
os.makedirs(fig_root,exist_ok=True)
### no bbox_inches='tight' - the layout above already reserves exactly what is
### needed, and a tight bbox would silently change the output pixel dimensions
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
