import os, numpy as np, cmocean, uxarray as ux, geovista as gv, pyvista
from scipy.spatial import cKDTree
from geovista.common import to_cartesian
pyvista.OFF_SCREEN = True
#-------------------------------------------------------------------------------
'''NOTES
salloc --nodes=1 --time=02:00:00 --account=e3sm
srun --pty bash
conda activate gv_headless   # headless env w/ vtk-osmesa (see chat history / memory for why)
time python code_eye-candy/plot.eye-candy.geovista-test.v1.py
'''
#-------------------------------------------------------------------------------
fig_file = 'figs_eye-candy/eye-candy.geovista-test.v1.png'

# DIN_LOC_ROOT = '/lustre/orion/cli115/world-shared/e3sm/inputdata' # OLCF
# DIN_LOC_ROOT = '/global/cfs/cdirs/e3sm/inputdata' # NERSC
DIN_LOC_ROOT = '/lcrc/group/e3sm/data/inputdata' # LCRC

# # v3LR data for testing
# ocn_grid_file = f'/lcrc/group/e3sm/data/inputdata/ocn/mpas-o/IcoswISC30E3r5/ocean.IcoswISC30E3r5.nomask.scrip.20231120.nc'
# ocn_data_root = '/lcrc/group/e3sm2/ac.jwolfe/E3SMv3/20260720.baseline.WCYCL2010.bluepulse.chrysalis/archive/ocn/hist'
# ocn_data_file = f'{ocn_data_root}/20260720.baseline.WCYCL2010.bluepulse.chrysalis.mpaso.hist.am.highFrequencyOutput.0157-01-01_00.00.00.nc'
# atm_grid_file = '/lcrc/group/e3sm/ac.whannah/files_grid/ne30pg2_scrip.nc'
# atm_data_root = '/lcrc/group/e3sm2/ac.jwolfe/E3SMv3/20260720.baseline.WCYCL2010.bluepulse.chrysalis/archive/atm/hist'
# atm_data_file = f'{ocn_data_root}/20260720.baseline.WCYCL2010.bluepulse.chrysalis.eam.h2.0157-01-01-00000.nc'
# topo_data_file = '/lcrc/group/e3sm/data/inputdata/atm/cam/topo/USGS-gtopo30_ne30np4pg2_x6t-SGH.c20210614.nc'
# topo_grid_file = '/lcrc/group/e3sm/ac.whannah/files_grid/ne30pg2_scrip.nc'

# v3HR data
ocn_grid_file = f'/lcrc/group/e3sm/data/inputdata/ocn/mpas-o/RRSwISC6to18E3r5/ocean.RRSwISC6to18E3r5.nomask.scrip.20240327.nc'
ocn_data_root = '/lcrc/group/e3sm2/ac.jwolfe/E3SMv3_dev/v3.HR.1950_control/archive/ocn/hist'
ocn_data_file = f'{ocn_data_root}/v3.HR.1950_control.mpaso.hist.am.highFrequencyOutput.2009-01-01_00.00.00.nc'
atm_grid_file = '/lcrc/group/e3sm/ac.whannah/files_grid/ne120pg2_scrip.nc'
atm_data_root = '/lcrc/group/e3sm2/ac.jwolfe/E3SMv3_dev/v3.HR.1950_control/archive/atm/hist'
# atm_data_file = f'{atm_data_root}/v3.HR.1950_control.eam.h2.2009-01-06-00000.nc'
# atm_data_file = f'{atm_data_root}/v3.HR.1950_control.eam.h2.2010-01-01-00000.nc'
atm_data_file = f'{atm_data_root}/v3.HR.1950_control.eam.h2.2012-01-21-00000.nc'
# atm_data_file = f'{atm_data_root}/v3.HR.1950_control.eam.h2.2014-01-11-00000.nc'
# atm_data_file = f'{atm_data_root}/v3.HR.1950_control.eam.h2.2018-01-21-00000.nc'
# atm_data_file = f'{atm_data_root}/v3.HR.1950_control.eam.h2.2019-01-16-00000.nc'
# atm_data_file = f'{atm_data_root}/v3.HR.1950_control.eam.h2.2020-01-11-00000.nc'
# topo_data_file = '/lcrc/group/e3sm/data/inputdata/atm/cam/topo/USGS-gtopo30_ne256np4pg2_x6t-SGH.c20210614.nc'
# topo_grid_file = '/lcrc/group/e3sm/ac.whannah/files_grid/ne256pg2_scrip.nc'
topo_data_file = '/lcrc/group/e3sm/public_html/inputdata/atm/cam/topo/USGS-gtopo30_ne1024np4pg2_x6t-SGH.c20210614.nc'
topo_grid_file = '/lcrc/group/e3sm/ac.whannah/files_grid/ne1024pg2_scrip.nc'


# target_lat, target_lon = 35.6762, 139.6503 # Tokyo
# target_lat, target_lon = 18.4331, 360.-66.0517 # Puerto Rico
target_lat, target_lon = 35, 360.-70
# target_lat, target_lon = 30, 360.-120. # Baja
# target_lat, target_lon = -60, 360.-55 # Falkland Islands

xyz_vector = to_cartesian(target_lon, target_lat, radius=1.0).squeeze()
# xyz_vector = to_cartesian(target_lon, target_lat, radius=0.8).squeeze()

#-------------------------------------------------------------------------------
print()
print(f' ocn_grid_file: {ocn_grid_file}')
print(f' ocn_data_file: {ocn_data_file}')
print()
#-------------------------------------------------------------------------------
# Read the native ocean grid and SSH field, then build a spherical mesh from the MPAS connectivity
uxds = ux.open_dataset(ocn_grid_file, ocn_data_file)
lons = uxds.uxgrid.node_lon.values
lats = uxds.uxgrid.node_lat.values
conn = uxds.uxgrid.face_node_connectivity.values
ssh = uxds["ssh"].isel(Time=0).values
sst = uxds["temperatureAtSurface"].isel(Time=0).values  # deg C

# A small fraction of cells (all under Antarctic ice shelves, e.g. Ronne/
# Filchner - lat < -62) have wildly unphysical ssh (down to ~-2300 m) since
# plain ssh isn't pressure-adjusted for the ice-shelf cavity; drop those
# faces entirely so they don't blow out the color scale / warp relief
ssh_threshold = 10.0  # meters; real ssh anomalies here are O(1) m
valid_face = np.abs(ssh) < ssh_threshold
print(f'masking out {(~valid_face).sum()} of {len(ssh)} faces with |ssh| >= {ssh_threshold} m')
conn = conn[valid_face]
ssh = ssh[valid_face]
sst = sst[valid_face]

# Color by sea surface temperature (like the reference image), but keep
# ssh driving the warp/relief - attach both as cell data up front so they
# go through triangulate()/smoothing together, consistently
ssh_mesh = gv.Transform.from_unstructured( lons, lats, connectivity=conn,
                                           data=ssh, name="ssh", start_index=0)
ssh_mesh.cell_data["sst"] = sst
#-------------------------------------------------------------------------------
# MPAS cells are irregular n-gons (5-8 sided). VTK triangulates them
# internally at render time to draw them, but per-vertex Phong normals get
# computed against the original n-gon topology - when the renderer's
# internal triangulation diagonals disagree with that, curved/lit polygons
# show a fine random "fan" faceting that looks like noise and has nothing
# to do with the ssh data. Triangulate explicitly first so everything
# downstream (normals, warp, shading) agrees on the same triangles.
ssh_mesh = ssh_mesh.triangulate()

# ssh comes in as face/cell data (one value per face); warp_by_scalar needs
# point data, so interpolate it onto the mesh points first
ssh_mesh = ssh_mesh.cell_data_to_point_data()

# Low-pass filter ssh first, using repeated point<->cell data round trips
# (a cheap Jacobi-style smoother that respects the actual mesh connectivity),
# so only the broad eddy structure drives the warp/color.
n_smooth_passes = 2 # 8
for _ in range(n_smooth_passes):
    ssh_mesh = ssh_mesh.point_data_to_cell_data().cell_data_to_point_data()
#-------------------------------------------------------------------------------
# Give the eddies a bit of relief. add_base_layer() draws an OPAQUE sphere
# at the default radius=1.0, same as this mesh's rest radius, and
# warp_by_scalar displaces points inward for negative values - at a large
# enough exaggeration factor that's enough to tunnel the ocean surface
# behind/inside the opaque base sphere, so blue marble's raw texture shows
# through there instead of any ssh color at all (looks like "masking",
# isn't). Fix: warp using ssh clipped/shifted to be >=0, so the ocean layer
# only ever bulges outward and can never sink below the base sphere; keep
# the real signed ssh for coloring, only the geometric displacement is
# shifted. Clip (rather than shift by the true global min) and tie the
# floor to the same clim lower bound already used for coloring - using the
# true global min (~-9 m, from the rare polar tail) as the floor would push
# the *entire* ocean outward by that same large near-constant baseline
# before any real local relief, growing a visible coastline gap as factor
# increases; ssh_clim[0] is representative of the actual displayed range.
# (this is the warp-only clim, distinct from sst_clim used for coloring below)
sst_clim = [0, 29]  # deg C
ssh_clim = [-3, 0.25]
ssh_warp_factor = 0.02
ssh_mesh["ssh_warp"] = np.clip(ssh_mesh["ssh"], ssh_clim[0], None) - ssh_clim[0]
ssh_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
ssh_mesh.warp_by_scalar(scalars="ssh_warp", inplace=True, factor=ssh_warp_factor)
#-------------------------------------------------------------------------------
# Land topography (native ne256pg2 physics grid - a different mesh than the
# ocean's, so this is a separate PolyData composited alongside ssh_mesh, not
# merged with it). PHIS is surface geopotential (m^2/s^2); divide by g for
# elevation in meters.
#
# NOTE: originally masked to LANDFRAC > 0.5 to avoid painting over the
# ocean, but that hard threshold produced a scrambled "swiss cheese" look
# over places like the Canadian Shield lake district, where thousands of
# sub-grid-cell lakes push neighboring cells randomly across the 0.5 cutoff
# - some cells get excluded (falling through to raw blue_marble) right next
# to included ones (elevation-colored), which reads as scrambled/corrupted
# data even though each individual cell's value is correct (verified: grid
# file and data file column ordering match exactly). Fix: don't mask at
# all - render the topo mesh globally (it already covers the whole planet;
# ocean columns just have near-zero elevation). ssh_mesh is added on top
# afterwards and already covers ~all ocean area, so it naturally occludes
# this layer over open ocean with no hard-threshold gap artifacts at
# coastlines. Real gaps (sea-ice/ice-shelf areas ssh_mesh's own mask
# dropped) fall through to blue_marble underneath, per your note.
uxtopo = ux.open_dataset(topo_grid_file, topo_data_file)
topo_lons = uxtopo.uxgrid.node_lon.values
topo_lats = uxtopo.uxgrid.node_lat.values
topo_conn = uxtopo.uxgrid.face_node_connectivity.values
topo_data = uxtopo["PHIS"].values / 9.80616  # m
topo_factor = 5e-6  # named up front now - needed below for the local shift math too

# Mask out topo columns that fall over the ocean, using the *ocean* mesh's
# own valid cell centers (not topo's independent LANDFRAC - that had the
# swiss-cheese lake problem earlier). topo (ne256pg2) and the ocean (MPAS)
# are unrelated grids with no shared indexing, so this is a real spatial
# join: nearest-neighbor via a KDTree on unit-sphere xyz (avoids lon
# wraparound/pole issues that a naive lat/lon distance would have), then
# threshold by distance. This should track the ocean model's actual
# coastline far more cleanly than an independent land-fraction estimate.
def _to_xyz(lon_deg, lat_deg):
    lon, lat = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    return np.stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], axis=-1)

ocean_face_lon_valid = uxds.uxgrid.face_lon.values[valid_face]
ocean_face_lat_valid = uxds.uxgrid.face_lat.values[valid_face]
ocean_tree = cKDTree(_to_xyz(ocean_face_lon_valid, ocean_face_lat_valid))
topo_face_lon = uxtopo.uxgrid.face_lon.values
topo_face_lat = uxtopo.uxgrid.face_lat.values
ocean_dist, ocean_idx = ocean_tree.query(_to_xyz(topo_face_lon, topo_face_lat), k=1, workers=-1)

# chord distance on a unit sphere ~= great-circle angle (radians) for small
# angles; convert a physical threshold (km) using Earth's radius (6371 km).
# The ocean mesh's cells are 6-18 km, so 15 km is a middle-to-coarse-end
# choice: big enough to fully cover coarse-resolution open ocean, small
# enough to leave most real coastal land alone. Tune as needed.
ocean_dist_threshold_km = 15.0
ocean_valid_mask = ocean_dist < (ocean_dist_threshold_km / 6371.0)
print(f'masking out {ocean_valid_mask.sum()} of {len(ocean_valid_mask)} topo faces within '
      f'{ocean_dist_threshold_km} km of a valid ocean cell')
land_valid = ~ocean_valid_mask

# Local coastal radius matching, instead of a single guessed global shift
# constant: reuse the same ocean_tree/ocean_dist above to look up, for
# every topo face, the ACTUAL nearby ocean cell's radius (from its own ssh
# warp), and use that as this topo face's local "sea level" baseline. This
# tracks each stretch of coastline individually instead of assuming one
# global number fits everywhere. Far from any coast (deep continental
# interior), "nearest ocean cell" stops being a meaningful match, so blend
# it down with distance to the same analytic global estimate used before
# (derived from ssh_clim/ssh_warp_factor, not guessed).
ocean_face_radius = 1.0 + (np.clip(ssh, ssh_clim[0], None) - ssh_clim[0]) * ssh_warp_factor
local_ocean_radius = ocean_face_radius[ocean_idx]  # per topo face, pre-mask
local_topo_shift = (local_ocean_radius - 1.0) / topo_factor

topo_baseline_shift = ((0.0 - ssh_clim[0]) * ssh_warp_factor) / topo_factor  # analytic global fallback
coastal_blend_dist_km = 100.0  # local match fades out to the fallback over this distance inland
coast_weight = np.clip(1.0 - (ocean_dist * 6371.0) / coastal_blend_dist_km, 0.0, 1.0)
topo_shift_field = coast_weight * local_topo_shift + (1.0 - coast_weight) * topo_baseline_shift

topo_data = topo_data[land_valid]
topo_conn = topo_conn[land_valid]
topo_shift_field = topo_shift_field[land_valid]

topo_mesh = gv.Transform.from_unstructured( topo_lons, topo_lats, connectivity=topo_conn, data=topo_data, name="topo", start_index=0)
topo_mesh.cell_data["local_shift"] = topo_shift_field
topo_mesh = topo_mesh.triangulate()  # see ssh_mesh note above re: n-gon/normals faceting
topo_mesh = topo_mesh.cell_data_to_point_data()

# A few smoothing passes (same technique as ssh) to soften the faceted,
# individually-visible-quad look at this grid's resolution
n_smooth_passes_topo = 4
for _ in range(n_smooth_passes_topo):
    topo_mesh = topo_mesh.point_data_to_cell_data().cell_data_to_point_data()

# Same z-fighting/occlusion risk as ssh vs the base sphere: clip topo
# at 0 before warping so land never dips inside the (radius=1.0) base
# sphere at the handful of below-sea-level columns (Death Valley-like spots)
# or slightly-negative smoothed values over open ocean. Baseline is now the
# per-point "local_shift" field computed above (smoothed along with
# everything else in the loop above) instead of one guessed constant.
topo_mesh["topo_warp"] = np.clip(topo_mesh["topo"], 0, None) + topo_mesh["local_shift"]
topo_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
topo_mesh.warp_by_scalar(scalars="topo_warp", inplace=True, factor=topo_factor)

elev_clim_max = 6500.0
# topo colormap is bathymetry<->topography colormap; so set clim symmetric about 0
topo_mesh["topo_display"] = np.sign(topo_mesh["topo"]) * np.sqrt(np.abs(topo_mesh["topo"]))
fac = 0.7 # factor to reduce limits to get white peaks over mountains
topo_clim = [-np.sqrt(elev_clim_max)*fac, np.sqrt(elev_clim_max)*fac]
#-------------------------------------------------------------------------------
# Translucent cloud layer from FLUT 
uxatm = ux.open_dataset(atm_grid_file, atm_data_file)
atm_lons = uxatm.uxgrid.node_lon.values
atm_lats = uxatm.uxgrid.node_lat.values
atm_conn = uxatm.uxgrid.face_node_connectivity.values
flut = uxatm["FLUT"].isel(time=0).values  # W/m^2

flut_threshold = 220.0  # mask out clear/thin-cloud sky above this
flut_valid = flut < flut_threshold
print(f'clouds: keeping {flut_valid.sum()} of {len(flut)} columns with FLUT < {flut_threshold}')
atm_conn = atm_conn[flut_valid]
flut = flut[flut_valid]

cloud_radius = 1.10 # 1.08  # fixed shell: tallest topo peaks (~1.16)

cloud_mesh = gv.Transform.from_unstructured( atm_lons, atm_lats, connectivity=atm_conn,
                                             data=flut, name="FLUT", radius=cloud_radius, start_index=0)
cloud_mesh = cloud_mesh.triangulate()
cloud_mesh = cloud_mesh.cell_data_to_point_data()

flut_min = 80.0  # (actual min ~68)
opacity = np.clip((flut_threshold - cloud_mesh["FLUT"]) / (flut_threshold - flut_min), 0.0, 1.0)
cloud_mesh["opacity"] = opacity
# gamma = 2.0  # higher = more aggressive push of the top end toward 1.0
# cloud_mesh["opacity"] = 1.0 - (1.0 - opacity)**gamma
cloud_mesh["opacity"] = np.clip(cloud_mesh["opacity"], 0.0, 0.98)

#-------------------------------------------------------------------------------
# Composite the scene: blue marble base (fills any gaps - see note above),
# then land topography, then the ocean SSH-warped/SST-colored layer on top
p = gv.GeoPlotter()
p.enable_anti_aliasing("ssaa")  # supersample AA - softens jagged coastlines/mesh edges
                                # (did NOT fix the n-gon faceting noise; see ssh_mesh.triangulate() above)
# p.add_base_layer(texture=gv.blue_marble())
p.add_mesh( topo_mesh, scalars="topo_display", 
            # cmap=cmocean.cm.topo, clim=topo_clim,
            cmap="topo", clim=topo_clim,
            show_scalar_bar=False, smooth_shading=True)
p.add_mesh( ssh_mesh, scalars="sst", 
            # cmap=cmocean.cm.thermal,
            # cmap="spectral", 
            # cmap="rainbow4",
            cmap="turbo",
            clim=sst_clim,
            show_scalar_bar=False, smooth_shading=True)
if 'cloud_mesh' in locals():
    p.add_mesh( cloud_mesh, color="white", opacity=cloud_mesh["opacity"], smooth_shading=True)
p.view_vector(xyz_vector)
p.camera.zoom(1.7)
#-------------------------------------------------------------------------------
os.makedirs(os.path.dirname(fig_file), exist_ok=True)
p.screenshot(fig_file,scale=1.5)
print(f'\n{fig_file}\n')
#-------------------------------------------------------------------------------
