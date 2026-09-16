
# headless geovista/pyvista env recipe (gv_headless)

Plain `conda install geovista pyvista` (e.g. in `ux_env2`) segfaults for any
real 3D rendering on a compute node with no GPU/X server (LCRC Chrysalis and
similar): VTK falls through GLX -> EGL -> OSMesa and none of them work, and
it crashes natively instead of raising a Python error. Recent conda-forge
`mesalib` (26.x) also dropped classic `libOSMesa`, so `conda install mesalib`
alone doesn't fix it either. Also, the latest `geovista` (0.5.3, as of
2026-09) has no upper pin on `pyvista` and breaks with pyvista's newer
internal API (`AttributeError: 'ActiveArrayInfoTuple' object has no
attribute '_namedtuple'` in `geovista.core.combine`).

Recipe to build a working headless env from scratch on any machine:

```shell
# 1. Python must be <=3.12 -- Kitware's vtk-osmesa wheels stop at cp312
conda create -n gv_headless -c conda-forge python=3.12 uxarray geovista pyvista numpy -y
conda activate gv_headless

# 2. Replace the regular (GLX/EGL-only) VTK build with Kitware's OSMesa
#    build. Must run AFTER conda, so pip's files win. --no-deps so pip
#    doesn't try to also pull in a different pyvista/numpy.
pip install --force-reinstall --no-deps --extra-index-url https://wheels.vtk.org vtk-osmesa

# 3. Pin pyvista back to a version geovista 0.5.3 actually works with
#    (pure python package, safe to swap without touching VTK).
pip install --no-deps "pyvista==0.44.2"
```

Sanity check (should render with no display/GPU at all, no segfault):
```python
import pyvista
pyvista.OFF_SCREEN = True
pl = pyvista.Plotter(off_screen=True)
pl.add_mesh(pyvista.Sphere())
pl.screenshot('/tmp/test.png')
print('ok')
```

Gotchas when building meshes from uxarray grids for geovista:
- Pair `face_node_connectivity` with `uxgrid.node_lon`/`node_lat`, NOT
  `face_lon`/`face_lat` -- connectivity indexes into the *node* table.
  Using face coords is a silent out-of-bounds VTK memory read (no
  validation anywhere) that only segfaults later inside a filter like
  `compute_normals`, which looks like an environment problem but isn't.
- Face/cell-centered fields (e.g. `ssh`) need `.cell_data_to_point_data()`
  before `warp_by_scalar`, which requires point data.

# where to get data?

https://visibleearth.nasa.gov/collection/1484/blue-marble

https://visibleearth.nasa.gov/images/73776/august-blue-marble-next-generation-w-topography-and-bathymetry


put jpeg images in one of these root directories:
```shell
/pscratch/sd/w/whannah/Blue_Marble/       # NERSC
/lcrc/group/e3sm/ac.whannah/Blue_Marble/  # LCRC
```


# gdal_translate

```shell
> gdal_translate --help
Usage: gdal_translate [--help] [--help-general] [--long-usage]
       [-ot {Byte/Int8/Int16/UInt16/UInt32/Int32/UInt64/Int64/Float32/Float64/
             CInt16/CInt32/CFloat32/CFloat64}] [-strict]
       [-if <format>]... [-of <format>]
       [-b <band>] [-mask <band>] [-expand {gray|rgb|rgba}]
       [-outsize <xsize>[%]|0 <ysize>[%]|0] [-tr <xres> <yres>]
       [-ovr <level>|AUTO|AUTO-<n>|NONE]
       [-r {nearest,bilinear,cubic,cubicspline,lanczos,average,mode}]
       [-unscale] [-scale[_bn] [<src_min> <src_max> [<dst_min> <dst_max>]]]... [-exponent[_bn] <exp_val>]...
       [-srcwin <xoff> <yoff> <xsize> <ysize>] [-epo] [-eco]
       [-projwin <ulx> <uly> <lrx> <lry>] [-projwin_srs <srs_def>]
       [-a_srs <srs_def>] [-a_coord_epoch <epoch>]
       [-a_ullr <ulx> <uly> <lrx> <lry>] [-a_nodata <value>]
       [-a_gt <gt0> <gt1> <gt2> <gt3> <gt4> <gt5>]
       [-a_scale <value>] [-a_offset <value>]
       [-nogcp] [-gcp <pixel> <line> <easting> <northing> [<elevation>]]...
       |-colorinterp{_bn} {red|green|blue|alpha|gray|undefined}]
       |-colorinterp {red|green|blue|alpha|gray|undefined},...]
       [-mo <META-TAG>=<VALUE>]... [-q] [-sds]
       [-co <NAME>=<VALUE>]... [-stats] [-norat] [-noxmp]
       [-oo <NAME>=<VALUE>]...
       <src_dataset> <dst_dataset>
```


```shell

DATA_ROOT=/pscratch/sd/w/whannah/Blue_Marble
# DATA_ROOT=/lcrc/group/e3sm/ac.whannah/Blue_Marble

FILE_PREFIX=world.200408.3x5400x2700
gdal_translate -ot Int16 -of netCDF ${DATA_ROOT}/${FILE_PREFIX}.png ${DATA_ROOT}/${FILE_PREFIX}.nc

FILE_PREFIX=world.200408.3x5400x2700
gdal_translate -ot Int16 -of netCDF ${DATA_ROOT}/${FILE_PREFIX}.jpg ${DATA_ROOT}/${FILE_PREFIX}.jpg.nc

FILE_PREFIX=world.200408x294x196
gdal_translate -ot Int16 -of netCDF ${DATA_ROOT}/${FILE_PREFIX}.jpg ${DATA_ROOT}/${FILE_PREFIX}.nc

FILE_PREFIX=world.200408.3x21600x10800.august
gdal_translate -ot Int16 -of netCDF ${DATA_ROOT}/${FILE_PREFIX}.png ${DATA_ROOT}/${FILE_PREFIX}.nc

FILE_PREFIX=world.200408.3x21600x10800.august
gdal_translate -ot Float32 -of netCDF ${DATA_ROOT}/${FILE_PREFIX}.png ${DATA_ROOT}/${FILE_PREFIX}.nc



FILE_PREFIX=world.topo.bathy.200408.3x21600x10800
# FILE_PREFIX=world.topo.bathy.200409.3x21600x10800
# FILE_PREFIX=world.topo.bathy.200411.3x21600x10800
gdal_translate -ot Int16 -outsize 21600 10800 -of netCDF ${DATA_ROOT}/${FILE_PREFIX}.jpg ${DATA_ROOT}/${FILE_PREFIX}.nc
# gdal_translate -ot Byte -of netCDF ${DATA_ROOT}/${FILE_PREFIX}.jpg ${DATA_ROOT}/${FILE_PREFIX}.byte.nc
# gdal_translate -ot Int16 -of netCDF -outsize 10800 5400 ${DATA_ROOT}/world.topo.bathy.200408.3x21600x10800.jpg ${DATA_ROOT}/world.topo.bathy.200408.3x10800x5400.nc

gdal_translate -ot Byte -of netCDF -outsize 10800 5400 ${DATA_ROOT}/world.topo.bathy.200408.3x21600x10800.jpg ${DATA_ROOT}/world.topo.bathy.200408.3x10800x5400.byte.nc

# gdal_translate -ot Int16 -of HDF4Image -r bilinear ${DATA_ROOT}/${FILE_PREFIX}.jpg ${DATA_ROOT}/${FILE_PREFIX}.bilin.hdf4

```

# grid files

```shell
# Create grid files for plotting blue marble data
ncremap -6 -G ttl='Equi-Angular grid 21600x10800'#latlon=10800,21600#lat_typ=uni#lon_typ=180_wst -g /pscratch/sd/w/whannah/files_grid/scrip_21600x10800.nc
ncremap -6 -G ttl='Equi-Angular grid 10800x5400'#latlon=5400,10800#lat_typ=uni#lon_typ=180_wst -g /pscratch/sd/w/whannah/files_grid/scrip_10800x5400.nc
ncremap -6 -G ttl='Equi-Angular grid 5400x2700'#latlon=2700,5400#lat_typ=uni#lon_typ=180_wst -g /pscratch/sd/w/whannah/files_grid/scrip_5400x2700.nc
ncremap -6 -G ttl='Equi-Angular grid 294x196'#latlon=196,294#lat_typ=uni#lon_typ=180_wst -g /pscratch/sd/w/whannah/files_grid/scrip_294x196.nc
```

# overlay method

OpenCV overlay from forum post
https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image
```python
import cv2

background = cv2.imread("background.png", cv2.IMREAD_UNCHANGED)
foreground = cv2.imread("overlay.png", cv2.IMREAD_UNCHANGED)

# normalize alpha channels from 0-255 to 0-1
alpha_background = background[:,:,3] / 255.0
alpha_foreground = foreground[:,:,3] / 255.0

# set adjusted colors
for color in range(0, 3):
    background[:,:,color] = alpha_foreground * foreground[:,:,color] + \
        alpha_background * background[:,:,color] * (1 - alpha_foreground)

# set adjusted alpha and denormalize back to 0-255
background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

# display the image
cv2.imshow("Composited image", background)
cv2.waitKey(0)
```