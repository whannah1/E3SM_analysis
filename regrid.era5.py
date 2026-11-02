import os, subprocess as sp, hapy
host = hapy.get_host()
#---------------------------------------------------------------------------------------------------
'''
GRID_ROOT=/lustre/orion/cli115/proj-shared/hannah6/files_grid
MAPS_ROOT=/lustre/orion/cli115/proj-shared/hannah6/files_map

DST_NE=30
SRC_GRID=${GRID_ROOT}/scrip_ERA5_721x1440.nc
DST_GRID=${GRID_ROOT}/ne${DST_NE}pg2_scrip.nc

MAP_ALG=traave
MAP_FILE=${MAPS_ROOT}/map_721x1440_to_ne${DST_NE}pg2_${MAP_ALG}.nc

# generate map file
ncremap -a ${MAP_ALG} --src_grd=${SRC_GRID} --dst_grd=${DST_GRID} -m ${MAP_FILE}

'''
#---------------------------------------------------------------------------------------------------
if host=='olcf':
  src_root = '/lustre/orion/cli115/world-shared/e3sm/diagnostics/observations/Atm/time-series/ERA5'
  dst_root = '/lustre/orion/cli115/proj-shared/hannah6/obs_data/ERA5'
  map_root = '/lustre/orion/cli115/proj-shared/hannah6/files_map'
# if host=='nersc':
#   src_root = ???
#   dst_root = ???
#   map_root = ???
# if host=='lcrc':
# if host=='llnl':
#---------------------------------------------------------------------------------------------------
dst_grid = 'ne30pg2'
dst_sub = f'diag_data_remap_{dst_grid}'
map_alg = 'traave'


map_file = f'{map_root}/map_721x1440_to_{dst_grid}_{map_alg}.nc'

var_list = []
# var_list.append('sp')
var_list.append('t2m')
# var_list.append('tas')
# var_list.append('ua')
#---------------------------------------------------------------------------------------------------
if not os.path.exists(f'{dst_root}/{dst_sub}'): os.makedirs(f'{dst_root}/{dst_sub}',exist_ok=True)
#---------------------------------------------------------------------------------------------------
for var in var_list: 
  
  src_file = f'{src_root}/{var}_197901_201912.nc'

  dst_file = src_file.replace(src_root,f'{dst_root}/{dst_sub}').replace('.nc',f'.remap_{dst_grid}.nc')

  cmd = f'ncremap'
  # cmd +=f' --overwrite' # doesn't work?
  cmd +=f' --map={map_file}'
  cmd +=f' --in_fl={src_file}'
  cmd +=f' --out_fl={dst_file}'

  print(f'\n  {hapy.tclr.GREEN}{cmd}{hapy.tclr.END}\n')

  try:
    sp.check_output(cmd,shell=True,universal_newlines=True)
  except sp.CalledProcessError as error:
    print(error.output)
    exit()

#---------------------------------------------------------------------------------------------------