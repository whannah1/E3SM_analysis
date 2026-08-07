#!/usr/bin/env python
class tclr: END,RED,GREEN,MAGENTA,CYAN = '\033[0m','\033[31m','\033[32m','\033[35m','\033[36m'
import sys, os, glob, subprocess as sp, argparse
import xarray as xr
# home = os.getenv('HOME')
#---------------------------------------------------------------------------------------------------
''' grid file creation

ncremap -G ttl='Equi-Angular grid 2x2 degree, dimensions 90x180, cell edges on Poles/Equator and Prime Meridian/Date Line'#latlon=90,180#lat_typ=uni#lon_typ=grn_wst -g $HOME/E3SM/data_grid/90x180_s2n_scrip.nc

# PMP grids
ncremap -G ttl='Equi-Angular grid 180x360'#latlon=180,360#lat_typ=uni#lon_typ=grn_wst -g $HOME/E3SM/data_grid/pmp_180x360_scrip.nc
ncremap -G ttl='CAM-FV grid 121x240'#latlon=121,240#lat_typ=fv#lon_typ=grn_ctr -g $HOME/E3SM/data_grid/pmp_121x240_scrip.nc
ncremap -G ttl='CAM-FV grid  72x144'#latlon=72,144#lat_typ=fv#lon_typ=grn_wst -g $HOME/E3SM/data_grid/pmp_72x144_scrip.nc

# command to create CMIP6 HighResMIP vertical pressure coordinate file
ncap2 -O -v -s 'defdim("plev",27);plev[$plev]={100000,97500,95000,92500,90000,87500,85000,82500,80000,77500,75000,70000,65000,60000,55000,50000,45000,40000,35000,30000,25000,22500,20000,17500,15000,12500,10000};' ~/E3SM/vert_grid_files/vrt_prs_CMIP6.nc
'''
#---------------------------------------------------------------------------------------------------
''' map file creation
SRC_GRID=$HOME/E3SM/data_grid/ne30pg2_scrip.nc
DST_GRID=$HOME/E3SM/data_grid/90x180_s2n_scrip.nc
MAP_FILE=$HOME/maps/map_ne30pg2_to_90x180_traave.nc
ncremap -a traave --src_grd=${SRC_GRID} --dst_grd=${DST_GRID} -m ${MAP_FILE}
'''
#---------------------------------------------------------------------------------------------------
'''

MAP_FILE=~/maps/map_ne30pg2_to_90x180_traave.nc
DATA_ROOT=/pscratch/sd/w/whannah/e3sm_scratch/pm-gpu

MAP_FILE=~/maps/map_ne30pg2_to_90x180_traave.nc
DATA_ROOT=/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch
REGRID_ARGS="--map=${MAP_FILE} --root=${DATA_ROOT} --sub=run --prefix=output.daily.AVERAGE.nhours_x24"

python regrid.eamxx.py ${REGRID_ARGS} --case=E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_0
python regrid.eamxx.py ${REGRID_ARGS} --case=E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_1

python regrid.eamxx.py ${REGRID_ARGS} --case=E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_0.c0_0.001
python regrid.eamxx.py ${REGRID_ARGS} --case=E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_0.c0_0.0001
python regrid.eamxx.py ${REGRID_ARGS} --case=E3SM.2026-ZM-ICW-00.F2010xx-ZM.ne30pg2.NN_4.zm_rad_0.c0_0



MAP_FILE=/lustre/orion/cli115/proj-shared/hannah6/files_map/map_ne256pg2_to_180x360_traave.nc
DATA_ROOT=/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch
REGRID_ARGS="--map=${MAP_FILE} --root=${DATA_ROOT} --sub=archive/atm/hist --prefix=output.daily.AVERAGE.nhours_x24"

python regrid.eamxx.py ${REGRID_ARGS} --case=E3SM.2026-ZM-BASE-00.ne256pg2.NN_128.F2010-SCREAMv1
python regrid.eamxx.py ${REGRID_ARGS} --case=E3SM.2026-ZM-BASE-00.ne256pg2.NN_128.F2010xx-ZM-CICE


'''
#---------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# parser.add_argument("-f", "--file", required=True, help="Path to input file")
parser.add_argument('--map',    required=True, dest='map_file', default=None,  help='map file')
parser.add_argument('--case',   required=True, dest='case',     default=None,  help='simulation name')
parser.add_argument('--root',   required=True, dest='case_root',default=None,  help='root directory for case')
parser.add_argument('--sub',    required=False,dest='case_sub', default='run', help='subfolder for data (default=run)')
parser.add_argument('--prefix', required=True, dest='prefix',   default=None,  help='file name prefix')
parser.add_argument('-n',       required=False,dest='num_file', default=None,  help='sets number of files to print')
# parser.add_argument('--vert',action='store_true', dest='vert_pressure_remap', default=False,help='enable vertical interpolation to pressure levels')
args = parser.parse_args()
#---------------------------------------------------------------------------------------------------
case        = args.case
map_file    = args.map_file
case_root   = args.case_root
src_sub     = args.case_sub
prefix      = args.prefix

#-------------------------------------------------------------------------------
execute   = True
overwrite = True
print_cmd = True
# write_log = False
# nlat_dst,nlon_dst =  90,180
#-------------------------------------------------------------------------------
ds = xr.open_dataset(map_file)
[dst_nlon,dst_nlat] = ds.dst_grid_dims.values
#-------------------------------------------------------------------------------
dst_sub = f'remap_{dst_nlat}x{dst_nlon}'
src_dir = f'{case_root}/{case}/{src_sub}'
dst_dir = f'{case_root}/{case}/{dst_sub}'
#-------------------------------------------------------------------------------
# if opts.vert_pressure_remap: 
#     odir_sub = f'data_remap_{nlat_dst}x{nlon_dst}_prs'
# else:
#     odir_sub = f'data_remap_{nlat_dst}x{nlon_dst}'
#-------------------------------------------------------------------------------
print()
print(f'  case      : {case}')
print(f'  dst_nlon  : {dst_nlon}')
print(f'  dst_nlat  : {dst_nlat}')
print(f'  case_root : {case_root}')
print(f'  src_dir   : {src_dir}')
print(f'  dst_dir   : {dst_dir}')
print()
#-------------------------------------------------------------------------------
if not os.path.exists(dst_dir): os.makedirs(dst_dir,exist_ok=True)
#-------------------------------------------------------------------------------
file_list = sorted(glob.glob(f'{src_dir}/{prefix}*'))
#-------------------------------------------------------------------------------
if len(file_list)==0: exit(f'\nNo files found for remapping...?\n') 
#-------------------------------------------------------------------------------
# for f in file_list: print(f)
# exit()
#---------------------------------------------------------------------------------------------------
for src_file in file_list : 

    dst_file = src_file.replace('.nc',f'.remap_{dst_nlat}x{dst_nlon}.nc')

    if overwrite and os.path.isfile(dst_file) : os.remove(dst_file)

    # Change sub-directory
    dst_file = dst_file.replace(f'/{src_sub}/',f'/{dst_sub}/')

    cmd = f'ncremap -m {map_file} -i {src_file} -o {dst_file}'

    # if opts.vert_pressure_remap:
    #     cmd += ' --vrt_fl=$HOME/E3SM/vert_grid_files/vrt_prs_CMIP6.nc'

    if print_cmd:
        print(f'\n  {tclr.GREEN}{cmd}{tclr.END}\n')
    else:
        src_file_name = src_file.replace(src_dir,'')
        dst_file_name = dst_file.replace(dst_dir,'')
        print(f'  {src_file_name}  >  {dst_file_name}')
    
    if execute: 
        # os.system(cmd)
        try:
            sp.check_output(cmd,shell=True,universal_newlines=True)
        except sp.CalledProcessError as error:
            print(error.output)
            exit()

#---------------------------------------------------------------------------------------------------
