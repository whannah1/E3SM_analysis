import os, glob, numpy as np, xarray as xr, copy, string, dask, pandas as pd, cmocean
import hapy
#-------------------------------------------------------------------------------
'''
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account=m4310
source activate ux_env
time python code_QBO/2026_SciDAC_calculate_qoi.py

sbatch -A m4310 -C cpu -q regular -t 00:30:00 -N 1 --wrap="python my_script.py"

sbatch --account=m4310 --constraint=cpu --qos=regular --nodes=1 --time=06:00:00 \
       --mail-user=hannah6@llnl.gov --mail-type=END,FAIL \
       --job-name=calc_qoi_b1 \
       --wrap="conda activate ux_env && python /global/homes/w/whannah/E3SM_analysis/code_QBO/2026_SciDAC_calculate_qoi.py  --block=1"

sbatch --account=m4310 --constraint=cpu --qos=regular --nodes=1 --time=06:00:00 \
       --mail-user=hannah6@llnl.gov --mail-type=END,FAIL \
       --job-name=calc_qoi_b2 \
       --wrap="conda activate ux_env && python /global/homes/w/whannah/E3SM_analysis/code_QBO/2026_SciDAC_calculate_qoi.py  --block=2"

sbatch --account=m4310 --constraint=cpu --qos=regular --nodes=1 --time=06:00:00 \
       --mail-user=hannah6@llnl.gov --mail-type=END,FAIL \
       --job-name=calc_qoi_b3 \
       --wrap="conda activate ux_env && python /global/homes/w/whannah/E3SM_analysis/code_QBO/2026_SciDAC_calculate_qoi.py  --block=3"

sbatch --account=m4310 --constraint=cpu --qos=regular --nodes=1 --time=06:00:00 \
       --mail-user=hannah6@llnl.gov --mail-type=END,FAIL \
       --job-name=calc_qoi_b4 \
       --wrap="conda activate ux_env && python /global/homes/w/whannah/E3SM_analysis/code_QBO/2026_SciDAC_calculate_qoi.py  --block=4"

sbatch --account=m4310 --constraint=cpu --qos=regular --nodes=1 --time=06:00:00 \
       --mail-user=hannah6@llnl.gov --mail-type=END,FAIL \
       --job-name=calc_qoi_b5 \
       --wrap="conda activate ux_env && python /global/homes/w/whannah/E3SM_analysis/code_QBO/2026_SciDAC_calculate_qoi.py  --block=5"

sbatch --account=m4310 --constraint=cpu --qos=regular --nodes=1 --time=06:00:00 \
       --mail-user=hannah6@llnl.gov --mail-type=END,FAIL \
       --job-name=calc_qoi_b6 \
       --wrap="conda activate ux_env && python /global/homes/w/whannah/E3SM_analysis/code_QBO/2026_SciDAC_calculate_qoi.py  --block=6"

'''
#-------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--block", type=int, required=True)
args = parser.parse_args()
block = args.block
#-------------------------------------------------------------------------------
case_list = []
opts_list = []
def add_case( case_in, **kwargs ):
   case_list.append(case_in)
   case_opts = {}
   for k, val in kwargs.items(): case_opts[k] = val
   opts_list.append(case_opts)
#-------------------------------------------------------------------------------
if os.path.exists( '/lcrc/group/e3sm' )      : obs_scratch_path = '/lcrc/group/e3sm/diagnostics/observations/Atm/time-series'
if os.path.exists( '/global/cfs/cdirs/e3sm' ): obs_scratch_path = '/global/cfs/cdirs/e3sm/diagnostics/observations/Atm/time-series'
#-------------------------------------------------------------------------------
# Create tarball of hov data:
# tar -czvf QBO.hov.2023_L80_ensemble.tar.gz data_temp/QBO.hov.* 
# tar -czvf QBO.hov.2023_AMIP.tar.gz data_temp/QBO.hov.v1.E3SM.2023-SCIDAC-v2-AMIP* 
#-------------------------------------------------------------------------------
verbose_indent = ' '*2
lat1,lat2 = -5,5

# lev = np.array([   0.1, 0.2, 0.5, 1.,    2.,    3.,    5.,    7.,   10.,   20.,   30.,   50.,   70.,  100.,  125.,  150.,])
plev_target_hov = np.array([ 1., 2., 3., 5., 7., 10., 20., 30., 50., 70., 100., 125., 150.])

plev_target_clm= np.array([ 1., 2., 3., 5., 7., 10, 20, 30, 50, 70, 100, 125, 150, 175, \
                             200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, \
                             750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])

# convert to Pa for vinth2p_simple
plev_target_hov = plev_target_hov*1e2
plev_target_clm = plev_target_clm*1e2

# add_case('ERA5',yr1=1984,yr2=1993)
# add_case('ERA5',yr1=1995,yr2=1999)
# add_case('ERA5',yr1=1995,yr2=2004)

calc_hov = True
calc_clm = True

ens_root = '/global/cfs/cdirs/m4310/data/sims/'

ens_num_list = []
if block==1:
   ens_num_list.append('0051')
   ens_num_list.append('0091')
   ens_num_list.append('0101')
   ens_num_list.append('0111')
if block==2:
   ens_num_list.append('0121')
   ens_num_list.append('0131')
   ens_num_list.append('0141')
   ens_num_list.append('0151')
if block==3:
   ens_num_list.append('0161')
   ens_num_list.append('0171')
   ens_num_list.append('0181')
   ens_num_list.append('0191')
if block==4:
   ens_num_list.append('0201')
   ens_num_list.append('0211')
   ens_num_list.append('0221')
   ens_num_list.append('0231')
if block==5:
   ens_num_list.append('0241')
   ens_num_list.append('0251')
   ens_num_list.append('0261')
   ens_num_list.append('0271')
if block==6:
   ens_num_list.append('0281')
   ens_num_list.append('0291')
   ens_num_list.append('0311')
   ens_num_list.append('0321')

# ens_num_list.append('0301')

for ens_num in ens_num_list:
   add_case(f'v3.LR.historical_{ens_num}',
            src_root=f'{ens_root}/v3.LR.historical_{ens_num}/remap_90x180_aave',
            dst_root=f'{ens_root}/v3.LR.historical_{ens_num}')

#---------------------------------------------------------------------------------------------------
print()
for c in range(len(case_list)):
   #----------------------------------------------------------------------------
   case = case_list[c]
   opts = opts_list[c] 
   yr1 = opts.get('yr1',None)
   yr2 = opts.get('yr2',None)
   src_root = opts.get('src_root',None)
   dst_root = opts.get('dst_root',None)
   #----------------------------------------------------------------------------
   dst_path_hov = f'{dst_root}/SciDAC_qoi_hov.{case}.lat_{lat1}_{lat2}.nc'
   dst_path_clm = f'{dst_root}/SciDAC_qoi_clm.{case}.lat_{lat1}_{lat2}.nc'
   if yr1 is not None:
      dst_path_hov.replace('.nc',f'.yr_{yr1}-{yr2}.nc')
      dst_path_clm.replace('.nc',f'.yr_{yr1}-{yr2}.nc')
   #----------------------------------------------------------------------------
   # out_root = '/global/cfs/cdirs/m4310/whannah/E3SM/2024-SciDAC-L80'
   # out_file = f'{out_root}/QBO.hov.{case}.lat_{lat1}_{lat2}.yr_{yr1}-{yr2}.nc'
   #----------------------------------------------------------------------------
   if case=='ERA5':
      print(f'{verbose_indent}case: {hapy.tclr.CYAN}{case}{hapy.tclr.END}  =>  {out_file}')
      ds = xr.open_dataset(f'{obs_scratch_path}/ERA5/ua_197901_201912.nc')
      area = QBO_methods.calculate_area(ds['lon'].values,ds['lat'].values,
                                        ds['lon_bnds'].values,ds['lat_bnds'].values)
   else:
      file_path = f'{src_root}/*.h0.*'
      file_list = sorted(glob.glob(file_path))
      
      # file_list = file_list[:2]
      # print(f'{hapy.tclr.RED}WARNING - truncating files for debugging - WARNING{hapy.tclr.END}')

      ds = xr.open_mfdataset(file_list)
   #----------------------------------------------------------------------------
   if yr1 is not None: ds = ds.where( ds['time.year']>=yr1, drop=True)
   if yr2 is not None: ds = ds.where( ds['time.year']<=yr2, drop=True)
   #----------------------------------------------------------------------------
   if calc_hov:
      print(f'{verbose_indent}loading data...')
      if case=='ERA5':
         area = xr.DataArray( area, coords=[ds['lat'],ds['lon']] )
         data = ds['ua']
      else:
         area = ds['area']
         data = hapy.vinth2p_simple( ds['U'], ds['hyam'], ds['hybm'], plev_target_hov, ds['PS'],
                                     interp_type='linear', extrapolate=True)
      #-------------------------------------------------------------------------
      # print(); print(ds['U'])
      # print(); print(data)
      # hapy.print_stat(data['lat'])
      # hapy.print_stat(data['lon'])
      # print(); print(data[0,:,0,0].values)
      # exit()
      #-------------------------------------------------------------------------
      print(f'{verbose_indent}calculating hovmoller...')
      data = data.sel(lat=slice(lat1,lat2))
      area = area.sel(lat=slice(lat1,lat2))
      data_hov = (data*area).sum(dim=('lon','lat')) / area.sum(dim=('lon','lat'))
      #-------------------------------------------------------------------------
      print(f'{verbose_indent}writing to file: {dst_path_hov}')
      ds_out_hov = xr.Dataset()
      ds_out_hov['U_hov']        = data_hov
      ds_out_hov['lat1']         = lat1
      ds_out_hov['lat2']         = lat2
      if opts.get('yr1'): ds_out_hov['yr1'] = opts['yr1']
      if opts.get('yr2'): ds_out_hov['yr2'] = opts['yr2']
      ds_out_hov.to_netcdf(path=dst_path_hov,mode='w')
   #----------------------------------------------------------------------------
   if calc_clm:
      print(f'{verbose_indent}calculating climatology...')
      var_list = ['U','T']
      ds_out_clm = xr.Dataset()
      if opts.get('yr1'): ds_out_clm['yr1'] = opts['yr1']
      if opts.get('yr2'): ds_out_clm['yr2'] = opts['yr2']
      for var in var_list:
         if case=='ERA5':
            raise NotImplementedError('climatology support for ERA5 not implemented!')
            # # obs_root,obs_file_prefix = '/global/cfs/cdirs/e3sm/diagnostics/observations/Atm/time-series/ERA5',''
            # obs_root,obs_file_prefix = '/global/cfs/cdirs/m4310/whannah/E3SM/2025-SciDAC-MF-pilot','ERA5_90x180_'
            # if var=='U': ivar='ua'; input_file = f'{obs_root}/{obs_file_prefix}ua_197901_201912.nc'
            # if var=='T': ivar='ta'; input_file = f'{obs_root}/{obs_file_prefix}ta_197901_201912.nc'
         else:
            # input_file = f'{case_root}/post/atm/90x180/clim/{nyr}yr/{case}_ANN_199501_{(1995+nyr-1)}12_clmo.nc'
            data = hapy.vinth2p_simple( ds['U'], ds['hyam'], ds['hybm'], plev_target_clm, ds['PS'],
                                        interp_type='linear', extrapolate=False)
            data = data.mean(dim='time')
            area = ds['area']
         # calculate zonal mean
         data_avg = (data*area).sum(dim=('lon')) / area.sum(dim=('lon'))
         # mask out antarctica
         data_avg = data.where( (data['lat']>-65) | (data['plev']<600) )
         # add to output dataset
         ds_out_clm[var] = data_avg
      #-------------------------------------------------------------------------
      print(f'{verbose_indent}writing to file: {dst_path_clm}')
      ds_out_clm.to_netcdf(path=dst_path_clm,mode='w')
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
