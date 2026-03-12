import os, subprocess as sp, numpy as np, xarray as xr, copy, string, dask, glob, cmocean
import pywt
# from statsmodels.tsa.arima.model import ARIMA
import QBO_diagnostic_methods as QBO_methods
import hapy; host = hapy.get_host()
#-------------------------------------------------------------------------------
# based on E3SM diagnostics package:
# https://github.com/E3SM-Project/e3sm_diags/blob/main/e3sm_diags/driver/qbo_driver.py
#-------------------------------------------------------------------------------
case_list,case_root,case_sub = [],[],[]
#-------------------------------------------------------------------------------
def add_case(case_in,root=None,s=None,**kwargs):
   global case_list
   case_list.append(case_in)
   case_root.append(root)
   case_sub.append(s)
#-------------------------------------------------------------------------------
if host=='nersc':
   yr1,yr2=1995,1999
   grp_msz,ens_msz = 200,10
   # add_case('ERA5', n='ERA5',clr='black',mrk='.',msz=grp_msz,yr1=yr1,yr2=yr2,gidx=0)
   tmp_scratch = f'/pscratch/sd/w/whannah/e3sm_scratch/pm-cpu'
   # add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8',           n='E3SM control',   clr='black', mrk='.',msz=grp_msz,yr1=yr1,yr2=yr2,gidx=1,root=tmp_scratch,s='run')
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_55', n='E3SM top_km_55', clr='red',   mrk='.',msz=grp_msz,yr1=yr1,yr2=yr2,gidx=1,root=tmp_scratch,s='run')
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_50', n='E3SM top_km_50', clr='orange',mrk='.',msz=grp_msz,yr1=yr1,yr2=yr2,gidx=1,root=tmp_scratch,s='run')
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_45', n='E3SM top_km_45', clr='green', mrk='.',msz=grp_msz,yr1=yr1,yr2=yr2,gidx=1,root=tmp_scratch,s='run')
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_40', n='E3SM top_km_40', clr='cyan',  mrk='.',msz=grp_msz,yr1=yr1,yr2=yr2,gidx=1,root=tmp_scratch,s='run')
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_35', n='E3SM top_km_35', clr='blue',  mrk='.',msz=grp_msz,yr1=yr1,yr2=yr2,gidx=1,root=tmp_scratch,s='run')
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_30', n='E3SM top_km_30', clr='purple',mrk='.',msz=grp_msz,yr1=yr1,yr2=yr2,gidx=1,root=tmp_scratch,s='run')
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_25', n='E3SM top_km_25', clr='pink',  mrk='.',msz=grp_msz,yr1=yr1,yr2=yr2,gidx=1,root=tmp_scratch,s='run')

#-------------------------------------------------------------------------------
if host=='lcrc':
   # yr1,yr2=1985,2014
   yr1,yr2=2021,2050
   ens_root = '/lcrc/group/e3sm2/ac.wlin/E3SMv3'
   # add_case('v3.LR.historical_0051',root=ens_root)
   add_case('v3.LR.historical_0091',root=ens_root)
   add_case('v3.LR.historical_0101',root=ens_root)
   add_case('v3.LR.historical_0111',root=ens_root)
   add_case('v3.LR.historical_0121',root=ens_root)
   add_case('v3.LR.historical_0131',root=ens_root)
   add_case('v3.LR.historical_0141',root=ens_root)
   add_case('v3.LR.historical_0151',root=ens_root)
   add_case('v3.LR.historical_0161',root=ens_root)
   add_case('v3.LR.historical_0171',root=ens_root)
   add_case('v3.LR.historical_0181',root=ens_root)
   add_case('v3.LR.historical_0191',root=ens_root)
   add_case('v3.LR.historical_0201',root=ens_root)
   add_case('v3.LR.historical_0211',root=ens_root)
   add_case('v3.LR.historical_0221',root=ens_root)
   add_case('v3.LR.historical_0231',root=ens_root)
   add_case('v3.LR.historical_0241',root=ens_root)
   add_case('v3.LR.historical_0251',root=ens_root)
   add_case('v3.LR.historical_0261',root=ens_root)
   add_case('v3.LR.historical_0271',root=ens_root)
   add_case('v3.LR.historical_0281',root=ens_root)
   add_case('v3.LR.historical_0291',root=ens_root)
   add_case('v3.LR.historical_0301',root=ens_root)
   add_case('v3.LR.historical_0311',root=ens_root)
   add_case('v3.LR.historical_0321',root=ens_root)

#-------------------------------------------------------------------------------

var = ['U']

plev_target = np.array([   1.,    2.,    3.,    5.,    7.,   10.,   20.,   30.,   50.,   70.,  100. ])
num_lev = len(plev_target)

use_native_data = True

# tmp_file_head = 'data_temp/QBO.wavelet_profile.v1'
# tmp_file_head = 'data_temp/QBO.wavelet_profile.v1.alt_h0' # based on native grid data (integer periods)
tmp_file_head = 'data_temp/QBO.wavelet_profile.v1.alt_period' # use finer period axis (0.1 month period interval)

lat1,lat2 = -5,5

period_min = 12*1
period_max = 12*5

# period_list = np.arange(period_min, period_max + 1) ; num_period = len(period_list)
period_list = np.arange(period_min, period_max + 1, 0.1) ; num_period = len(period_list)


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def get_psd_from_wavelet_pywt(data):
   global period_list
   deg = 6
   widths = deg / ( 2 * np.pi / period_list )
   [cfs, freq] = pywt.cwt(data, scales=widths, wavelet='cmor1.5-1.0')
   psd = np.mean( np.square( np.abs(cfs) ), axis=1)
   period = 1 / freq
   return (period, psd)
#---------------------------------------------------------------------------------------------------
# def get_alpha_sigma2(data_in):
#    ## Fit AR1 model to estimate the autocorrelation (alpha) and variance (sigma2)
#    mod = ARIMA( data_in, order=(1,0,0) )
#    res = mod.fit()
#    return (res.params[1],res.params[2])
#---------------------------------------------------------------------------------------------------
def get_tmp_file(case,var,yr1,yr2):
   return f'{tmp_file_head}.{case}.{var}.{yr1}-{yr2}.nc'
#---------------------------------------------------------------------------------------------------
def mask_data(ds,data,lat_name='lat',lon_name='lon'):
   global lat1,lat2
   
   if 'ncol' in data.dims:
      tmp_data = np.ones([len(ds['ncol'])],dtype=bool)
      tmp_dims,tmp_coords = ('ncol'),{'ncol':data['ncol']}
   else:
      tmp_data = np.ones([len(ds[lat_name]),len(ds[lon_name])],dtype=bool)
      tmp_dims,tmp_coords = (lat_name,lon_name),{lat_name:ds[lat_name],lon_name:ds[lon_name]}
   mask = xr.DataArray( tmp_data, coords=tmp_coords, dims=tmp_dims )
   mask = mask & (ds[lat_name]>= lat1) & (ds[lat_name]<= lat2)
   data = data.where( mask.compute(), drop=True)
   return data
#---------------------------------------------------------------------------------------------------
num_var,num_case = len(var),len(case_list)
for v in range(num_var):
   hapy.print_line()
   print(f'  var: {hapy.tclr.MAGENTA}{var[v]}{hapy.tclr.END}')
   #----------------------------------------------------------------------------
   for c in range(num_case):
      tmp_file = get_tmp_file(case_list[c],var[v],yr1,yr2)
      print(f'    case: {hapy.tclr.GREEN}{case_list[c]}{hapy.tclr.END}  =>  {tmp_file}')

      xy_dims = ('lon','lat')

      if case_list[c]=='ERA5':
         #-------------------------------------------------------------------
         # obs_root = '/global/cfs/cdirs/e3sm/diagnostics/observations/Atm'
         obs_root = '/lcrc/group/e3sm/diagnostics/observations/Atm'
         input_file_name = f'{obs_root}/time-series/ERA5/ua_197901_201912.nc'
         #-------------------------------------------------------------------
         # open dataset for temporal subset within [yr1:yr2]
         ds = xr.open_dataset( input_file_name )
         ds = ds.isel(time=slice( (12*(yr1-1979)),(12*(yr2+1-1979)), ))
         #-------------------------------------------------------------------
         # load data
         area = QBO_methods.calculate_area(ds['lon'].values,ds['lat'].values,ds['lon_bnds'].values,ds['lat_bnds'].values)
         area = xr.DataArray( area, coords=[ds['lat'],ds['lon']] )  
         data = ds['ua']
         data = data.rename({'plev':'lev'})
         data['lev'] = data['lev']/1e2
         data = data.sel(lev=plev_target)
         #-------------------------------------------------------------------
         # apply mask and perform spatial average
         data = mask_data(ds,data)
         area = mask_data(ds,area)
         data_avg = ( (data*area).sum(dim=xy_dims) / area.sum(dim=xy_dims) )
      else:
         #-------------------------------------------------------------------
         # normally we should use the postprocessed/remapped U file, 
         # but in a pinch we can still use the native grid h0 files
         if use_native_data:
            # data_sub = 'archive/atm/hist'
            file_path = f'{case_root[c]}/{case_list[c]}/{case_sub[c]}/*.eam.h0.*'
            file_list = sorted(glob.glob(file_path))
         else:
            # data_sub = 'post/atm/180x360_aave/ts/monthly/30yr'
            file_path = f'{case_root[c]}/{case_list[c]}/{case_sub[c]}/U_{yr1}01_{yr2}12.nc'
            file_list = sorted(glob.glob(file_path))
         #-------------------------------------------------------------------
         if file_list==[]: raise ValueError(f'ERROR: file_list is empty - file_path: {file_path}')
         #-------------------------------------------------------------------
         ds = xr.open_mfdataset( file_list )
         #-------------------------------------------------------------------
         ds = ds.where( ds['time.year']>=yr1, drop=True)
         ds = ds.where( ds['time.year']<=yr2, drop=True)
         #-------------------------------------------------------------------
         # # load data
         # data = he.interpolate_to_pressure(ds,data_mlev=ds[var[v]],
         #                                   lev=plev_target,ds_ps=ds,ps_var='PS',
         #                                   interp_type=2,extrap_flag=True)#.isel(lev=0)
         #-------------------------------------------------------------------
         # load data
         ps_var = None
         if 'ps' in ds: ps_var = 'ps'
         if 'PS' in ds: ps_var = 'PS'
         data = hapy.vinth2p_simple( ds[var[v]], ds['hyam'], ds['hybm'], plev_target, ds[ps_var],
                                     interp_type='log', extrapolate=True)
         data = data.rename({'plev':'lev'})
         #-------------------------------------------------------------------
         # apply mask and perform spatial average
         data = mask_data(ds,data)
         area = mask_data(ds,ds['area'])
         if 'ncol' in data.dims:
            data_avg = ( (data*area).sum(dim='ncol') / area.sum(dim='ncol') )
         else:
            data_avg = ( (data*area).sum(dim=xy_dims) / area.sum(dim=xy_dims) )
      #----------------------------------------------------------------------
      # convert to anomalies
      data_avg = data_avg - data_avg.mean(dim='time')
      # detrend in time
      fit = xr.polyval(data_avg['time'], data_avg.polyfit(dim='time', deg=1).polyfit_coefficients)
      data_avg = data_avg - fit
      #-------------------------------------------------------------------------
      # calculate wavelet spectra
      wavelet_spec = np.full([num_lev,num_period],np.nan)
      for k in range(num_lev):
         ( period, wavelet_spec[k,:] ) = get_psd_from_wavelet_pywt(data_avg.isel(lev=k).values)
      #----------------------------------------------------------------------
      period_coord = np.array(period_list)
      # ds_out = xr.Dataset( coords=data_avg.coords )
      ds_out = xr.Dataset()
      ds_out['period']       = xr.DataArray(period_coord, coords={'period':period_coord})
      ds_out['wavelet_spec'] = xr.DataArray(wavelet_spec,coords={'lev':plev_target,'period':period_coord,})
      ds_out.to_netcdf(path=tmp_file,mode='w')
      ds_out.close()
      #----------------------------------------------------------------------
      ds.close()
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
