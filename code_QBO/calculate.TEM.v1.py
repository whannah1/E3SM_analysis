import os, copy, glob, xarray as xr, numpy as np, numba
class tclr:END,RED,GREEN,MAGENTA,CYAN = '\033[0m','\033[31m','\033[32m','\033[35m','\033[36m'
import hapy; host = hapy.get_host()
#---------------------------------------------------------------------------------------------------
case,case_dir,case_sub = [],[],[]
def add_case(case_in,n=None,p=None,s=None,g=None,d=0,c='black',m=0):
   global case,case_dir,case_sub
   case.append(case_in); case_dir.append(p); case_sub.append(s)
#---------------------------------------------------------------------------------------------------

overwrite = True

#-------------------------------------------------------------------------------
if host=='nersc':
   pressure_level_file = '../publication_scripts/2025_qbo/grid_files/vrt_prs_ERA5.nc'

   tmp_scratch = f'/pscratch/sd/w/whannah/e3sm_scratch/pm-cpu'
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8',           n='E3SM control',   p=tmp_scratch,s='run')
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_55', n='E3SM top_km_55', p=tmp_scratch,s='run')
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_50', n='E3SM top_km_50', p=tmp_scratch,s='run')
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_45', n='E3SM top_km_45', p=tmp_scratch,s='run')
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_40', n='E3SM top_km_40', p=tmp_scratch,s='run')
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_35', n='E3SM top_km_35', p=tmp_scratch,s='run')
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_30', n='E3SM top_km_30', p=tmp_scratch,s='run')
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_25', n='E3SM top_km_25', p=tmp_scratch,s='run')
   # add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_20', n='E3SM top_km_20', p=tmp_scratch,s='run')
   htype = '.eam.h0.'

#---------------------------------------------------------------------------------------------------
def hybrid_to_pressure(ds,var_names,pressure_level_file):
   '''
   input: 
      ds: dataset, read from model output 
      var_names: variable names 
      pressure_level_file: pressure levels file directory (in pa) 
   output: 
      ds_out: dataset that interpolated to pressure levels (in pa) 
   '''
   plev = xr.open_dataset(pressure_level_file).plev
   pressure = ds['hyam']*ds['P0']+ds['hybm']*ds['PSzm']
   pressure = pressure.transpose("time","lev","zalat","zalon")
   log_p = np.log(pressure)
   log_plev = np.log(plev)
   
   ds_out = xr.Dataset()
   for var_name in var_names:
      var = ds[var_name]
      s = var.values.shape
      var_new = np.zeros([s[0],len(plev),s[2],s[3]])
      for i in range(s[0]):
         for j in range(s[2]):
            for k in range(s[3]):
               var_new[i,:,j,k] = np.interp(log_plev, log_p[i,:,j,k], var[i,:,j,k])
      var_new = xr.DataArray(var_new, dims=('time', 'plev', 'zalat', 'zalon')
                  , coords={'time':var.coords['time'],
                         'plev':plev.coords['plev'],
                        'zalat':var.coords['zalat'],
                        'zalon':var.coords['zalon']})
      var_new.attrs = var.attrs
      ds_out[var_name] = var_new
   return ds_out 
#---------------------------------------------------------------------------------------------------
def calculate_tem(ds):
   '''
   input: 
      ds: xarray dataset from E3SM model output 
         required variables: 'Uzm','Vzm','Wzm','THzm','VTHzm','WTHzm','UVzm','UWzm','PSzm','P0','hyam','hybm'
   output:         
      ds_out: TEM terms 
   '''
   # constants
   H     = 7e3          # m         assumed mean scale heightof the atmosphere
   P0    = 101325       # Pa        surface pressure
   Rd    = 287.058      # J/kg/K    gas constant for dry air
   cp    = 1004.64      # J/kg/K    specific heat for dry air
   g     = 9.80665      # m/s       global average of gravity at MSLP
   a     = 6.37123e6    # m         Earth's radius
   omega = 7.29212e-5   # 1/s       Earth's rotation rate
   pi    = 3.14159
   
   #----------------------------------------------------------------------------
   # basic zonal means and anomalies
   ds = hybrid_to_pressure(ds,['Uzm','Vzm','Wzm','THzm','VTHzm','WTHzm','UVzm','UWzm'],pressure_level_file)
   ds = ds.rename({'zalat':'lat'})
   ds['plev'] = ds['plev'].values
   nlat = len(ds[ 'lat'].values)
   nlev = len(ds['plev'].values)

   dlat = ds['lat']
   rlat = np.deg2rad(dlat)
   rlat['lat'] = rlat
   cos_lat = np.cos(rlat)
   #----------------------------------------------------------------------------
   # basic zonal means and anomalies

   TH_b = ds['THzm'].mean(dim='zalon')                     # K
   U_b  = ds['Uzm'].mean(dim='zalon')                      # m/s 
   V_b  = ds['Vzm'].mean(dim='zalon')                      # m/s
   W_b  = ds['Wzm'].mean(dim='zalon')                      # pa/s, or kg/(m·s3)

   UV   = ds['UVzm'].mean(dim='zalon')                     # m2/ s2 
   UW   = ds['UWzm'].mean(dim='zalon')                     # m pa/ s2, or kg/(s4)
   VTH  = ds['VTHzm'].mean(dim='zalon')                    # K m/s
   WTH  = ds['WTHzm'].mean(dim='zalon')                    # K pa/s, or K kg/(m·s3)


   # make sure coordinate data is lat in radians for da.differentiate()
   TH_b['lat'] = rlat
   U_b ['lat'] = rlat
   V_b ['lat'] = rlat
   W_b ['lat'] = rlat

   UV ['lat'] = rlat
   UW ['lat'] = rlat
   VTH['lat'] = rlat
   WTH['lat'] = rlat
   
   #----------------------------------------------------------------------------
   # EP flux vectors

   dTHdp = TH_b.differentiate('plev')                                   # Kms2/kg
   dUdp  =  U_b.differentiate('plev')                                   # m2s/kg
   dUdy  = (U_b*cos_lat).differentiate('lat') / (a*cos_lat)             # s-1

   # eddy stream function
   gamma = VTH / dTHdp                           # kg/s3

   fcor = 2*omega*np.sin(rlat)                                          # s-1

   ### original version based on Gerber and Manzini
   F_y = a*cos_lat * (       dUdp *gamma - UV )  # m3/s2
   F_z = a*cos_lat * ( (fcor-dUdy)*gamma - UW )  # kgm/s4

   F_y = F_y.transpose('time','plev','lat')    
   F_z = F_z.transpose('time','plev','lat')

   # # replace inf values with nan
   # F_z = F_z.where(np.isinf(F_z),np.nan,F_z)

   
   #----------------------------------------------------------------------------
   def replace_inf(Y): return xr.DataArray( replace_inf_numba(Y.values), coords=Y.coords )
   @numba.njit()
   def replace_inf_numba(X):
      [ntime,nlev,nlat] = X.shape
      for t in range(ntime):
         for k in range(nlev):
            for i in range(nlat):
               if np.isinf(X[t,k,i]): X[t,k,i] = np.nan
      return X
   # #----------------------------------------------------------------------------
   
   F_z   = replace_inf(F_z)
   gamma = replace_inf(gamma)

   #----------------------------------------------------------------------------
   # EP flux divergence
   
   dFydy = (F_y*cos_lat).differentiate('lat') / (a*cos_lat)             # m2/s2
   dFzdp = F_z.fillna(np.nan).differentiate('plev')                                    # m2/s2
   
   EP_div = ( dFydy + dFzdp ) / (a*cos_lat)                             # m/s2  already / (rho0 ae cos(rlat))
   
   #----------------------------------------------------------------------------
   # # EP flux divergence transformed to log-pressure
   
   # F_y_lp = F_y * ds['plev']/P0                                         # m3/s2
   # F_z_lp = F_z * -H/P0                                                 # m3/s2

   # dFydy_lp = (F_y_lp*cos_lat).differentiate('lat') / (a*cos_lat)       # m2/s2
   # dFzdp_lp = F_z_lp.differentiate('plev')                              # ?m4/kg

   # EP_div_lp = dFydy_lp + dFzdp_lp

   #----------------------------------------------------------------------------
   # TEM meridional and vertical  velocities

   dgamma_dy = (gamma*cos_lat).differentiate('lat') / (a*cos_lat)       # kg/(ms3)
   dgamma_dp = gamma.differentiate('plev')                              # m/s

   V_star = V_b - dgamma_dp                                             # m/s
   W_star = W_b + dgamma_dy                                             # kg/(ms3), or Pa/s

   #-------------------------------------------------------------------------
   # TEM mass stream function

   dp_int = xr.full_like(ds['plev'],-1)
   for k in range(1,nlev-1):
      pint1 = ( ds['plev'][k-1] + ds['plev'][k-0] ) / 2.
      pint2 = ( ds['plev'][k-0] + ds['plev'][k+1] ) / 2.
      dp_int[k] =  pint1 - pint2

   gamma_mass = xr.full_like(U_b,np.nan)
   for k in range(1,nlev-1):
      tmp_integral = xr.full_like(U_b.mean(dim='plev'),0)
      for kk in range(1,k):
         tmp_integral[:,:] = tmp_integral[:,:] + ( V_b[:,kk,:]*dp_int[kk] - gamma[:,kk,] )
      gamma_mass[:,k,:] = ( (2*pi*a*cos_lat[:]/g) * tmp_integral[:,:] ).transpose('time','lat')

   #----------------------------------------------------------------------------
   # TEM northward and upward advection
   
   dUdt_y = V_star * ( fcor - dUdy )                          # m/s2
   dUdt_z = -1 * W_star * dUdp                                # m/s2   

   #----------------------------------------------------------------------------
   # Create output datset and add variables      
   ds_out = xr.Dataset()
   ds_out['vtem']         = V_star
   ds_out['wtem']         = W_star
   ds_out['psitem']       = gamma_mass 
   ds_out['epfy']         = F_y
   ds_out['epfz']         = F_z
   ds_out['depfydy']      = dFydy 
   ds_out['depfzdz']      = dFzdp 
   # ds_out['depfydy']      = dFydy_lp 
   # ds_out['depfzdz']      = dFzdp_lp 
   ds_out['utendepfd']    = EP_div
   # ds_out['utendepfd_lp'] = EP_div_lp
   ds_out['utendvtem']    = dUdt_y
   ds_out['utendwtem']    = dUdt_z
   ds_out['u']            = U_b
   #----------------------------------------------------------------------------
   # add variable long names
   ds_out['vtem']        .attrs['long_name'] = 'Transformed Eulerian mean northward wind'
   ds_out['wtem']        .attrs['long_name'] = 'Transformed Eulerian mean upward wind'
   ds_out['psitem']      .attrs['long_name'] = 'Transformed Eulerian mean mass stream function'
   ds_out['epfy']        .attrs['long_name'] = 'Northward component of the Eliassen-Palm flux'
   ds_out['epfz']        .attrs['long_name'] = 'Upward component of the Eliassen-Palm flux'
   ds_out['depfydy']     .attrs['long_name'] = 'Meridional derivative of northward component of the Eliassen-Palm flux'
   ds_out['depfzdz']     .attrs['long_name'] = 'Vertical derivative of upward component of the Eliassen-Palm flux'
   ds_out['utendepfd']   .attrs['long_name'] = 'Tendency of eastward wind due to TEM Eliassen-Palm flux divergence'
   # ds_out['utendepfd_lp'].attrs['long_name'] = 'Tendency of eastward wind due to TEM Eliassen-Palm flux divergence (log-pressure)'
   ds_out['utendvtem']   .attrs['long_name'] = 'Tendency of eastward wind due to TEM northward wind advection and coriolis'
   ds_out['utendwtem']   .attrs['long_name'] = 'Tendency of eastward wind due to TEM upward wind advection'

   #----------------------------------------------------------------------------
   # add variable units
   ds_out['vtem']        .attrs['units'] = 'm/s'
   ds_out['wtem']        .attrs['units'] = 'pa/s'
   ds_out['psitem']      .attrs['units'] = 'kg/s'
   ds_out['epfy']        .attrs['units'] = 'm3/s2'
   ds_out['epfz']        .attrs['units'] = 'm3/s2'
   ds_out['depfydy']     .attrs['units'] = 'm/s2'
   ds_out['depfzdz']     .attrs['units'] = 'm/s2'
   ds_out['utendepfd']   .attrs['units'] = 'm/s2'
   # ds_out['utendepfd_lp'].attrs['units'] = 'm/s2'
   ds_out['utendvtem']   .attrs['units'] = 'm/s2'
   ds_out['utendwtem']   .attrs['units'] = 'm/s2'
   ds_out['u']           .attrs['units'] = 'm/s'

   ds_out['lat'] = dlat # use latitude in degrees for output
   # ds_out = ds_out.rename({'plev':'lev'})
   # ds_out['lev'] = ds_out['lev'].values/100

   ds_out['plev'] = ds_out['plev'].values/100
   
   return ds_out
      
#---------------------------------------------------------------------------------------------------
num_case = len(case)
for c in range(num_case):
   print(f'    case: {tclr.CYAN}{case[c]}{tclr.END}')

   dir_src = f'{case_dir[c]}/{case[c]}/{case_sub[c]}'
   dir_dst = f'{case_dir[c]}/{case[c]}/tem'

   if not os.path.exists(dir_dst): os.mkdir(dir_dst)
   file_path = f'{dir_src}/*{htype}*'
   file_list = sorted(glob.glob(file_path)) 
   #file_list.remove(file_list[-1]) # last file is empty

   if 'num_files' in locals(): file_list = file_list[:num_files]
   #----------------------------------------------------------------------------
   if file_list==[]: print('ERROR: Empty file list:'); print(); print(file_path); exit()
   #----------------------------------------------------------------------------
   # loop through files to calculate TEM terms 
   for f in file_list: 
      f_out = f.replace(f'.{htype}.',f'.{htype}-tem.').replace(dir_src,dir_dst)
      
      if not overwrite:
         if os.path.exists(f_out):
            continue

      ds = xr.open_dataset(f)[['Uzm','Vzm','Wzm','THzm','VTHzm','WTHzm','UVzm','UWzm','PSzm','P0','hyam','hybm']]
      ds_out = calculate_tem(ds)
      
      # exit('\nSTOPPING\n')

      print(' '*4+f'writing to file: {f_out}')
      ds_out.to_netcdf(path=f_out,mode='w')

#---------------------------------------------------------------------------------------------------
   
