import os, subprocess as sp, numpy as np, xarray as xr, copy, string, dask, glob, pandas as pd, cmocean
import matplotlib, matplotlib.pyplot as plt, matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm; matplotlib.use('Agg')
import hapy; host = hapy.get_host()
#-------------------------------------------------------------------------------
case_name,case,case_dir,case_sub,case_grid,clr,dsh,mrk = [],[],[],[],[],[],[],[]
def add_case(case_in,n=None,p=None,s=None,g=None,d=0,c='black',m=0):
   global name,case,case_dir,case_sub,clr,dsh,mrk
   if n is None:
      tmp_name = ''
   else:
      tmp_name = n
   case.append(case_in); case_name.append(tmp_name)
   case_dir.append(p); case_sub.append(s); case_grid.append(g)
   dsh.append(d) ; clr.append(c) ; mrk.append(m)
#-------------------------------------------------------------------------------
# Create tarball of hov data:
# tar -czvf QBO.hov.2023_L80_ensemble.tar.gz data_temp/QBO.hov.* 
# tar -czvf QBO.hov.2023_AMIP.tar.gz data_temp/QBO.hov.v1.E3SM.2023-SCIDAC-v2-AMIP* 
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
if host=='nersc':
   # add_case('ERA5', n='ERA5')

   tmp_scratch = f'/pscratch/sd/w/whannah/e3sm_scratch/pm-cpu'
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8',           n='E3SM control', p=tmp_scratch,s='run')
   # add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_20', n='E3SM top_km_20', p=tmp_scratch,s='run')
   # add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_25', n='E3SM top_km_25', p=tmp_scratch,s='run')
   add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_30', n='E3SM top_km_30', p=tmp_scratch,s='run')
   # add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_35', n='E3SM top_km_35', p=tmp_scratch,s='run')
   # add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_40', n='E3SM top_km_40', p=tmp_scratch,s='run')
   # add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_45', n='E3SM top_km_45', p=tmp_scratch,s='run')
   # add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_50', n='E3SM top_km_50', p=tmp_scratch,s='run')
   # add_case('E3SM.2026-model-top-test-00.ne30pg2.F20TR.NN_8.top_km_55', n='E3SM top_km_55', p=tmp_scratch,s='run')
   htype = '.eam.h0.'

#-------------------------------------------------------------------------------
# ERA5 levels
# lev = np.array([   1.,    2.,    3.,    5.,    7.,   10.,   20.,   30.,   50.,   70.,
#                  100.,  125.,  150.,  175.,  200.,  225.,  250.,  300.,  350.,  400.,
#                  450.,  500.,  550.,  600.,  650.,  700.,  750.,  775.,  800.,  825.,
#                  850.,  875.,  900.,  925.,  950.,  975., 1000.])
# lev = np.array([   0.1, 0.2, 0.5, 1.,    2.,    3.,    5.,    7.,   10.,   20.,   30.,   50.,   70.,  100.,  125.,  150.,])
plev_target = np.array([   1.,    2.,    3.,    5.,    7.,   10.,   20.,   30.,   50.,   70.,  100.,  125.,  150.,])*1e2
zlev_target = np.arange(10e3,60e3,2e3)
#-------------------------------------------------------------------------------
var, var_str = [],[]
def add_var(var_in,s=None):
   var.append(var_in)
   var_str.append(var_in if s is None else s)
#-------------------------------------------------------------------------------

add_var('U',s='Zonal Wind [m/s]') # first variable is always used for contours
# add_var('OMEGA')
# add_var('T')
# add_var('O3')

recalculate = False

#-------------------------------------------------------------------------------

fig_file = 'figs_QBO/QBO.hov.v1.png'

tmp_file_root = 'data_temp'
tmp_file_head = 'QBO.hov.v1'


lat1,lat2 = -5,5


yr1,yr2=1995,1999

use_height_coord = True

print_stats = True
print_time  = False

var_x_case = False

use_common_label_bar = True

# num_plot_col = len(var)
num_plot_col = 1

# write_to_file = True

#---------------------------------------------------------------------------------------------------
# Set up plot resources
#---------------------------------------------------------------------------------------------------
num_var,num_case = len(var),len(case)
if 'htype' not in locals(): htype = 'h0'
if 'lev'   not in locals(): lev = np.array([0])
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def get_comp(case):
   comp = 'eam'
   if 'PI-CPL.v1' in case: comp = 'cam'
   return comp
#---------------------------------------------------------------------------------------------------
def calculate_obs_area(lon,lat,lon_bnds,lat_bnds):
   re = 6.37122e06  # radius of earth
   nlat,nlon = len(lat),len(lon)
   area = np.empty((nlat,nlon),np.float64)
   for j in range(nlat):
      for i in range(nlon):
         dlon = np.absolute( lon_bnds[j,1] - lon_bnds[j,0] )
         dlat = np.absolute( lat_bnds[j,1] - lat_bnds[j,0] )
         dx = re*dlon*np.pi/180.
         dy = re*dlat*np.pi/180.
         area[j,i] = dx*dy
   return area
#---------------------------------------------------------------------------------------------------
def get_tmp_file_name(case_in,var_in):
   global lat1,lat2,yr1,yr2
   tmp_file = f'{tmp_file_root}/{tmp_file_head}.{case_in}.{var_in}.{yr1}-{yr2}.lat_{lat1}_{lat2}.nc'
   return tmp_file
#---------------------------------------------------------------------------------------------------
def write_file(case_in,var_in,data_avg,Z_avg=None):
   global yr1,yr2
   tmp_file = get_tmp_file_name(case_in,var_in)
   print(''+' '*4+f'writing to file: {tmp_file}')
   ds_out = xr.Dataset( coords=data_avg.coords )
   ds_out[var[v]] = data_avg
   ds_out.to_netcdf(path=tmp_file,mode='w')
   # print('done.')
   return
#---------------------------------------------------------------------------------------------------
def load_file(case_in,var_in):
   global yr1,yr2
   tmp_file = get_tmp_file_name(case_in,var_in)
   print(''+' '*4+f'reading from file: {tmp_file}')
   ds = xr.open_dataset( tmp_file )
   data_avg = ds[var_in]

   # if 'yr1' in globals(): data_avg = data_avg.where( data_avg['time.year']>=yr1, drop=True)
   # if 'yr2' in globals(): data_avg = data_avg.where( data_avg['time.year']<=yr2, drop=True)

   return data_avg
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
#---------------------------------------------------------------------------------------------------

def _finalize_ax(ax, tvec, lvec, vstr, title, panel_label, use_height_coord):
   """Apply axis formatting equivalent to the original NCL resources."""
   ax.set_title(f'({panel_label})  {title}', fontsize=9, loc='left')
   ax.set_title(vstr, fontsize=9, loc='right')

   ax.set_xlabel('Year', fontsize=9)

   if use_height_coord:
      ax.set_ylabel('Height [km]', fontsize=9)
      # ax.set_ylim(bottom=20)
   else:
      ax.set_ylabel('Pressure [mb]', fontsize=9)
      ax.set_ylim(5, 100)
      ax.invert_yaxis()
      tm_vals = [1, 10, 50, 100, 200]
      ax.set_yticks([t for t in tm_vals if 5 <= t <= 100])
      ax.set_yticklabels([str(t) for t in tm_vals if 5 <= t <= 100], fontsize=8)
      ax.set_yscale('log')

   xb_tm_vals = np.arange(1800, 2200, 2)
   valid_xticks = [t for t in xb_tm_vals if tvec.min() <= t <= tvec.max()]
   ax.set_xticks(valid_xticks)
   ax.set_xticklabels([str(int(t)) for t in valid_xticks], fontsize=7, rotation=0, ha='right')
   ax.tick_params(axis='both', which='both', labelsize=8)

#---------------------------------------------------------------------------------------------------

# Storage for all data (collected across var loop, used for plotting)
all_data   = {}   # all_data[v][c]   = 2D numpy array (lev x time)
all_time   = {}   # all_time[v][c]   = 1D float array
all_vlev    = {}   # all_vlev[v][c]    = 1D float array
all_clev   = {}   # contour levels per variable

for v in range(num_var):
   hapy.print_line()
   print('\n'+' '*2+f'var: '+var[v])
   tvar = var[v]
   data_list,time_list,vlev_list = [],[],[]
   
   for c in range(num_case):
      print('\n'+' '*4+f'case: {hapy.tclr.CYAN}{case[c]}{hapy.tclr.END}')

      if recalculate:

         if case[c]=='ERA5':

            lat_name,lon_name = 'lat','lon'
            xy_dims = (lon_name,lat_name)
            obs_root = '/global/cfs/cdirs/e3sm/diagnostics/observations/Atm/time-series/ERA5'
            # if var[v]=='U' : input_file_name = f'{obs_root}/ua_197901_201912.nc'
            # if var[v]=='O3': input_file_name = f'{obs_root}/tro3_197901_201912.nc'
            if var[v]=='U' : obs_var,input_file_name =  'ua', f'{obs_root}/ua_197901_201912.nc'
            if var[v]=='V' : obs_var,input_file_name =  'va', f'{obs_root}/va_197901_201912.nc'
            if var[v]=='T' : obs_var,input_file_name =  'ta', f'{obs_root}/ta_197901_201912.nc'
            if var[v]=='Q' : obs_var,input_file_name = 'hus', f'{obs_root}/hus_197901_201912.nc'
            if var[v]=='O3': obs_var,input_file_name = 'tro3',f'{obs_root}/tro3_197901_201912.nc'

            ds = xr.open_dataset( input_file_name )

            # era5_yr0 = 1979
            # ds = ds.isel(time=slice( (12*(yr1-era5_yr0)),(12*(yr2+1-era5_yr0)), ))

            ds = ds.where( ds['time.year']>=yr1, drop=True)
            ds = ds.where( ds['time.year']<=yr2, drop=True)

            area = hapy.calculate_area_from_latlon(ds['lon'].values,ds['lat'].values,ds['lon_bnds'].values,ds['lat_bnds'].values)
            area = xr.DataArray( area, coords=[ds['lat'],ds['lon']] )  
            data = ds[obs_var]
            data = data.rename({'plev':'lev'})
            data['lev'] = data['lev']#/1e2
            data = data.sel(lev=lev)

            data = mask_data(ds,data)
            area = mask_data(ds,area)
            data_avg = ( (data*area).sum(dim=xy_dims) / area.sum(dim=xy_dims) )

         else:
            #-------------------------------------------------------------------
            comp = 'eam'
            # file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/*.{comp}.{htype}.*'
            # file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/{case[c]}.{comp}.{htype}.*' # used for certain v3HR cases
            file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/*{htype}*.nc' # used for SCREAM
            file_list = sorted(glob.glob(file_path))
            # #-------------------------------------------------------------------
            # for f in file_list: print(f)
            # exit()
            #-------------------------------------------------------------------
            # # subset files that fall within [yr1:yr2]
            # file_list_all = file_list ; file_list = []
            # for f in range(len(file_list_all)):
            #    ds = xr.open_dataset( file_list_all[f] )
            #    yr = int(ds['time.year'].values[0])
            #    # if 'remap' in file_list_all[f]:
            #    #    yr = int(file_list_all[f][-24:-24+4])
            #    # else:
            #    #    yr = int(file_list_all[f][-10:-10+4])
            #    if yr>=yr1 and yr<=yr2: file_list.append(file_list_all[f])
            #    # print(); print(file_list_all[f])
            #    # print(); print(yr)
            #    # exit()
            #-------------------------------------------------------------------
            # subset files that fall within [yr1:yr2]
            file_list_all = file_list ; file_list = []
            for f in range(len(file_list_all)):
               ds = xr.open_dataset( file_list_all[f] )
               fyr1 = int(ds['time.year'].values[0])
               fyr2 = int(ds['time.year'].values[-1])
               if (fyr1>=yr1 and fyr2<=yr2) or (fyr2>=yr1 and fyr2<=yr2):
                  file_list.append(file_list_all[f])
            #-------------------------------------------------------------------
            # for f in file_list: print(f)
            # exit()
            #-------------------------------------------------------------------
            ds = xr.open_mfdataset( file_list )
            ds = ds.where( ds['time.year']>=yr1, drop=True)
            ds = ds.where( ds['time.year']<=yr2, drop=True)
            #-------------------------------------------------------------------
            tvar = var[v]
            # if '2025-v3HR-QBO-01' in case[c]:
            #    if var[v]=='UTGWSPEC':
            #       tvar = 'FUTGWSPEC'
            #-------------------------------------------------------------------
            if use_height_coord:
               print(''+' '*4+'interpolating to height coordinate...')
               target_heights = np.arange(10e3,55e3+250,200)
               z_var = None
               if 'z_mid' in ds: z_var = 'z_mid'
               if 'Z3'    in ds: z_var = 'Z3'
               data_interp_hgt = hapy.interp_to_height( ds[tvar], ds[z_var], zlev_target,
                                                        lev_dim='lev', height_dim='zlev',extrapolate=False)
               data = data_interp_hgt
            else:
               print(''+' '*4+'interpolating to pressure coordinate...')
               ps_var = None
               if 'ps' in ds: ps_var = 'ps'
               if 'PS' in ds: ps_var = 'PS'
               data = hapy.vinth2p_simple( ds[tvar], ds['hyam'], ds['hybm'], plev_target, ds[ps_var],
                                           interp_type='linear', extrapolate=False)
            #-------------------------------------------------------------------
            data = mask_data(ds,data)
            area = mask_data(ds,ds['area'])
            data_avg = ( (data*area).sum(dim='ncol') / area.sum(dim='ncol') )
            #-------------------------------------------------------------------
            if var[v]=='O3': data_avg = data_avg*1e6 # mol/mol => ppmv
            #-------------------------------------------------------------------
            # adjust time to represent the middle of the month instead of the end
            time = data_avg.time.values.copy()
            time_orig = copy.deepcopy(time)
            for i,t in enumerate(time):
               dt = None
               if i==0:
                  dt = pd.Timedelta('15 days')
               else:
                  dt = ( time_orig[i] - time_orig[i-1] ) / 2
               time[i] = time_orig[i] - dt
            data_avg['time'] = time
         #----------------------------------------------------------------------
         write_file(case[c],var[v],data_avg)
      else:
         data_avg = load_file(case[c],var[v])

      #-------------------------------------------------------------------------
      # adjust units
      if var[v] in ['PUTEND','UTGWSPEC','FUTGWSPEC','BUTGWSPEC','UTGWORO']: data_avg = data_avg*86400. # m/s/s => m/s/day
      #-------------------------------------------------------------------------
      # convert to anomaly
      if var[v] in ['O3','T']:
         data_avg = data_avg - data_avg.mean(dim='time')
      #-------------------------------------------------------------------------

      data_list.append( data_avg.transpose().values )
      if use_height_coord:
         vlev_list.append( data_avg['zlev'].values/1e3 )
      else:
         vlev_list.append( data_avg['plev'].values/1e2 )
      time = data_avg['time.year'].values + data_avg['time.month'].values/12.
      time_list.append( time )

   #----------------------------------------------------------------------------
   # print stats after time averaging
   print()
   for c in range(num_case):
      hapy.print_stat(data_list[c],name=f'{case_name[c]:40} {var[v]}',stat='naxsh',indent='    ',compact=True)
   #----------------------------------------------------------------------------

   # Determine contour levels for this variable
   if num_var==1:
      if var[v]=='U' : clev = np.linspace(-50,50,num=21)
   else:
      if var[v]=='U' : dc = 8 ; clev = np.arange(dc*-20,dc*20+dc,dc)
   if var[v]=='T'      : clev = np.arange(-5,5+1,1)
   if var[v]=='Q'      : clev = np.arange(2,40+2,2)*1e-4
   if var[v]=='O3'     : clev = np.linspace(-2,2,21)
   if var[v]=='OMEGA'  : clev = np.linspace(-1,1,21)*0.001
   if var[v] in ['PUTEND','UTGWSPEC','BUTGWSPEC','UTGWORO']: clev = np.linspace(-0.2,0.2,11)

   all_data[v] = data_list
   all_time[v] = time_list
   all_vlev[v] = vlev_list
   all_clev[v] = clev

#---------------------------------------------------------------------------------------------------
# Finalize plot
#---------------------------------------------------------------------------------------------------

if num_var==1:
   num_panels = num_case
else:
   num_panels = num_case * (num_var - 1)

nrows = int(np.ceil(num_panels / float(num_plot_col)))
ncols = num_plot_col

panel_labels = list(string.ascii_lowercase)

# Choose colormap per variable
def get_cmap(vname):
   if vname in ['U','T','O3','OMEGA','PUTEND','UTGWSPEC','BUTGWSPEC','UTGWORO']:
      return cmocean.cm.balance
   if vname == 'Q':
      return cmocean.cm.rain
   return cmocean.cm.balance

# Collect the global min/max across all cases for shared colorbar
v_fill = 0 if num_var==1 else 1
global_clev = all_clev[v_fill]
cmap = get_cmap(var[v_fill])
norm = BoundaryNorm(global_clev, ncolors=cmap.N, clip=False)

fig_height = max(3.5 * nrows, 4)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                         figsize=(10*ncols, fig_height),
                         squeeze=False)
axes_flat = axes.flatten()

ip = 0
for c in range(num_case):

   if num_var == 1:
      ax = axes_flat[ip]
      v_fill = 0
      data2d  = all_data[v_fill][c]
      tvec    = all_time[v_fill][c]
      lvec    = all_vlev[v_fill][c]
      clev    = all_clev[v_fill]
      cmap_v  = get_cmap(var[v_fill])
      norm_v  = BoundaryNorm(clev, ncolors=cmap_v.N, clip=False)

      cf = ax.contourf(tvec, lvec, data2d, levels=clev, cmap=cmap_v, norm=norm_v, extend='both')

      # overlay zero contour
      ax.contour(tvec, lvec, data2d, levels=[0], colors='k', linewidths=0.8, linestyles='--')

      _finalize_ax(ax, tvec, lvec, var_str[v_fill], case_name[c], panel_labels[ip],
                   use_height_coord)

      if not use_common_label_bar:
         fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, fraction=0.03)

      ip += 1

   else:
      # multi-var: overlay contour (var[0]) on fill (var[1..])
      for vv in range(1, num_var):
         ax = axes_flat[ip]
         v_fill = vv
         v_cont = 0
         data_fill = all_data[v_fill][c]
         data_cont = all_data[v_cont][c]
         tvec      = all_time[v_fill][c]
         lvec      = all_vlev[v_fill][c]
         tvec_c    = all_time[v_cont][c]
         lvec_c    = all_vlev[v_cont][c]
         clev_fill = all_clev[v_fill]
         clev_cont = all_clev[v_cont]
         cmap_v    = get_cmap(var[v_fill])
         norm_v    = BoundaryNorm(clev_fill, ncolors=cmap_v.N, clip=False)

         cf = ax.contourf(tvec, lvec, data_fill, levels=clev_fill, cmap=cmap_v, norm=norm_v, extend='both')

         # overlay contour lines for var[0], dashed negative
         neg_levels = [l for l in clev_cont if l < 0]
         pos_levels = [l for l in clev_cont if l > 0]
         if neg_levels:
            ax.contour(tvec_c, lvec_c, data_cont, levels=neg_levels,
                       colors='k', linewidths=1.5, linestyles='--')
         if pos_levels:
            ax.contour(tvec_c, lvec_c, data_cont, levels=pos_levels,
                       colors='k', linewidths=1.5, linestyles='-')

         _finalize_ax(ax, tvec, lvec, var_str[v_fill], case_name[c], panel_labels[ip],
                      use_height_coord)

         if not use_common_label_bar:
            fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, fraction=0.03)

         ip += 1

# Hide any unused axes
for idx in range(ip, len(axes_flat)):
   axes_flat[idx].set_visible(False)

# Shared colorbar
if use_common_label_bar:
   # Use the last contourf for the colorbar
   cbar = fig.colorbar(cf, ax=axes_flat[:ip], orientation='vertical',
                       fraction=0.02, pad=0.03, shrink=0.8)
   cbar.ax.tick_params(labelsize=8)

plt.savefig(fig_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n{fig_file}\n')

# hapy.trim_png(fig_file)
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------