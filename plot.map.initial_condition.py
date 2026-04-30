import os, subprocess as sp, numpy as np, xarray as xr, dask, copy, string, cmocean, glob
import uxarray as ux, cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import hapy
host = hapy.get_host()
#-------------------------------------------------------------------------------
file_list = []
opt_list = []
def add_case( init_file, **kwargs ):
   file_list.append(init_file)
   tmp_opts = {}
   for k, val in kwargs.items(): tmp_opts[k] = val
   opt_list.append(tmp_opts)
#-------------------------------------------------------------------------------
var_list,lev_list,var_str = [],[],[]
def add_var(var_name,lev=0,s=None): 
   var_list.append(var_name); lev_list.append(lev); 
   var_str.append(var_name if s is None else s)
#-------------------------------------------------------------------------------
# fig_file = os.getenv('HOME')+'/E3SM_analysis/figs_clim/clim.map.v1.png'
fig_file = 'map.initial_condition.png'
#-------------------------------------------------------------------------------
if host=='nersc':

   ### IC files for vertical grid test
   tmp_root = '/global/cfs/cdirs/e3sm/whannah/files_init'
   tmp_scrip_file = '/global/cfs/cdirs/e3sm/mapping/grids/ne256np4_scrip_c20190127.nc'
   add_case(f'{tmp_root}/screami_ne256np4L128_ifs-20200120_20220914.L128v3.1_alpha_1.0_pm_100.nc', n='alpha_1.0_pm_100', scrip_file=tmp_scrip_file)
   # add_case(f'{tmp_root}/screami_ne256np4L128_ifs-20200120_20220914.L128v3.1_alpha_1.5_pm_100.nc', n='alpha_1.0_pm_100', scrip_file=tmp_scrip_file)
   add_case(f'{tmp_root}/screami_ne256np4L128_ifs-20200120_20220914.L128v3.1_alpha_2.0_pm_100.nc', n='alpha_1.0_pm_100', scrip_file=tmp_scrip_file)
   # add_case(f'{tmp_root}/screami_ne256np4L128_ifs-20200120_20220914.L128v3.1_alpha_2.5_pm_100.nc', n='alpha_1.0_pm_100', scrip_file=tmp_scrip_file)
   # add_case(f'{tmp_root}/screami_ne256np4L128_ifs-20200120_20220914.L128v3.1_alpha_3.0_pm_100.nc', n='alpha_1.0_pm_100', scrip_file=tmp_scrip_file)

#-------------------------------------------------------------------------------
# if host=='olcf':

#-------------------------------------------------------------------------------
# if host=='nersc':
   
#-------------------------------------------------------------------------------
# if host=='lcrc':

#-------------------------------------------------------------------------------
# if host=='llnl':

#-------------------------------------------------------------------------------

add_var('T_mid',lev=-80)

#-------------------------------------------------------------------------------
# lat1,lat2 = -30,30
# lat1,lat2,lon1,lon2 = -10,70,180,340             # N. America
# lat1,lat2,lon1,lon2 = 0,40,360-90,360-10         # Atlantic/carribean
# lat1,lat2,lon1,lon2 = 25, 50, 360-125, 360-75    # CONUS
# lat1,lat2,lon1,lon2 = -40,40,90,240              # MC + West Pac
# lat1,lat2,lon1,lon2 = 0,60,50,120                # India
# lat1,lat2,lon1,lon2 = -40,30,360-120,360-40      # S. America
# lat1,lat2,lon1,lon2 = -20,15,360-90,360-60       # S. America - zoom in on Andes
#-------------------------------------------------------------------------------

# plot_diff,add_diff = False,False
plot_diff,add_diff = True,False

print_stats          = True
var_x_case           = False

# num_plot_col         = len(case)
num_plot_col         = 1#len(var)

use_common_label_bar = False

if 'use_snapshot' not in locals(): use_snapshot,ss_t = False,-1

#---------------------------------------------------------------------------------------------------
# Set up plot resources
num_var,num_file = len(var_list),len(file_list)

if 'subtitle_font_height' not in locals(): subtitle_font_height = 0.01

#---------------------------------------------------------------------------------------------------
# set up figure objects
subplot_kwargs = {}
# subplot_kwargs['projection'] = ccrs.Robinson(central_longitude=180)
subplot_kwargs['projection'] = ccrs.PlateCarree(central_longitude=180)
# lat_min, lat_max = -90, -30
# subplot_kwargs['projection'] = ccrs.Orthographic(central_latitude=-90)
(d1,d2) = (num_var,num_file) if var_x_case else (num_file,num_var)
# dx=10;figsize = (dx*num_var,dx*num_file) if var_x_case else (dx*num_file,dx*num_var)

fdx,fdy=5,10;figsize = (fdx*num_file,fdy*num_var) if var_x_case else (fdx*num_var,fdy*num_file)
title_fontsize,lable_fontsize = 20,18

# figsize = (30,30); title_fontsize,lable_fontsize = 25,15

fig, axs = plt.subplots(d1,d2, subplot_kw=subplot_kwargs, figsize=figsize, squeeze=False )

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
   hapy.print_line()
   print(' '*2+'var: '+hapy.tclr.MAGENTA+var_list[v]+hapy.tclr.END)
   data_list = []
   glb_avg_list = []
   grid_ds_list = []
   for f in range(num_file):
      print(' '*4+'file: '+hapy.tclr.GREEN+file_list[f]+hapy.tclr.END)
      #-------------------------------------------------------------------------
      # ds = ux.open_mfdataset(opt_list['scrip_file'], file_list, data_vars='minimal')
      grid_ds = xr.open_dataset(opt_list[f]['scrip_file'])
      ds = xr.open_mfdataset(file_list[f], data_vars='minimal')
      ds = ds.isel(time=0)
      # ds = ds.isel(valid_time=0)

      data = ds[var_list[v]]

      # mask = ((ds['lat'] >= lat_min) & (ds['lat'] <= lat_max))
      # mask.load()
      # data = data.where(mask, drop=True)
      # data.load()

      # hapy.print_stat(data,name=var_list[v],stat='naxsh',indent='    ',compact=True)

      
      # grid_ds = grid_ds.where(mask.rename({'ncol':'grid_size'}), drop=True)
      grid_ds.load()
      grid_ds_list.append(grid_ds)

      #-------------------------------------------------------------------------
      # # adjust units
      # if var_list[v]=='FRONTGF': data = data*86400e6 # K^2/M^2/S > K^2/KM^2/day
      #-------------------------------------------------------------------------
      if 'lev' in data.dims:
         if lev_list[v] is not None:
            # data = data.isel(lev=0)
            # if requested lev<0 use model levels without interpolation
            if lev_list[v]<0:
               approx_plev = data.lev.isel(lev=np.absolute(lev_list[v])).values
               print(' '*4+f'selecting model level ~ approx_plev: {approx_plev}') 
               data = data.isel(lev=np.absolute(lev_list[v]),drop=True)
      #-------------------------------------------------------------------------
      data.load()
      data_list.append( data )
      #-------------------------------------------------------------------------
      # # print stats before time averaging
      # if print_stats: hapy.print_stat(data,name=var_list[v],stat='naxsh',indent='    ',compact=True)
      #-------------------------------------------------------------------------
      # # average over time dimension
      # if 'time' in data.dims : 
      #    hapy.print_time_length(data.time,indent=' '*6,print_span=True, print_length=False)
      #    if use_snapshot:
      #       data = data.isel(time=ss_t,drop=True)
      #       print(' '*4+f'{hapy.tclr.RED}WARNING - snapshot mode enabled{hapy.tclr.END}')
      #    else:
      #       data = data.mean(dim='time')
      #-------------------------------------------------------------------------
      # # Calculate area weighted global mean
      # if 'area' in locals() :
      #    gbl_mean = ( (data*area).sum() / area.sum() ).values 
      #    print(hapy.tcolor.CYAN+f'      Area Weighted Global Mean : {gbl_mean:6.4}'+hapy.tcolor.END)
      # else:
      #    glb_avg_list.append(None)
      #-------------------------------------------------------------------------
      # # print stats after time averaging
      # if print_stats: hapy.print_stat(data,name=var_list[v],stat='naxsh',indent='    ',compact=True)
      #-------------------------------------------------------------------------
      # # append to data lists
      # data_list.append( data )
      #-------------------------------------------------------------------------
      # # save baseline for diff map
      # if plot_diff :
      #    if c==diff_base:
      #       data_baseline = data.copy()
   #----------------------------------------------------------------------------
   # calculate common limits for consistent contour levels
   data_min = np.min([np.nanmin(d) for d in data_list])
   data_max = np.max([np.nanmax(d) for d in data_list])

   if plot_diff:
      # tmp_data = copy.deepcopy(data_list)
      # for c in range(num_file): tmp_data[c] = data_list[c] - data_list[0]
      # # diff_data_max = np.max([np.nanmax(d) for d in tmp_data])
      # # diff_data_min = np.min([np.nanmin(d) for d in tmp_data])
      # diff_data_max = np.max([np.nanmax(np.absolute(d)) for d in tmp_data])
      # diff_data_min = -1*diff_data_max
      for f in range(num_file):
         if f!=0:
            data_list[f] = data_list[f] - data_list[0]
      #    hapy.print_stat(data_list[f],name=var_list[v],stat='naxsh',indent='    ',compact=True)
      # exit()
   #----------------------------------------------------------------------------
   # set color bar levels
   clev = None
   # clev = np.logspace( -5, -1, num=40)
   # if var_list[v] in ['PS','sp']: clev = np.linspace( 800e2, 1020e2, num=40)
   # if var_list[v] in ['PS','sp']: clev = np.arange(600e2,1040e2+2e2,10e2)
   #----------------------------------------------------------------------------
   # set color map
   cmap = 'viridis'
   # cmap = cmocean.cm.rain
   cmap = cmocean.cm.balance
   #----------------------------------------------------------------------------
   for f in range(num_file):
      #-------------------------------------------------------------------------
      img_kwargs = {}
      img_kwargs['origin'] = 'lower'
      img_kwargs['cmap']   = cmap

      # img_kwargs['vmin']   = -1000 
      # img_kwargs['vmax']   =  1000 

      if plot_diff and f!=0:
         img_kwargs['cmap']   = cmocean.cm.balance
         # img_kwargs['vmin']   = diff_data_min
         # img_kwargs['vmax']   = diff_data_max
         clev = None

      if clev is not None: img_kwargs['norm'] = mcolors.BoundaryNorm(clev, ncolors=256)

      ax = axs[v,c] if var_x_case else axs[f,v]
      ax.coastlines(linewidth=0.2,edgecolor='white')
      ax.set_title(opt_list[f]['n'],fontsize=title_fontsize, loc='left')
      ax.set_title(var_str[v],      fontsize=title_fontsize, loc='right')
      ax.set_global()

      # if plot_diff and c>0:
      #    rst = hapy.to_raster(data_list[c]-data_list[0], grid_ds_list[c], ax=ax, use_spherical=False)
      # else:
      #    rst = hapy.to_raster(data_list[c], grid_ds_list[c], ax=ax, use_spherical=False)

      rst = hapy.to_raster(data_list[f], grid_ds_list[f], ax=ax, use_spherical=False, pixel_ratio=20.0)
      img = ax.imshow(rst, extent=ax.get_xlim()+ax.get_ylim(), **img_kwargs)

      cbar = fig.colorbar(img, ax=ax, fraction=0.02, orientation='vertical')
      cbar.ax.tick_params(labelsize=lable_fontsize)

#---------------------------------------------------------------------------------------------------
# Finalize plot
fig.savefig(fig_file, dpi=100, bbox_inches='tight')
plt.close(fig)

print(f'\n{fig_file}\n')

#---------------------------------------------------------------------------------------------------
