import os, subprocess as sp, numpy as np, xarray as xr, dask, copy, string, cmocean, glob
import uxarray as ux, cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import hapy
host = hapy.get_host()
xr.set_options(use_new_combine_kwarg_defaults=True)
#-------------------------------------------------------------------------------
name,case,case_dir,case_sub = [],[],[],[]
scrip_file_list = []
def add_case(case_in,n=None,p=None,s=None,g=None,c=None,scrip_file=None):
   global name,case,case_dir,case_sub
   tmp_name = case_in if n is None else n
   case.append(case_in); name.append(tmp_name); 
   case_dir.append(p); case_sub.append(s);
   scrip_file_list.append(scrip_file)
#-------------------------------------------------------------------------------
var,lev_list,var_str = [],[],[]
var_opts_list = []
def add_var(var_name,lev=0,s=None,**kwargs): 
   var.append(var_name); lev_list.append(lev); 
   var_str.append(var_name if s is None else s)
   var_opts = {}
   for k, val in kwargs.items(): var_opts[k] = val
   var_opts_list.append(var_opts)
#-------------------------------------------------------------------------------
# fig_file = os.getenv('HOME')+'/E3SM_analysis/figs_clim/clim.map.v1.png'
fig_file = 'figs_sfc-osc/sfc-osc.map.v1.png'
#-------------------------------------------------------------------------------
# if host=='olcf':

   # scrip_file_path = os.getenv('HOME')+f'/E3SM/data_grid/ne30pg2_scrip.nc'

   ### 2023 coriolis test
   # add_case('E3SM.2023-coriolis-test.GNUGPU.ne30pg2_oECv3.F2010-MMF1.coriolis-off',n='MMF 2D')
   # add_case('E3SM.2023-coriolis-test.GNUGPU.ne30pg2_oECv3.F2010-MMF1.NXY_32_1.coriolis-on',n='MMF 2D + non-trad cor')

#-------------------------------------------------------------------------------
if host=='nersc':
   
   ## 2025 scidac multi-fidelity tests
   tmp_scratch = '/pscratch/sd/w/whannah/e3sm_scratch/pm-gpu'
   tmp_grid_file = '/pscratch/sd/w/whannah/files_grid/scrip_ne256pg2.nc'
   add_case('E3SM.2026-osc-test-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_32',n='ne256 dt_phys=10min',p=tmp_scratch,s='run',scrip_file=tmp_grid_file)
   htype,first_file,num_files = 'eam.h1',-1,1
   use_snapshot,ss_t = True,-1

   add_var('T_2m_atm_backtend',                     s='T2m backtend AVERAGE',              htype='output.scream.2D.AVERAGE.ndays_x1.')
   add_var('T_2m_atm_backtend2',                    s='T2m atm_backtend2 AVERAGE',         htype='output.scream.2D.AVERAGE.ndays_x1.')
   add_var('T_2m_atm_backtend2_product',            s='T2m atm_backtend2_product AVERAGE', htype='output.scream.2D.AVERAGE.ndays_x1.')
   add_var('T_2m_nf1.0_mac2_atm_osc_intermittency', s='T2m osc_intermittency AVERAGE',     htype='output.scream.2D.AVERAGE.ndays_x1.')

#-------------------------------------------------------------------------------
# if host=='lcrc':

   # add_case('E3SM.2025-MF-test-00.ne22pg2.F20TR.NN_2',n='ne22 test',p='/lcrc/group/e3sm/ac.whannah/scratch/chrys/nersc_runs',s='run',scrip_file=f'/home/ac.whannah/E3SM/data_grid/ne22pg2_scrip.nc')

#-------------------------------------------------------------------------------
# if host=='llnl':
#    scrip_file_path = '/g/g19/hannah6/files_grid/ne30pg2_scrip.nc'

#    ### STRONG tests
#    add_case('v3.2026-STRONG-ENS-TEST-00.start_2018-07-04.seed_113355',n='00 seed_113355',p='/p/vast1/strong/hannah6',s='run',scrip_file=scrip_file_path)

#    htype,first_file,num_files = 'eam.h1',-1,1

#-------------------------------------------------------------------------------
if host=='alcf':

   tmp_grid_file = '/lus/flare/projects/E3SM_Dec/whannah/E3SM_grid_support/files_grid/ne256pg2_scrip.nc'
   tmp_scratch = '/lus/flare/projects/E3SM_Dec/whannah/scratch'
   add_case('E3SM.2026-osc-test-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_256',         n='ne256 dt_phys=10min',p=tmp_scratch,s='run',scrip_file=tmp_grid_file)
   # add_case('E3SM.2026-osc-test-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_256.NCPL_288',n='ne256 dt_phys=5min', p=tmp_scratch,s='run',scrip_file=tmp_grid_file)
   first_file,num_files = 5,0

   add_var('T_2m_atm_backtend',          s='T2m backtend MAX', htype='output.scream.2D.MAX.ndays_x1.')
   # add_var('wind_speed_10m_atm_backtend',s='wsp backtend MAX', htype='output.scream.2D.MAX.ndays_x1.')
   # add_var('surf_sens_flux_atm_backtend',s='shf backtend MAX', htype='output.scream.2D.MAX.ndays_x1.')

   # add_var('T_2m_atm_backtend',          s='T2m backtend MIN', htype='output.scream.2D.MIN.ndays_x1.')
   # add_var('wind_speed_10m_atm_backtend',s='wsp backtend MIN', htype='output.scream.2D.MIN.ndays_x1.')
   # add_var('surf_sens_flux_atm_backtend',s='shf backtend MIN', htype='output.scream.2D.MIN.ndays_x1.')

   add_var('T_2m_atm_backtend',          s='T2m backtend AVERAGE', htype='output.scream.2D.AVERAGE.ndays_x1.')
   # add_var('wind_speed_10m_atm_backtend',s='wsp backtend AVERAGE', htype='output.scream.2D.AVERAGE.ndays_x1.')
   # add_var('surf_sens_flux_atm_backtend',s='shf backtend AVERAGE', htype='output.scream.2D.AVERAGE.ndays_x1.')

   add_var('T_2m_atm_backtend',          s='T2m backtend MAX', htype='output.scream.2D.MAX.ndays_x1.',method='std')

#-------------------------------------------------------------------------------

# add_var('PS')

# add_var('TS')
# add_var('PRECT',   s='Precipitation')
# add_var('TGCLDLWP',s='Liq Water Path')
# add_var('TGCLDIWP',s='Ice Water Path')
# add_var('P-E')
# add_var('TMQ')
# add_var('LHFLX')
# add_var('U10')

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

plot_diff,add_diff = False,False
# plot_diff,add_diff = True,False

print_stats          = True
var_x_case           = True

# num_plot_col         = len(case)
num_plot_col         = 1#len(var)

use_common_label_bar = False

if 'use_snapshot' not in locals(): use_snapshot,ss_t = False,-1

#---------------------------------------------------------------------------------------------------
# Set up plot resources
if case==[]: raise ValueError('ERROR - case list is empty!')
num_var,num_case = len(var),len(case)

if 'subtitle_font_height' not in locals(): subtitle_font_height = 0.01

diff_base = 0

#---------------------------------------------------------------------------------------------------
# set up figure objects
subplot_kwargs = {}
# subplot_kwargs['projection'] = ccrs.Robinson(central_longitude=180)
subplot_kwargs['projection'] = ccrs.PlateCarree()#central_longitude=180)
(d1,d2) = (num_var,num_case) if var_x_case else (num_case,num_var)
fdx,fdy=20,10;figsize = (fdx*num_case,fdy*num_var) if var_x_case else (fdx*num_var,fdy*num_case)
fig, axs = plt.subplots(d1,d2, subplot_kw=subplot_kwargs, figsize=figsize, squeeze=False )

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
   var_opts = var_opts_list[v]
   hapy.print_line()
   print(' '*2+'var: '+hapy.tclr.MAGENTA+var[v]+hapy.tclr.END)
   data_list = []
   glb_avg_list = []
   lat_list,lon_list = [],[]
   if 'lev_list' in locals(): lev = lev_list[v]
   for c in range(num_case):
      print(' '*4+'case: '+hapy.tclr.GREEN+case[c]+hapy.tclr.END)
      #-------------------------------------------------------------------------
      htype_tmp = None
      if 'htype' in globals(): htype_tmp = htype
      if 'htype' in var_opts: htype_tmp = var_opts['htype']
      #-------------------------------------------------------------------------
      # file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/{case[c]}*{htype}*'
      file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/*{htype_tmp}*'
      file_list = sorted(glob.glob(file_path))
      if 'first_file' in locals(): 
         if first_file!=0:
            file_list = file_list[first_file:]
      if 'num_files'  in locals(): 
         if num_files!=0:
            file_list = file_list[:num_files]
      #-------------------------------------------------------------------------
      if file_list==[]: print('ERROR: Empty file list:'); print(); print(file_path); exit()
      #-------------------------------------------------------------------------
      # print();print(' '*6+'file_list:')
      # for f in file_list: print(' '*6+f'{f}')
      # print()
      #-------------------------------------------------------------------------
      # ds = xr.open_mfdataset( file_list )
      # data = ds[var[v]]
      #-------------------------------------------------------------------------
      uxds = ux.open_mfdataset(scrip_file_list[c], file_list)
      data = uxds[var[v]]
      #-------------------------------------------------------------------------
      # adjust units
      if var[v]=='FRONTGF': data = data*86400e6 # K^2/M^2/S > K^2/KM^2/day
      #-------------------------------------------------------------------------
      if 'lev' in data.dims:
         # data = data.isel(lev=0)
         # if requested lev<0 use model levels without interpolation
         if lev_list[v]<0: data = data.isel(lev=np.absolute(lev_list),drop=True)
      #-------------------------------------------------------------------------
      # # print stats before time averaging
      # if print_stats: hapy.print_stat(data,name=var[v],stat='naxsh',indent='    ',compact=True)
      #-------------------------------------------------------------------------
      # average over time dimension
      if 'time' in data.dims : 
         hapy.print_time_length(data.time,indent=' '*6,print_span=True, print_length=False)
         if use_snapshot:
            data = data.isel(time=ss_t,drop=True)
            print(' '*4+f'{hapy.tclr.RED}WARNING - snapshot mode enabled{hapy.tclr.END}')
         else:
            if 'backtend' in var[v]:
               if 'method' in var_opts:
                  if var_opts['method']=='std': data = data.std(dim='time')
               else:
                  if '.MAX.'     in var_opts['htype']: data = data.max(dim='time')
                  if '.MIN.'     in var_opts['htype']: data = data.min(dim='time')
                  if '.AVERAGE.' in var_opts['htype']: data = data.mean(dim='time')

               print(f'{hapy.tclr.RED}WARNING - assuming case index order - WARNING{hapy.tclr.END}')
               if c==0: data = data*10.*60.
               if c==1: data = data* 5.*60.

            else:
               data = data.mean(dim='time')
      #-------------------------------------------------------------------------
      # Calculate area weighted global mean
      if 'area' in locals() :
         gbl_mean = ( (data*area).sum() / area.sum() ).values 
         print(hapy.tcolor.CYAN+f'      Area Weighted Global Mean : {gbl_mean:6.4}'+hapy.tcolor.END)
      else:
         glb_avg_list.append(None)
      #-------------------------------------------------------------------------
      # print stats after time averaging
      if print_stats: hapy.print_stat(data,name=var[v],stat='naxsh',indent='    ',compact=True)
      #-------------------------------------------------------------------------
      # append to data lists
      data_list.append( data )
      #-------------------------------------------------------------------------
      # save baseline for diff map
      if plot_diff :
         if c==diff_base:
            data_baseline = data.copy()
   #----------------------------------------------------------------------------
   # calculate common limits for consistent contour levels
   data_min = np.min([np.nanmin(d) for d in data_list])
   data_max = np.max([np.nanmax(d) for d in data_list])
   #----------------------------------------------------------------------------
   if plot_diff:
      tmp_data = copy.deepcopy(data_list)
      for c in range(num_case): tmp_data[c] = data_list[c] - data_list[diff_base]
      # diff_data_max = np.max([np.nanmax(d) for d in tmp_data])
      # diff_data_min = np.min([np.nanmin(d) for d in tmp_data])
      diff_data_max = np.max([np.nanmax(np.absolute(d)) for d in tmp_data])
      diff_data_min = -1*diff_data_max
      for c in range(num_case):
         if c!=diff_base:
            data_list[c] = data_list[c] - data_list[diff_base]
   #----------------------------------------------------------------------------
   # set color bar levels
   clev = None
   if var[v]=='FRONTGF': clev = np.logspace( -5, -1, num=40)
   #----------------------------------------------------------------------------
   # set color map
   # cmap = 'viridis'
   # cmap = cmocean.cm.rain
   cmap = cmocean.cm.amp
   # cmap = cmocean.cm.balance
   #----------------------------------------------------------------------------
   if 'backtend' in var[v] and '.AVERAGE.' in var_opts['htype']:
      cmap = cmocean.cm.balance
      data_mag_max = max(abs(data_min),abs(data_max))
      data_min = data_mag_max*-1
      data_max = data_mag_max
   #----------------------------------------------------------------------------
   for c in range(num_case):
      #-------------------------------------------------------------------------
      img_kwargs = {}
      img_kwargs['origin'] = 'lower'
      img_kwargs['extent'] = [-180, 180, -90, 90]
      img_kwargs['cmap']   = cmap

      if plot_diff and c!=diff_base:
         img_kwargs['cmap']   = cmocean.cm.balance
         img_kwargs['vmin']   = diff_data_min
         img_kwargs['vmax']   = diff_data_max
         clev = None

      if clev is None:
         img_kwargs['vmin'],img_kwargs['vmax'] = data_min, data_max
      else:
         img_kwargs['norm'] = mcolors.BoundaryNorm(clev, ncolors=256)


      # print()
      # print(img_kwargs)
      # print()

      ax = axs[v,c] if var_x_case else axs[c,v]
      ax.coastlines(linewidth=0.2,edgecolor='white')
      ax.set_title(name[c],   fontsize=20, loc='left')
      ax.set_title(var_str[v],fontsize=20, loc='right')
      ax.set_global()

      img = ax.imshow(data_list[c].to_raster(ax=ax), **img_kwargs)

      # orientation = 'vertical' if var_x_case else 'horizontal'
      # if c==num_case-1: fig.colorbar(img, ax=ax, fraction=0.02, orientation=orientation)

      cbar = fig.colorbar(img, ax=ax, fraction=0.02, orientation='vertical')
      cbar.ax.tick_params(labelsize=15)

      # if var_x_case:
      #    if c==num_case-1: fig.colorbar(img, ax=ax, fraction=0.02, orientation='vertical')
      # else:
      #    fig.colorbar(img, ax=ax, fraction=0.05, orientation='horizontal')
      

#---------------------------------------------------------------------------------------------------
# Finalize plot
fig.savefig(fig_file, dpi=100, bbox_inches='tight')
plt.close(fig)

print(f'\n{fig_file}\n')

#---------------------------------------------------------------------------------------------------
