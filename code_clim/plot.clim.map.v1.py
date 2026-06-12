from time import perf_counter
timer_start = perf_counter()
import os, subprocess as sp, numpy as np, xarray as xr, dask, copy, string, cmocean, glob
import uxarray as ux, cartopy.crs as ccrs, matplotlib.pyplot as plt, matplotlib.colors as mcolors
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
   if scrip_file is None: raise ValueError('scrip_file cannot be None')
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
fig_file = 'figs_clim/clim.map.v1.png'
#-------------------------------------------------------------------------------
if host=='olcf':

   scrip_file_path = os.getenv('HOME')+f'/E3SM/data_grid/ne30pg2_scrip.nc'

   ### 2023 coriolis test
   # add_case('E3SM.2023-coriolis-test.GNUGPU.ne30pg2_oECv3.F2010-MMF1.coriolis-off',n='MMF 2D')
   # add_case('E3SM.2023-coriolis-test.GNUGPU.ne30pg2_oECv3.F2010-MMF1.NXY_32_1.coriolis-on',n='MMF 2D + non-trad cor')

#-------------------------------------------------------------------------------
if host=='nersc':
   
   ### 2025 scidac multi-fidelity tests
   # add_case('E3SM.2025-MF-test-00.ne18pg2.F20TR.NN_12',n='ne18',p='/pscratch/sd/w/whannah/e3sm_scratch/pm-cpu',s='run',scrip_file='/global/cfs/cdirs/m4310/whannah/files_grid/ne18pg2_scrip.nc')

   # ### Wandi's frontogensis gradient correction
   # tmp_path,tmp_sub = '/pscratch/sd/w/wandiyu/e3sm_scratch/pm-cpu','run'
   # scrip_file = os.getenv('HOME')+f'/E3SM/data_grid/ne30pg2_scrip.nc'
   # # add_case('v3.LR.GW.fronto.correction.nofix.Jan.20251104',n='control',   p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
   # add_case('v3.LR.GW.fronto.correction.pfix.Jan.20251104', n='p-grad fix', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
   # add_case('v3.LR.GW.fronto.correction.zfix.Jan.20251104', n='z-grad fix', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
   # htype,first_file,num_files = 'eam.h0',0,0

   # add_case('v3.LR.GW.fronto.correction.nofix.Jan.20251104',n='control',   p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
   # htype,first_file,num_files = 'eam.h1',-1,1
   # use_snapshot,ss_t = True,-1

   ### ERA5 remapped surface pressure for 2026 CONUS-RRM
   # tmp_path,tmp_sub = '/pscratch/sd/m/meng/hiccup','tmp'
   # scrip_file = '/global/cfs/cdirs/e3sm/2026-INCITE-CONUS-RRM/files_grid/2026-incite-conus-1024x2-np4_scrip.nc'
   # # add_case('v3.LR.GW.fronto.correction.nofix.Jan.20251104',n='control',   p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
   # add_case('incite_conus', n='', p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
   # # htype,first_file,num_files = 'eam.h0',0,1


   tmp_path,tmp_sub = '/pscratch/sd/w/whannah/scream_scratch/pm-gpu','run'
   # scrip_file = os.getenv('HOME')+f'/E3SM/data_grid/ne4pg2_scrip.nc'
   # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne4pg2.NN_1.zm_cxx.debug', n='zm_cxx',  p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
   # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne4pg2.NN_1.zm_f90.debug', n='zm_f90',  p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
   scrip_file = os.getenv('HOME')+f'/E3SM/data_grid/ne30pg2_scrip.nc'
   # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_cxx',   p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
   # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_f90',   p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
   # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_f90.zm_detr_0',n='zm_f90.zm_detr_0',p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
   # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_f90.zm_detr_1',n='ZM-F90',p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
   # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_cxx.zm_detr_0',n='zm_cxx.zm_detr_0',p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
   # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_cxx.zm_detr_1',n='ZM-C++.',p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
   # add_case('E3SM.2026-ZM-DEV-00.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_cxx.zm_detr_1.mcsp_mod_1',n='zm_cxx.zm_detr_1.mcsp_mod_1',p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
   add_case('E3SM.2026-ZM-DEV-01.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_f90',n='zm_f90',  p=tmp_path,s=tmp_sub,scrip_file=scrip_file)
   add_case('E3SM.2026-ZM-DEV-01.F2010xx-ZM.GPU.ne30pg2.NN_4.zm_cxx',n='zm_cxx',  p=tmp_path,s=tmp_sub,scrip_file=scrip_file)

   # use_snapshot,ss_t = True,4
   # htype,first_file,num_files = 'output.2D.INSTANT.nsteps_x1',0,1
   # htype,first_file,num_files = 'output.1ma.AVERAGE.nmonths_x1',0,1
   # htype,first_file,num_files = 'output.monthly.AVERAGE.nmonths_x1',0,12
   htype,first_file,num_files = 'output.daily.AVERAGE.nhours_x24',0,0

#-------------------------------------------------------------------------------
if host=='lcrc':

   # tmp_grid_file = '/lcrc/group/e3sm/ac.whannah/scratch/chrys/files_grid/ne256pg2_scrip.nc'

   tmp_grid_file = '/lcrc/group/e3sm/ac.whannah/scratch/chrys/files_grid/scrip_ne30pg2.nc'
   tmp_scratch = '/lcrc/group/e3sm/ac.whannah/scratch/chrys'
   add_case('E3SM.2026-ZM-MOD-00.F2010.ne30pg2.NN_8',         n='E3SM ncin=1',  p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
   # add_case('E3SM.2026-ZM-MOD-00.F2010.ne30pg2.NN_8.ncin_5',  n='E3SM ncin=5',  p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
   add_case('E3SM.2026-ZM-MOD-00.F2010.ne30pg2.NN_8.ncin_20', n='E3SM ncin=20', p=tmp_scratch, s='run',scrip_file=tmp_grid_file)
   htype,first_file,num_files = 'eam.h1',0,0
   add_var('PCONVT',htype='eam.h1')

#-------------------------------------------------------------------------------
if host=='llnl':
   scrip_file_path = '/g/g19/hannah6/files_grid/ne30pg2_scrip.nc'

   ### STRONG tests
   add_case('v3.2026-STRONG-ENS-TEST-00.start_2018-07-04.seed_113355',n='00 seed_113355',p='/p/vast1/strong/hannah6',s='run',scrip_file=scrip_file_path)
   # add_case('v3.2026-STRONG-ENS-TEST-00.start_2018-07-04.seed_224466',n='00 seed_224466',p='/p/vast1/strong/hannah6',s='run',scrip_file=scrip_file_path)
   # add_case('v3.2026-STRONG-ENS-TEST-01.start_2018-07-04.seed_113355',n='01 seed_113355',p='/p/vast1/strong/hannah6',s='run',scrip_file=scrip_file_path)
   # add_case('v3.2026-STRONG-ENS-TEST-01.start_2018-07-04.seed_224466',n='01 seed_224466',p='/p/vast1/strong/hannah6',s='run',scrip_file=scrip_file_path)
   htype,first_file,num_files = 'eam.h1',0,0

#-------------------------------------------------------------------------------
if host=='alcf':

   tmp_grid_file = '/lus/flare/projects/E3SM_Dec/whannah/E3SM_grid_support/files_grid/ne256pg2_scrip.nc'
   tmp_scratch = '/lus/flare/projects/E3SM_Dec/whannah/scratch'
   add_case('E3SM.2026-osc-test-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_256',         n='ne256 dt_phys=10min',p=tmp_scratch,s='run',scrip_file=tmp_grid_file)
   # add_case('E3SM.2026-osc-test-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_256.NCPL_288',n='ne256 dt_phys=5min', p=tmp_scratch,s='run',scrip_file=tmp_grid_file)
   first_file,num_files = 5,0

#-------------------------------------------------------------------------------

# add_var('sp',s='ncremap',htype='erasfc.tr.sp.20250630_00.nc')
# add_var('sp',s='moab',   htype='erasfc.mb.sp.20250630_00.nc')

add_var('zm_activity')
add_var('zm_cape')
# add_var('zm_dcape')
add_var('precip_total_surf_mass_flux')
# add_var('zm_prec_frac')
# add_var('zm_prec')
add_var('LW_flux_up_at_model_top')
# add_var('zm_detr_liq')
# add_var('zm_detr_ice')

add_var('VapWaterPath')
add_var('LiqWaterPath')
# add_var('IceWaterPath')
# add_var('RainWaterPath')
# add_var('T_2m')
# add_var('zm_cape')


# add_var('TS')
# add_var('PRECT',s='Precipitation')
# add_var('TGCLDLWP',s='Liq Water Path')
# add_var('TGCLDIWP',s='Ice Water Path')
# add_var('P-E')
# add_var('TMQ')
# add_var('LHFLX')
# add_var('FSNT'); add_var('FLNT')
# add_var('LWCF'); add_var('SWCF')
# add_var('U10')

# add_var('T',     lev=500)
# add_var('U',     lev=500)
# add_var('CLDLIQ',lev=500)
# add_var('CLDICE',lev=500)
# add_var('Q',     lev=900)
# add_var('OMEGA',lev=500)

# add_var('FRONTGF',lev=-57,s='FRONTGF [K^2/km^2/day]')

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
# plot_diff,add_diff = True,False
plot_diff,add_diff = True,True

print_stats          = True
var_x_case           = False

# num_plot_col         = len(case)
# num_plot_col         = 1#len(var)

use_common_label_bar = False

if 'use_snapshot' not in locals(): use_snapshot,ss_t = False,-1

#---------------------------------------------------------------------------------------------------
# Set up plot resources
if case==[]: raise ValueError('ERROR - case list is empty!')
num_var,num_case = len(var),len(case)


diff_base = 0

if 'first_file' not in locals(): first_file = None
if 'num_files'  not in locals(): num_files  = None

#---------------------------------------------------------------------------------------------------
# build the list of panels along the case dimension
#    each panel is a tuple of (kind, case_index) where kind is 'full' or 'diff'
case_panels = []
if plot_diff and add_diff:
   # retain all full-field panels and append difference panels
   for c in range(num_case): case_panels.append(('full',c))
   for c in range(num_case):
      if c!=diff_base: case_panels.append(('diff',c))
elif plot_diff:
   # replace non-baseline full fields with differences (no added panels)
   for c in range(num_case):
      if c==diff_base: case_panels.append(('full',c))
      else:            case_panels.append(('diff',c))
else:
   for c in range(num_case): case_panels.append(('full',c))
num_panel = len(case_panels)

#---------------------------------------------------------------------------------------------------
# set up figure objects
subplot_kwargs = {}
# subplot_kwargs['projection'] = ccrs.Orthographic(central_latitude=-85); lat_min, lat_max = -90, -80
subplot_kwargs['projection'] = ccrs.PlateCarree(central_longitude=180)
# subplot_kwargs['projection'] = ccrs.PlateCarree()
(d1,d2) = (num_var,num_panel) if var_x_case else (num_panel,num_var)
fdx,fdy=20,10;figsize = (fdx*num_panel,fdy*num_var) if var_x_case else (fdx*num_var,fdy*num_panel)
# title_fontsize,lable_fontsize = 20,18
title_fontsize,lable_fontsize = 30,25

fig, axs = plt.subplots(d1,d2, subplot_kw=subplot_kwargs, figsize=figsize, squeeze=False )

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
   var_opts = var_opts_list[v]
   hapy.print_line()
   print(' '*2+'var: '+hapy.tclr.MAGENTA+var[v]+hapy.tclr.END)
   data_list = []
   glb_avg_list = []
   grid_ds_list = []
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
      if first_file is not None and first_file!=0: file_list = file_list[first_file:]
      if num_files  is not None and num_files !=0: file_list = file_list[:num_files]
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
      uxds = ux.open_mfdataset(scrip_file_list[c], file_list, data_vars='minimal')

      # if 'valid_time' in uxds.dims:
      #    if len(uxds['valid_time'])==1:
      #       uxds = uxds.isel(valid_time=0)

      # hapy.check_invalid_values(uxds, variables=[var[v]], check_range=None)
      # exit()

      if var[v]=='zm_prec_frac':
         data = uxds['zm_prec'] / uxds['precip_total_surf_mass_flux']
      else:
         data = uxds[var[v]]

      # # lon_min, lon_max = -180, 180
      # mask = ((uxds['lat'] >= lat_min) & (uxds['lat'] <= lat_max))
      # # mask.load()
      # data = data.where(mask, drop=True)

      # if print_stats: hapy.print_stat(data,name=var[v],stat='naxsh',indent='    ',compact=True)

      # if 'time' in data.dims:
      #    data = data.mean(dim='time')
      #    # data = data.min(dim='time')

      # grid_ds = xr.open_dataset(scrip_file_list[c])
      # # grid_ds = grid_ds.where(mask, drop=True)
      # grid_ds_list.append(grid_ds)

      #-------------------------------------------------------------------------
      # if 'lev' in data.dims:
      #    # data = data.isel(lev=0)
      #    # if requested lev<0 use model levels without interpolation
      #    if lev_list[v]<0: data = data.isel(lev=np.absolute(lev_list),drop=True)
      #-------------------------------------------------------------------------
      # integrate over levels
      if 'lev' in data.dims:
         print(' '*4+f'{hapy.tclr.RED}WARNING - summing over lev dimension{hapy.tclr.END}')
         data = data.sum(dim='lev')
      #-------------------------------------------------------------------------
      # adjust units
      if var[v]=='FRONTGF': data = data*86400e6 # K^2/M^2/S > K^2/KM^2/day
      if 'prec' in var[v] and var[v]!='zm_prec_frac': data = data*86400*1e3 # m/s > mm/day
      #-------------------------------------------------------------------------
      if use_snapshot and 'time' in data.dims:
         data = data.isel(time=ss_t,drop=True)
         print(' '*4+f'{hapy.tclr.RED}WARNING - snapshot mode enabled{hapy.tclr.END}')
      #-------------------------------------------------------------------------
      # print stats before time averaging
      if print_stats: hapy.print_stat(data,name=var[v],stat='naxh',indent='    ',compact=True)
      #-------------------------------------------------------------------------
      # average over time dimension
      if 'time' in data.dims : 
         # hapy.print_time_length(data.time,indent=' '*6,print_span=True, print_length=False)
         # if use_snapshot:
         #    data = data.isel(time=ss_t,drop=True)
         #    print(' '*4+f'{hapy.tclr.RED}WARNING - snapshot mode enabled{hapy.tclr.END}')
         if not use_snapshot:
            data = data.mean(dim='time')
      #-------------------------------------------------------------------------
      # # Calculate area weighted global mean
      # if 'area' in locals() :
      #    gbl_mean = ( (data*area).sum() / area.sum() ).values 
      #    print(hapy.tcolor.CYAN+f'      Area Weighted Global Mean : {gbl_mean:6.4}'+hapy.tcolor.END)
      # else:
      #    glb_avg_list.append(None)
      #-------------------------------------------------------------------------
      # # print stats after time averaging
      # if print_stats: hapy.print_stat(data,name=var[v],stat='naxsh',indent='    ',compact=True)
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
   # compute differences relative to the baseline case (data_list keeps full fields)
   if plot_diff:
      diff_list = [ data_list[c] - data_list[diff_base] for c in range(num_case) ]
      diff_data_max = np.max([np.nanmax(np.absolute(d)) for d in diff_list])
      diff_data_min = -1*diff_data_max
   #----------------------------------------------------------------------------
   # set color bar levels
   clev = None
   if var[v]=='FRONTGF': clev = np.logspace( -5, -1, num=40)
   if 'prec' in var[v]: clev = clev = np.arange(0,20+1,1)
   # if 'prec' in var[v]: clev = np.logspace( np.log10(0.1), np.log10(50), num=40)
   if var[v]=='zm_prec_frac': clev = None
   # if var[v]=='zm_prec_frac': clev = np.linspace( 0, 1, num=20)
   # if var[v]=='zm_prec_frac': clev = np.logspace( np.log10(0.1), np.log10(1.0), num=40)
   # if var[v] in ['PS','sp']: clev = np.linspace( 800e2, 1020e2, num=40)
   if var[v] in ['PS','sp']: clev = np.arange(600e2,1040e2+2e2,10e2)
   if var[v]=='PCONVT': clev = np.arange(100e2,600e2+5e2,5e2)

   # if var[v]=='PCONVT': clev = np.logspace( np.log10(50e2), np.log10(500e2), num=40)
   #----------------------------------------------------------------------------
   # set color map
   cmap = 'viridis'
   # cmap = cmocean.cm.rain
   # cmap = cmocean.cm.amp
   # cmap = cmocean.cm.balance
   if var[v]=='PCONVT': cmap = cmocean.cm.rain_r
   #----------------------------------------------------------------------------
   for p in range(num_panel):
      kind,c = case_panels[p]
      #-------------------------------------------------------------------------
      img_kwargs = {}
      img_kwargs['origin'] = 'lower'
      # img_kwargs['extent'] = [-180, 180, -90, 90]
      img_kwargs['cmap']   = cmap

      if kind=='diff':
         plot_data = diff_list[c]
         img_kwargs['cmap']   = cmocean.cm.balance
         img_kwargs['vmin']   = diff_data_min
         img_kwargs['vmax']   = diff_data_max
      else:
         plot_data = data_list[c]
         if clev is None:
            img_kwargs['vmin'],img_kwargs['vmax'] = data_min, data_max
         else:
            img_kwargs['norm'] = mcolors.BoundaryNorm(clev, ncolors=256)


      # print()
      # print(img_kwargs)
      # print()

      ax = axs[v,p] if var_x_case else axs[p,v]
      ax.coastlines(linewidth=0.2,edgecolor='white')
      if kind=='diff':
         ax.set_title(f'{name[c]} - {name[diff_base]}',fontsize=title_fontsize, loc='left')
      else:
         ax.set_title(name[c],fontsize=title_fontsize, loc='left')
      ax.set_title(var_str[v],fontsize=title_fontsize, loc='right')
      ax.set_global()

      # rst = hapy.to_raster(plot_data, grid_ds_list[c], ax)
      # img = ax.imshow(rst, extent=ax.get_xlim() + ax.get_ylim(), **img_kwargs)

      img = ax.imshow(plot_data.to_raster(ax=ax), extent=ax.get_xlim() + ax.get_ylim(), **img_kwargs)

      # orientation = 'vertical' if var_x_case else 'horizontal'
      # if c==num_case-1: fig.colorbar(img, ax=ax, fraction=0.02, orientation=orientation)

      cbar = fig.colorbar(img, ax=ax, fraction=0.02, orientation='vertical')
      cbar.ax.tick_params(labelsize=lable_fontsize)

      # if var_x_case:
      #    if c==num_case-1: fig.colorbar(img, ax=ax, fraction=0.02, orientation='vertical')
      # else:
      #    fig.colorbar(img, ax=ax, fraction=0.05, orientation='horizontal')
      

#---------------------------------------------------------------------------------------------------
# Finalize plot
fig.savefig(fig_file, dpi=100, bbox_inches='tight')
plt.close(fig)

print(f'\n{fig_file}')

#---------------------------------------------------------------------------------------------------
etime = perf_counter()-timer_start
time_str = f'{etime:10.1f} sec'
if etime>60       : time_str += f' ({(etime/60):4.1f} min)'
if etime>(2*3600) : time_str += f' ({(etime/3600):.1f} hr)'
print(f'\n{hapy.tclr.YELLOW}elapsed time: {time_str} {hapy.tclr.END}')
print()
#---------------------------------------------------------------------------------------------------
