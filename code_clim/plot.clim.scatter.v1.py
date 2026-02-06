import os, sys, subprocess as sp, numpy as np, xarray as xr, dask, copy, string, cmocean, glob
import uxarray as ux, matplotlib.pyplot as plt
import hapy
host = hapy.get_host()
# Suppress getfattr warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
#---------------------------------------------------------------------------------------------------
case_opts_list = []
def add_case(case_in,name=None,root=None,sub=None,d=0,c='black',m=0,**kwargs):
   case_opts = {}
   case_opts['case']       = case_in
   case_opts['name']       = case_in if name is None else name
   case_opts['root']       = root
   case_opts['sub']        = sub
   case_opts['grid_file']  = grid_file
   case_opts['dsh']        = d
   case_opts['clr']        = c
   case_opts['mrk']        = m
   for k, val in kwargs.items(): case_opts[k] = val
   case_opts_list.append(case_opts)
#---------------------------------------------------------------------------------------------------
var_opts_list = []
def add_var(xvar, yvar, xn=None, yn=None, htype=None,**kwargs):
   var_opts = {}; var_opts['xvar'] = xvar; var_opts['yvar'] = yvar
   var_opts['xvar_name'] = xvar if xn is None else xn
   var_opts['yvar_name'] = yvar if yn is None else yn
   var_opts['htype'] = htype
   for k, val in kwargs.items(): var_opts[k] = val
   var_opts_list.append(var_opts)
#---------------------------------------------------------------------------------------------------
if host=='olcf':
   grid_file_path = os.getenv('HOME')+f'/E3SM/data_grid/ne30pg2_scrip.nc'

#-------------------------------------------------------------------------------
if host=='alcf':
   ### EAMxx auto-calibration blitz
   scratch = '/lus/flare/projects/E3SM_Dec/whannah/scratch'
   grid_file = '/lus/flare/projects/E3SM_Dec/whannah/files_grid/ne30pg2_scrip.nc'
   add_case('2025-EACB-v3.111.ne32.NN_4.3a93ec49ed14',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.112.ne32.NN_4.153fee23cda2',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.113.ne32.NN_4.cbe575556cc8',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.114.ne32.NN_4.d0fe74c62bdc',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.115.ne32.NN_4.54999e06f405',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.116.ne32.NN_4.90969b9df7e6',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.117.ne32.NN_4.948b14d05c7c',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.118.ne32.NN_4.f8923f26da36',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.119.ne32.NN_4.d7a5c0322396',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.120.ne32.NN_4.e72914b29dbc',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.121.ne32.NN_4.ac89952b256d',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.122.ne32.NN_4.0dbb60f76219',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.123.ne32.NN_4.386fadecd170',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.124.ne32.NN_4.92f68e135713',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.125.ne32.NN_4.bc9bd98d6242',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.126.ne32.NN_4.15d1bda56c9b',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.127.ne32.NN_4.9f07c47d18a4',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.128.ne32.NN_4.54177dd468de',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.129.ne32.NN_4.62f559a82923',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.130.ne32.NN_4.21add96b03d0',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.131.ne32.NN_4.ce424d0fab83',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.132.ne32.NN_4.0afeeaf5f3bc',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.133.ne32.NN_4.6d180a087fb4',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.134.ne32.NN_4.c0cb5393290f',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.135.ne32.NN_4.013df5972706',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.136.ne32.NN_4.baa6cfad6ed6',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.137.ne32.NN_4.7a7e4faae9d5',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.138.ne32.NN_4.f041a73ea952',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.139.ne32.NN_4.9a748ddc1710',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.140.ne32.NN_4.2c6d4180d71e',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.141.ne32.NN_4.a82031039812',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v3.142.ne32.NN_4.ab410c45fdd9',name='',clr='red',root=scratch,sub='run',grid_file=grid_file)

   add_case('2025-EACB-v4.111.ne32.NN_4.3a93ec49ed14',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.111.ne32.NN_4.153fee23cda2',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.112.ne32.NN_4.cbe575556cc8',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.113.ne32.NN_4.d0fe74c62bdc',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.114.ne32.NN_4.54999e06f405',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.115.ne32.NN_4.90969b9df7e6',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.116.ne32.NN_4.948b14d05c7c',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.117.ne32.NN_4.f8923f26da36',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.118.ne32.NN_4.d7a5c0322396',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.119.ne32.NN_4.e72914b29dbc',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.120.ne32.NN_4.ac89952b256d',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.121.ne32.NN_4.0dbb60f76219',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.122.ne32.NN_4.386fadecd170',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.123.ne32.NN_4.92f68e135713',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.124.ne32.NN_4.bc9bd98d6242',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.125.ne32.NN_4.15d1bda56c9b',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.126.ne32.NN_4.9f07c47d18a4',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.127.ne32.NN_4.54177dd468de',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.128.ne32.NN_4.62f559a82923',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.129.ne32.NN_4.21add96b03d0',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.130.ne32.NN_4.ce424d0fab83',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.131.ne32.NN_4.0afeeaf5f3bc',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.132.ne32.NN_4.6d180a087fb4',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.133.ne32.NN_4.c0cb5393290f',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.134.ne32.NN_4.013df5972706',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.135.ne32.NN_4.baa6cfad6ed6',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.136.ne32.NN_4.7a7e4faae9d5',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.137.ne32.NN_4.f041a73ea952',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.138.ne32.NN_4.9a748ddc1710',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.139.ne32.NN_4.2c6d4180d71e',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.140.ne32.NN_4.a82031039812',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)
   add_case('2025-EACB-v4.141.ne32.NN_4.ab410c45fdd9',name='',clr='blue',root=scratch,sub='run',grid_file=grid_file)

   htype,first_file,num_files = '1ma_ne30pg2.AVERAGE.nmonths_x1',0,12
   # use_snapshot,ss_t = True,-1

   

#-------------------------------------------------------------------------------
if host=='nersc':
   ### Wandi's frontogensis gradient correction
   tmp_path,tmp_sub = '/pscratch/sd/w/wandiyu/e3sm_scratch/pm-cpu','run'
   grid_file = os.getenv('HOME')+f'/E3SM/data_grid/ne30pg2_scrip.nc'
   # add_case('v3.LR.GW.fronto.correction.nofix.Jan.20251104',n='control',   p=tmp_path,s=tmp_sub,grid_file=grid_file)
   add_case('v3.LR.GW.fronto.correction.pfix.Jan.20251104', n='p-grad fix', p=tmp_path,s=tmp_sub,grid_file=grid_file)
   add_case('v3.LR.GW.fronto.correction.zfix.Jan.20251104', n='z-grad fix', p=tmp_path,s=tmp_sub,grid_file=grid_file)
   htype,first_file,num_files = 'eam.h0',0,1
   # use_snapshot,ss_t = True,-1

#-------------------------------------------------------------------------------
if host=='lcrc':
   add_case('E3SM.2025-MF-test-00.ne22pg2.F20TR.NN_2',n='ne22 test',p='/lcrc/group/e3sm/ac.whannah/scratch/chrys/nersc_runs',s='run',grid_file=f'/home/ac.whannah/E3SM/data_grid/ne22pg2_scrip.nc')

#-------------------------------------------------------------------------------
if host=='llnl':
   ### STRONG tests
   grid_file_path = '/g/g19/hannah6/files_grid/ne30pg2_scrip.nc'
   add_case('v3.2026-STRONG-ENS-TEST-00.start_2018-07-04.seed_113355',n='00 seed_113355',p='/p/vast1/strong/hannah6',s='run',grid_file=grid_file_path)
   # add_case('v3.2026-STRONG-ENS-TEST-00.start_2018-07-04.seed_224466',n='00 seed_224466',p='/p/vast1/strong/hannah6',s='run',grid_file=grid_file_path)
   htype,first_file,num_files = 'eam.h1',-1,1
   # use_snapshot,ss_t = True,-1

#---------------------------------------------------------------------------------------------------

# add_var(xvar='LW_flux_up_at_model_top', yvar='SW_flux_up_at_model_top', htype=htype)
# add_var(xvar='LongwaveCloudForcing',    yvar='ShortwaveCloudForcing',   htype=htype)
# add_var(xvar='LiqWaterPath',             yvar='IceWaterPath',              htype=htype)
# add_var(xvar='VapWaterPath',             yvar='precip_liq_surf_mass_flux', htype=htype)
add_var(xvar='VapWaterPath',             yvar='LW_flux_up_at_model_top', htype=htype)
# add_var(xvar='cldfrac_tot_for_analysis', yvar='cldfrac_ice_for_analysis',  htype=htype)
# add_var(xvar='isccp_cldtot', yvar='???',  htype=htype)

num_plot_col = 2

#---------------------------------------------------------------------------------------------------
# lat1,lat2 = -30,30

# xlat,xlon,dy,dx = 60,120,10,10;
# if 'xlat' in locals(): lat1,lat2,lon1,lon2 = xlat-dy,xlat+dy,xlon-dx,xlon+dx
#---------------------------------------------------------------------------------------------------

fig_file = 'figs_clim/clim.scatter.v1.png'
tmp_head = 'data_temp/clim.scatter.v1'

recalculate = True

# plot_diff   = False
print_stats = True

# ncol_chunk_size = int(100e3)

#---------------------------------------------------------------------------------------------------
num_var = len(var_opts_list)
num_case = len(case_opts_list)

if 'num_plot_col' not in locals(): num_plot_col = len(num_var)
if 'use_snapshot' not in locals(): use_snapshot = False
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def get_file_list(case,case_dir,case_sub,file_type):
   file_path = f'{case_dir}/{case}/{case_sub}/*{file_type}*'
   file_list = sorted(glob.glob(file_path))
   if 'first_file' in globals(): file_list = file_list[first_file:]
   if 'num_files' in globals(): file_list = file_list[:num_files]
   if file_list==[]:
      print('ERROR: file list is empty?')
      print(f'file_path: {file_path}')
      print(f'file_list: {file_list}')
      exit()
   return file_list
#---------------------------------------------------------------------------------------------------
def get_data(ds,var):
   global comp_idx_list
   tvar = var
   if var=='qci'          : tvar = 'qc'
   if var=='horiz_winds_u': tvar = 'horiz_winds'
   if var=='horiz_winds_v': tvar = 'horiz_winds'
   data = ds[tvar]#.load()
   #----------------------------------------------------------------------------
   # if 'dim2' in data.dims: data = data.isel(dim2=?)
   #----------------------------------------------------------------------------
   if var=='horiz_winds_u': data = data.isel(dim2=0)
   if var=='horiz_winds_v': data = data.isel(dim2=1)
   if var=='qci'          : data += ds['qi'] 
   # if 'lev' in data.dims : data = interpolate_to_pressure(ds,data)
   return data
#---------------------------------------------------------------------------------------------------
# set up figure objects
subplot_kwargs = {}
figsize = (5*num_var,5)
fig, axs = plt.subplots(1,num_var, figsize=figsize, squeeze=False )
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# comp_list_list,lag_list_list = [],[]
for v in range(num_var):
   var_opts = var_opts_list[v]
   xvar = var_opts['xvar']
   yvar = var_opts['yvar']
   print(' '*2+f'vars: {hapy.tclr.GREEN}{xvar}{hapy.tclr.END} / {hapy.tclr.GREEN}{yvar}{hapy.tclr.END}')
   #----------------------------------------------------------------------------
   x_list = []
   y_list = []
   #----------------------------------------------------------------------------
   for c in range(num_case):
      case_opts = case_opts_list[c]
      case = case_opts['case']
      tmp_file = f'{tmp_head}.{case}.{xvar}.{yvar}.f0_{first_file}.nf_{num_files}.nc'
      # print(' '*4+f'case: {hapy.tclr.CYAN}{case}{hapy.tclr.END}')
      print(' '*4+f'case: {hapy.tclr.CYAN}{case}{hapy.tclr.END}  =>  {tmp_file}')
      # if recalculate:
      if True:
         #----------------------------------------------------------------------
         htype = var_opts['htype']
         file_list = get_file_list(case,case_opts['root'],case_opts['sub'],htype)
         ds = ux.open_mfdataset(case_opts['grid_file'], file_list, data_vars='all')
         data_x = get_data(ds,xvar)
         data_y = get_data(ds,yvar)
         area   = get_data(ds,'area').isel(time=0,missing_dims='ignore')
         #----------------------------------------------------------------------
         # print(); print(data_x)
         # print(); print(data_y)
         # print(); print(area)
         # exit()
         #----------------------------------------------------------------------
         data_x = data_x.mean(dim='time', skipna=True)
         data_y = data_y.mean(dim='time', skipna=True)
         #----------------------------------------------------------------------
         data_x = ( (data_x*area).sum(dim='n_face') / area.sum(dim='n_face') )
         data_y = ( (data_y*area).sum(dim='n_face') / area.sum(dim='n_face') )
         # #----------------------------------------------------------------------
         # if 'lat1' in locals() or 'lon1' in locals():
         #    lat = get_data(ds,'lat').isel(time=0,missing_dims='ignore')
         #    lon = get_data(ds,'lon').isel(time=0,missing_dims='ignore')
         #    ncol = ds['ncol']
         #    mask = xr.DataArray( np.ones([len(ncol)],dtype=bool), coords=[('ncol', ncol.values)], dims='ncol' )
         #    if 'lat1' in locals(): mask = mask & (lat>=lat1) & (lat<=lat2)
         #    if 'lon1' in locals(): mask = mask & (lon>=lon1) & (lon<=lon2)
         #    mask.load()
         #    data_x = data_x.where(mask,drop=True)
         #    data_y = data_y.where(mask,drop=True)
         #    # area = area.where(mask,drop=True)
         #----------------------------------------------------------------------
         #----------------------------------------------------------------------
      #    #----------------------------------------------------------------------
      #    # Write to file 
      #    if os.path.isfile(tmp_file) : os.remove(tmp_file)
      #    tmp_ds = xr.Dataset()
      #    tmp_ds[var[v]] = data
      #    tmp_ds.to_netcdf(path=tmp_file,mode='w')
      # else:
      #    tmp_ds = xr.open_dataset( tmp_file )
      #    data = tmp_ds[var[v]]
      #-------------------------------------------------------------------------
      # adjust units
      # if unit_fac_list[v] is not None: comp = comp*unit_fac_list[v]
      #-------------------------------------------------------------------------
      # if print_stats: hapy.print_stat(data,compact=True)
      #-------------------------------------------------------------------------
      x_list.append(data_x)
      y_list.append(data_y)
   #----------------------------------------------------------------------------
   # Create plot

   ip = v
   
   x_min = np.min([np.nanmin(d) for d in x_list])
   x_max = np.max([np.nanmax(d) for d in x_list])
   y_min = np.min([np.nanmin(d) for d in y_list])
   y_max = np.max([np.nanmax(d) for d in y_list])

   x_mag = np.abs(x_max-x_min)
   y_mag = np.abs(y_max-y_min)

   extent = [None]*4
   extent[0] = x_min - x_mag*0.02
   extent[1] = x_max + x_mag*0.02
   extent[2] = y_min - y_mag*0.02
   extent[3] = y_max + y_mag*0.02

   #----------------------------------------------------------------------------
   ax = axs[0,v]

   # ax.scatter(x_list, y_list, color='blue', alpha=0.6, s=100)

   for c in range(num_case):
      ax.scatter(x_list[c], y_list[c], color=case_opts_list[c]['clr'], alpha=0.6, s=100)

   # Add labels and title
   ax.set_xlabel(var_opts['xvar_name'], fontsize=12)
   ax.set_ylabel(var_opts['yvar_name'], fontsize=12)
   # ax.set_title('Scatter Plot', fontsize=14, fontweight='bold')

   # #----------------------------------------------------------------------------
   # for c in range(num_case):
   #    tres.tiXAxisString = var1_list[v]
   #    tres.tiYAxisString = var2_list[v]
   #    tres.xyMarkerColor = clr[c]
   #    tres.xyMarker      = mrk[c]

   #    px = np.ma.masked_invalid( x_list[c].stack().values )
   #    py = np.ma.masked_invalid( y_list[c].stack().values )
      
   #    tplot = ngl.xy(wks, px, py, tres)
      
   #    if c==0:
   #       plot[v] = tplot
   #    else:
   #       ngl.overlay( plot[v], tplot )
   #    #-------------------------------------------------------------------------
   #    # add linear fit

   #    # simple and fast method for regression coeff and intercept value
   #    a = np.cov( px.flatten(), py.flatten() )[1,0] / np.var( px )
   #    b = np.nanmean(py) - a*np.nanmean(px)

   #    print()
   #    print(f'    case: {case}')
   #    print(f'    linear regression a: {a}    b: {b}\n')

   #    case_opts['clr']
   #    case_opts['dsh']

   #    px_range = np.abs( np.max(px) - np.min(px) )
   #    lx = np.array([-1e2*px_range,1e2*px_range])

   #    # tplot = ngl.xy(wks, lx, lx*a+b , tres)
   #    # ngl.overlay( plot[v], tplot )

#-------------------------------------------------------------------------------
# Add legend?

#---------------------------------------------------------------------------------------------------
# Finalize plot
fig.savefig(fig_file, dpi=100, bbox_inches='tight')
plt.close(fig)
print(f'\n{fig_file}\n')
#---------------------------------------------------------------------------------------------------
