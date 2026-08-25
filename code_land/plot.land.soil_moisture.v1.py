import os, glob, numpy as np, xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import hapy
host = hapy.get_host()
#--------------------------------------------------------------------------------------------------- 
# plots area-weighted soil moisture (or soil temperature) for each soil level vs time
# to gauge land spin-up state, with optional regional subsetting
#--------------------------------------------------------------------------------------------------- 
case, opts_list = [], []
def add_case(case_in, n=None, p=None, s=None, htype='elm.h0', comp='elm', grid_file=None, **kwargs):
   case.append(case_in)
   case_opts = {}
   case_opts['n']         = case_in if n is None else n
   case_opts['p']         = p
   case_opts['s']         = s
   case_opts['htype']     = htype
   case_opts['comp']      = comp
   case_opts['grid_file'] = grid_file
   for k, val in kwargs.items(): case_opts[k] = val
   opts_list.append(case_opts)

var_list, var_str_list = [], []
def add_var(var_name, var_str=None):
   var_list.append(var_name)
   var_str_list.append(var_name if var_str is None else var_str)
#--------------------------------------------------------------------------------------------------- 
if host=='lcrc':
   tmp_scratch = '/lcrc/group/e3sm/ac.whannah/scratch/chrys'

#-------------------------------------------------------------------------------
if host=='llnl':
   tmp_scratch = '/p/lustre2/hannah6/e3sm_scratch'
   add_case('ELM_spinup.ICRUELM.ne0np4-2024-nimbus-iraq-32x3-pg2.20-yr.2015-10-01' ,n='iraq-32x3', p=tmp_scratch,s='run')
   add_case('ELM_spinup.ICRUELM.ne0np4-2024-nimbus-iraq-64x3-pg2.20-yr.2015-10-01' ,n='iraq-64x3', p=tmp_scratch,s='run')
   add_case('ELM_spinup.ICRUELM.ne0np4-2024-nimbus-iraq-128x3-pg2.20-yr.2015-10-01',n='iraq-128x3',p=tmp_scratch,s='run')

#-------------------------------------------------------------------------------
if host=='olcf':
   tmp_scratch = '/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch'
   tmp_grid    = '/lustre/orion/cli115/world-shared/e3sm/2026-INCITE-CONUS-RRM/files_grid/2026-incite-conus-1024x2-pg2_scrip.nc'
   add_case('ELM_spinup.2026-INCITE-CONUS-RRM.IERA5ELM.conus1024x2v1pg2_RRSwISC6to18E3r5',
            n='1024x2 ELM spinup',p=tmp_scratch,s='run',htype='elm.r.',grid_file=tmp_grid)
   # first_file,num_files = 0,12*2

'''
ls -1 /lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch/ELM_spinup.2026-INCITE-CONUS-RRM.IERA5ELM.conus1024x2v1pg2_RRSwISC6to18E3r5/run/*elm.r.*
'''
#-------------------------------------------------------------------------------

add_var('ZWT')
# add_var('H2OSOI_LIQ')
# add_var('H2OSOI')
# add_var('TSOI')

fig_file = 'figs_land/land.soil_moisture.v1.png'

#--------------------------------------------------------------------------------------------------- 
# optional regional subset
# set use_region_subset=False to use the full global domain
# longitudes use the 0-360 convention (e.g. California is ~235-246)
#--------------------------------------------------------------------------------------------------- 
use_region_subset = False

### California
# lat1,lat2 = 32.5, 42.0
# lon1,lon2 = 360-124.5, 360-114.0    # 235.5, 246.0

### other regions
# lat1,lat2,lon1,lon2 = 25, 50, 360-125, 360-75    # CONUS
# lat1,lat2,lon1,lon2 = -45, 15, 360-90, 360-30    # S. America
# clat,clon, dx = 36, -97, 2.5                      # ARM SGP
# lat1,lat2,lon1,lon2 = clat-dx,clat+dx,360+clon-dx,360+clon+dx
#--------------------------------------------------------------------------------------------------- 

# first_file,num_files = 0,12*20

print_stats = False

convert_to_annual_mean = False
#--------------------------------------------------------------------------------------------------- 
num_var  = len(var_list)
num_case = len(case)

### colormap used to color lines by soil level (shallow -> deep)
line_cmap = plt.get_cmap('viridis')

### day of year at the start of each month for the noleap calendar used by ELM
noleap_month_start = np.cumsum([0,31,28,31,30,31,30,31,31,30,31,30])
#--------------------------------------------------------------------------------------------------- 
def get_file_list(case, case_dir, case_sub, htype):
   file_path = f'{case_dir}/{case}/{case_sub}/*{htype}*'
   file_list = sorted(glob.glob(file_path))
   if 'first_file' in globals(): file_list = file_list[first_file:]
   if 'num_files'  in globals(): file_list = file_list[:num_files]
   #----------------------------------------------------------------------------
   # print('WARNING - non-standard file list truncation! - WARNING')
   # file_list = file_list[::30]
   #----------------------------------------------------------------------------
   if file_list==[]:
      print('ERROR: file list is empty?')
      print(f'file_path: {file_path}')
      exit()
   print()
   for f in file_list: print(f'  {f}')
   print()
   return file_list
#--------------------------------------------------------------------------------------------------- 
def load_history_data(file_list, var_name):
   """ area weighted spatial mean of an ELM history file variable
       returns time [years since start], data of shape (time,lev), level values, level label """
   ds = xr.open_mfdataset(file_list, data_vars='minimal')

   data = ds[var_name]
   area = ds['area'].astype(np.double)
   if 'time' in area.dims: area = area.isel(time=0, drop=True)

   ### use a consistent spatial dimension name (ELM unstructured grid uses lndgrid)
   if 'lndgrid' in data.dims: data = data.rename({'lndgrid':'ncol'})
   if 'lndgrid' in area.dims: area = area.rename({'lndgrid':'ncol'})

   ### use a consistent vertical dimension name
   if 'levgrnd' in data.dims: data = data.rename({'levgrnd':'lev'})
   if 'levtot'  in data.dims: data = data.rename({'levtot':'lev'})

   #----------------------------------------------------------------
   # optionally subset to a region using the 1D lat/lon coordinates
   if use_region_subset:
      lat, lon = ds['lat'], ds['lon']
      if 'lndgrid' in lat.dims: lat = lat.rename({'lndgrid':'ncol'})
      if 'lndgrid' in lon.dims: lon = lon.rename({'lndgrid':'ncol'})
      mask = (lat>=lat1) & (lat<=lat2) & (lon>=lon1) & (lon<=lon2)
      mask.load()
      data = data.where(mask, drop=True)
      area = area.where(mask, drop=True)

   if print_stats: hapy.print_stat(data,name=var_name,stat='naxsh',indent='    ',compact=True)

   #----------------------------------------------------------------
   # area weighted spatial mean
   if 'ncol' in data.dims:
      gbl_data = ( (data*area).sum(dim='ncol') / area.sum(dim='ncol') )
   elif 'lat' in data.dims:
      gbl_data = ( (data*area).sum(dim=('lat','lon')) / area.sum(dim=('lat','lon')) )
   gbl_data = gbl_data.compute()

   if convert_to_annual_mean:
      gbl_data = gbl_data.resample(time='YE').mean(dim='time')

   ### variables without soil levels get a single level so the plotting is uniform
   if 'lev' not in gbl_data.dims: gbl_data = gbl_data.expand_dims('lev',axis=-1)
   gbl_data = gbl_data.transpose('time','lev')

   ### time in years relative to the start of the run
   time = ( gbl_data['time'] - gbl_data['time'][0] ).astype('float') / 86400e9 / 365

   if 'lev' in gbl_data.coords:
      lev_vals, lev_label = gbl_data['lev'].values, 'soil layer depth [m]'
   else:
      lev_vals, lev_label = np.arange(gbl_data.sizes['lev'],dtype=np.double), 'soil layer index'

   return time.values, gbl_data.values, lev_vals, lev_label
#--------------------------------------------------------------------------------------------------- 
def get_restart_file_date(file_name):
   """ parse the date from an ELM restart file name: *.elm.r.YYYY-MM-DD-SSSSS.nc """
   yr,mn,dy,sec = os.path.basename(file_name).split('.elm.r.')[-1].replace('.nc','').split('-')
   return int(yr),int(mn),int(dy),int(sec)
#--------------------------------------------------------------------------------------------------- 
def get_restart_column_weight(ds, grid_file):
   """ area weight of each ELM column = its weight within the gridcell * the gridcell area
       restart files carry no area variable, so gridcell area is taken from the SCRIP grid file
       and mapped through the sparse gridcell->grid and column->gridcell indices """
   if grid_file is None:
      print('ERROR: reading elm.r files requires a SCRIP grid_file to supply the gridcell area')
      exit()
   ### these index variables are integers but decode to float due to the fill value attributes
   wgt_gridcell = ds['cols1d_wtxy'].values
   col_gridcell = ds['cols1d_gridcell_index'].values.astype(np.int64)   # 1-based gridcell of each column
   grid_index   = ds['grid1d_ixy'].values.astype(np.int64)              # 1-based SCRIP index of each gridcell

   grid_area = xr.open_dataset(grid_file)['grid_area'].values
   wgt = wgt_gridcell * grid_area[grid_index-1][col_gridcell-1]

   #----------------------------------------------------------------
   # optionally subset to a region using the 1D column lat/lon
   if use_region_subset:
      lat, lon = ds['cols1d_lat'].values, ds['cols1d_lon'].values
      wgt = np.where( (lat>=lat1)&(lat<=lat2)&(lon>=lon1)&(lon<=lon2), wgt, 0. )

   return np.where( np.isfinite(wgt), wgt, 0. )
#--------------------------------------------------------------------------------------------------- 
def load_restart_data(file_list, var_name, grid_file):
   """ ELM restart files hold a single time each with no area or time coordinate, so each file
       is reduced to an area weighted mean here and the results are stacked along time
       returns time [years since start], data of shape (time,lev), level values, level label """
   wgt, year_list, time_list, data_list = None, [], [], []
   for f in file_list:
      print(f'      reading {os.path.basename(f)} ...',flush=True)
      ds = xr.open_dataset(f, decode_timedelta=False)

      ### the column weights are static, so only build them once
      if wgt is None: wgt = get_restart_column_weight(ds, grid_file)

      data = ds[var_name].values
      ### treat everything as (column,lev) - column variables without levels get a single level
      if data.ndim==1: data = data[:,None]

      if print_stats: hapy.print_stat(data,name=var_name,stat='naxsh',indent='    ',compact=True)

      ### inactive columns (lake, urban, glacier, ...) are filled and drop out of the mean
      gbl_data = np.full(data.shape[1],np.nan)
      for k in range(data.shape[1]):
         msk = np.isfinite(data[:,k]) & (wgt>0)
         if np.any(msk): gbl_data[k] = np.sum(wgt[msk]*data[msk,k]) / np.sum(wgt[msk])

      yr,mn,dy,sec = get_restart_file_date(f)
      year_list.append(yr)
      time_list.append( yr + ( noleap_month_start[mn-1] + (dy-1) + sec/86400. )/365. )
      data_list.append(gbl_data)
      ds.close()

   time = np.array(time_list)
   data = np.array(data_list)

   if convert_to_annual_mean:
      year = np.array(year_list)
      year_vals = np.unique(year)
      data = np.array([ np.nanmean(data[year==y,:],axis=0) for y in year_vals ])
      time = np.array([ np.mean(  time[year==y]          ) for y in year_vals ])

   ### time in years relative to the start of the run
   time = time - time[0]

   ### restart files provide no soil level depths, so color the lines by level index
   return time, data, np.arange(data.shape[1],dtype=np.double), 'soil layer index'
#--------------------------------------------------------------------------------------------------- 
# Load data
#--------------------------------------------------------------------------------------------------- 
# nested lists indexed by [var][case]
all_time,all_data,all_levs,all_maxlev,all_levlab = [],[],[],[],[]

for v in range(num_var):
   hapy.print_line()
   print(f'  var: {hapy.tclr.MAGENTA}{var_list[v]}{hapy.tclr.END}')
   time_list,data_list,lev_vals_list = [],[],[]
   max_lev, lev_label = None, ''
   for c in range(num_case):
      case_opts = opts_list[c]
      print(f'    case: {hapy.tclr.CYAN}{case[c]}{hapy.tclr.END}')

      #----------------------------------------------------------------
      # read the data and reduce it to an area weighted mean of shape (time,lev)
      file_list = get_file_list(case[c], case_opts['p'], case_opts['s'], case_opts['htype'])
      if 'elm.r' in case_opts['htype']:
         time,gbl_data,lev_vals,lev_label = load_restart_data(file_list, var_list[v], case_opts['grid_file'])
      else:
         time,gbl_data,lev_vals,lev_label = load_history_data(file_list, var_list[v])

      #----------------------------------------------------------------
      # determine how many soil levels to plot (set once from first case)
      # for soil moisture omit the deepest near-zero levels
      if c==0:
         max_lev = gbl_data.shape[1]
         if var_list[v] in ['H2OSOI']:
            for k in range(1,gbl_data.shape[1]) :
               if np.all( gbl_data[:,k] < 0.001 ) : continue
               max_lev = k+1

      data_list.append( gbl_data[:,:max_lev] )
      time_list.append( time )
      lev_vals_list.append( lev_vals[:max_lev] )

   all_time.append(time_list)
   all_data.append(data_list)
   all_levs.append(lev_vals_list)
   all_maxlev.append(max_lev)
   all_levlab.append(lev_label)
#--------------------------------------------------------------------------------------------------- 
# Create plot
#--------------------------------------------------------------------------------------------------- 
### panel layout - rows = variables, cols = cases (2x2 special case for 4 cases)
nrows,ncols = num_var,num_case
if num_var==1 and num_case==4 : nrows,ncols = 2,2

dpi = 150
fig_w,fig_h = 7*ncols, 4*nrows
fig, axs = plt.subplots(nrows, ncols, figsize=(fig_w,fig_h), dpi=dpi, squeeze=False)
fig.subplots_adjust(wspace=0.25, hspace=0.30)

var_unit = {'H2OSOI':'mm3/mm3', 'TSOI':'K', 'ZWT':'m', 'H2OSOI_LIQ':'kg/m2', 'T_SOISNO':'K'}

for v in range(num_var):
   time_list = all_time[v]
   data_list = all_data[v]
   lev_vals_list = all_levs[v]
   max_lev = all_maxlev[v]

   # ----------------------------------------------------------------
   # common axis limits across cases - pad degenerate limits so the axis is valid
   ymin = np.min([np.nanmin(d) for d in data_list])
   ymax = np.max([np.nanmax(d) for d in data_list])
   xmin = np.min([np.nanmin(t) for t in time_list])
   xmax = np.max([np.nanmax(t) for t in time_list])
   if ymin==ymax : ymin,ymax = ymin-0.5,ymax+0.5
   if xmin==xmax : xmin,xmax = xmin-0.5,xmax+0.5

   # ----------------------------------------------------------------
   # color lines by soil layer depth (shallow -> deep)
   depth = lev_vals_list[0]
   dmin,dmax = float(np.nanmin(depth)),float(np.nanmax(depth))
   if dmin==dmax : dmin,dmax = dmin-0.5,dmax+0.5
   norm = mcolors.Normalize(vmin=dmin, vmax=dmax)

   for c in range(num_case):
      ### default layout is rows=variables, cols=cases
      ### the 2x2 special case (1 var, 4 cases) wraps cases across the grid
      if num_var==1 and num_case==4: row,col = c//ncols, c%ncols
      else:                          row,col = v, c
      ax = axs[row][col]

      for k in range(max_lev):
         ax.plot( time_list[c], data_list[c][:,k],
                  color=line_cmap(norm(lev_vals_list[c][k])), linewidth=2.0 )

      ax.set_xlim(xmin,xmax)
      ax.set_ylim(ymin,ymax)
      ax.set_xlabel('Time [years]', fontsize=12)
      ax.tick_params(labelsize=10)
      ax.grid(alpha=0.3)

      ax.set_title(opts_list[c]['n'], loc='left',  fontsize=14)
      ax.set_title(var_str_list[v],   loc='right', fontsize=14)

      # ----------------------------------------------------------------
      # colorbar indicating soil layer depth - only useful with multiple levels
      if max_lev>1:
         sm = cm.ScalarMappable(cmap=line_cmap, norm=norm); sm.set_array([])
         cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
         cbar.set_label(all_levlab[v], fontsize=10)
         cbar.ax.tick_params(labelsize=9)

      unit = var_unit.get(var_list[v],'')
      if unit!='': ax.set_ylabel(f'{var_str_list[v]} [{unit}]', fontsize=12)
      else:        ax.set_ylabel(var_str_list[v], fontsize=12)

   # ----------------------------------------------------------------
   # hide any unused panels (e.g. 2x2 layout with <4 cases)
   for ip in range(num_case, nrows*ncols):
      axs[ip//ncols][ip%ncols].set_visible(False)
#--------------------------------------------------------------------------------------------------- 
# Finalize plot
#--------------------------------------------------------------------------------------------------- 
os.makedirs(os.path.dirname(fig_file) or '.', exist_ok=True)
fig.savefig(fig_file, dpi=dpi, bbox_inches='tight')
plt.close(fig)

print(f'\n{fig_file}\n')
#--------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------
