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
def add_case(case_in, n=None, p=None, s=None, htype='elm.h0', comp='elm', **kwargs):
   case.append(case_in)
   case_opts = {}
   case_opts['n']     = case_in if n is None else n
   case_opts['p']     = p
   case_opts['s']     = s
   case_opts['htype'] = htype
   case_opts['comp']  = comp
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

add_var('H2OSOI')
# add_var('TSOI')

fig_file = 'figs_land/land.soil_moisture.v1.png'

#--------------------------------------------------------------------------------------------------- 
# optional regional subset
# set use_region_subset=False to use the full global domain
# longitudes use the 0-360 convention (e.g. California is ~235-246)
#--------------------------------------------------------------------------------------------------- 
use_region_subset = True

### California
lat1,lat2 = 32.5, 42.0
lon1,lon2 = 360-124.5, 360-114.0    # 235.5, 246.0

### other regions
# lat1,lat2,lon1,lon2 = 25, 50, 360-125, 360-75    # CONUS
# lat1,lat2,lon1,lon2 = -45, 15, 360-90, 360-30    # S. America
# clat,clon, dx = 36, -97, 2.5                      # ARM SGP
# lat1,lat2,lon1,lon2 = clat-dx,clat+dx,360+clon-dx,360+clon+dx
#--------------------------------------------------------------------------------------------------- 

first_file,num_files = 0,12*20

print_stats = False

convert_to_annual_mean = True
#--------------------------------------------------------------------------------------------------- 
num_var  = len(var_list)
num_case = len(case)

### colormap used to color lines by soil level (shallow -> deep)
line_cmap = plt.get_cmap('viridis')
#--------------------------------------------------------------------------------------------------- 
def get_file_list(case, case_dir, case_sub, htype):
   file_path = f'{case_dir}/{case}/{case_sub}/*{htype}*'
   file_list = sorted(glob.glob(file_path))
   if 'first_file' in globals(): file_list = file_list[first_file:]
   if 'num_files'  in globals(): file_list = file_list[:num_files]
   if file_list==[]:
      print('ERROR: file list is empty?')
      print(f'file_path: {file_path}')
      exit()
   return file_list
#--------------------------------------------------------------------------------------------------- 
# Load data
#--------------------------------------------------------------------------------------------------- 
# nested lists indexed by [var][case]
all_time,all_data,all_levs,all_maxlev = [],[],[],[]

for v in range(num_var):
   hapy.print_line()
   print(f'  var: {hapy.tclr.MAGENTA}{var_list[v]}{hapy.tclr.END}')
   time_list,data_list,lev_vals_list = [],[],[]
   max_lev = None
   for c in range(num_case):
      case_opts = opts_list[c]
      print(f'    case: {hapy.tclr.CYAN}{case[c]}{hapy.tclr.END}')

      #----------------------------------------------------------------
      # read the data
      file_list = get_file_list(case[c], case_opts['p'], case_opts['s'], case_opts['htype'])
      ds = xr.open_mfdataset(file_list, data_vars='minimal')

      data = ds[var_list[v]]
      area = ds['area'].astype(np.double)
      if 'time' in area.dims: area = area.isel(time=0, drop=True)

      ### use a consistent spatial dimension name (ELM unstructured grid uses lndgrid)
      if 'lndgrid' in data.dims:
         data = data.rename({'lndgrid':'ncol'})
         area = area.rename({'lndgrid':'ncol'})

      data = data.rename({'levgrnd':'lev'})

      #----------------------------------------------------------------
      # optionally subset to a region using the 1D lat/lon coordinates
      if use_region_subset:
         lat = ds['lat']
         lon = ds['lon']
         if 'lndgrid' in lat.dims: lat = lat.rename({'lndgrid':'ncol'})
         if 'lndgrid' in lon.dims: lon = lon.rename({'lndgrid':'ncol'})
         mask = (lat>=lat1) & (lat<=lat2) & (lon>=lon1) & (lon<=lon2)
         mask.load()
         data = data.where(mask, drop=True)
         area = area.where(mask, drop=True)

      if print_stats: hapy.print_stat(data,name=var_list[v],stat='naxsh',indent='    ',compact=True)

      #----------------------------------------------------------------
      # area weighted spatial mean
      if 'ncol' in data.dims:
         gbl_data = ( (data*area).sum(dim='ncol') / area.sum(dim='ncol') )
      elif 'lat' in data.dims:
         gbl_data = ( (data*area).sum(dim=('lat','lon')) / area.sum(dim=('lat','lon')) )
      gbl_data = gbl_data.compute()

      if convert_to_annual_mean:
         gbl_data = gbl_data.resample(time='YE').mean(dim='time')

      #----------------------------------------------------------------
      # determine how many soil levels to plot (set once from first case)
      # for soil moisture omit the deepest near-zero levels
      if c==0:
         max_lev = len(gbl_data.lev)
         if var_list[v] in ['H2OSOI']:
            max_lev = 0
            for k in range(1,len(gbl_data.lev)) :
               if np.all( gbl_data[:,k].values < 0.001 ) : continue
               max_lev = k

      #----------------------------------------------------------------
      # time in years relative to the start of the run
      time = ( gbl_data['time'] - gbl_data['time'][0] ).astype('float') / 86400e9 / 365

      data_list.append( gbl_data[:,:max_lev].values )
      time_list.append( time.values )
      lev_vals_list.append( gbl_data['lev'][:max_lev].values )

   all_time.append(time_list)
   all_data.append(data_list)
   all_levs.append(lev_vals_list)
   all_maxlev.append(max_lev)
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

var_unit = {'H2OSOI':'mm3/mm3', 'TSOI':'K'}

for v in range(num_var):
   time_list = all_time[v]
   data_list = all_data[v]
   lev_vals_list = all_levs[v]
   max_lev = all_maxlev[v]

   # ----------------------------------------------------------------
   # common axis limits across cases
   ymin = np.min([np.nanmin(d) for d in data_list])
   ymax = np.max([np.nanmax(d) for d in data_list])
   xmin = np.min([np.nanmin(t) for t in time_list])
   xmax = np.max([np.nanmax(t) for t in time_list])

   # ----------------------------------------------------------------
   # color lines by soil layer depth (shallow -> deep)
   depth = lev_vals_list[0]
   norm = mcolors.Normalize(vmin=float(np.nanmin(depth)), vmax=float(np.nanmax(depth)))

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
      # colorbar indicating soil layer depth
      sm = cm.ScalarMappable(cmap=line_cmap, norm=norm); sm.set_array([])
      cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
      cbar.set_label('soil layer depth [m]', fontsize=10)
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
fig.savefig(fig_file, dpi=dpi, bbox_inches='tight')
plt.close(fig)

print(f'\n{fig_file}\n')
#--------------------------------------------------------------------------------------------------- 
#--------------------------------------------------------------------------------------------------- 
