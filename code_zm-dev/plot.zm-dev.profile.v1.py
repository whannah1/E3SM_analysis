import os, glob, copy, xarray as xr, numpy as np, cmocean
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import hapy
# host = hapy.get_host()
xr.set_options(use_new_combine_kwarg_defaults=True)
#-------------------------------------------------------------------------------
'''
time python -i  code_zm-dev/plot.zm-dev.profile.v1.py
exec(open('code_zm-dev/plot.zm-dev.profile.v1.py').read())
'''
#-------------------------------------------------------------------------------
case,case_opts_list = [],[]
def add_case(case_in,**kwargs):
   case.append(case_in)
   case_opts = {}
   for k, val in kwargs.items(): case_opts[k] = val
   case_opts_list.append(case_opts)
#---------------------------------------------------------------------------------------------------
var,var_opts_list = [],[]
def add_var(var_name,**kwargs):
   var.append(var_name)
   var_opts = {}
   for k, val in kwargs.items(): var_opts[k] = val
   if 'str' not in var_opts: var_opts['str'] = var_name
   var_opts_list.append(var_opts)
#---------------------------------------------------------------------------------------------------

tmp_scratch = '/lcrc/group/e3sm/ac.whannah/scratch/chrys'
add_case('E3SM.2026-ZM-MOD-00.F2010.ne30pg2.NN_8', n='E3SM', p=tmp_scratch, s='run')
first_file,num_files = 8,1

#---------------------------------------------------------------------------------------------------

htype_tmp = 'eam.h1'

add_var('ZM_ENTR_UP',htype=htype_tmp)
add_var('ZM_DETR_UP',htype=htype_tmp)
# add_var('ZM_ENTR_DN',htype=htype_tmp)
add_var('ZMMU',      htype=htype_tmp)
# add_var('ZMMD',      htype=htype_tmp)
add_var('ZMDT',      htype=htype_tmp)
# add_var('ZMDQ',      htype=htype_tmp)


#---------------------------------------------------------------------------------------------------

fig_file = 'figs_zm-dev/zm-dev.profile.v1.png'
tmp_file_head = 'data_temp/zm-dev.profile.v2'

recalculate = True

# plot_diff   = False
# use_height  = False
print_stats = True
var_x_case = True
# num_plot_col = 2

# ncol_chunk_size = int(100e3)

#---------------------------------------------------------------------------------------------------
# set up figure objects
num_var,num_case = len(var),len(case)
fdx,fdy=20,10;figsize = (fdx*num_case,fdy*num_var) if var_x_case else (fdx*num_var,fdy*num_case)
title_fontsize,lable_fontsize = 20,18
(d1,d2) = (num_var,num_case) if var_x_case else (num_case,num_var)
if 'num_plot_col' in locals():
   (d1,d2) = ( int(np.ceil((num_var*num_case)/float(num_plot_col))), num_plot_col )
fig, axs = plt.subplots( d1, d2, figsize=figsize, squeeze=False )
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def get_file_list(case,case_opts,var_opts):
   case_dir = case_opts['p']
   case_sub = case_opts['s']
   htype    = var_opts['htype']
   #----------------------------------------------------------------------------
   file_path = f'{case_dir}/{case}/{case_sub}/*{htype}*'
   file_list = sorted(glob.glob(file_path))
   if 'first_file' in globals():file_list = file_list[first_file:]
   if 'num_files' in globals():
      if num_files>0: file_list = file_list[:num_files]
   #----------------------------------------------------------------------------
   if file_list==[]:
      print('ERROR: file list is empty?')
      print(f'file_path: {file_path}')
      print(f'file_list: {file_list}'); exit()
   #----------------------------------------------------------------------------
   return file_list
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
data_list_list,lev_list_list = [],[]
amax_list = [None]*num_case
for v in range(num_var):
   var_opts = var_opts_list[v]
   hapy.print_line()
   print(' '*2+f'var: {hapy.tclr.GREEN}{var[v]}{hapy.tclr.END}')
   #----------------------------------------------------------------------------
   data_list = []
   time_list = []
   lev_list = []
   #----------------------------------------------------------------------------
   for c in range(num_case):
      case_opts = case_opts_list[c]
      # print(' '*4+f'case: {hapy.tclr.CYAN}{case[c]}{hapy.tclr.END}')
      tmp_file = f'{tmp_file_head}.{case[c]}.{var[v]}.f0_{first_file}.nf_{num_files}.nc'
      print(' '*4+f'case: {hapy.tclr.CYAN}{case[c]}{hapy.tclr.END}  =>  {tmp_file}')
      #-------------------------------------------------------------------------
      if recalculate:
      # if True:
         #----------------------------------------------------------------------
         file_list = get_file_list(case[c],case_opts,var_opts)
         ds = xr.open_mfdataset( file_list )
         # ds = ds.mean(dim='time', skipna=True)
         #----------------------------------------------------------------------
         # lat  = get_data(ds,'lat').isel(time=0,missing_dims='ignore')
         # area = get_data(ds,'area').isel(time=0,missing_dims='ignore')
         data = ds[var[v]]
         #----------------------------------------------------------------------
         # for n in range(7):
         #    amax = data.argmax(dim=['time','lev','ncol'])
         #    data[amax['time'],amax['lev'],amax['ncol']] = 0.
         # data = data.isel(time=amax['time'],ncol=amax['ncol'])
         # amax = data.argmax()
         # data = data.isel(ncol=np.max(amax.values))
         if v==0 and c==0:
            # amax = ds['PRECZ'].var(dim='time').argmax(dim='ncol')
            # amax_list[c] = amax+2
            amax_list[c] = 4651
            lat_loc = ds.lat.isel(ncol=amax_list[c]).values
            lon_loc = ds.lon.isel(ncol=amax_list[c]).values
            print('\n'+' '*4+f'amax ncol/lat/lon : {amax_list[c]} / {lat_loc:5.2f} / {lon_loc:5.2f}\n')
            
         #----------------------------------------------------------------------
         data = data.isel(ncol=amax_list[c])
         #----------------------------------------------------------------------
         # if 'lat1' in locals() or 'lon1' in locals():
         #    lat = get_data(ds,'lat').isel(time=0,missing_dims='ignore')
         #    lon = get_data(ds,'lon').isel(time=0,missing_dims='ignore')
         #    ncol = ds['ncol']
         #    mask = xr.DataArray( np.ones([len(ncol)],dtype=bool), coords=[('ncol', ncol.values)], dims='ncol' )
         #    if 'lat1' in locals(): mask = mask & (lat>=lat1) & (lat<=lat2)
         #    if 'lon1' in locals(): mask = mask & (lon>=lon1) & (lon<=lon2)
         #    mask.load()
         #    data = data.where(mask,drop=True)
         #    area = area.where(mask,drop=True)
         #----------------------------------------------------------------------
         # data = ( (data*area).sum(dim='ncol') / area.sum(dim='ncol') )
         #----------------------------------------------------------------------
         # if use_height: 
         #    hght = ???
         #    hght = ( (hght*area).sum(dim='ncol') / area.sum(dim='ncol') )
         #----------------------------------------------------------------------
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
      if 'unit_fac' in var_opts:
         if var_opts['unit_fac']:
            data = data*var_opts['unit_fac']
      #-------------------------------------------------------------------------
      if print_stats: hapy.print_stat(data,compact=True)
      #-------------------------------------------------------------------------
      data_list.append( data.values )
      time_list.append( data['time'].values )
      lev_list.append( data['lev'].values )
      # if not use_height: lev_list.append( data['lev'].values )
      # if     use_height: lev_list.append( hght.values )
   #----------------------------------------------------------------------------
   # Create plot

   ip = v
   
   data_min = np.min([np.nanmin(d) for d in data_list])
   data_max = np.max([np.nanmax(d) for d in data_list])

   #----------------------------------------------------------------------------
   # print the profiles
   # print()
   # tmp_data = data_list[c][-48,:]
   # for i,d in enumerate(tmp_data):
   #    print(f'{i:3}  {d}')
   # print()

   #----------------------------------------------------------------------------
   # if plot_diff:
   #    baseline = copy.deepcopy(data_list[0])
   #    for c in range(num_case): 
   #       data_list[c] = data_list[c] - baseline
   #----------------------------------------------------------------------------
   # set color bar levels
   clev = None
   # if var[v]=='FRONTGF': clev = np.logspace( -5, -1, num=40)
   # if var[v] in ['PS','sp']: clev = np.linspace( 800e2, 1020e2, num=40)
   # if var[v] in ['PS','sp']: clev = np.arange(600e2,1040e2+2e2,10e2)

   if var[v]=='ZM_ENTR_UP': clev = np.logspace( np.log10(0.0001e-3), np.log10(1.0e-3), num=40)
   if var[v]=='ZM_DETR_UP': clev = np.logspace( np.log10(0.001e-3), np.log10(0.5e-3), num=40)
   if var[v]=='ZMDT'      : clev = np.linspace( -0.002, 0.002, num=51)

   #----------------------------------------------------------------------------
   # if var[v] in ['ZM_ENTR_UP','ZM_DETR_UP']: 
   #    for c in range(num_case):
   #       data_list[c] = np.absolute(data_list[c])
   if var[v]=='ZMDT':
      for c in range(num_case):
         # data_list[c] = np.ma.masked_invalid( data_list[c] )
         data_list[c] = np.ma.masked_where(data_list[c]==0, data_list[c])
   #----------------------------------------------------------------------------
   # set color map
   cmap = 'viridis'
   # cmap = cmocean.cm.rain
   if var[v]=='ZMDT': cmap = cmocean.cm.balance
   #----------------------------------------------------------------------------
   for c in range(num_case):
      #-------------------------------------------------------------------------
      img_kwargs = {}
      # img_kwargs['origin'] = 'lower'
      img_kwargs['cmap']   = cmap

      # if plot_diff and c!=diff_base:
      #    img_kwargs['cmap']   = cmocean.cm.balance
      #    img_kwargs['vmin']   = diff_data_min
      #    img_kwargs['vmax']   = diff_data_max
      #    clev = None

      if clev is not None: img_kwargs['norm'] = mcolors.BoundaryNorm(clev, ncolors=256)

      data = data_list[c]
      lev = lev_list[c]
      
      time_vals   = time_list[c]
      time_idx    = np.arange(len(time_vals))
      time_labels = [t.strftime('%m/%d %Hz') for t in time_vals]

      ax = axs[v,c] if var_x_case else axs[c,v]
      ax.set_title(case_opts_list[c]['n'],   fontsize=title_fontsize, loc='left')
      ax.set_title(var_opts_list[v]['str'],  fontsize=title_fontsize, loc='right')
      ax.set_xlabel('Time')
      ax.set_ylabel('Reference Pressure Level')
      ax.set_ylim(lev.max(), lev.min())   # invert: surface at bottom, TOA at top
      
      img = ax.pcolormesh( time_idx, lev, data.T, **img_kwargs)

      # ax.set_xticks(time_idx)
      # ax.set_xticklabels(time_labels, fontsize=10, rotation=45, ha='right')

      tick_stride = max(1, len(time_idx) // 20)
      ax.set_xticks(time_idx[::tick_stride])
      ax.set_xticklabels(time_labels[::tick_stride], rotation=45, ha='right')

      cbar = fig.colorbar(img, ax=ax, fraction=0.02, orientation='vertical')
      cbar.ax.tick_params(labelsize=lable_fontsize)

#-------------------------------------------------------------------------------
# Add legend

#---------------------------------------------------------------------------------------------------
# Finalize plot
fig.savefig(fig_file, dpi=100, bbox_inches='tight')
plt.close(fig)
print(f'\n{fig_file}\n')
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
