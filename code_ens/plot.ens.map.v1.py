#---------------------------------------------------------------------------------------------------
# Plot lat/lon maps for E3SM "multi-instance" ensembles (i.e. NINST>1, where each
# ensemble member writes its own history files with an instance tag like "_0001")
# ensemble members are treated like the case list in code_clim/plot.clim.map.v1.py
#---------------------------------------------------------------------------------------------------
from time import perf_counter
timer_start = perf_counter()
import os, subprocess as sp, numpy as np, xarray as xr, dask, copy, string, cmocean, glob
import cartopy.crs as ccrs, matplotlib.pyplot as plt, matplotlib.colors as mcolors
import hapy
host = hapy.get_host()
xr.set_options(use_new_combine_kwarg_defaults=True)
#---------------------------------------------------------------------------------------------------
name,case,case_dir,case_sub,inst_list = [],[],[],[],[]
scrip_file_list = []
def add_case(case_in,n=None,p=None,s=None,inst=None,scrip_file=None):
    global name,case,case_dir,case_sub,inst_list
    tmp_name = case_in if n is None else n
    case.append(case_in); name.append(tmp_name);
    case_dir.append(p); case_sub.append(s); inst_list.append(inst)
    if scrip_file is None: raise ValueError('scrip_file cannot be None')
    scrip_file_list.append(scrip_file)
def add_ens(case_in,num_inst,n=None,p=None,s=None,scrip_file=None,inst_start=1):
    # add each instance of a multi-instance ensemble as a separate "case"
    tmp_name = case_in if n is None else n
    for i in range(inst_start,inst_start+num_inst):
        add_case(case_in,n=f'{tmp_name} #{i:04d}',p=p,s=s,inst=f'{i:04d}',scrip_file=scrip_file)
#---------------------------------------------------------------------------------------------------
var,lev_list,var_str = [],[],[]
var_opts_list = []
def add_var(var_name,lev=None,s=None,**kwargs):
    var.append(var_name); lev_list.append(lev);
    var_str.append(var_name if s is None else s)
    var_opts = {}
    for k, val in kwargs.items(): var_opts[k] = val
    var_opts_list.append(var_opts)
#---------------------------------------------------------------------------------------------------
fig_file = 'figs_ens/ens.map.v1.png'
#---------------------------------------------------------------------------------------------------
# htype is a file name pattern where '{inst}' is replaced by the instance number, e.g.
#   EAMxx : htype = 'output.1da_{inst}.AVERAGE'
#   EAM   : htype = 'eam_{inst}.h0'
#---------------------------------------------------------------------------------------------------
if host=='alcf':

    tmp_scratch = '/lus/flare/projects/E3SM_Dec/whannah/scratch'
    tmp_grid_dir = '/lus/flare/projects/E3SM_Dec/whannah/files_grid'

    add_ens('E3SM.2026-v4-MI-TEST-00.ne30pg2.NINST_2', 2, n='MI-TEST-00', p=tmp_scratch, s='run',
            scrip_file=f'{tmp_grid_dir}/ne30pg2_scrip.nc')

    htype = 'output.1da_{inst}.AVERAGE'
    first_file,num_files = 0,0

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

add_var('precip_total_surf_mass_flux',s='precip')
add_var('zm_prec')
# add_var('LW_flux_up_at_model_top')
# add_var('VapWaterPath')
# add_var('LiqWaterPath')
# add_var('IceWaterPath')
# add_var('T_2m')
# add_var('SeaLevelPressure',s='PSL')
# add_var('ps')
# add_var('surf_evap')
# add_var('surf_sens_flux')
# add_var('wind_speed_10m')
# add_var('U_at_850hPa')
# add_var('U_at_200hPa')

# add_var('TS')
# add_var('PRECT',s='Precipitation')
# add_var('TMQ')
# add_var('T', lev=500)

#---------------------------------------------------------------------------------------------------

add_ens_mean    = False  # add panel with ensemble mean
add_ens_std     = False  # add panel with ensemble standard deviation
remove_ens_mean = False # replace each member panel with its anomaly from the ensemble mean (no extra panels)
diff_from_first = True # show full field of 1st member, other members as difference from 1st member

print_stats  = True
var_x_case   = True     # True => var x case ; False => case x var

# when set, panels are filled in order (all panels for 1st var, then 2nd var, etc. if var_x_case)
# and wrapped onto rows with this many columns
num_plot_col = None
# num_plot_col = 2

if 'use_snapshot' not in locals(): use_snapshot,ss_t = False,-1

#---------------------------------------------------------------------------------------------------
# Set up plot resources
if case==[]: raise ValueError('ERROR - case list is empty!')
if var==[]: raise ValueError('ERROR - var list is empty!')
num_var,num_case = len(var),len(case)

if 'first_file' not in locals(): first_file = None
if 'num_files'  not in locals(): num_files  = None

if (add_ens_std or remove_ens_mean or diff_from_first) and num_case<2:
    raise ValueError('ensemble std dev, anomalies, and differences require at least 2 members')
if remove_ens_mean and diff_from_first:
    raise ValueError('remove_ens_mean and diff_from_first cannot both be enabled')

#---------------------------------------------------------------------------------------------------
# build the list of panels along the case dimension
#    each panel is a tuple of (kind, case_index) where kind is 'full', 'anom', 'diff', 'mean', or 'std'
case_panels = []
for c in range(num_case):
    if   remove_ens_mean:      case_panels.append(('anom',c))
    elif diff_from_first and c>0: case_panels.append(('diff',c))
    else:                      case_panels.append(('full',c))
if add_ens_mean: case_panels.append(('mean',None))
if add_ens_std:  case_panels.append(('std',None))
num_panel = len(case_panels)

# list of (var_index, panel_index) in plotting order
if var_x_case:
    plot_order = [ (v,p) for v in range(num_var) for p in range(num_panel) ]
else:
    plot_order = [ (v,p) for p in range(num_panel) for v in range(num_var) ]

#---------------------------------------------------------------------------------------------------
# set up figure objects
subplot_kwargs = {}
subplot_kwargs['projection'] = ccrs.PlateCarree(central_longitude=180)
# subplot_kwargs['projection'] = ccrs.Robinson(central_longitude=180)
# subplot_kwargs['projection'] = ccrs.Orthographic(central_latitude=-85)

if num_plot_col is None:
    (nrows,ncols) = (num_var,num_panel) if var_x_case else (num_panel,num_var)
else:
    num_plots = len(plot_order)
    ncols = min(num_plot_col, num_plots)
    nrows = int(np.ceil(num_plots / float(ncols)))

fdx,fdy = 10,5
figsize = (fdx*ncols,fdy*nrows)
title_fontsize,lable_fontsize = 20,18

fig, axs = plt.subplots(nrows,ncols, subplot_kw=subplot_kwargs, figsize=figsize, squeeze=False )
axs_flat = axs.flatten()
for ax in axs_flat[len(plot_order):]: ax.set_visible(False)

def get_ax(v,p):
    if num_plot_col is None:
        return axs[v,p] if var_x_case else axs[p,v]
    return axs_flat[plot_order.index((v,p))]

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
    var_opts = var_opts_list[v]
    hapy.print_line()
    print(' '*2+'var: '+hapy.tclr.MAGENTA+var[v]+hapy.tclr.END)
    data_list = []
    grid_ds_list = []
    for c in range(num_case):
        print(' '*4+'case: '+hapy.tclr.GREEN+case[c]+hapy.tclr.END+f'  inst: {inst_list[c]}')
        #-------------------------------------------------------------------------
        htype_tmp = None
        if 'htype' in globals(): htype_tmp = htype
        if 'htype' in var_opts: htype_tmp = var_opts['htype']
        if inst_list[c] is not None: htype_tmp = htype_tmp.replace('{inst}',inst_list[c])
        #-------------------------------------------------------------------------
        file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/*{htype_tmp}*'
        file_list = sorted(glob.glob(file_path))
        if first_file is not None and first_file!=0: file_list = file_list[first_file:]
        if num_files  is not None and num_files !=0: file_list = file_list[:num_files]
        #-------------------------------------------------------------------------
        if file_list==[]: print('ERROR: Empty file list:'); print(); print(file_path); exit()
        #-------------------------------------------------------------------------
        ds = xr.open_mfdataset( file_list, data_vars='minimal' )

        if var[v]=='zm_prec_frac':
            data = ds['zm_prec'] / ds['precip_total_surf_mass_flux']
        elif var[v]=='RESTOM':
            data = ds['SW_flux_dn_at_model_top'] \
                  -ds['SW_flux_up_at_model_top'] \
                  -ds['LW_flux_up_at_model_top']
        else:
            data = ds[var[v]]

        grid_ds = xr.open_dataset(scrip_file_list[c])
        grid_ds_list.append(grid_ds)

        #-----------------------------------------------------------------------
        if 'lev' in data.dims:
            if lev_list[v] is None:
                raise ValueError('lev_list[v] is None - not sure what to do...')
            if isinstance(lev_list[v], str):
                if lev_list[v]=='sum':
                    data = data.sum(dim='lev')
            elif isinstance(lev_list[v], (int, float)):
                if lev_list[v]>0:
                    ps_var = next((v for v in ['ps', 'PS'] if v in ds), None)
                    if ps_var is None: raise ValueError('No sfc pressure variable found in dataset?')
                    data = hapy.vinth2p_simple(data, ds['hyam'], ds['hybm'], lev_list[v], ds[ps_var],
                                                        interp_type='log', extrapolate=False)
                if lev_list[v]<0:
                    # if requested lev<0 use model levels without interpolation
                    klev_tmp = np.absolute(lev_list[v])
                    plev_tmp = data['lev'].values[klev_tmp]
                    print(' '*4+f'{hapy.tclr.YELLOW}selecting model level {klev_tmp} (plev~{plev_tmp}mb){hapy.tclr.END}')
                    data = data.isel(lev=klev_tmp,drop=True)
        #-----------------------------------------------------------------------
        # adjust units
        if var[v]=='FRONTGF': data = data*86400e6 # K^2/M^2/S > K^2/KM^2/day
        if 'prec' in var[v] and var[v]!='zm_prec_frac': data = data*86400*1e3 # m/s > mm/day
        #-------------------------------------------------------------------------
        if use_snapshot and 'time' in data.dims:
            data = data.isel(time=ss_t,drop=True)
            print(' '*4+f'{hapy.tclr.RED}WARNING - snapshot mode enabled{hapy.tclr.END}')
        #-----------------------------------------------------------------------
        if 'dim2' in data.dims: data = data.isel(dim2=0)
        #-----------------------------------------------------------------------
        # print stats before time averaging
        if print_stats: hapy.print_stat(data,name=var[v],stat='naxh',indent='    ',compact=True)
        #-----------------------------------------------------------------------
        # average over time dimension
        if 'time' in data.dims : data = data.mean(dim='time')
        #-----------------------------------------------------------------------
        # load into memory once - otherwise the min/max, the ensemble stats, and
        # to_raster each trigger a separate re-read of the input files
        data.load()
        #-------------------------------------------------------------------------
        data_list.append( data )
    #----------------------------------------------------------------------------
    # ensemble statistics - members are assumed to share the same grid
    ens_data = xr.concat(data_list, dim='ens')
    ens_mean = ens_data.mean(dim='ens')
    if add_ens_std: ens_std = ens_data.std(dim='ens', ddof=1)
    if remove_ens_mean:
        anom_list = [ d - ens_mean for d in data_list ]
        anom_max = np.max([np.nanmax(np.absolute(d)) for d in anom_list])
    if diff_from_first:
        diff_list = [ d - data_list[0] for d in data_list ]
        diff_max = np.max([np.nanmax(np.absolute(d)) for d in diff_list[1:]])
    #----------------------------------------------------------------------------
    # calculate common limits for consistent contour levels
    data_min = np.min([np.nanmin(d) for d in data_list])
    data_max = np.max([np.nanmax(d) for d in data_list])
    #----------------------------------------------------------------------------
    # set color bar levels
    clev = None
    if var[v]=='FRONTGF': clev = np.logspace( -5, -1, num=40)
    if 'prec' in var[v]: clev = np.arange(0,20+1,1)
    if var[v]=='zm_prec_frac': clev = None
    if var[v] in ['PS','ps','sp']: clev = np.arange(600e2,1040e2+2e2,10e2)
    if var[v]=='PCONVT': clev = np.arange(100e2,600e2+5e2,5e2)
    #----------------------------------------------------------------------------
    # set color map
    cmap = 'viridis'
    if var[v]=='PCONVT': cmap = cmocean.cm.rain_r
    #----------------------------------------------------------------------------
    for p in range(num_panel):
        kind,c = case_panels[p]
        cg = 0 if c is None else c  # grid index for ensemble stat panels
        #-------------------------------------------------------------------------
        img_kwargs = {}
        img_kwargs['origin'] = 'lower'
        img_kwargs['cmap']   = cmap

        if kind=='anom':
            plot_data = anom_list[c]
            img_kwargs['cmap']   = cmocean.cm.balance
            img_kwargs['vmin']   = -1*anom_max
            img_kwargs['vmax']   = anom_max
        elif kind=='diff':
            plot_data = diff_list[c]
            img_kwargs['cmap']   = cmocean.cm.balance
            img_kwargs['vmin']   = -1*diff_max
            img_kwargs['vmax']   = diff_max
        elif kind=='std':
            plot_data = ens_std
            img_kwargs['cmap']   = cmocean.cm.amp
            img_kwargs['vmin']   = 0
            img_kwargs['vmax']   = np.nanmax(ens_std)
        else:
            plot_data = data_list[c] if kind=='full' else ens_mean
            if clev is None:
                img_kwargs['vmin'],img_kwargs['vmax'] = data_min, data_max
            else:
                img_kwargs['norm'] = mcolors.BoundaryNorm(clev, ncolors=256)

        ax = get_ax(v,p)
        ax.coastlines(linewidth=0.2,edgecolor='white')
        if   kind=='anom': left_str = f'{name[c]} - ens mean'
        elif kind=='diff':
            # abbreviate the baseline name if it is a member of the same ensemble
            if case[c]==case[0] and inst_list[0] is not None:
                left_str = f'{name[c]} - #{inst_list[0]}'
            else:
                left_str = f'{name[c]} - {name[0]}'
        elif kind=='mean': left_str = 'ens mean'
        elif kind=='std':  left_str = 'ens std dev'
        else:              left_str = name[c]
        ax.set_title(left_str,fontsize=title_fontsize, loc='left')
        ax.set_title(var_str[v],fontsize=title_fontsize, loc='right')

        # add global mean value
        area = grid_ds_list[cg].grid_area.values
        gbl_mean = ( (plot_data*area).sum() / area.sum() ).values
        ax.set_title(f'mean: {gbl_mean:8.2f}', fontsize=int(title_fontsize*0.5), \
                     loc='center', y=0.9, backgroundcolor='white')

        ax.set_global()

        rst = hapy.to_raster(plot_data, grid_ds_list[cg], ax)

        img = ax.imshow(rst, extent=ax.get_xlim() + ax.get_ylim(), **img_kwargs)

        cbar = fig.colorbar(img, ax=ax, fraction=0.02, orientation='vertical')
        cbar.ax.tick_params(labelsize=lable_fontsize)

#---------------------------------------------------------------------------------------------------
# Finalize plot
os.makedirs(os.path.dirname(fig_file), exist_ok=True)
fig.savefig(fig_file, dpi=150, bbox_inches='tight')
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
