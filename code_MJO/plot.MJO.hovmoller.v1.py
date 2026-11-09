#---------------------------------------------------------------------------------------------------
# MJO time-longitude (Hovmoller) diagram
#
#   meridionally average daily data over an equatorial band onto longitude bins, remove the
#   time mean at each longitude (optionally band-pass filter 20-100 day), and plot
#   time vs longitude for each variable/case - optional overlay contour variable (e.g. U850)
#---------------------------------------------------------------------------------------------------
from time import perf_counter
timer_start = perf_counter()
import os, glob, string, warnings, numpy as np, xarray as xr, numba, cmocean
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import hapy; host = hapy.get_host()
xr.set_options(use_new_combine_kwarg_defaults=True)
#---------------------------------------------------------------------------------------------------
case,case_opts_list = [],[]
def add_case(case_in,**kwargs):
    """ p=path, s=sub-dir, n=name, htype/yr1/yr2 can also be set per-case """
    case.append(case_in)
    case_opts = dict(kwargs)
    if 'n' not in case_opts: case_opts['n'] = case_in
    case_opts_list.append(case_opts)
#---------------------------------------------------------------------------------------------------
var,var_opts_list = [],[]
def add_var(var_name,**kwargs):
    """ shaded variable + optional overlay contour variable
        supported options:
          s / str   - string used for the plot label
          htype     - override the global history stream for this variable
          cvar      - variable name for the overlaid contour lines
          cvar_s / cvar_str - string used for the contour variable label
          cvar_htype- history stream for the contour variable
          clev      - explicit contour levels for the shaded field
          cclev     - explicit contour levels for the overlay contours
          cmap      - explicit color map for the shaded field
    """
    var.append(var_name)
    var_opts = dict(kwargs)
    if 's'   in var_opts: var_opts['str'] = var_opts.pop('s')
    if 'str' not in var_opts: var_opts['str'] = var_name
    if 'cvar_s' in var_opts: var_opts['cvar_str'] = var_opts.pop('cvar_s')
    if 'cvar' in var_opts and 'cvar_str' not in var_opts: var_opts['cvar_str'] = var_opts['cvar']
    var_opts_list.append(var_opts)
#---------------------------------------------------------------------------------------------------
if host=='lcrc':

    ### MCSP sensitivity tests
    tmp_path,tmp_sub = '/lcrc/group/e3sm/ac.whannah/scratch/chrys','run'
    htype = 'output.1da.AVERAGE.ndays_x1'
    prefix = 'E3SM.2026-MCSP-SS-00.F2010xx-ZM-CICE.NN_4.ne30pg2.ufs_1'
    add_case(f'{prefix}.usm_0.mtc_0.3',                n='usm=0 mtc=0.3',              p=tmp_path,s=tmp_sub)
    add_case(f'{prefix}.usm_1.son_20.sox_60.mtc_1',    n='usm=1 mtc=1.0',              p=tmp_path,s=tmp_sub)
    add_case(f'{prefix}.usm_1.son_20.sox_60.mtc_1.2',  n='usm=1 mtc=1.2',              p=tmp_path,s=tmp_sub)
    add_case(f'{prefix}.use_1.evn_2.5e-06.evx_3.5e-06',n='use=1 evn=2.5e-6 evx=3.5e-6',p=tmp_path,s=tmp_sub)
    add_case(f'{prefix}.use_1.evn_2.5e-06.evx_4.5e-06',n='use=1 evn=2.5e-6 evx=4.5e-6',p=tmp_path,s=tmp_sub)
    yr1,yr2 = 2,2

    kwargs = {'cvar':'U_at_850hPa','cvar_str':'850mb U'}
    add_var('LW_flux_up_at_model_top',    s='OLR [W/m2]',      cmap=cmocean.cm.balance, **kwargs)
    add_var('precip_total_surf_mass_flux',s='precip [mm/day]', cmap=plt.get_cmap('BrBG'))
    add_var('U_at_850hPa',                s='850mb U [m/s]',   cmap=cmocean.cm.balance)
    # add_var('U_at_200hPa',                s='200mb U [m/s]',   cmap=cmocean.cm.balance)
    # add_var('VapWaterPath',               s='VWP [kg/m2]',     cmap=plt.get_cmap('BrBG'))
    # add_var('zm_prec',                    s='ZM precip [mm/day]', cmap=plt.get_cmap('BrBG'))

#---------------------------------------------------------------------------------------------------
if host=='olcf':

    tmp_path,tmp_sub = '/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch','archive/atm/hist_ne30'
    htype = 'output.ne30pg2.1da.AVERAGE.nhours_x24'
    prefix = 'E3SM.2026-ZM-BASE-01.ne256pg2.NN_256.F2010xx-ZM-CICE.COSP'
    add_case(f'{prefix}', n='ne256+ZM', p=tmp_path,s=tmp_sub)
    yr1,yr2 = 2,6

    kwargs = {'cvar':'U_at_850hPa','cvar_str':'850mb U'}
    add_var('LW_flux_up_at_model_top',    s='OLR [W/m2]',      **kwargs)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

recalculate = True
create_plot = True

if 'recalculate' not in locals(): recalculate = False
if 'create_plot' not in locals(): create_plot = False

#---------------------------------------------------------------------------------------------------

fig_file      = 'figs_MJO/MJO.hovmoller.v1.png'
tmp_file_root = 'data_temp'
tmp_file_head = 'MJO.hovmoller.v1'

### meridional extent of the equatorial average
lat1,lat2 = -10,10
# lat1,lat2 = -15,15

### longitude bins used for the meridional average
dlon = 2.5
lon_bins = np.arange(dlon/2.,360.,dlon)

### the last file of a running case may still be open for writing - skip it
skip_last_file = True

### anomalies relative to the time mean at each longitude (record is too short for a seasonal cycle)
remove_mean = True

### 20-100 day band pass filter
filter_data = False
nwgt    = 121
fhi,flo = 20.,100.

### time window to plot [days] - None => plot the full record
###   plot_from_end=True  => last num_days of each case (avoids spin-up, uses the newest data)
###   plot_from_end=False => num_days starting from day_offset
num_days      = 180
plot_from_end = True
day_offset    = 0

### plot options
plot_lon1,plot_lon2 = 30,240
print_stats  = True
var_x_case   = True
use_common_label_bar = True

#---------------------------------------------------------------------------------------------------
if case==[]: raise ValueError('ERROR - case list is empty!')
if var ==[]: raise ValueError('ERROR - var list is empty!')
num_var,num_case = len(var),len(case)

if 'htype' not in locals(): htype = None
if 'yr1'   not in locals(): yr1   = None
if 'yr2'   not in locals(): yr2   = None

### band-pass filter weights
if filter_data:
    wgt = hapy.filter_wgts_lp_lanczos(nwgt, fc_lp=1./fhi, fc_hp=1./flo)
    win_width = int( np.floor(len(wgt)/2) )
    tvals0 = np.arange( 0-win_width, 0+win_width+1, dtype=np.int64 )

tmp_file_tail = f'lat_{lat1}_{lat2}.dlon_{dlon}'
if yr1 is not None and yr2 is not None: tmp_file_tail = f'{yr1}-{yr2}.'+tmp_file_tail
if skip_last_file: tmp_file_tail += '.skip_last'

FONT_SM,FONT_MD,FONT_LG = 7,8,10

os.makedirs(tmp_file_root, exist_ok=True)
os.makedirs(os.path.dirname(fig_file), exist_ok=True)

#---------------------------------------------------------------------------------------------------
# temporal filtering
#---------------------------------------------------------------------------------------------------
@numba.njit()
def filter_numba(data,wgt,tvals0,win_width):
    """ brute force convolution of the Lanczos weights along the leading (time) dimension """
    (ntime,ncol) = data.shape
    data_filt = np.full((ntime,ncol),np.nan)
    for t in range(win_width, ntime-win_width):
        tvals = tvals0 + t
        for i in range(ncol):
            data_filt[t,i] = np.sum( data[tvals,i] * wgt )
    return data_filt
#---------------------------------------------------------------------------------------------------
# area weighted meridional average onto longitude bins
#---------------------------------------------------------------------------------------------------
@numba.njit()
def meridional_avg(data, lon, area, lon_bins, dlon):
    """ input data should have dimensions (time,ncol) - returns (time,nbin)
        NaNs are skipped (e.g. U850 is missing where 850 hPa is below ground) """
    (ntime,ncol) = data.shape
    nbin = len(lon_bins)
    data_avg = np.zeros( (ntime,nbin) )
    wsum     = np.zeros( (ntime,nbin) )
    lon0 = lon_bins[0]-dlon/2.
    for i in range(ncol):
        b = int( np.floor( (lon[i]-lon0)/dlon ) )
        if b<0 or b>=nbin: continue
        for t in range(ntime):
            if np.isnan(data[t,i]): continue
            wsum[t,b]     += area[i]
            data_avg[t,b] += data[t,i]*area[i]
    for b in range(nbin):
        for t in range(ntime):
            if wsum[t,b]>0:
                data_avg[t,b] = data_avg[t,b]/wsum[t,b]
            else:
                data_avg[t,b] = np.nan
    return data_avg
#---------------------------------------------------------------------------------------------------
# data loading helpers
#---------------------------------------------------------------------------------------------------
def get_file_list(case_in,case_opts,htype_in):
    file_path = f'{case_opts["p"]}/{case_in}/{case_opts["s"]}/*{htype_in}*'
    file_list = sorted(glob.glob(file_path))
    if skip_last_file: file_list = file_list[:-1]
    if file_list==[]: raise ValueError(f'No files found for path: {file_path}')
    return file_list
#---------------------------------------------------------------------------------------------------
def open_case_dataset(case_in,case_opts,htype_in):
    """ open the history files and apply the year subset """
    file_list = get_file_list(case_in,case_opts,htype_in)
    print(' '*6+f'num files: {len(file_list)}')
    ds = xr.open_mfdataset( file_list, data_vars='minimal' )
    #-------------------------------------------------------------------------
    # use the mid-point of the averaging interval for the time stamp - eamxx
    # output uses the right end point which shifts things by half an interval
    if 'time_bnds' in ds:
        tbnd = ds['time_bnds'].load()
        ds = ds.assign_coords( time=tbnd.mean(dim=tbnd.dims[-1]).values )
    #-------------------------------------------------------------------------
    yr1_loc = case_opts.get('yr1',yr1)
    yr2_loc = case_opts.get('yr2',yr2)
    if yr1_loc is not None: ds = ds.where( ds['time.year']>=yr1_loc, drop=True)
    if yr2_loc is not None: ds = ds.where( ds['time.year']<=yr2_loc, drop=True)
    if len(ds['time'])==0: raise ValueError(f'No data found for {case_in} in [{yr1_loc}:{yr2_loc}]')
    return ds
#---------------------------------------------------------------------------------------------------
def adjust_units(data,var_name):
    if 'prec' in var_name.lower() and 'frac' not in var_name.lower():
        data = data*86400*1e3   # m/s => mm/day
    return data
#---------------------------------------------------------------------------------------------------
def calc_hovmoller(case_in,case_opts,var_name,htype_in):
    """ load the equatorial band and average onto longitude bins => (time,lon) """
    ds = open_case_dataset(case_in,case_opts,htype_in)
    data = ds[var_name]
    if 'ncol' not in data.dims:
        raise ValueError('expected an unstructured (ncol) grid - use the native/ne30 history files')
    #-------------------------------------------------------------------------
    lat = ds['lat'].load()
    if 'time' in lat.dims: lat = lat.isel(time=0)
    mask = ((lat>=lat1) & (lat<=lat2)).values
    data = data.isel(ncol=np.where(mask)[0])
    lon  = ds['lon'].values
    area = ds['area'].values
    if lon.ndim>1:  lon  = lon[0,:]
    if area.ndim>1: area = area[0,:]
    lon  = np.mod(lon[mask],360.)
    area = area[mask]
    #-------------------------------------------------------------------------
    # convert to daily mean if the data is sub-daily
    if len(data['time'])>1:
        dtime = ( data['time'][1]-data['time'][0] ).values / np.timedelta64(1,'s')
        if dtime < 86400.:
            print(' '*6+'resampling to daily mean...')
            data = data.resample(time='D').mean(dim='time')
    #-------------------------------------------------------------------------
    data = adjust_units(data,var_name)
    data.load()
    if print_stats: hapy.print_stat(data,name=var_name,stat='naxsh',indent=' '*6,compact=True)
    #-------------------------------------------------------------------------
    hov = meridional_avg( np.ascontiguousarray(data.values,dtype=np.float64),
                          np.ascontiguousarray(lon,dtype=np.float64),
                          np.ascontiguousarray(area,dtype=np.float64),
                          lon_bins, float(dlon) )
    hov_da = xr.DataArray( hov, coords=[('time',data['time'].values),('lon',lon_bins)],
                           dims=['time','lon'], name=var_name )
    return hov_da
#---------------------------------------------------------------------------------------------------
def get_tmp_file(case_in,var_name):
    return f'{tmp_file_root}/{tmp_file_head}.{case_in}.{var_name}.{tmp_file_tail}.nc'
#---------------------------------------------------------------------------------------------------
calculated_files = set()   # avoid recalculating a variable used as both shaded and contour field
def get_hovmoller(case_in,case_opts,var_name,htype_in):
    """ read the cached hovmoller or recalculate it, then compute anomalies / filter / window """
    tmp_file = get_tmp_file(case_in,var_name)
    if recalculate and tmp_file not in calculated_files:
        calculated_files.add(tmp_file)
        print(' '*4+f'calculating hovmoller ({hapy.tclr.CYAN}{var_name}{hapy.tclr.END})...')
        hov = calc_hovmoller(case_in,case_opts,var_name,htype_in)
        print(' '*4+f'writing to file: {tmp_file}')
        hov.to_netcdf(path=tmp_file,mode='w')
    else:
        print(' '*4+f'reading from file: {tmp_file}')
        hov = xr.open_dataset( tmp_file )[var_name]
        hov.load()
    #-------------------------------------------------------------------------
    if remove_mean: hov = hov - hov.mean(dim='time')
    if filter_data:
        if len(hov['time'])<=2*win_width:
            raise ValueError(f'record too short ({len(hov["time"])} days) for the {nwgt} weight filter')
        hov = hov.copy( data=filter_numba(np.ascontiguousarray(hov.values),wgt,tvals0,win_width) )
        hov = hov.isel(time=slice(win_width,-win_width))     # drop NaN edges
    #-------------------------------------------------------------------------
    if num_days is not None:
        nt = len(hov['time'])
        if nt<num_days:
            print(' '*6+f'{hapy.tclr.YELLOW}WARNING - only {nt} days available (num_days={num_days}){hapy.tclr.END}')
        if plot_from_end: hov = hov.isel(time=slice(max(0,nt-num_days),nt))
        else:             hov = hov.isel(time=slice(day_offset,day_offset+num_days))
    return hov
#---------------------------------------------------------------------------------------------------
def nice_cntr_levels(data_min, data_max, max_steps=21, aboutZero=False):
    """ Return (cmin,cmax,cint) with 'nice' round numbers, optionally symmetric about zero """
    if aboutZero:
        mag = max(abs(data_min),abs(data_max))
        raw_int = 2*mag/(max_steps-1)
    else:
        raw_int = (data_max-data_min)/(max_steps-1)
    if raw_int<=0: raw_int = 1.
    exp  = np.floor(np.log10(raw_int))
    frac = raw_int/10**exp
    nice = np.array([1,2,2.5,5,10])
    cint = nice[np.argmin(np.abs(nice-frac))]*10**exp
    if aboutZero:
        n = int(np.ceil(mag/cint))
        cmin,cmax = -n*cint, n*cint
    else:
        cmin = np.floor(data_min/cint)*cint
        cmax = np.ceil( data_max/cint)*cint
    return cmin,cmax,cint
#---------------------------------------------------------------------------------------------------
def get_clev(data_list, nlev=21, pct=98):
    """ symmetric contour levels spanning all cases - percentile avoids outliers dominating """
    vals = np.concatenate([ np.abs(d.values[np.isfinite(d.values)]).ravel() for d in data_list ])
    mag = np.percentile(vals,pct)
    cmin,cmax,cint = nice_cntr_levels(-mag,mag,max_steps=nlev,aboutZero=True)
    clev = np.arange(cmin,cmax+cint/2,cint)
    if len(clev)<3: clev = np.linspace(cmin,cmax,num=nlev)
    return clev
#---------------------------------------------------------------------------------------------------
def format_ax(ax, hov, title):
    ax.set_title(title, fontsize=FONT_MD, loc='left')
    ax.set_xlim(plot_lon1,plot_lon2)
    xt_val = np.arange(0,360+30,30)
    xt_str = [ '0' if t%360==0 else ('180' if t==180 else (f'{t}E' if t<180 else f'{360-t}W'))
               for t in xt_val ]
    keep = [ (t>=plot_lon1) and (t<=plot_lon2) for t in xt_val ]
    ax.set_xticks( xt_val[keep] )
    ax.set_xticklabels( [s for s,k in zip(xt_str,keep) if k], fontsize=FONT_SM )
    #-------------------------------------------------------------------------
    # time runs downward - label with dates of the first of each month
    ntime = len(hov['time'])
    ax.set_ylim(ntime-1,0)
    time = hov['time'].values
    yt_val = [ t for t in range(ntime) if time[t].day==1 ]
    if yt_val==[]: yt_val = [0]
    ax.set_yticks(yt_val)
    ax.set_yticklabels([ f'{time[t].year:04d}-{time[t].month:02d}-{time[t].day:02d}' for t in yt_val ],
                       fontsize=FONT_SM)
    ax.tick_params(axis='both',which='both',labelsize=FONT_SM)
#---------------------------------------------------------------------------------------------------
# load the data
#---------------------------------------------------------------------------------------------------
sdata_all,cdata_all = {},{}
for v in range(num_var):
    var_opts = var_opts_list[v]
    hapy.print_line()
    print(' '*2+f'var: {hapy.tclr.MAGENTA}{var[v]}{hapy.tclr.END}')
    sdata_list,cdata_list = [],[]
    for c in range(num_case):
        case_opts = case_opts_list[c]
        print(); print(' '*4+f'case: {hapy.tclr.GREEN}{case[c]}{hapy.tclr.END}')
        htype_svar = var_opts.get('htype',      case_opts.get('htype',htype))
        htype_cvar = var_opts.get('cvar_htype', htype_svar)
        sdata_list.append( get_hovmoller(case[c],case_opts,var[v],htype_svar) )
        if var_opts.get('cvar',None) is not None:
            cdata_list.append( get_hovmoller(case[c],case_opts,var_opts['cvar'],htype_cvar) )
    sdata_all[v] = sdata_list
    cdata_all[v] = cdata_list

#---------------------------------------------------------------------------------------------------
if not create_plot: exit('\nExiting because create_plot is False.\n')
#---------------------------------------------------------------------------------------------------
# Create plot
#---------------------------------------------------------------------------------------------------
(d1,d2) = (num_var,num_case) if var_x_case else (num_case,num_var)
fdx,fdy = 3.5,5
fig, axs = plt.subplots(d1,d2, figsize=(fdx*d2,fdy*d1), squeeze=False)
fig.subplots_adjust(hspace=0.25, wspace=0.35)

panel_labels = list(string.ascii_lowercase)

for v in range(num_var):
    var_opts   = var_opts_list[v]
    sdata_list = sdata_all[v]
    cdata_list = cdata_all[v]
    use_cvar   = var_opts.get('cvar',None) is not None
    #-----------------------------------------------------------------------
    sclev = var_opts.get('clev',None)
    if sclev is None: sclev = get_clev(sdata_list)
    cclev = None
    if use_cvar:
        cclev = var_opts.get('cclev',None)
        if cclev is None: cclev = get_clev(cdata_list,nlev=9)
    cmap = var_opts.get('cmap',cmocean.cm.balance)
    norm = BoundaryNorm(sclev, ncolors=cmap.N, extend='both')
    #-----------------------------------------------------------------------
    for c in range(num_case):
        ax = axs[v,c] if var_x_case else axs[c,v]
        sdata = sdata_list[c]
        lon = sdata['lon'].values
        tt  = np.arange(len(sdata['time']))
        cf = ax.contourf(lon, tt, sdata.values, levels=sclev, cmap=cmap, norm=norm, extend='both')
        #-------------------------------------------------------------------
        # overlay the contour variable - negative values are dashed, zero omitted
        if use_cvar:
            cdata = cdata_list[c]
            ct = np.arange(len(cdata['time']))
            neg_lev = [l for l in cclev if l<0]
            pos_lev = [l for l in cclev if l>0]
            if neg_lev: ax.contour(cdata['lon'].values, ct, cdata.values,
                                   levels=neg_lev, colors='black', linewidths=0.5, linestyles='--')
            if pos_lev: ax.contour(cdata['lon'].values, ct, cdata.values,
                                   levels=pos_lev, colors='black', linewidths=0.5, linestyles='-')
        #-------------------------------------------------------------------
        p = v*num_case+c if var_x_case else c*num_var+v
        format_ax(ax, sdata, f'({panel_labels[p%26]}) {case_opts_list[c]["n"]}')
        if c==0 and var_x_case:
            ax.annotate(var_opts['str'], xy=(-0.42,0.5), xycoords='axes fraction', rotation=90,
                        ha='center', va='center', fontsize=FONT_LG)
        if not use_common_label_bar:
            cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.06, fraction=0.04)
            cbar.ax.tick_params(labelsize=FONT_SM)
    #-----------------------------------------------------------------------
    if use_common_label_bar:
        ax_list = [ axs[v,c] for c in range(num_case) ] if var_x_case else \
                  [ axs[c,v] for c in range(num_case) ]
        cbar = fig.colorbar(cf, ax=ax_list, orientation='vertical', fraction=0.02, pad=0.01, shrink=0.9)
        cbar.ax.tick_params(labelsize=FONT_SM)
        clabel = var_opts['str']+(' anomaly' if remove_mean else '')
        if use_cvar: clabel += f'\ncontours: {var_opts["cvar_str"]} (dashed<0, int={cclev[1]-cclev[0]:g})'
        cbar.set_label(clabel, fontsize=FONT_SM)

#---------------------------------------------------------------------------------------------------
# Finalize plot
#---------------------------------------------------------------------------------------------------
sup = f'{lat1}:{lat2} lat avg'
if remove_mean: sup += ', anomaly from time mean'
if filter_data: sup += f', {int(fhi)}-{int(flo)} day filtered'
top = max( a.get_position().y1 for a in axs.flat )
fig.suptitle(sup, fontsize=FONT_LG, y=top+0.03)

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
