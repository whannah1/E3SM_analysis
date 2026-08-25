#---------------------------------------------------------------------------------------------------
# MJO lag-longitude regression
#
#   band-pass filter (20-100 day) the daily data, build a normalized MJO index from an
#   equatorial box average, and regress the meridionally averaged fields onto that index
#   at a range of time lags => lag vs longitude contour plots
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
          lev       - pressure level [Pa] (>0) or model level index (<0) or 'sum'
          htype     - override the global history stream for this variable
          cvar      - variable name for the overlaid contour lines
          cvar_s / cvar_str - string used for the contour variable label
          cvar_lev  - level for the contour variable
          cvar_htype- history stream for the contour variable
          clev      - explicit contour levels for the shaded field
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
if host=='olcf':

    ### ZM baseline simulations
    tmp_path,tmp_sub = '/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch','archive/atm/hist_ne30'
    htype = 'output.ne30pg2.1da.AVERAGE.nhours_x24'
    prefix = 'E3SM.2026-ZM-BASE-01.ne256pg2.NN_256.F2010xx-ZM-CICE.COSP'
    add_case(f'{prefix}',                            n='ne256+ZM',             p=tmp_path,s=tmp_sub)
    add_case(f'{prefix}.L128v3.6',                   n='ne256+ZM+L128v4',      p=tmp_path,s=tmp_sub)
    add_case(f'{prefix}.L128v3.6.phys_order_swap_1', n='ne256+ZM+L128v4+swap', p=tmp_path,s=tmp_sub)
    yr1,yr2 = 2,6

    ### index variable - OLR is the standard choice (sign is flipped below)
    ind_var,ind_htype = 'LW_flux_up_at_model_top',None
    # ind_var,ind_htype = 'precip_total_surf_mass_flux',None

    kwargs = {'cvar':'U_at_850hPa','cvar_str':'850mb U'}
    add_var('precip_total_surf_mass_flux',s='precip [mm/day]',      **kwargs)
    # add_var('LW_flux_up_at_model_top',    s='OLR [W/m2]',           **kwargs)
    # add_var('U_at_850hPa',                s='850mb U [m/s]',        **kwargs)
    # add_var('U_at_200hPa',                s='200mb U [m/s]',        **kwargs)
    # add_var('VapWaterPath',               s='VWP [kg/m2]',          **kwargs)
    add_var('zm_prec',                    s='ZM Precip [mm/day]',   **kwargs)

#---------------------------------------------------------------------------------------------------
if host=='nersc':

    tmp_path,tmp_sub = '/pscratch/sd/w/whannah/e3sm_scratch/pm-cpu','run'
    # add_case('E3SM.2025-MF-test-00.ne30pg2.F20TR.NN_32',n='ne30',p=tmp_path,s=tmp_sub)
    htype = 'eam.h1'
    yr1,yr2 = 1985,1994

    ind_var,ind_htype = 'FLNT',None

    add_var('PRECT',s='Precipitation [mm/day]',cvar='U',cvar_str='850mb U',cvar_lev=850e2)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
fig_file      = 'figs_MJO/MJO.lag_regression.horz.v1.png'
tmp_file_root = 'data_temp'
tmp_file_head = 'MJO.lag_regression.horz.v1'

### meridional extent of the equatorial average
lat1,lat2 = -5,5

### maximum time lag [days]
mxlag = 25

### regression index box => ilon +/- iwid
ilon,iwid = 90,5
# ilon,iwid = 150,10

### longitude bins used for the meridional average
dlon = 2
lon_bins = np.arange(dlon/2.,360.,dlon)

### only use base times that fall in these months (set to None to use all)
regression_months = [10,11,12,1,2,3]
# regression_months = None

### 20-100 day band pass filter
filter_data = True
nwgt    = 121
fhi,flo = 20.,100.

recalculate_ind = True
recalculate_reg = True
create_plot     = True

### plot options
plot_lon1,plot_lon2 = 0,360
print_stats  = True
var_x_case   = True
use_common_label_bar = False

plot_diff,add_diff = False,False
# plot_diff,add_diff = True,False   # diff panels replace the full field for non-baseline cases
# plot_diff,add_diff = True,True    # diff panels are added alongside the full fields

#---------------------------------------------------------------------------------------------------
if case==[]: raise ValueError('ERROR - case list is empty!')
if var ==[]: raise ValueError('ERROR - var list is empty!')
num_var,num_case = len(var),len(case)

diff_base = 0

if 'htype'    not in locals(): htype    = None
if 'yr1'      not in locals(): yr1      = None
if 'yr2'      not in locals(): yr2      = None
if 'ind_var'  not in locals(): raise ValueError('ERROR - ind_var was not specified!')
if 'ind_htype' not in locals(): ind_htype = None

### band-pass filter weights
fc_lp,fc_hp = 1./fhi, 1./flo
wgt = hapy.filter_wgts_lp_lanczos(nwgt, fc_lp=fc_lp, fc_hp=fc_hp)
win_width = int( np.floor(len(wgt)/2) )
tvals0 = np.arange( 0-win_width, 0+win_width+1, dtype=np.int64 )

tmp_file_tail = f'lat_{lat1}_{lat2}.ilon_{ilon}.iwid_{iwid}'
if yr1 is not None and yr2 is not None: tmp_file_tail = f'{yr1}-{yr2}.'+tmp_file_tail
if filter_data: tmp_file_tail += f'.filt_{int(fhi)}-{int(flo)}'

FONT_SM,FONT_MD,FONT_LG = 7,8,10

os.makedirs(tmp_file_root, exist_ok=True)
os.makedirs(os.path.dirname(fig_file), exist_ok=True)

#---------------------------------------------------------------------------------------------------
# temporal filtering
#---------------------------------------------------------------------------------------------------
@numba.njit()
def filter_numba(data,wgt,tvals0,win_width):
    """ brute force convolution of the Lanczos weights along the leading (time) dimension """
    if data.ndim==1:
        ntime,ncol = data.shape[0], 1
        data_filt = np.full((ntime),np.nan)
    else:
        (ntime,ncol) = data.shape
        data_filt = np.full((ntime,ncol),np.nan)
    for t in range(ntime):
        if t >= win_width and t <= (ntime-1-win_width) :
            tvals = tvals0 + t
            for i in range(ncol):
                if data.ndim==1: data_filt[t]   = np.sum( data[tvals]   * wgt )
                if data.ndim==2: data_filt[t,i] = np.sum( data[tvals,i] * wgt )
    return data_filt
#---------------------------------------------------------------------------------------------------
# meridional average onto longitude bins
#---------------------------------------------------------------------------------------------------
@numba.njit()
def meridional_avg(data, lon, lon_bins, dlon):
    """ input data should have dimensions (time,ncol) - returns (time,nbin) """
    (ntime,ncol) = data.shape
    nbin = len(lon_bins)
    data_avg = np.zeros( (ntime,nbin) )
    cnt      = np.zeros( (nbin) )
    for i in range(ncol):
        b = -1
        for bb in range(nbin):
            if lon[i]>=(lon_bins[bb]-dlon/2.) and lon[i]<(lon_bins[bb]+dlon/2.):
                b = bb
                break
        if b<0: continue
        cnt[b] += 1
        for t in range(ntime): data_avg[t,b] += data[t,i]
    for b in range(nbin):
        for t in range(ntime):
            if cnt[b]>0:
                data_avg[t,b] = data_avg[t,b]/cnt[b]
            else:
                data_avg[t,b] = np.nan
    return data_avg
#---------------------------------------------------------------------------------------------------
# lag regression
#---------------------------------------------------------------------------------------------------
@numba.njit()
def lag_regression_numba(data_in, index, lag_coord, base_mask):
    """ regress data_in(time,lon) onto index(time) at each lag

        NOTE - unlike the original pyngl version, the covariance is normalized by the
        variance of the *index* (i.e. the standard regression coefficient), so the
        result has the units of the input data per standard deviation of the index
    """
    (ntime,nbin) = data_in.shape
    nlag = len(lag_coord)
    reg_out = np.full( (nlag,nbin), np.nan )
    ### collect the valid base times
    nvalid = 0
    for t in range(ntime):
        if base_mask[t]: nvalid += 1
    tvals = np.empty( (nvalid), dtype=np.int64 )
    n = 0
    for t in range(ntime):
        if base_mask[t]:
            tvals[n] = t
            n += 1
    ### index statistics only need to be computed once
    ind_avg = 0.
    for n in range(nvalid): ind_avg += index[tvals[n]]
    ind_avg = ind_avg/nvalid
    ind_var = 0.
    for n in range(nvalid): ind_var += (index[tvals[n]]-ind_avg)**2
    ind_var = ind_var/nvalid
    ### regress at each lag
    for b in range(nbin):
        for l in range(nlag):
            data_avg,valid = 0.,True
            for n in range(nvalid):
                dtmp = data_in[tvals[n]+lag_coord[l],b]
                if np.isnan(dtmp): valid = False; break
                data_avg += dtmp
            if not valid: continue
            data_avg = data_avg/nvalid
            cov = 0.
            for n in range(nvalid):
                cov += (data_in[tvals[n]+lag_coord[l],b]-data_avg) * (index[tvals[n]]-ind_avg)
            reg_out[l,b] = ( cov/nvalid ) / ind_var
    return reg_out
#---------------------------------------------------------------------------------------------------
# data loading helpers
#---------------------------------------------------------------------------------------------------
def get_file_list(case_in,case_opts,htype_in):
    file_path = f'{case_opts["p"]}/{case_in}/{case_opts["s"]}/*{htype_in}*'
    file_list = sorted(glob.glob(file_path))
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
def get_var_data(ds,var_name):
    """ pull the variable out of the dataset - includes a few derived quantities """
    if var_name=='RESTOM':
        return ds['SW_flux_dn_at_model_top'] \
              -ds['SW_flux_up_at_model_top'] \
              -ds['LW_flux_up_at_model_top']
    if var_name=='P-E':
        return ds['PRECT']*86400*1e3 - ds['QFLX']*86400
    return ds[var_name]
#---------------------------------------------------------------------------------------------------
def select_level(ds,data,lev_in,var_name):
    """ same level handling as plot.clim.map.v1.py """
    if 'lev' not in data.dims: return data
    if lev_in is None: raise ValueError(f'{var_name} has a lev dimension but no lev was specified')
    if isinstance(lev_in,str):
        if lev_in=='sum': return data.sum(dim='lev')
        raise ValueError(f'unsupported lev option: {lev_in}')
    if lev_in>0:
        ps_var = next((n for n in ['ps','PS'] if n in ds), None)
        if ps_var is None: raise ValueError('No sfc pressure variable found in dataset?')
        data = hapy.vinth2p_simple(data, ds['hyam'], ds['hybm'], lev_in, ds[ps_var],
                                   interp_type='log', extrapolate=False)
        if 'plev' in data.dims: data = data.isel(plev=0,drop=True)
        if 'lev'  in data.dims: data = data.isel(lev=0,drop=True)
    else:
        klev = int(np.absolute(lev_in))
        plev = data['lev'].values[klev]
        print(' '*6+f'{hapy.tclr.YELLOW}selecting model level {klev} (plev~{plev}){hapy.tclr.END}')
        data = data.isel(lev=klev,drop=True)
    return data
#---------------------------------------------------------------------------------------------------
def adjust_units(data,var_name):
    if 'prec' in var_name.lower() and 'frac' not in var_name.lower():
        data = data*86400*1e3   # m/s => mm/day
    return data
#---------------------------------------------------------------------------------------------------
def subset_lat_lon(ds,data,lon1=None,lon2=None):
    """ reduce to the equatorial band (and optionally a longitude box) """
    lat = ds['lat'].load()
    mask = (lat>=lat1) & (lat<=lat2)
    if lon1 is not None and lon2 is not None:
        lon = ds['lon'].load()
        lon1_tmp,lon2_tmp = np.mod(lon1,360), np.mod(lon2,360)
        if lon1_tmp<=lon2_tmp: mask = mask & (lon>=lon1_tmp) & (lon<=lon2_tmp)
        else:                  mask = mask & ((lon>=lon1_tmp) | (lon<=lon2_tmp))
    if 'ncol' not in data.dims:
        raise ValueError('expected an unstructured (ncol) grid - use the native/ne30 history files')
    data = data.where( mask.compute(), drop=True )
    return data
#---------------------------------------------------------------------------------------------------
def load_daily(ds,var_name,lev_in,lon1=None,lon2=None):
    """ load the variable, subset in space, and convert to daily mean """
    data = get_var_data(ds,var_name)
    data = select_level(ds,data,lev_in,var_name)
    data = subset_lat_lon(ds,data,lon1=lon1,lon2=lon2)
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
    return data
#---------------------------------------------------------------------------------------------------
def get_ind_file(case_in):
    return f'{tmp_file_root}/{tmp_file_head}.{case_in}.index_{ind_var}.{tmp_file_tail}.nc'
def get_reg_file(case_in,var_name):
    return f'{tmp_file_root}/{tmp_file_head}.{case_in}.{var_name}.{tmp_file_tail}.nc'
#---------------------------------------------------------------------------------------------------
def calc_index(ds,case_in):
    """ normalized band-pass filtered MJO index from the equatorial index box """
    print(' '*4+f'calculating index ({hapy.tclr.CYAN}{ind_var}{hapy.tclr.END})...')
    ind = load_daily(ds,ind_var,None,lon1=ilon-iwid,lon2=ilon+iwid)
    time = ind['time'].values
    #-------------------------------------------------------------------------
    if filter_data:
        print(' '*4+f'filtering index ({int(fhi)}-{int(flo)} day)...')
        ind_filt = filter_numba( np.ascontiguousarray(ind.values,dtype=np.float64),
                                 wgt, tvals0, win_width )
    else:
        ind_filt = np.ascontiguousarray(ind.values,dtype=np.float64)
    #-------------------------------------------------------------------------
    with warnings.catch_warnings():                          # edge times are all NaN
        warnings.simplefilter('ignore',category=RuntimeWarning)
        ind_filt = np.nanmean(ind_filt,axis=1)               # average over the index box
    if ind_var in ['FLNT','FLUT','LW_flux_up_at_model_top']:
        ind_filt = ind_filt * -1                             # flip sign of OLR
    ind_filt = ind_filt - np.nanmean(ind_filt)               # normalize
    ind_filt = ind_filt / np.nanstd(ind_filt)
    #-------------------------------------------------------------------------
    ind_da = xr.DataArray( ind_filt, coords=[('time',time)], dims=['time'], name=ind_var )
    return ind_da
#---------------------------------------------------------------------------------------------------
def calc_regression(ds,ind,var_name,lev_in):
    """ meridionally average, filter, and regress onto the index """
    print(' '*4+f'loading data ({hapy.tclr.CYAN}{var_name}{hapy.tclr.END})...')
    data = load_daily(ds,var_name,lev_in)
    if print_stats: hapy.print_stat(data,name=var_name,stat='naxsh',indent=' '*6,compact=True)
    #-------------------------------------------------------------------------
    print(' '*4+'averaging onto longitude bins...')
    data_mavg = meridional_avg( np.ascontiguousarray(data.values,dtype=np.float64),
                                np.ascontiguousarray(data['lon'].values,dtype=np.float64),
                                lon_bins, float(dlon) )
    #-------------------------------------------------------------------------
    if filter_data:
        print(' '*4+f'filtering data ({int(fhi)}-{int(flo)} day)...')
        data_mavg = filter_numba(data_mavg,wgt,tvals0,win_width)
    #-------------------------------------------------------------------------
    # align the index with the data time coordinate
    ind_tmp = ind.sel(time=data['time'],method='nearest')
    index = np.ascontiguousarray(ind_tmp.values,dtype=np.float64)
    #-------------------------------------------------------------------------
    # base times must have a valid index value, valid data at every lag (the
    # filter leaves NaNs at each end of the record), and fall within the requested months
    ntime = data_mavg.shape[0]
    time_valid = np.isfinite(data_mavg).any(axis=1) & np.isfinite(index)
    base_mask = np.ones([ntime],dtype=bool)
    for lag in range(-mxlag,mxlag+1):
        base_mask = base_mask & np.roll(time_valid,-lag)
    if mxlag>0:                  # np.roll wraps around - discard the wrapped edges
        base_mask[:mxlag]  = False
        base_mask[-mxlag:] = False
    if regression_months is not None:
        base_mask = base_mask & np.isin( data['time.month'].values, regression_months )
    print(' '*4+f'regressing data on index  ({np.sum(base_mask)} base times)...')
    if np.sum(base_mask)<10: raise ValueError('not enough valid base times for the regression!')
    #-------------------------------------------------------------------------
    lag_coord = np.arange( -mxlag, mxlag+1, 1, dtype=np.int64 )
    reg = lag_regression_numba(data_mavg, index, lag_coord, base_mask)
    #-------------------------------------------------------------------------
    reg_ds = xr.Dataset()
    reg_ds[f'{var_name}_reg'] = xr.DataArray( reg, coords=[('lag',lag_coord),('lon',lon_bins)],
                                              dims=['lag','lon'] )
    return reg_ds
#---------------------------------------------------------------------------------------------------
def get_regression(case_in,case_opts,var_name,lev_in,htype_in,ind):
    """ read the cached regression or recalculate it """
    reg_file = get_reg_file(case_in,var_name)
    if recalculate_reg:
        ds = open_case_dataset(case_in,case_opts,htype_in)
        reg_ds = calc_regression(ds,ind,var_name,lev_in)
        print(' '*4+f'writing regression to file: {reg_file}')
        reg_ds.to_netcdf(path=reg_file,mode='w')
    else:
        print(' '*4+f'reading regression from file: {reg_file}')
        reg_ds = xr.open_dataset( reg_file )
    return reg_ds[f'{var_name}_reg']
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
def get_clev(data_list, nlev=21):
    """ symmetric contour levels spanning all cases """
    dmin = np.min([np.nanmin(d) for d in data_list])
    dmax = np.max([np.nanmax(d) for d in data_list])
    cmin,cmax,cint = nice_cntr_levels(dmin,dmax,max_steps=nlev,aboutZero=True)
    clev = np.arange(cmin,cmax+cint/2,cint)
    if len(clev)<3: clev = np.linspace(cmin,cmax,num=nlev)
    return clev
#---------------------------------------------------------------------------------------------------
def format_ax(ax, title_left, title_right, mean_str=None):
    ax.set_title(title_left,  fontsize=FONT_MD, loc='left')
    ax.set_title(title_right, fontsize=FONT_MD, loc='right')
    ax.set_xlim(plot_lon1,plot_lon2)
    ax.set_ylim(-mxlag,mxlag)
    xt_val = np.arange(0,360+60,60)
    xt_str = ['0','60E','120E','180','120W','60W','0']
    keep = [ (t>=plot_lon1) and (t<=plot_lon2) for t in xt_val ]
    ax.set_xticks( xt_val[keep] )
    ax.set_xticklabels( [s for s,k in zip(xt_str,keep) if k], fontsize=FONT_MD )
    ax.set_yticks( np.arange(-mxlag,mxlag+5,5) )
    ax.tick_params(axis='both',which='both',labelsize=FONT_MD)
    ax.set_xlabel('Longitude',fontsize=FONT_MD)
    ax.set_ylabel('Lag [days]',fontsize=FONT_MD)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(ilon, color='black', linewidth=0.5, linestyle='dotted')
    if mean_str is not None:
        ax.text(0.5,0.95,mean_str,transform=ax.transAxes,ha='center',va='top',
                fontsize=FONT_SM,backgroundcolor='white')
#---------------------------------------------------------------------------------------------------
# build the list of panels along the case dimension
#    each panel is a tuple of (kind, case_index) where kind is 'full' or 'diff'
case_panels = []
if plot_diff and add_diff:
    for c in range(num_case): case_panels.append(('full',c))
    for c in range(num_case):
        if c!=diff_base: case_panels.append(('diff',c))
elif plot_diff:
    for c in range(num_case):
        case_panels.append( ('full',c) if c==diff_base else ('diff',c) )
else:
    for c in range(num_case): case_panels.append(('full',c))
num_panel = len(case_panels)

#---------------------------------------------------------------------------------------------------
# Calculate the MJO index for each case
#---------------------------------------------------------------------------------------------------
ind_list = []
for c in range(num_case):
    case_opts = case_opts_list[c]
    hapy.print_line()
    print(' '*2+f'case: {hapy.tclr.GREEN}{case[c]}{hapy.tclr.END}')
    ind_file = get_ind_file(case[c])
    ind_htype_loc = ind_htype if ind_htype is not None else case_opts.get('htype',htype)
    if recalculate_ind:
        ds = open_case_dataset(case[c],case_opts,ind_htype_loc)
        hapy.print_time_length(ds,indent=' '*4)
        ind = calc_index(ds,case[c])
        print(' '*4+f'writing index to file: {ind_file}')
        ind.to_netcdf(path=ind_file,mode='w')
    else:
        print(' '*4+f'reading index from file: {ind_file}')
        ind = xr.open_dataset( ind_file )[ind_var]
        ind.load()
    if print_stats: hapy.print_stat(ind,name='MJO index',stat='naxsh',indent=' '*6,compact=True)
    ind_list.append(ind)

#---------------------------------------------------------------------------------------------------
# Calculate the lagged regressions
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
        #-----------------------------------------------------------------------
        htype_svar = var_opts.get('htype',      case_opts.get('htype',htype))
        htype_cvar = var_opts.get('cvar_htype', htype_svar)
        #-----------------------------------------------------------------------
        sdata_list.append( get_regression(case[c],case_opts,var[v],var_opts.get('lev',None),
                                          htype_svar,ind_list[c]) )
        #-----------------------------------------------------------------------
        if var_opts.get('cvar',None) is not None:
            cdata_list.append( get_regression(case[c],case_opts,var_opts['cvar'],
                                              var_opts.get('cvar_lev',None),
                                              htype_cvar,ind_list[c]) )
    sdata_all[v] = sdata_list
    cdata_all[v] = cdata_list

#---------------------------------------------------------------------------------------------------
if not create_plot: exit('\nExiting because create_plot is False.\n')
#---------------------------------------------------------------------------------------------------
# Create plot
#---------------------------------------------------------------------------------------------------
(d1,d2) = (num_var,num_panel) if var_x_case else (num_panel,num_var)
fdx,fdy = 6,3
fig, axs = plt.subplots(d1,d2, figsize=(fdx*d2,fdy*d1), squeeze=False)
fig.subplots_adjust(hspace=0.5, wspace=0.25)

panel_labels = list(string.ascii_lowercase)

for v in range(num_var):
    var_opts   = var_opts_list[v]
    sdata_list = sdata_all[v]
    cdata_list = cdata_all[v]
    use_cvar   = var_opts.get('cvar',None) is not None
    #-----------------------------------------------------------------------
    if print_stats:
        print()
        for c in range(num_case):
            hapy.print_stat(np.ma.masked_invalid(sdata_list[c].values),
                            name=f'{case_opts_list[c]["n"]:24} {var[v]}',
                            stat='naxsh',indent=' '*4,compact=True)
    #-----------------------------------------------------------------------
    sdiff_list = [ sdata_list[c] - sdata_list[diff_base] for c in range(num_case) ] if plot_diff else None
    cdiff_list = None
    if plot_diff and use_cvar:
        cdiff_list = [ cdata_list[c] - cdata_list[diff_base] for c in range(num_case) ]
    #-----------------------------------------------------------------------
    # contour levels are common across all cases for a consistent comparison
    sclev = var_opts.get('clev',None)
    if sclev is None: sclev = get_clev(sdata_list)
    cclev = get_clev(cdata_list,nlev=11) if use_cvar else None
    sclev_diff = get_clev([ sdiff_list[c] for c in range(num_case) if c!=diff_base ]) \
                 if (plot_diff and num_case>1) else None
    #-----------------------------------------------------------------------
    cmap = var_opts.get('cmap',cmocean.cm.balance)
    #-----------------------------------------------------------------------
    for p in range(num_panel):
        kind,c = case_panels[p]
        ax = axs[v,p] if var_x_case else axs[p,v]
        #-------------------------------------------------------------------
        if kind=='diff':
            sdata = sdiff_list[c]
            clev_tmp = sclev_diff if sclev_diff is not None else sclev
            title_left = f'({panel_labels[p]})  {case_opts_list[c]["n"]} - {case_opts_list[diff_base]["n"]}'
        else:
            sdata = sdata_list[c]
            clev_tmp = sclev
            title_left = f'({panel_labels[p]})  {case_opts_list[c]["n"]}'
        #-------------------------------------------------------------------
        lon = sdata['lon'].values
        lag = sdata['lag'].values
        norm = BoundaryNorm(clev_tmp, ncolors=cmap.N, clip=False)
        cf = ax.contourf(lon, lag, sdata.values, levels=clev_tmp, cmap=cmap, norm=norm, extend='both')
        #-------------------------------------------------------------------
        # overlay the contour variable - negative values are dashed
        if use_cvar:
            cdata = cdiff_list[c] if kind=='diff' else cdata_list[c]
            clev_c = cclev
            neg_lev = [l for l in clev_c if l<0]
            pos_lev = [l for l in clev_c if l>0]
            if neg_lev: ax.contour(cdata['lon'].values, cdata['lag'].values, cdata.values,
                                   levels=neg_lev, colors='black', linewidths=0.6, linestyles='--')
            if pos_lev: ax.contour(cdata['lon'].values, cdata['lag'].values, cdata.values,
                                   levels=pos_lev, colors='black', linewidths=0.6, linestyles='-')
        #-------------------------------------------------------------------
        title_right = var_opts['str']
        if use_cvar: title_right += f'  /  {var_opts["cvar_str"]}'
        format_ax(ax, title_left, title_right)
        #-------------------------------------------------------------------
        if not use_common_label_bar:
            cbar = fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, fraction=0.03)
            cbar.ax.tick_params(labelsize=FONT_SM)
    #-----------------------------------------------------------------------
    if use_common_label_bar:
        ax_list = [ axs[v,p] for p in range(num_panel) ] if var_x_case else \
                  [ axs[p,v] for p in range(num_panel) ]
        cbar = fig.colorbar(cf, ax=ax_list, orientation='vertical', fraction=0.02, pad=0.02, shrink=0.8)
        cbar.ax.tick_params(labelsize=FONT_SM)

#---------------------------------------------------------------------------------------------------
# Finalize plot
#---------------------------------------------------------------------------------------------------
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
