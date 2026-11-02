from time import perf_counter
timer_start = perf_counter()
import os, numpy as np, xarray as xr, glob
import matplotlib.pyplot as plt
import hapy
host = hapy.get_host()
xr.set_options(use_new_combine_kwarg_defaults=True)
#---------------------------------------------------------------------------------------------------
name,case,case_dir,case_sub,clr = [],[],[],[],[]
def add_case(case_in,n=None,p=None,s=None,c=None):
    global name,case,case_dir,case_sub,clr
    tmp_name = case_in if n is None else n
    case.append(case_in); name.append(tmp_name)
    case_dir.append(p); case_sub.append(s); clr.append(c)
#---------------------------------------------------------------------------------------------------
var,var_str = [],[]
var_opts_list = []
def add_var(var_name,s=None,**kwargs):
    var.append(var_name)
    var_str.append(var_name if s is None else s)
    var_opts = {}
    for k, val in kwargs.items(): var_opts[k] = val
    var_opts_list.append(var_opts)
#---------------------------------------------------------------------------------------------------
fig_file = 'figs_clim/clim.bias.v1.png'
#---------------------------------------------------------------------------------------------------
if host=='olcf':

    ### TMS parameter sensitivity tests
    tmp_path,tmp_sub = '/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch','run'
    land_frac_file = '/lustre/orion/cli115/world-shared/e3sm/inputdata/share/domains/domain.lnd.ne30pg2_gx1v6.190806.nc'
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_0',                   n='noTMS',                 c='tab:gray',  p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1',                   n='w/TMS',                 c='red',  p=tmp_path,s=tmp_sub)

    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.5',           n='w/TMS+ohf=0.5',        c='tab:cyan',  p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.rlf_0.035',         n='w/TMS+rlf=0.035',      c='tab:purple',p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.5',           n='w/TMS+dsf=0.5',        c='tab:orange',p=tmp_path,s=tmp_sub)
    
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_1.5.dsf_0.5',   n='w/TMS+ohf=1.5+dsf=0.5',c='tab:red',   p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohn_0.5',           n='w/TMS+ohn=0.5',        c='tab:green', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohn_2',             n='w/TMS+ohn=2',          c='tab:olive', p=tmp_path,s=tmp_sub)
    
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25',            n='w/TMS ohf=0.25',       c='blue',p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.rlf_0.01',            n='w/TMS rlf=0.01',       c='fuchsia',p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.25',            n='w/TMS dsf=0.25',       c='peru',p=tmp_path,s=tmp_sub)

    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.4.rlf_0.07.dsf_0.4',  n='w/TMS ohf=0.4+rlf=0.07+dsf=0.4',   c='aquamarine', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.75.rlf_0.04.dsf_0.25',n='w/TMS ohf=0.75+rlf=0.04+dsf=0.25', c='teal',       p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25.rlf_0.04.dsf_0.75',n='w/TMS ohf=0.25+rlf=0.04+dsf=0.75', c='gold',  p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.5.rlf_0.04.dsf_0.5',  n='w/TMS ohf=0.5+rlf=0.04+dsf=0.5',   c='skyblue',    p=tmp_path,s=tmp_sub)

    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.05.wsn_1',    n='w/TMS+dsf=0.05+wsn=1', c='green', p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.1.wsn_1',     n='w/TMS+dsf=0.1+wsn=1',  c='blue',  p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.2.wsn_1',     n='w/TMS+dsf=0.2+wsn=1',  c='cyan',  p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.4.wsn_1',     n='w/TMS+dsf=0.4+wsn=1',  c='purple',p=tmp_path,s=tmp_sub)

    htype,first_file,num_files = 'output.1ma.AVERAGE.nmonths_x1',None,60

    obs_root = '/lustre/orion/cli115/proj-shared/hannah6/obs_data/ERA5/diag_data_remap_ne30pg2'
    add_var('T_2m', s='T_2m [K]', obs_file=f'{obs_root}/t2m_197901_201912.remap_ne30pg2.nc', obs_var='t2m')
    add_var('T_2m', s='T_2m [K]', obs_file=f'{obs_root}/t2m_197901_201912.remap_ne30pg2.nc', obs_var='t2m', stat='rmse')

#---------------------------------------------------------------------------------------------------
season_list = ['ANN','DJF','JJA']

# default statistic to plot (override per variable with add_var(...,stat=...))
# 'bias' => box plot of spatial bias distribution / 'rmse' => bar plot of area-weighted RMSE
stat_default = 'bias'

# region for the area-weighted average: 'global' / 'land' / 'ocean'
region = 'land'
land_frac_thresh = 1.0   # land/ocean classification threshold for the land fraction

# box = 25-75th percentiles, whiskers = whis_pct percentiles, marker = area-weighted mean
# (percentiles are area-weighted, using the region mask)
whis_pct = (5,95)

# optionally restrict obs years used for the climatology (None => use all)
obs_yr1,obs_yr2 = None,None
#---------------------------------------------------------------------------------------------------
if case==[]: raise ValueError('ERROR - case list is empty!')
if var==[]: raise ValueError('ERROR - var list is empty!')
for v in range(len(var)):
    var_opts_list[v].setdefault('stat', stat_default)
    if var_opts_list[v]['stat'] not in ['bias','rmse']:
        raise ValueError(f'stat not recognized for {var[v]}: {var_opts_list[v]["stat"]}')
num_var,num_case,num_season = len(var),len(case),len(season_list)

if 'first_file' not in locals(): first_file = None
if 'num_files'  not in locals(): num_files  = None

season_months = {'ANN':list(range(1,13)), 'DJF':[12,1,2], 'MAM':[3,4,5], 'JJA':[6,7,8], 'SON':[9,10,11]}
#---------------------------------------------------------------------------------------------------
def get_month(ds):
    # model output time stamps are at the end of the averaging interval,
    # so use the start of the time bounds to identify the month
    if 'time_bnds' in ds:
        return ds['time_bnds'].isel(dim2=0).compute().dt.month
    return ds['time'].dt.month
#---------------------------------------------------------------------------------------------------
def seasonal_clim(data, month):
    # simple mean of monthly means for each season
    clim = {}
    for s in season_list:
        clim[s] = data.where(month.isin(season_months[s]), drop=True).mean(dim='time').load()
    return clim
#---------------------------------------------------------------------------------------------------
def weighted_percentile(x, w, pct):
    # percentile of x using weights w (midpoint of each weight interval in the CDF)
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    cdf = (np.cumsum(w) - 0.5*w) / w.sum()
    return np.interp(np.asarray(pct)/100., cdf, x)
#---------------------------------------------------------------------------------------------------
def calc_box_stats(diff, wgt):
    # area-weighted box plot statistics in the format expected by ax.bxp()
    valid = (wgt>0) & np.isfinite(diff)
    x, w = diff[valid], wgt[valid]
    p = weighted_percentile(x, w, [whis_pct[0],25,50,75,whis_pct[1]])
    return {'whislo':p[0], 'q1':p[1], 'med':p[2], 'q3':p[3], 'whishi':p[4],
            'mean':float((x*w).sum()/w.sum()), 'fliers':[]}
#---------------------------------------------------------------------------------------------------
def calc_rmse(diff, wgt):
    # area-weighted root mean square error
    valid = (wgt>0) & np.isfinite(diff)
    x, w = diff[valid], wgt[valid]
    return float(np.sqrt((w*x**2).sum()/w.sum()))
#---------------------------------------------------------------------------------------------------
# area weights with optional land/ocean mask
lfrac_ds = xr.open_dataset(land_frac_file)
area = xr.DataArray(lfrac_ds['area'].values.ravel(), dims='ncol')
lfrac = xr.DataArray(lfrac_ds['frac'].values.ravel(), dims='ncol')
if   region=='global': wgt = area
elif region=='land'  : wgt = area.where(lfrac>=land_frac_thresh, 0.)
elif region=='ocean' : wgt = area.where(lfrac< land_frac_thresh, 0.)
else: raise ValueError(f'region not recognized: {region}')
#---------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(num_var, 1, figsize=(10,5*num_var), squeeze=False)
axs = axs[:,0]

box_width = 0.8/num_case
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
    var_opts = var_opts_list[v]
    stat = var_opts['stat']
    hapy.print_line()
    print(' '*2+'var: '+hapy.tclr.MAGENTA+var[v]+hapy.tclr.END+f'  ({stat})')
    #----------------------------------------------------------------------------
    # obs climatology
    obs_ds = xr.open_dataset(var_opts['obs_file'])
    obs = obs_ds[var_opts['obs_var']]
    if obs_yr1 is not None: obs = obs.where(obs['time.year']>=obs_yr1, drop=True)
    if obs_yr2 is not None: obs = obs.where(obs['time.year']<=obs_yr2, drop=True)
    obs_clim = seasonal_clim(obs, obs['time.month'])
    #----------------------------------------------------------------------------
    box_stats = [[None]*num_season for c in range(num_case)]
    for c in range(num_case):
        print(' '*4+'case: '+hapy.tclr.GREEN+case[c]+hapy.tclr.END)
        htype_tmp = var_opts['htype'] if 'htype' in var_opts else htype
        file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/*{htype_tmp}*'
        file_list = sorted(glob.glob(file_path))
        if first_file is not None and first_file!=0: file_list = file_list[first_file:]
        if num_files  is not None and num_files !=0: file_list = file_list[:num_files]
        if file_list==[]: print('ERROR: Empty file list:'); print(); print(file_path); exit()
        #-------------------------------------------------------------------------
        # load the variable once and close the files - leaving many lazy datasets open
        # exceeds xarray's file cache (128) and can hang when files are re-opened
        with xr.open_mfdataset( file_list, data_vars='minimal' ) as ds:
            data = ds[var[v]].load()
            month = get_month(ds).load()
        mod_clim = seasonal_clim(data, month)
        #-------------------------------------------------------------------------
        for s,season in enumerate(season_list):
            diff = mod_clim[season].values - obs_clim[season].values
            if stat=='bias':
                bs = calc_box_stats(diff, wgt.values)
                box_stats[c][s] = bs
                print(' '*6+f'{season}  mean: {bs["mean"]:7.3f}  median: {bs["med"]:7.3f}'
                           +f'  IQR: [{bs["q1"]:7.3f},{bs["q3"]:7.3f}]'
                           +f'  p{whis_pct[0]}-p{whis_pct[1]}: [{bs["whislo"]:7.3f},{bs["whishi"]:7.3f}]')
            if stat=='rmse':
                box_stats[c][s] = calc_rmse(diff, wgt.values)
                print(' '*6+f'{season}  rmse: {box_stats[c][s]:7.3f}')
    #----------------------------------------------------------------------------
    ax = axs[v]
    x = np.arange(num_season)
    for c in range(num_case):
        xc = x - 0.4 + box_width*(c+0.5)
        if stat=='bias':
            ax.bxp(box_stats[c], positions=xc, widths=box_width*0.8, showmeans=True, showfliers=False,
                   patch_artist=True, manage_ticks=False,
                   boxprops=dict(facecolor=clr[c], edgecolor=clr[c], alpha=0.6),
                   medianprops=dict(color='black', linewidth=1.5),
                   whiskerprops=dict(color=clr[c], linewidth=1.5),
                   capprops=dict(color=clr[c], linewidth=1.5),
                   meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black', markersize=5))
            ax.bar(np.nan, np.nan, color=clr[c], alpha=0.6, label=name[c]) # legend proxy
        if stat=='rmse':
            ax.bar(xc, box_stats[c], width=box_width*0.8, color=clr[c], alpha=0.6,
                   edgecolor=clr[c], label=name[c])
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(season_list, fontsize=14)
    ax.set_ylabel(stat.upper() if stat=='rmse' else stat, fontsize=12)
    ax.set_title(var_str[v], fontsize=16, loc='left')
    rtitle = f'{region} (vs ERA5)'
    if stat=='bias': rtitle += f'  whiskers: p{whis_pct[0]}-p{whis_pct[1]}'
    ax.set_title(rtitle, fontsize=12, loc='right')
    ax.grid(axis='y', linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    if v==0: ax.legend(fontsize=10, frameon=False, loc='upper left', bbox_to_anchor=(1.01,1))
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
