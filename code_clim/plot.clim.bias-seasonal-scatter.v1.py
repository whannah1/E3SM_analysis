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
fig_file = 'figs_clim/clim.bias-seasonal-scatter.v1.png'
#---------------------------------------------------------------------------------------------------
if host=='olcf':

    ### TMS parameter sensitivity tests
    tmp_path,tmp_sub = '/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch','run'
    land_frac_file = '/lustre/orion/cli115/world-shared/e3sm/inputdata/share/domains/domain.lnd.ne30pg2_gx1v6.190806.nc'
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_0', n='noTMS', c='tab:gray',  p=tmp_path,s=tmp_sub)
    
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1',                           n='w/TMS',                            c='red', p=tmp_path,s=tmp_sub)

    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.5',                   n='w/TMS+ohf=0.5',                    c='tab:cyan',   p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.rlf_0.035',                 n='w/TMS+rlf=0.035',                  c='tab:purple', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.5',                   n='w/TMS+dsf=0.5',                    c='tab:orange', p=tmp_path,s=tmp_sub)
    
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_1.5.dsf_0.5',           n='w/TMS+ohf=1.5+dsf=0.5',            c='tab:red',    p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohn_0.5',                   n='w/TMS+ohn=0.5',                    c='tab:green',  p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohn_2',                     n='w/TMS+ohn=2',                      c='tab:olive',  p=tmp_path,s=tmp_sub)
    
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25',                  n='w/TMS+ohf=0.25',                   c='blue',       p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.rlf_0.01',                  n='w/TMS+rlf=0.01',                   c='fuchsia',    p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.25',                  n='w/TMS+dsf=0.25',                   c='peru',       p=tmp_path,s=tmp_sub)

    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.4.rlf_0.07.dsf_0.4',  n='w/TMS+ohf=0.4+rlf=0.07+dsf=0.4',   c='aquamarine', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.75.rlf_0.04.dsf_0.25',n='w/TMS+ohf=0.75+rlf=0.04+dsf=0.25', c='teal',       p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25.rlf_0.04.dsf_0.75',n='w/TMS+ohf=0.25+rlf=0.04+dsf=0.75', c='gold',       p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.5.rlf_0.04.dsf_0.5',  n='w/TMS+ohf=0.5+rlf=0.04+dsf=0.5',   c='skyblue',    p=tmp_path,s=tmp_sub)

    # # recolor previous cases to highlight new parameter choices below
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1',                             n='w/TMS',                            c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.5',                     n='w/TMS+ohf=0.5',                    c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.rlf_0.035',                   n='w/TMS+rlf=0.035',                  c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.5',                     n='w/TMS+dsf=0.5',                    c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_1.5.dsf_0.5',             n='w/TMS+ohf=1.5+dsf=0.5',            c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohn_0.5',                     n='w/TMS+ohn=0.5',                    c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohn_2',                       n='w/TMS+ohn=2',                      c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25',                    n='w/TMS+ohf=0.25',                   c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.rlf_0.01',                    n='w/TMS+rlf=0.01',                   c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.25',                    n='w/TMS+dsf=0.25',                   c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.4.rlf_0.07.dsf_0.4',    n='w/TMS+ohf=0.4+rlf=0.07+dsf=0.4',   c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.75.rlf_0.04.dsf_0.25',  n='w/TMS+ohf=0.75+rlf=0.04+dsf=0.25', c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25.rlf_0.04.dsf_0.75',  n='w/TMS+ohf=0.25+rlf=0.04+dsf=0.75', c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.5.rlf_0.04.dsf_0.5',    n='w/TMS+ohf=0.5+rlf=0.04+dsf=0.5',   c='whitesmoke',p=tmp_path,s=tmp_sub)

    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.5.rlf_0.15.wsn_1',            n='w/TMS+ohf=0.50+rlf=0.15+wsn=1',         c='navy',      p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25.wsn_1',                    n='w/TMS+ohf=0.25+wsn=1',                  c='tab:blue',  p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25.rlf_0.3.wsn_1',            n='w/TMS+ohf=0.25+rlf=0.3+wsn=1',          c='tab:brown', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25.rlf_0.04.dsf_0.75.wsn_1',  n='w/TMS+ohf=0.25+rlf=0.04+dsf=0.75+wsn=1',c='maroon',    p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25.rlf_0.3.dsf_0.75',         n='w/TMS+ohf=0.25+rlf=0.3+dsf=0.75',       c='tab:pink',  p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25.dsf_0.75',                 n='w/TMS+ohf=0.25+dsf=0.75',               c='lime',      p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25.rlf_0.3',                  n='w/TMS+ohf=0.25+rlf=0.3',                c='darkgreen', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.5.rlf_0.15',                  n='w/TMS+ohf=0.50+rlf=0.15',               c='indigo',    p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.5.wsn_4',                     n='w/TMS+dsf=0.50+wsn=4',                  c='rosybrown', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.wsn_1',                             n='w/TMS+wsn=1',                           c='slateblue', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.wsn_4',                             n='w/TMS+wsn=4',                           c='darkkhaki', p=tmp_path,s=tmp_sub)
    # add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.wsn_10',                            n='w/TMS+wsn=10',                          c='black',     p=tmp_path,s=tmp_sub)

    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.5.rlf_0.15.wsn_1',            n='w/TMS+ohf=0.50+rlf=0.15+wsn=1',         c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25.wsn_1',                    n='w/TMS+ohf=0.25+wsn=1',                  c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25.rlf_0.3.wsn_1',            n='w/TMS+ohf=0.25+rlf=0.3+wsn=1',          c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25.rlf_0.04.dsf_0.75.wsn_1',  n='w/TMS+ohf=0.25+rlf=0.04+dsf=0.75+wsn=1',c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25.rlf_0.3.dsf_0.75',         n='w/TMS+ohf=0.25+rlf=0.3+dsf=0.75',       c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25.dsf_0.75',                 n='w/TMS+ohf=0.25+dsf=0.75',               c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.25.rlf_0.3',                  n='w/TMS+ohf=0.25+rlf=0.3',                c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.ohf_0.5.rlf_0.15',                  n='w/TMS+ohf=0.50+rlf=0.15',               c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.5.wsn_4',                     n='w/TMS+dsf=0.50+wsn=4',                  c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.wsn_1',                             n='w/TMS+wsn=1',                           c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.wsn_4',                             n='w/TMS+wsn=4',                           c='whitesmoke',p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.wsn_10',                            n='w/TMS+wsn=10',                          c='whitesmoke',p=tmp_path,s=tmp_sub)

    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.05.wsn_1',                    n='w/TMS+dsf=0.05+wsn=1', c='green',   p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.1.wsn_1',                     n='w/TMS+dsf=0.1+wsn=1',  c='blue',    p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.2.wsn_1',                     n='w/TMS+dsf=0.2+wsn=1',  c='cyan',    p=tmp_path,s=tmp_sub)
    add_case('E3SM.2026-TMS-TEST-01.ne30pg2.rmf_1.tms_1.dsf_0.4.wsn_1',                     n='w/TMS+dsf=0.4+wsn=1',  c='purple',  p=tmp_path,s=tmp_sub)
    

    htype,first_file,num_files = 'output.1ma.AVERAGE.nmonths_x1',None,60
    # htype,first_file,num_files = 'output.1ma.AVERAGE.nmonths_x1',0,12*3
    # htype,first_file,num_files = 'output.1ma.AVERAGE.nmonths_x1',12*2,12*3

    obs_root = '/lustre/orion/cli115/proj-shared/hannah6/obs_data/ERA5/diag_data_remap_ne30pg2'
    
    # add_var('T_2m', s='T_2m [K]', obs_file=f'{obs_root}/t2m_197901_201912.remap_ne30pg2.nc', obs_var='t2m',stat='rmse')
    add_var('T_2m', s='T_2m [K]', obs_file=f'{obs_root}/t2m_197901_201912.remap_ne30pg2.nc', obs_var='t2m',stat='bias_pos')
    add_var('T_2m', s='T_2m [K]', obs_file=f'{obs_root}/t2m_197901_201912.remap_ne30pg2.nc', obs_var='t2m',stat='bias_neg')

#---------------------------------------------------------------------------------------------------
# seasons for the x and y axes of the scatter plot
season_x,season_y = 'DJF','JJA'
season_list = [season_x,season_y]

# default statistic to plot (override per variable with add_var(...,stat=...))
# 'rmse'     => area-weighted RMSE
# 'bias_pos' => area-weighted sum of positive biases only, normalized by total region area
# 'bias_neg' => area-weighted sum of negative biases only, normalized by total region area
# (bias_pos + bias_neg = area-weighted mean bias, so these show the magnitude of each sign before cancellation)
stat_default = 'rmse'

# region for the area-weighted RMSE: 'global' / 'land' / 'ocean'
region = 'land'
land_frac_thresh = 1.0   # land/ocean classification threshold for the land fraction

# optionally restrict obs years used for the climatology (None => use all)
obs_yr1,obs_yr2 = None,None
#---------------------------------------------------------------------------------------------------
if case==[]: raise ValueError('ERROR - case list is empty!')
if var==[]: raise ValueError('ERROR - var list is empty!')
for v in range(len(var)):
    var_opts_list[v].setdefault('stat', stat_default)
    if var_opts_list[v]['stat'] not in ['rmse','bias_pos','bias_neg']:
        raise ValueError(f'stat not recognized for {var[v]}: {var_opts_list[v]["stat"]}')
num_var,num_case = len(var),len(case)

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
def calc_rmse(diff, wgt):
    # area-weighted root mean square error
    valid = (wgt>0) & np.isfinite(diff)
    x, w = diff[valid], wgt[valid]
    return float(np.sqrt((w*x**2).sum()/w.sum()))
#---------------------------------------------------------------------------------------------------
def calc_bias_sign(diff, wgt, sign):
    # area-weighted sum of biases with only one sign, normalized by the total region area
    # so that the positive and negative parts add up to the area-weighted mean bias
    valid = (wgt>0) & np.isfinite(diff)
    x, w = diff[valid], wgt[valid]
    if sign>0: x = np.where(x>0, x, 0.)
    if sign<0: x = np.where(x<0, x, 0.)
    return float((w*x).sum()/w.sum())
#---------------------------------------------------------------------------------------------------
def calc_stat(diff, wgt, stat):
    if stat=='rmse'    : return calc_rmse(diff, wgt)
    if stat=='bias_pos': return calc_bias_sign(diff, wgt, 1)
    if stat=='bias_neg': return calc_bias_sign(diff, wgt,-1)
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
fig, axs = plt.subplots(1, num_var, figsize=(7*num_var,6), squeeze=False)
axs = axs[0,:]
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
    var_opts = var_opts_list[v]
    stat = var_opts['stat']
    stat_str = stat.upper() if stat=='rmse' else stat
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
    vals = {s:np.full(num_case,np.nan) for s in season_list}
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
        for season in season_list:
            diff = mod_clim[season].values - obs_clim[season].values
            vals[season][c] = calc_stat(diff, wgt.values, stat)
        print(' '*6+f'{season_x} {stat}: {vals[season_x][c]:7.3f}   {season_y} {stat}: {vals[season_y][c]:7.3f}')
    #----------------------------------------------------------------------------
    # print table of plotted values
    name_len = max(len(n) for n in name)
    print()
    print(' '*2+f'{var_str[v]} - {region} {stat_str} (vs ERA5)')
    print(' '*2+f'{"case":{name_len}}  {season_x:>8}  {season_y:>8}  {"mean":>8}')
    print(' '*2+'-'*(name_len+30))
    for c in range(num_case):
        vals_mean = (vals[season_x][c]+vals[season_y][c])/2
        print(' '*2+f'{name[c]:{name_len}}  {vals[season_x][c]:8.3f}  {vals[season_y][c]:8.3f}  {vals_mean:8.3f}')
    #----------------------------------------------------------------------------
    ax = axs[v]
    for c in range(num_case):
        ax.scatter(vals[season_x][c], vals[season_y][c], s=120, color=clr[c],
                   edgecolor='black', linewidth=0.8, zorder=3, label=name[c])
    # use the same range on both axes so the 1:1 line is meaningful
    vmin = min(np.nanmin(vals[season_x]), np.nanmin(vals[season_y]))
    vmax = max(np.nanmax(vals[season_x]), np.nanmax(vals[season_y]))
    pad = 0.1*(vmax-vmin) if vmax>vmin else 0.1*abs(vmax)
    lim = (vmin-pad, vmax+pad)
    ax.plot(lim, lim, color='gray', linewidth=0.8, linestyle='--', zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.set_xlabel(f'{season_x} {stat_str}', fontsize=12)
    ax.set_ylabel(f'{season_y} {stat_str}', fontsize=12)
    ax.set_title(var_str[v], fontsize=16, loc='left')
    ax.set_title(f'{region} (vs ERA5)', fontsize=12, loc='right')
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    if v==num_var-1: ax.legend(fontsize=10, frameon=False, loc='upper left', bbox_to_anchor=(1.01,1))
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
