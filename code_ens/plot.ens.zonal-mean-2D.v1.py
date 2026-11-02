#---------------------------------------------------------------------------------------------------
# Plot zonal means of 2D fields for an ensemble of standalone cases (i.e. "bundled" runs where
# each member is a normal case with its own case directory - not the multi-instance capability)
# thin colored line for each member and a thick black line for the ensemble mean
#---------------------------------------------------------------------------------------------------
from time import perf_counter
timer_start = perf_counter()
import os, glob, warnings, numpy as np, xarray as xr
import matplotlib; matplotlib.use('Agg')
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
if host=='alcf':

    ### E3SMv4 ZM tuning ensemble (bundled runs)
    tmp_scratch = '/lus/flare/projects/E3SM_Dec/whannah/scratch'
    grid = 'ne30' # ne30 / ne256
    ens_name = f'ZM-TUNE-00.{grid}'
    if grid=='ne30':
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE',                     n='default',               p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.alfa_0.1',            n='alfa_0.1',              p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.alfa_0.2',            n='alfa_0.2',              p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.alfa_0.3',            n='alfa_0.3',              p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.c0l_0.001',           n='c0l_0.001',             p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.c0l_0.003',           n='c0l_0.003',             p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.c0l_0.003.c0o_0.003', n='c0l_0.003.c0o_0.003',   p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.c0o_0.001',           n='c0o_0.001',             p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.c0o_0.003',           n='c0o_0.003',             p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cape_250',            n='cape_250',              p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cape_500',            n='cape_500',              p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cape_750',            n='cape_750',              p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1',               n='cda_1',                 p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.dcape_0.01',          n='dcape_0.01',            p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.dcape_0.05',          n='dcape_0.05',            p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.dcape_0.1',           n='dcape_0.1',             p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ke_1e-5',             n='ke_1e-5',               p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ke_5e-6',             n='ke_5e-6',               p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ke_7.5e-6',           n='ke_7.5e-6',             p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.mtc_0.3.mqc_0.3',     n='mtc_0.3.mqc_0.3',       p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.mtc_0.5.mqc_0',       n='mtc_0.5.mqc_0',         p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.mtc_0.5.mqc_0.3',     n='mtc_0.5.mqc_0.3',       p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.mtc_0.5.mqc_0.5',     n='mtc_0.5.mqc_0.5',       p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ncin_2',              n='ncin_2',                p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ncin_4',              n='ncin_4',                p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ncin_6',              n='ncin_6',                p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tau_1800',            n='tau_1800',              p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tau_7200',            n='tau_7200',              p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tka_0.8',             n='tka_0.8',               p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tka_1.6',             n='tka_1.6',               p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tpf_1',               n='tpf_1',                 p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tpf_3',               n='tpf_3',                 p=tmp_scratch, s='run')
    if grid=='ne256':
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE',                     n='default',               p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.alfa_0.1',            n='alfa_0.1',              p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.alfa_0.2',            n='alfa_0.2',              p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.alfa_0.3',            n='alfa_0.3',              p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.c0l_0.001',           n='c0l_0.001',             p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.c0l_0.003',           n='c0l_0.003',             p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.c0l_0.003.c0o_0.003', n='c0l_0.003.c0o_0.003',   p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.c0o_0.001',           n='c0o_0.001',             p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.c0o_0.003',           n='c0o_0.003',             p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.cape_250',            n='cape_250',              p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.cape_500',            n='cape_500',              p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.cape_750',            n='cape_750',              p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.cda_1',               n='cda_1',                 p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.dcape_0.01',          n='dcape_0.01',            p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.dcape_0.05',          n='dcape_0.05',            p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.dcape_0.1',           n='dcape_0.1',             p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.ke_1e-5',             n='ke_1e-5',               p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.ke_5e-6',             n='ke_5e-6',               p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.ke_7.5e-6',           n='ke_7.5e-6',             p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.mtc_0.3.mqc_0.3',     n='mtc_0.3.mqc_0.3',       p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.mtc_0.5.mqc_0',       n='mtc_0.5.mqc_0',         p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.mtc_0.5.mqc_0.3',     n='mtc_0.5.mqc_0.3',       p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.mtc_0.5.mqc_0.5',     n='mtc_0.5.mqc_0.5',       p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.ncin_2',              n='ncin_2',                p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.ncin_4',              n='ncin_4',                p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.ncin_6',              n='ncin_6',                p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.tau_1800',            n='tau_1800',              p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.tau_7200',            n='tau_7200',              p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.tka_0.8',             n='tka_0.8',               p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.tka_1.6',             n='tka_1.6',               p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.tpf_1',               n='tpf_1',                 p=tmp_scratch, s='run')
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne256.F2010xx-ZM-CICE.tpf_3',               n='tpf_3',                 p=tmp_scratch, s='run')

    # NOTE - ne256 output is remapped to ne30pg2 online
    htype = 'output.1ma.AVERAGE'
    first_file,num_files = 0,None

#---------------------------------------------------------------------------------------------------

add_var('RESTOM')
add_var('SW_net_at_model_top', s='Net SW (TOA)')
add_var('LW_flux_up_at_model_top', s='OLR')
# add_var('ShortwaveCloudForcing', s='SWCF')
# add_var('LongwaveCloudForcing', s='LWCF')
# add_var('precip_total_surf_mass_flux', s='precip')
# add_var('zm_prec')
# add_var('zm_prec_frac')
# add_var('VapWaterPath')
# add_var('LiqWaterPath')
# add_var('IceWaterPath')
# add_var('T_2m')
# add_var('surf_evap', unit_fac=86400)

#---------------------------------------------------------------------------------------------------

season = 'ANN'  # ANN / DJF / JJA / MAM / SON

lat1,lat2,dlat = -90,90,2

use_sinlat         = True   # x-axis spaced by sin(lat) to make areas visually comparable
diff_from_ens_mean = False  # plot each member as anomaly from ensemble mean
add_legend         = True   # legend includes global mean of each member
print_stats        = True

num_plot_col = 2

fig_file = f'figs_ens/ens.zonal-mean-2D.v1.{ens_name}.{season}.png'

#---------------------------------------------------------------------------------------------------
season_months = {'ANN':None, 'DJF':[12,1,2], 'MAM':[3,4,5], 'JJA':[6,7,8], 'SON':[9,10,11]}
if season not in season_months: raise ValueError(f'season not supported: {season}')
#---------------------------------------------------------------------------------------------------
# Set up plot resources
if case==[]: raise ValueError('ERROR - case list is empty!')
if var==[]: raise ValueError('ERROR - var list is empty!')
num_var,num_case = len(var),len(case)

if 'first_file' not in locals(): first_file = None
if 'num_files'  not in locals(): num_files  = None

clr = hapy.fill_color_list(clr)
#---------------------------------------------------------------------------------------------------
def get_file_list(c):
    file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/*{htype}*'
    file_list = sorted(glob.glob(file_path))
    if first_file is not None and first_file!=0: file_list = file_list[first_file:]
    if num_files  is not None and num_files !=0: file_list = file_list[:num_files]
    return file_list
#---------------------------------------------------------------------------------------------------
def get_data(ds,var_name):
    if var_name=='RESTOM':
        data = ds['SW_flux_dn_at_model_top'] \
              -ds['SW_flux_up_at_model_top'] \
              -ds['LW_flux_up_at_model_top']
    elif var_name=='SW_net_at_model_top':
        data = ds['SW_flux_dn_at_model_top'] - ds['SW_flux_up_at_model_top']
    elif var_name=='zm_prec_frac':
        data = ds['zm_prec'] / ds['precip_total_surf_mass_flux']
    else:
        data = ds[var_name]
    if 'prec' in var_name and var_name!='zm_prec_frac': data = data*86400*1e3 # m/s > mm/day
    return data
#---------------------------------------------------------------------------------------------------
def time_mean(data,ds):
    # EAMxx time stamps are at the end of the averaging window, so use the
    # left edge of time_bnds for month selection and weight by window length
    if 'time' not in data.dims: return data
    tbnd = ds['time_bnds']
    bdim = [d for d in tbnd.dims if d!='time'][0]
    month = tbnd.isel({bdim:0}).dt.month.values
    dt = np.array([ d.total_seconds() for d in (tbnd.isel({bdim:1})-tbnd.isel({bdim:0})).values ])
    wgt = xr.DataArray(dt, dims='time')
    if season_months[season] is not None:
        mask = np.isin(month, season_months[season])
        if not mask.any(): return None
        data,wgt = data.isel(time=mask), wgt.isel(time=mask)
    return (data*wgt).sum(dim='time') / wgt.sum()
#---------------------------------------------------------------------------------------------------
# load data - each case is opened once for all variables
zm_data   = [[None]*num_case for v in range(num_var)]  # zonal means
gbl_means = [[None]*num_case for v in range(num_var)]  # global means
for c in range(num_case):
    hapy.print_line()
    print(' '*2+'case: '+hapy.tclr.GREEN+case[c]+hapy.tclr.END)
    file_list = get_file_list(c)
    if file_list==[]:
        print(' '*4+f'{hapy.tclr.RED}WARNING - no files found - skipping{hapy.tclr.END}')
        continue
    print(' '*4+f'num files: {len(file_list)}')
    ds = xr.open_mfdataset( file_list, data_vars='minimal' )
    area = ds['area']
    if 'time' in area.dims: area = area.isel(time=0)
    area.load()
    lat = ds['lat']
    if 'time' in lat.dims: lat = lat.isel(time=0)
    lat.load()
    #----------------------------------------------------------------------------
    for v in range(num_var):
        try:
            data = get_data(ds,var[v])
        except KeyError as e:
            print(' '*4+f'{hapy.tclr.RED}WARNING - variable not found: {e} - skipping{hapy.tclr.END}')
            continue
        if 'unit_fac' in var_opts_list[v]: data = data*var_opts_list[v]['unit_fac']
        data = time_mean(data,ds)
        if data is None:
            print(' '*4+f'{hapy.tclr.RED}WARNING - no data for season {season} - skipping{hapy.tclr.END}')
            continue
        data.load()
        #-------------------------------------------------------------------------
        gbl_means[v][c] = float( (data*area).sum() / area.sum() )
        if print_stats: print(' '*4+f'{var[v]:30}  global mean: {gbl_means[v][c]:10.4f}')
        #-------------------------------------------------------------------------
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            bin_ds = hapy.bin_YbyX(data, lat, bin_min=lat1, bin_max=lat2, bin_spc=dlat, wgt=area)
        zm_data[v][c] = bin_ds['bin_val']
#---------------------------------------------------------------------------------------------------
# set up figure objects
ncols = min(num_plot_col, num_var)
nrows = int(np.ceil(num_var / float(ncols)))
fig, axs = plt.subplots(nrows,ncols, figsize=(8*ncols,5*nrows), squeeze=False )
axs_flat = axs.flatten()
for ax in axs_flat[num_var:]: ax.set_visible(False)

lat_ticks = np.array([-90,-60,-30,0,30,60,90])
#---------------------------------------------------------------------------------------------------
hapy.print_line()
print(f'  ensemble mean (season: {season})')
for v in range(num_var):
    ax = axs_flat[v]
    valid = [c for c in range(num_case) if zm_data[v][c] is not None]
    if valid==[]:
        print(' '*4+f'{hapy.tclr.RED}WARNING - no valid data for {var[v]}{hapy.tclr.END}')
        ax.set_visible(False)
        continue
    ens_mean = xr.concat([zm_data[v][c] for c in valid], dim='ens').mean(dim='ens')
    gbl_ens_mean = np.mean([gbl_means[v][c] for c in valid])
    print(' '*4+f'{var[v]:30}  global mean: {gbl_ens_mean:10.4f}   ({len(valid)} members)')
    #----------------------------------------------------------------------------
    lat_bins = ens_mean['bin'].values
    xx = np.sin(np.deg2rad(lat_bins)) if use_sinlat else lat_bins
    for c in valid:
        yy = zm_data[v][c] - ens_mean if diff_from_ens_mean else zm_data[v][c]
        ax.plot(xx, yy, color=clr[c], linewidth=0.8, label=f'{name[c]} ({gbl_means[v][c]:.2f})')
    if diff_from_ens_mean:
        ax.axhline(0, color='black', linewidth=2.5)
    else:
        ax.plot(xx, ens_mean, color='black', linewidth=2.5, label=f'ens mean ({gbl_ens_mean:.2f})')
    #----------------------------------------------------------------------------
    if var[v]=='RESTOM' and not diff_from_ens_mean: ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    if use_sinlat:
        ax.set_xticks(np.sin(np.deg2rad(lat_ticks)))
        ax.set_xlim(np.sin(np.deg2rad(lat1)), np.sin(np.deg2rad(lat2)))
    else:
        ax.set_xticks(lat_ticks)
        ax.set_xlim(lat1, lat2)
    ax.set_xticklabels([f'{t}°' for t in lat_ticks])
    ax.set_xlabel('Latitude')
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_title(ens_name, loc='left')
    ax.set_title(var_str[v] + (' (anom from ens mean)' if diff_from_ens_mean else ''), loc='right')
    ax.set_title(season, loc='center')
    if add_legend:
        ax.legend(loc='center left', bbox_to_anchor=(1.01,0.5), fontsize=6, ncol=1, frameon=False)
#---------------------------------------------------------------------------------------------------
# Finalize plot
fig.tight_layout()
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
