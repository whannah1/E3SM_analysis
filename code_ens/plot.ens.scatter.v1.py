#---------------------------------------------------------------------------------------------------
# Scatter plot of global (or regional) means for an ensemble of standalone cases
# (i.e. "bundled" runs where each member is a normal case with its own case directory)
# one dot per case - one panel per pair of x/y variables
#---------------------------------------------------------------------------------------------------
from time import perf_counter
timer_start = perf_counter()
import os, glob, numpy as np, xarray as xr
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
var_opts_list = []
def add_var(xvar,yvar,xn=None,yn=None,**kwargs):
    var_opts = {}; var_opts['xvar'] = xvar; var_opts['yvar'] = yvar
    var_opts['xvar_name'] = xvar if xn is None else xn
    var_opts['yvar_name'] = yvar if yn is None else yn
    for k, val in kwargs.items(): var_opts[k] = val
    var_opts_list.append(var_opts)
#---------------------------------------------------------------------------------------------------
if host=='alcf':

    ### E3SMv4 ZM tuning ensemble (bundled runs)
    tmp_scratch = '/lus/flare/projects/E3SM_Dec/whannah/scratch'
    grid = 'ne30' # ne30 / ne256
    ens_name = f'2026-v4-ZM-TUNE-00.{grid}'
    if grid=='ne30':
        tmp_path,tmp_sub = '/lus/flare/projects/E3SM_Dec/whannah/scratch/bundles/E3SM.2026-v4-ZM-TUNE-00.ne30/cases','run'
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE',                    n='default',  p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cape_100',           n='cape_100', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cape_200',           n='cape_200', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cape_300',           n='cape_300', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cape_400',           n='cape_400', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.dcape_0.005',        n='dcape_0.005', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.dcape_0.01',         n='dcape_0.01', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.dcape_0.05',         n='dcape_0.05', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.dcape_0.1',          n='dcape_0.1', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.mtc_0.5.mqc_0',      n='mtc_0.5.mqc_0', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.mtc_0.3.mqc_0.3',    n='mtc_0.3.mqc_0.3', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.mtc_0.5.mqc_0.3',    n='mtc_0.5.mqc_0.3', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.mtc_0.5.mqc_0.5',    n='mtc_0.5.mqc_0.5', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ke_5e-6',            n='ke_5e-6', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ke_7.5e-6',          n='ke_7.5e-6', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ke_1e-5',            n='ke_1e-5', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.alfa_0.1',           n='alfa_0.1', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.alfa_0.2',           n='alfa_0.2', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.alfa_0.3',           n='alfa_0.3', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ncin_2',             n='ncin_2', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.ncin_4',             n='ncin_4', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tau_1200',           n='tau_1200', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tau_1800',           n='tau_1800', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tau_2400',           n='tau_2400', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tpf_1',              n='tpf_1', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tpf_3',              n='tpf_3', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tka_0.8',            n='tka_0.8', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.tka_1.6',            n='tka_1.6', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.c0l_0.001.c0o_0.002',n='c0l_0.001.c0o_0.002', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.c0l_0.003.c0o_0.002',n='c0l_0.003.c0o_0.002', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.c0l_0.002.c0o_0.003',n='c0l_0.002.c0o_0.003', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.c0l_0.002.c0o_0.001',n='c0l_0.002.c0o_0.001', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1',              n='cda_1', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.cape_100',     n='cda_1.cape_100', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.cape_200',     n='cda_1.cape_200', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.cape_300',     n='cda_1.cape_300', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.cape_400',     n='cda_1.cape_400', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.dcape_0.005',  n='cda_1.dcape_0.005', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.dcape_0.01',   n='cda_1.dcape_0.01', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.dcape_0.05',   n='cda_1.dcape_0.05', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.dcape_0.1',    n='cda_1.dcape_0.1', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.ke_5e-6',      n='cda_1.ke_5e-6', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.ke_7.5e-6',    n='cda_1.ke_7.5e-6', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.ke_1e-5',      n='cda_1.ke_1e-5', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.alfa_0.1',     n='cda_1.alfa_0.1', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.alfa_0.2',     n='cda_1.alfa_0.2', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.alfa_0.3',     n='cda_1.alfa_0.3', p=tmp_path,s=tmp_sub)
        add_case('E3SM.2026-v4-ZM-TUNE-00.ne30.F2010xx-ZM-CICE.cda_1.tau_1800',     n='cda_1.tau_1800', p=tmp_path,s=tmp_sub)
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

add_var('ShortwaveCloudForcing','LongwaveCloudForcing',xn='SWCF',yn='LWCF')
add_var('precip_total_surf_mass_flux','RESTOM',xn='precip')
# add_var('zm_prec_frac','RESTOM')
# add_var('LiqWaterPath','ShortwaveCloudForcing',xn='LWP',yn='SWCF')
# add_var('IceWaterPath','LongwaveCloudForcing',xn='IWP',yn='LWCF')
# add_var('VapWaterPath','LW_flux_up_at_model_top',xn='TMQ',yn='OLR')

#---------------------------------------------------------------------------------------------------

season = 'ANN'  # ANN / DJF / JJA / MAM / SON

# optional regional means
# lat1,lat2 = -30,30

add_labels  = True   # label each dot with the case name
add_fit     = True   # add linear regression line and correlation
print_stats = True

num_plot_col = 2

fig_file = f'figs_ens/ens.scatter.v1.{ens_name}.{season}.png'

#---------------------------------------------------------------------------------------------------
season_months = {'ANN':None, 'DJF':[12,1,2], 'MAM':[3,4,5], 'JJA':[6,7,8], 'SON':[9,10,11]}
if season not in season_months: raise ValueError(f'season not supported: {season}')
#---------------------------------------------------------------------------------------------------
# Set up plot resources
if case==[]: raise ValueError('ERROR - case list is empty!')
if var_opts_list==[]: raise ValueError('ERROR - var list is empty!')
num_var,num_case = len(var_opts_list),len(case)

if 'first_file' not in locals(): first_file = None
if 'num_files'  not in locals(): num_files  = None
if 'lat1'       not in locals(): lat1,lat2  = None,None

clr = hapy.fill_color_list(clr)

# unique list of variables to load
var_list = []
for vo in var_opts_list:
    for k in ['xvar','yvar']:
        if vo[k] not in var_list: var_list.append(vo[k])
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
gbl_means = {}  # keyed by (var,case index)
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
    if lat1 is not None:
        lat = ds['lat']
        if 'time' in lat.dims: lat = lat.isel(time=0)
        area = area.where( (lat>=lat1) & (lat<=lat2), 0 )
    area.load()
    #----------------------------------------------------------------------------
    for tvar in var_list:
        try:
            data = get_data(ds,tvar)
        except KeyError as e:
            print(' '*4+f'{hapy.tclr.RED}WARNING - variable not found: {e} - skipping{hapy.tclr.END}')
            continue
        data = time_mean(data,ds)
        if data is None:
            print(' '*4+f'{hapy.tclr.RED}WARNING - no data for season {season} - skipping{hapy.tclr.END}')
            continue
        gbl_means[(tvar,c)] = float( (data*area).sum() / area.sum() )
        if print_stats: print(' '*4+f'{tvar:30}  mean: {gbl_means[(tvar,c)]:10.4f}')
#---------------------------------------------------------------------------------------------------
# set up figure objects
ncols = min(num_plot_col, num_var)
nrows = int(np.ceil(num_var / float(ncols)))
fig, axs = plt.subplots(nrows,ncols, figsize=(7*ncols,6*nrows), squeeze=False )
axs_flat = axs.flatten()
for ax in axs_flat[num_var:]: ax.set_visible(False)

region_str = 'global' if lat1 is None else f'{lat1}:{lat2}°N'
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
    var_opts = var_opts_list[v]
    xvar,yvar = var_opts['xvar'],var_opts['yvar']
    ax = axs_flat[v]
    valid = [c for c in range(num_case) if (xvar,c) in gbl_means and (yvar,c) in gbl_means]
    if valid==[]:
        print(' '*4+f'{hapy.tclr.RED}WARNING - no valid data for {xvar} / {yvar}{hapy.tclr.END}')
        ax.set_visible(False)
        continue
    xx = np.array([gbl_means[(xvar,c)] for c in valid])
    yy = np.array([gbl_means[(yvar,c)] for c in valid])
    #----------------------------------------------------------------------------
    for i,c in enumerate(valid):
        # highlight the default/control case (first case) with a black star
        if c==0:
            ax.scatter(xx[i], yy[i], color='black', marker='*', s=250, zorder=3)
        else:
            ax.scatter(xx[i], yy[i], color=clr[c], edgecolor='black', linewidth=0.5, s=80, zorder=2)
        if add_labels:
            ax.annotate(name[c], (xx[i],yy[i]), xytext=(4,4), textcoords='offset points', fontsize=6)
    #----------------------------------------------------------------------------
    if add_fit and len(valid)>2:
        a,b = np.polyfit(xx, yy, 1)
        r = np.corrcoef(xx, yy)[0,1]
        lx = np.array([xx.min(),xx.max()])
        ax.plot(lx, a*lx+b, color='gray', linewidth=1, linestyle='--', zorder=1)
        ax.set_title(f'r = {r:.2f}  slope = {a:.3g}', loc='center', fontsize=10)
    #----------------------------------------------------------------------------
    if yvar=='RESTOM': ax.axhline(0, color='gray', linewidth=0.5)
    if xvar=='RESTOM': ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel(var_opts['xvar_name'], fontsize=12)
    ax.set_ylabel(var_opts['yvar_name'], fontsize=12)
    ax.set_title(f'{ens_name}', loc='left', fontsize=10)
    ax.set_title(f'{season} {region_str}', loc='right', fontsize=10)
    ax.grid(linewidth=0.3, alpha=0.5)
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
