import os, glob, copy, xarray as xr, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import hapy; host = hapy.get_host()
#---------------------------------------------------------------------------------------------------
# add_case() puts each case into a "group". Cases in the same group are
# connected by a line, and each group gets a single entry in the legend.
#
# key kwargs:
#   group      - group id used to bundle cases together (default: the case name,
#                i.e. each case is its own group)
#   group_name - legend label for the group. The LAST case added to a group that
#                specifies group_name wins, so the final case can set the label.
#   dx         - x-axis value for this case (a measure of grid spacing)
#   clr / ls / mrk - per-group style overrides (last non-None in a group wins)
#---------------------------------------------------------------------------------------------------
case,case_opts_list = [],[]
def add_case(case_in,**kwargs):
    case.append(case_in)
    case_opts = {}
    for k, val in kwargs.items(): case_opts[k] = val
    if 'n'     not in case_opts: case_opts['n'] = case_in
    if 'group' not in case_opts: case_opts['group'] = case_in
    case_opts_list.append(case_opts)
#---------------------------------------------------------------------------------------------------
# add_var() defines a panel. The y-axis value of each dot is a scalar operator
# reduction of the variable, controlled by the optional "method" kwarg:
#   mean           - area-weighted spatial + time mean
#   variance       - variance over time and space
#   variance_space - spatial variance, then time mean
#   variance_time  - temporal variance of the spatial mean
#   std            - spatial standard deviation, then time mean
# (any leftover vertical dimension is reduced with a mean)
#---------------------------------------------------------------------------------------------------
var,var_opts_list = [],[]
def add_var(var_name,**kwargs):
    var.append(var_name)
    var_opts = {}
    for k, val in kwargs.items(): var_opts[k] = val
    if 'str' not in var_opts: var_opts['str'] = var_name
    var_opts_list.append(var_opts)
#---------------------------------------------------------------------------------------------------
P0    = 101325       # Pa        surface pressure
Rd    = 287.058      # J/kg/K    gas constant for dry air
cp    = 1004.64      # J/kg/K    specific heat for dry air
Lv    = 2.5104e6     # J / kg    latent heat of vaporization / evaporation
g     = 9.80665      # m/s       global average of gravity at MSLP
#---------------------------------------------------------------------------------------------------
host = None
if os.path.exists('/global/cfs/cdirs'): host = 'nersc'
if os.path.exists('/lustre/orion'):     host = 'olcf'
#---------------------------------------------------------------------------------------------------
if host=='nersc':
    scratch_cpu = '/pscratch/sd/w/whannah/scream_scratch/pm-cpu'
    scratch_gpu = '/pscratch/sd/w/whannah/scream_scratch/pm-gpu'

    # -------------------------------------------------------------------
    # no-zm group
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_04.ne67.lx_600km.dt_75.enable_zm_0',  group='no-zm', lx=600, ne=67, clr='black', p=scratch_gpu,s='run', group_name='no-zm')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_03.ne50.lx_600km.dt_100.enable_zm_0', group='no-zm', lx=600, ne=50, clr='black', p=scratch_gpu,s='run', group_name='no-zm')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_02.ne34.lx_600km.dt_150.enable_zm_0', group='no-zm', lx=600, ne=34, clr='black', p=scratch_gpu,s='run', group_name='no-zm')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_02.ne25.lx_600km.dt_225.enable_zm_0', group='no-zm', lx=600, ne=25, clr='black', p=scratch_gpu,s='run', group_name='no-zm')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_0', group='no-zm', lx=600, ne=17, clr='black', p=scratch_gpu,s='run', group_name='no-zm')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne13.lx_600km.dt_360.enable_zm_0', group='no-zm', lx=600, ne=13, clr='black', p=scratch_gpu,s='run', group_name='no-zm')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_0',  group='no-zm', lx=600, ne= 9, clr='black', p=scratch_gpu,s='run', group_name='no-zm')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne7.lx_600km.dt_720.enable_zm_0',  group='no-zm', lx=600, ne= 7, clr='black', p=scratch_gpu,s='run', group_name='no-zm')
    # add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne5.lx_600km.dt_900.enable_zm_0',  group='no-zm', lx=600, ne= 5, clr='black', p=scratch_gpu,s='run', group_name='no-zm')


    # zm-ctl group
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_04.ne67.lx_600km.dt_75.enable_zm_1',  group='zm-ctl', lx=600, ne=67, clr='magenta', p=scratch_gpu,s='run', group_name='zm-ctl')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_03.ne50.lx_600km.dt_100.enable_zm_1', group='zm-ctl', lx=600, ne=50, clr='magenta', p=scratch_gpu,s='run', group_name='zm-ctl')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_02.ne34.lx_600km.dt_150.enable_zm_1', group='zm-ctl', lx=600, ne=34, clr='magenta', p=scratch_gpu,s='run', group_name='zm-ctl')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_02.ne25.lx_600km.dt_225.enable_zm_1', group='zm-ctl', lx=600, ne=25, clr='magenta', p=scratch_gpu,s='run', group_name='zm-ctl')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1', group='zm-ctl', lx=600, ne=17, clr='magenta', p=scratch_gpu,s='run', group_name='zm-ctl')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne13.lx_600km.dt_360.enable_zm_1', group='zm-ctl', lx=600, ne=13, clr='magenta', p=scratch_gpu,s='run', group_name='zm-ctl')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1',  group='zm-ctl', lx=600, ne= 9, clr='magenta', p=scratch_gpu,s='run', group_name='zm-ctl')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne7.lx_600km.dt_720.enable_zm_1',  group='zm-ctl', lx=600, ne= 7, clr='magenta', p=scratch_gpu,s='run', group_name='zm-ctl')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne5.lx_600km.dt_900.enable_zm_1',  group='zm-ctl', lx=600, ne= 5, clr='magenta', p=scratch_gpu,s='run', group_name='zm-ctl')

    # mcta_3 group
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_04.ne67.lx_600km.dt_75.enable_zm_1.mcta_3',  group='zm mcta=3', lx=600, ne=67, clr='cyan', p=scratch_gpu,s='run', group_name='zm mcta=3')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_03.ne50.lx_600km.dt_100.enable_zm_1.mcta_3', group='zm mcta=3', lx=600, ne=50, clr='cyan', p=scratch_gpu,s='run', group_name='zm mcta=3')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_02.ne34.lx_600km.dt_150.enable_zm_1.mcta_3', group='zm mcta=3', lx=600, ne=34, clr='cyan', p=scratch_gpu,s='run', group_name='zm mcta=3')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_02.ne25.lx_600km.dt_225.enable_zm_1.mcta_3', group='zm mcta=3', lx=600, ne=25, clr='cyan', p=scratch_gpu,s='run', group_name='zm mcta=3')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.mcta_3', group='zm mcta=3', lx=600, ne=17, clr='cyan', p=scratch_gpu,s='run', group_name='zm mcta=3')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne13.lx_600km.dt_360.enable_zm_1.mcta_3', group='zm mcta=3', lx=600, ne=13, clr='cyan', p=scratch_gpu,s='run', group_name='zm mcta=3')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mcta_3',  group='zm mcta=3', lx=600, ne= 9, clr='cyan', p=scratch_gpu,s='run', group_name='zm mcta=3')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne7.lx_600km.dt_720.enable_zm_1.mcta_3',  group='zm mcta=3', lx=600, ne= 7, clr='cyan', p=scratch_gpu,s='run', group_name='zm mcta=3')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne5.lx_600km.dt_900.enable_zm_1.mcta_3',  group='zm mcta=3', lx=600, ne= 5, clr='cyan', p=scratch_gpu,s='run', group_name='zm mcta=3')

    # # mcta_2 group
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_04.ne67.lx_600km.dt_75.enable_zm_1.mcta_2',  group='zm mcta=2', lx=600, ne=67, clr='olive', p=scratch_gpu,s='run', group_name='zm mcta=2')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_03.ne50.lx_600km.dt_100.enable_zm_1.mcta_2', group='zm mcta=2', lx=600, ne=50, clr='olive', p=scratch_gpu,s='run', group_name='zm mcta=2')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_02.ne34.lx_600km.dt_150.enable_zm_1.mcta_2', group='zm mcta=2', lx=600, ne=34, clr='olive', p=scratch_gpu,s='run', group_name='zm mcta=2')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_02.ne25.lx_600km.dt_225.enable_zm_1.mcta_2', group='zm mcta=2', lx=600, ne=25, clr='olive', p=scratch_gpu,s='run', group_name='zm mcta=2')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.mcta_2', group='zm mcta=2', lx=600, ne=17, clr='olive', p=scratch_gpu,s='run', group_name='zm mcta=2')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne13.lx_600km.dt_360.enable_zm_1.mcta_2', group='zm mcta=2', lx=600, ne=13, clr='olive', p=scratch_gpu,s='run', group_name='zm mcta=2')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mcta_2',  group='zm mcta=2', lx=600, ne= 9, clr='olive', p=scratch_gpu,s='run', group_name='zm mcta=2')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne7.lx_600km.dt_720.enable_zm_1.mcta_2',  group='zm mcta=2', lx=600, ne= 7, clr='olive', p=scratch_gpu,s='run', group_name='zm mcta=2')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne5.lx_600km.dt_900.enable_zm_1.mcta_2',  group='zm mcta=2', lx=600, ne= 5, clr='olive', p=scratch_gpu,s='run', group_name='zm mcta=2')

    # mcta_3 + mit_3600 group
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_04.ne67.lx_600km.dt_75.enable_zm_1.mcta_3.mit_3600',  group='zm mcta=3 mit=3600', lx=600, ne=67, clr='darkblue', p=scratch_gpu,s='run', group_name='zm mcta=3  mit=3600')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_03.ne50.lx_600km.dt_100.enable_zm_1.mcta_3.mit_3600', group='zm mcta=3 mit=3600', lx=600, ne=50, clr='darkblue', p=scratch_gpu,s='run', group_name='zm mcta=3  mit=3600')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_02.ne34.lx_600km.dt_150.enable_zm_1.mcta_3.mit_3600', group='zm mcta=3 mit=3600', lx=600, ne=34, clr='darkblue', p=scratch_gpu,s='run', group_name='zm mcta=3  mit=3600')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_02.ne25.lx_600km.dt_225.enable_zm_1.mcta_3.mit_3600', group='zm mcta=3 mit=3600', lx=600, ne=25, clr='darkblue', p=scratch_gpu,s='run', group_name='zm mcta=3  mit=3600')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne17.lx_600km.dt_300.enable_zm_1.mcta_3.mit_3600', group='zm mcta=3 mit=3600', lx=600, ne=17, clr='darkblue', p=scratch_gpu,s='run', group_name='zm mcta=3  mit=3600')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne13.lx_600km.dt_360.enable_zm_1.mcta_3.mit_3600', group='zm mcta=3 mit=3600', lx=600, ne=13, clr='darkblue', p=scratch_gpu,s='run', group_name='zm mcta=3  mit=3600')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne9.lx_600km.dt_600.enable_zm_1.mcta_3.mit_3600',  group='zm mcta=3 mit=3600', lx=600, ne= 9, clr='darkblue', p=scratch_gpu,s='run', group_name='zm mcta=3  mit=3600')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne7.lx_600km.dt_720.enable_zm_1.mcta_3.mit_3600',  group='zm mcta=3 mit=3600', lx=600, ne= 7, clr='darkblue', p=scratch_gpu,s='run', group_name='zm mcta=3  mit=3600')
    add_case('DP.2026-CTAL-GATE-01.arch_GPU.NN_01.ne5.lx_600km.dt_900.enable_zm_1.mcta_3.mit_3600',  group='zm mcta=3 mit=3600', lx=600, ne= 5, clr='darkblue', p=scratch_gpu,s='run', group_name='zm mcta=3  mit=3600')

    htype = 'output.1D.1hr.AVERAGE.nhours_x1'
    first_file,num_files = 20,None

#---------------------------------------------------------------------------------------------------

# add_var('precip_total_surf_mass_flux', str='precip mean', method='mean')

# add_var('CAPE', str='CAPE variance', method='variance')
# add_var('precip_total_surf_mass_flux', str='precip variance', method='variance')

add_var('zm_activity', str='ZM activity', method='mean')
add_var('zm_prec', str='ZM precip fraction', method='mean')

# add_var('LW_flux_up_at_model_top',  str='OLR variance',method='variance')
# add_var('VapWaterPath',             str='VWP variance',method='variance')
# add_var('LiqWaterPath',             str='LWP variance',method='variance')
# add_var('IceWaterPath',             str='IWP variance',method='variance')

# add_var('LW_flux_up_at_model_top',  str='OLR mean',method='mean')
# add_var('VapWaterPath',             str='VWP mean',method='mean')
# add_var('LiqWaterPath',             str='LWP mean',method='mean')
# add_var('IceWaterPath',             str='IWP mean',method='mean')
# add_var('RainWaterPath',            str='RWP mean',method='mean')

# add_var('T_mid', str='T_mid variance', method='variance')
# add_var('qv',    str='qv variance',    method='variance')

#---------------------------------------------------------------------------------------------------
fig_file = 'figs_dp/dp.resolution_sensitivity.v1.png'

print_stat = False
log_xaxis  = False
xaxis_label = 'np4 Grid Spacing [km]'

# num_plot_col = len(var)
num_plot_col = 3

#---------------------------------------------------------------------------------------------------
num_case, num_var = len(case), len(var)

if 'first_file' not in globals(): first_file = None
if 'num_files'  not in globals(): num_files = None
#---------------------------------------------------------------------------------------------------
# Build the group structure - preserve order of first appearance
#---------------------------------------------------------------------------------------------------
group_ids = []
for c in range(num_case):
    gid = case_opts_list[c]['group']
    if gid not in group_ids: group_ids.append(gid)
num_group = len(group_ids)

# legend label per group - last case that specifies group_name wins
group_name = {}
for c in range(num_case):
    gid = case_opts_list[c]['group']
    gn  = case_opts_list[c].get('group_name', None)
    if gn is not None: group_name[gid] = gn
for gid in group_ids:
    if gid not in group_name: group_name[gid] = str(gid)

# style per group - last non-None case wins, then fill missing colors
group_clr = {gid: None for gid in group_ids}
group_ls  = {gid: None for gid in group_ids}
group_mrk = {gid: 'o'  for gid in group_ids}
for c in range(num_case):
    gid = case_opts_list[c]['group']
    if case_opts_list[c].get('clr', None) is not None: group_clr[gid] = case_opts_list[c]['clr']
    if case_opts_list[c].get('ls',  None) is not None: group_ls [gid] = case_opts_list[c]['ls']
    if case_opts_list[c].get('mrk', None) is not None: group_mrk[gid] = case_opts_list[c]['mrk']

clr_list = hapy.fill_color_list([group_clr[gid] for gid in group_ids])
for i,gid in enumerate(group_ids): group_clr[gid] = clr_list[i]
#---------------------------------------------------------------------------------------------------
def get_data(ds,var):
    tvar = var
    if var=='horiz_winds_u': tvar = 'horiz_winds'
    if var=='horiz_winds_v': tvar = 'horiz_winds'
    if var=='qcri'         : tvar = 'qc'
    if var=='ncri'         : tvar = 'nc'
    if var=='qci'          : tvar = 'qc'
    if var=='nci'          : tvar = 'nc'
    if var=='S'            : tvar = 'T_mid'
    if var=='dS/dp'        : tvar = 'T_mid'
    if var=='MSE'          : tvar = 'T_mid'
    if var=='dz_mid'       : tvar = 'z_mid'
    if var=='pme'          : tvar = 'precip_total_surf_mass_flux'
    #-------------------------------------------------------------------------
    if var=='precip_total_surf_mass_flux' and 'precip_total_surf_mass_flux' not in ds.variables:
        data = ds['precip_liq_surf_mass_flux'] + ds['precip_ice_surf_mass_flux']
    elif var=='CAPE':
        data = hapy.calc_cape( ds['T_mid'], ds['qv'], pressure=None,
                               ps=ds['ps'], hyam=ds['hyam'], hybm=ds['hybm'], p0=1e5,
                               lev_dim='lev', humidity_type='specific', virtual=True,
                               return_cin=False, n_substeps=20)
    elif var=='zm_prec':
        # convert ZM precip into fraction of total
        data = ds[var] / ds['precip_total_surf_mass_flux']
    else:
        data = ds[tvar]
    #-------------------------------------------------------------------------
    if var=='horiz_winds_u': data = data.isel(dim2=0)
    if var=='horiz_winds_v': data = data.isel(dim2=1)
    if var=='qcri'         : data = data + ds['qr'] + ds['qi']
    if var=='ncri'         : data = data + ds['nr'] + ds['ni']
    if var=='qci'          : data = data + ds['qi']
    if var=='nci'          : data = data + ds['ni']
    if var=='S'            : data = data + ds['z_mid']*g/cp
    if var=='dS/dp'        : data = data + ds['z_mid']*g/cp; data = data.differentiate(coord='lev')*-1
    if var=='MSE'          : data = data + ds['z_mid']*g/cp + ds['qv']*Lv/cp
    if var=='dz_mid'       : data = data.diff(dim='lev')*-1
    if var=='pme'          : data = data - ds['surf_evap']/1e3
    #-------------------------------------------------------------------------
    # unit conversions
    if 'precip' in var: data = data*86400.*1e3
    if var=='pme'                                : data = data*86400.*1e3
    if var in ['qv','qc','qr','qi','qcri','qci'] : data = data*1e3
    if var=='RelativeHumidity'                   : data = data*100
    return data
#---------------------------------------------------------------------------------------------------
def reduce_to_scalar(data, area, method):
    #-------------------------------------------------------------------------
    # apply the requested operator reduction over time / space
    #-------------------------------------------------------------------------
    if method == 'mean':
        data = data.mean(dim='time')
        data = ( (data*area).sum(dim='ncol') / area.sum(dim='ncol') )
    elif method == 'variance':
        data = data.var(dim=['time','ncol'])
    elif method == 'variance_space':
        if len(data.ncol)==1: raise ValueError('no valid ncol dimension - variance_space is not a valid operator')
        data = data.var(dim='ncol').mean(dim='time')
    elif method == 'variance_time':
        if len(data.time)==1: raise ValueError('no valid time dimension - variance_time is not a valid operator')
        data = ( (data*area).sum(dim='ncol') / area.sum(dim='ncol') ).var(dim='time')
    elif method == 'std':
        data = data.std(dim='ncol').mean(dim='time')
    else:
        raise ValueError(f'method "{method}" is not a valid operator reduction')
    #-------------------------------------------------------------------------
    # collapse any leftover dimension (e.g. lev) down to a single scalar
    #-------------------------------------------------------------------------
    if data.ndim > 0: data = data.mean()
    return float(data.values)
#---------------------------------------------------------------------------------------------------
# Build figure
#---------------------------------------------------------------------------------------------------
nrows = int(np.ceil(num_var / float(num_plot_col)))
ncols = num_plot_col
fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5*ncols, 4*nrows),
                                 squeeze=False)
axes_flat = axes.flatten()
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
    var_opts = var_opts_list[v]
    var_method = var_opts.get('method', 'mean')
    print(' '*2+f'var: {hapy.tclr.GREEN}{var[v]}{hapy.tclr.END}')

    # per-case scalar y-value and x-value (grid spacing)
    yval = np.full(num_case, np.nan)
    xval = np.full(num_case, np.nan)

    for c in range(num_case):
        case_opts = case_opts_list[c]
        print(' '*4+f'case: {hapy.tclr.CYAN}{case[c]}{hapy.tclr.END}')

        case_dir = case_opts['p']
        case_sub = case_opts['s']
        htype_loc = var_opts.get('htype', None)
        if htype_loc is None:
            if 'htype' in globals():
                htype_loc = htype
            else:
                raise ValueError('ERROR: no valid htype identified')

        file_path = f'{case_dir}/{case[c]}/{case_sub}/*{htype_loc}*'
        file_list = sorted(glob.glob(file_path))

        if first_file is not None: file_list = file_list[first_file:]
        if num_files  is not None: file_list = file_list[:num_files]
        if file_list==[]:
            print('ERROR: file list is empty?')
            print(f'file_path: {file_path}')
            exit()
        #---------------------------------------------------------------------
        ds = xr.open_mfdataset(file_list)
        #---------------------------------------------------------------------
        if 'area' in ds.variables:
            area = ds['area'].isel(time=0, missing_dims='ignore')
        elif 'area_PG2' in ds.variables:
            area = ds['area_PG2'].isel(time=0, missing_dims='ignore')
        else:
            raise ValueError('area variable not found in dataset?')
        #---------------------------------------------------------------------
        data = get_data(ds, var[v])
        if print_stat: hapy.print_stat(data, compact=True)
        #---------------------------------------------------------------------
        # optional level selection for 3D fields
        lev_sel = var_opts.get('lev', None)
        if lev_sel is not None and 'lev' in data.dims:
            data = data.sel(lev=lev_sel, method='nearest')
        #---------------------------------------------------------------------
        yval[c] = reduce_to_scalar(data, area, var_method)
        # xval[c] = case_opts.get('dx', np.nan)
        ne = float(case_opts.get('ne', np.nan))
        lx = float(case_opts.get('lx', np.nan))
        xval[c] = lx/(ne*3) # approx. np4 grid spacing
        # xval[c] = lx/(ne*2) # approx. pg2 grid spacing

    #-------------------------------------------------------------------------
    ax = axes_flat[v]

    for gid in group_ids:
        # indices of the cases in this group, sorted by grid spacing
        idx = [c for c in range(num_case) if case_opts_list[c]['group']==gid]
        idx = sorted(idx, key=lambda c: xval[c])
        gx = np.array([xval[c] for c in idx])
        gy = np.array([yval[c] for c in idx])
        ax.plot(gx, gy,
                color     = group_clr[gid],
                linestyle = group_ls[gid] if group_ls[gid] is not None else 'solid',
                marker    = group_mrk[gid],
                markersize= 7,
                linewidth = 1.5,
                label     = group_name[gid])

    fontsize = 12
    if log_xaxis: ax.set_xscale('log')
    ax.set_xlabel(xaxis_label, fontsize=fontsize)
    ax.set_title(var_opts.get('str'), loc='right', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.grid(True, alpha=0.3)

    if num_group > 1:
        ax.legend(fontsize=8, loc='best', framealpha=0.6)

# Hide unused axes
for i in range(num_var, len(axes_flat)):
    axes_flat[i].set_visible(False)

#---------------------------------------------------------------------------------------------------
plt.tight_layout(w_pad=2, h_pad=3)
os.makedirs(os.path.dirname(fig_file) or '.', exist_ok=True)
fig.savefig(fig_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved figure: {fig_file}')
#---------------------------------------------------------------------------------------------------
