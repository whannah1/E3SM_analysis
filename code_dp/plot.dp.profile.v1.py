import os, glob, copy, xarray as xr, numpy as np, numba
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import hapy; host = hapy.get_host()
#---------------------------------------------------------------------------------------------------
case_name,case,case_dir,case_sub,case_grid,clr,dsh,mrk = [],[],[],[],[],[],[],[]
lx_list,ne_list = [],[]
def add_case(case_in,n=None,p=None,s=None,g=None,d=0,c='black',m=0,lx=None,ne=None):
    global name,case,case_dir,case_sub,clr,dsh,mrk
    tmp_name = '' if n is None else n
    case.append(case_in); case_name.append(tmp_name)
    case_dir.append(p); case_sub.append(s); case_grid.append(g)
    dsh.append(d) ; clr.append(c) ; mrk.append(m)
    lx_list.append(lx); ne_list.append(ne)
#-------------------------------------------------------------------------------
var, var_str, file_type_list = [], [], []
method_list = []
def add_var(var_name,file_type,n=None,method='mean'):
    var.append(var_name)
    file_type_list.append(file_type)
    if n is None:
        var_str.append(var_name)
    else:
        var_str.append(n)
    method_list.append(method)
#-------------------------------------------------------------------------------
P0    = 101325       # Pa        surface pressure
Rd    = 287.058      # J/kg/K    gas constant for dry air
cp    = 1004.64      # J/kg/K    specific heat for dry air
Lv    = 2.5104e6     # J / kg    latent heat of vaporization / evaporation
g     = 9.80665      # m/s       global average of gravity at MSLP
#-------------------------------------------------------------------------------
host = None
if os.path.exists('/global/cfs/cdirs'): host = 'nersc'
if os.path.exists('/lustre/orion'):     host = 'olcf'
#-------------------------------------------------------------------------------
if host=='nersc':
    scratch_cpu = '/pscratch/sd/w/whannah/scream_scratch/pm-cpu'
    scratch_gpu = '/pscratch/sd/w/whannah/scream_scratch/pm-gpu'

    # add_case('DP.2026-RCE-01.NN_01.ne_022.lx_200.dt_60.L128_cntrl',ne=22,lx=200,c='black',d=1,n='RCE cntrl lx=200 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_01.ne_022.lx_200.dt_60.L128_pbias',ne=22,lx=200,c='red',  d=1,n='RCE pbias lx=200 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_01.ne_022.lx_200.dt_60.L128_tbias',ne=22,lx=200,c='blue', d=1,n='RCE tbias lx=200 dx=3km', p=scratch_gpu,s='run')

    # add_case('DP.2026-RCE-01.NN_04.ne_067.lx_600.dt_60.L128_cntrl',ne=67,lx=600,c='black',d=0,n='RCE cntrl lx=600 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_04.ne_067.lx_600.dt_60.L128_pbias',ne=67,lx=600,c='red',  d=0,n='RCE pbias lx=600 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_04.ne_067.lx_600.dt_60.L128_tbias',ne=67,lx=600,c='blue', d=0,n='RCE tbias lx=600 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_04.ne_067.lx_600.dt_60.L128_cntrl.qi2qc_1',ne=67,lx=600,c='magenta', d=0,n='RCE qi2qc lx=600 dx=3km', p=scratch_gpu,s='run')

    add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1',        ne=67,lx=600,c='black', d=0,n='GATE control 600/3km', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_pbias',  ne=67,lx=600,c='blue',  d=0,n='GATE p-bias 600/3km', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha3', ne=67,lx=600,c='green', d=0,n='GATE alpha3 600/3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1.qi2qc_1',ne=67,lx=600,c='green', d=0,n='GATE qi2qc 600/3km', p=scratch_gpu,s='run')

    htype = 'output.scream.1D.1hr.AVERAGE.nhours_x1'
    first_file,num_files = -10,None
    # first_file,num_files = -30,10

#-------------------------------------------------------------------------------

# add_var('dz_mid',             htype, n='dz')
# add_var('S',                  htype, n=None)
# add_var('MSE',                htype, n=None)
# add_var('T_mid',              htype, n=None)
# add_var('z_mid',              htype, n=None)
# add_var('omega',              htype, n=None)
add_var('dS/dp',              htype, n='-dS/dp')
add_var('qv',                 htype, n=None)
add_var('RelativeHumidity',   htype, n=None)
add_var('qc',                 htype, n=None)
add_var('qr',                 htype, n=None)
add_var('qi',                 htype, n=None)

# add_var('T_mid',              htype, n='T_mid variance',method='variance')
# add_var('qv',                 htype, n='qv variance',method='variance')
# add_var('qc',                 htype, n='qc variance',method='variance')
# add_var('qr',                 htype, n='qr variance',method='variance')
# add_var('qi',                 htype, n='qi variance',method='variance')

#-------------------------------------------------------------------------------
fig_file = 'figs_dp/dp.profile.v1.png'

print_stat = False
num_plot_col = 3#len(var)
use_height_coord = False
skip_first_twelve_hours = False

# Map NGL integer dash indices to matplotlib linestyle strings
_dash_map = {0: 'solid', 1: 'dashed', 2: 'dotted', 3: 'dashdot'}
def dash_to_ls(d): return _dash_map.get(d, 'solid')

#---------------------------------------------------------------------------------------------------
num_case, num_var = len(case), len(var)

if 'first_file' not in globals(): first_file = None
if 'num_files'  not in globals(): num_files = None
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
    data = ds[tvar]
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
    # unit conversions
    if 'precip' in var: data = data*86400.*1e3
    if var in ['qv','qc','qr','qi','qcri','qci']: data = data*1e3
    if var=='RelativeHumidity': data = data*100
    return data
#---------------------------------------------------------------------------------------------------
@numba.njit()
def get_coords(ncol,lx=None):
    nx = int(np.sqrt(ncol))
    ne = int(nx/2)
    uxi = np.linspace(0,lx,nx+1)
    uxc = uxi[:-1] + np.diff(uxi)/2
    xc = np.full(ncol,np.nan)
    yc = np.full(ncol,np.nan)
    for ny in range(ne):
        for nx in range(ne):
            for nyj in range(2):
                for nxi in range(2):
                    g = ny*4*ne + nx*4 + nyj*2 + nxi
                    i = nx*2+nxi
                    j = ny*2+nyj
                    xc[g] = uxc[i]
                    yc[g] = uxc[j]
    return xc,yc
#---------------------------------------------------------------------------------------------------
@numba.njit()
def reorg_data(data,xc,yc,ncol,nlev):
    nx = int(np.sqrt(ncol))
    ne = int(nx/2)
    xx = np.zeros((nx,nx))
    yy = np.zeros((nx,nx))
    data_out = np.zeros((nx,nx,nlev))
    for ny in range(ne):
        for nx in range(ne):
            for nyj in range(2):
                for nxi in range(2):
                    g = ny*4*ne + nx*4 + nyj*2 + nxi
                    i = nx*2+nxi
                    j = ny*2+nyj
                    xx[i,j] = xc[g]
                    yy[i,j] = yc[g]
                    data_out[i,j,:] = data[g,:]
    return data_out,xx,yy
#---------------------------------------------------------------------------------------------------

# Build figure
nrows = int(np.ceil(num_var / float(num_plot_col)))
ncols = num_plot_col
fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5*ncols, 5*nrows),
                                 squeeze=False)
axes_flat = axes.flatten()

#---------------------------------------------------------------------------------------------------
for v in range(num_var):
    print(' '*2+f'var: {hapy.tclr.GREEN}{var[v]}{hapy.tclr.END}')
    data_list = []
    hght_list = []

    for c in range(num_case):
        print(' '*4+f'case: {hapy.tclr.CYAN}{case[c]}{hapy.tclr.END}')
        # file_list = get_file_list(case[c],case_dir[c],case_sub[c],file_type_list[v])
        
        file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/*{file_type_list[v]}*'
        file_list = sorted(glob.glob(file_path))

        if first_file is not None: file_list = file_list[first_file:]
        if num_files  is not None: file_list = file_list[:num_files]
        #-------------------------------------------------------------------------
        # print()
        # for f in file_list: print(f'  {f}')
        # print()
        #-------------------------------------------------------------------------
        ds = xr.open_mfdataset(file_list)
        #-------------------------------------------------------------------------
        if skip_first_twelve_hours:
            print(' '*4+f'{hapy.tclr.RED}WARNING - skipping first 12 hours{hapy.tclr.END}')
            ds = ds.isel(time=slice(12,))
        #-------------------------------------------------------------------------
        if 'area' in ds.variables:
          area = ds['area'].isel(time=0, missing_dims='ignore')
        elif 'area_PG2' in ds.variables:
          area = ds['area_PG2'].isel(time=0, missing_dims='ignore')
        else:
          raise ValueError('area variable not found in dataset?')
        #-------------------------------------------------------------------------
        data = get_data(ds, var[v])
        if print_stat: hapy.print_stat(data, compact=True)
        #-------------------------------------------------------------------------
        # data = data.isel(lev=slice(4,-1))
        #-------------------------------------------------------------------------
        if method_list[v] == 'mean':
            data = data.mean(dim='time')
            data = ( (data*area).sum(dim='ncol') / area.sum(dim='ncol') )
        if method_list[v] == 'variance':
            data = data.var(dim=['time','ncol'])
            # data = data.var(dim='ncol')#.mean(dim='time')
            # data = data.var(dim='time')
            # data = ( (data*area).sum(dim='ncol') / area.sum(dim='ncol') )
        #-------------------------------------------------------------------------
        if use_height_coord:
            hght = get_data(ds,'z_mid')
            if var[v] == 'dz_mid': hght = hght[1:]
            data['lev'] = hght/1e3
            dz = 100e-3 ; data = data.interp(lev=np.arange(dz,18+dz,dz))
        #-------------------------------------------------------------------------
        data_list.append(data.values)
        hght_list.append(data['lev'].values)

    #----------------------------------------------------------------------------
    ax = axes_flat[v]

    data_min = np.min([np.nanmin(d) for d in data_list])
    data_max = np.max([np.nanmax(d) for d in data_list])

    # Per-variable x-axis overrides
    if var[v] == 'dS/dp': data_max = 0.1
    if var[v] == 'MSE':   data_max = 350.0

    for c in range(num_case):
        ax.plot(data_list[c], hght_list[c],
                  color=clr[c],
                  linestyle=dash_to_ls(dsh[c]),
                  linewidth=1.5,
                  label=case_name[c])

    # Vertical zero line
    ylims = (hght_list[0].min(), hght_list[0].max())
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='solid')

    ax.set_xlim(data_min, data_max)
    ax.set_ylim(ylims)

    if use_height_coord:
        ax.set_ylabel('Height [km]', fontsize=9)
    else:
        ax.set_ylabel('Pressure [hPa]', fontsize=9)
        ax.invert_yaxis()

    ax.set_title(var_str[v], loc='right', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.xaxis.get_major_formatter().set_useOffset(False)

    if num_case > 1:
        ax.legend(fontsize=7, loc='best', framealpha=0.6)

# Hide unused axes
for i in range(num_var, len(axes_flat)):
    axes_flat[i].set_visible(False)

#---------------------------------------------------------------------------------------------------
# Separate legend figure
#---------------------------------------------------------------------------------------------------
# if num_case > 1:
#    legend_file = fig_file + '.legend.png'
#    fig_lgd, ax_lgd = plt.subplots(figsize=(4, 0.4*num_case))
#    ax_lgd.axis('off')
#    handles = [mlines.Line2D([], [], color=clr[c], linestyle=dash_to_ls(dsh[c]),
#                              linewidth=2, label='    '+case_name[c])
#               for c in range(num_case)]
#    ax_lgd.legend(handles=handles, loc='center', fontsize=9, frameon=False)
#    os.makedirs(os.path.dirname(legend_file) or '.', exist_ok=True)
#    fig_lgd.savefig(legend_file, dpi=150, bbox_inches='tight')
#    plt.close(fig_lgd)
#    print(f'Saved legend: {legend_file}')

#---------------------------------------------------------------------------------------------------
plt.tight_layout(w_pad=2, h_pad=3)
os.makedirs(os.path.dirname(fig_file) or '.', exist_ok=True)
fig.savefig(fig_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved figure: {fig_file}')
#---------------------------------------------------------------------------------------------------