import os, glob, copy, numpy as np, xarray as xr, cmocean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import hapy
host = hapy.get_host()
htype = None
#-------------------------------------------------------------------------------
case_name,case,case_dir,case_sub,case_grid,clr,dsh,mrk = [],[],[],[],[],[],[],[]
lx_list,ne_list = [],[]
def add_case(case_in,n=None,p=None,s=None,g=None,d=0,c='black',m=0,lx=None,ne=None,**kwargs):
    global name,case,case_dir,case_sub,clr,dsh,mrk
    tmp_name = '' if n is None else n
    case.append(case_in); case_name.append(tmp_name)
    case_dir.append(p); case_sub.append(s); case_grid.append(g)
    dsh.append(d) ; clr.append(c) ; mrk.append(m)
    lx_list.append(lx); ne_list.append(ne)
#-------------------------------------------------------------------------------
var, var_str, htype_list = [], [], []
mlev_list = []
def add_var(var_name,htype,n=None,mlev=None):
    var.append(var_name)
    htype_list.append(htype)
    var_str.append(var_name if n is None else n)
    mlev_list.append(mlev)
#-------------------------------------------------------------------------------
if host=='nersc':
    scratch_cpu = '/pscratch/sd/w/whannah/scream_scratch/pm-cpu'
    scratch_gpu = '/pscratch/sd/w/whannah/scream_scratch/pm-gpu'

    # add_case('DP.2026-RCE-01.NN_01.ne_022.lx_200.dt_60.L128_cntrl',ne=22,lx=200,c='red',  d=0,n='RCE cntrl lx=200 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_01.ne_022.lx_200.dt_60.L128_tbias',ne=22,lx=200,c='blue', d=0,n='RCE tbias lx=200 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_01.ne_022.lx_200.dt_60.L128_pbias',ne=22,lx=200,c='green',d=0,n='RCE pbias lx=200 dx=3km', p=scratch_gpu,s='run')

    # add_case('DP.2026-RCE-01.NN_04.ne_067.lx_600.dt_60.L128_cntrl',ne=67,lx=600,c='black',d=0,n='RCE cntrl lx=600 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_04.ne_067.lx_600.dt_60.L128_pbias',ne=67,lx=600,c='red',  d=0,n='RCE pbias lx=600 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_04.ne_067.lx_600.dt_60.L128_tbias',ne=67,lx=600,c='blue', d=0,n='RCE tbias lx=600 dx=3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-RCE-01.NN_04.ne_067.lx_600.dt_60.L128_cntrl.qi2qc_1',ne=67,lx=600,c='magenta', d=0,n='RCE qi2qc lx=600 dx=3km', p=scratch_gpu,s='run')


    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1',        ne=67,lx=600,c='black', d=0,n='GATE control 600/3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_pbias',  ne=67,lx=600,c='blue',  d=0,n='GATE p-bias 600/3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha3', ne=67,lx=600,c='green', d=0,n='GATE alpha3 600/3km', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1.qi2qc_1',ne=67,lx=600,c='green', d=0,n='GATE qi2qc 600/3km', p=scratch_gpu,s='run')

    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_300', n='a=1 pm=300', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_400', n='a=1 pm=400', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_500', n='a=1 pm=500', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_600', n='a=1 pm=600', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_700', n='a=1 pm=700', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_800', n='a=1 pm=800', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_900', n='a=1 pm=900', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_990', n='a=1 pm=990', p=scratch_gpu, s='run')

    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_500',   n='a=1 pm=500',   clr='red',   ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_700',   n='a=1 pm=700',   clr='green', ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_900',   n='a=1 pm=900',   clr='blue',  ls='solid', p=scratch_gpu, s='run')

    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1.5_pm_300', n='a=1.5 pm=300', clr='red',   ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_2.0_pm_300', n='a=2.0 pm=300', clr='green', ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_2.5_pm_300', n='a=2.5 pm=300', clr='blue',  ls='dashed', p=scratch_gpu, s='run')

    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_1.0_pm_100', n='a=1.0 pm=100',clr='red',    ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_1.5_pm_100', n='a=1.5 pm=100',clr='orange', ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_2.0_pm_100', n='a=2.0 pm=100',clr='green',  ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_2.5_pm_100', n='a=2.5 pm=100',clr='blue',   ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_3.0_pm_100', n='a=3.0 pm=100',clr='purple', ls='solid', p=scratch_gpu,s='run')

    # add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_1.0_pm_100',  n='a=1.0 pm=100',clr='red',    ls='dashed', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_1.5_pm_100',  n='a=1.5 pm=100',clr='orange', ls='dashed', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_2.0_pm_100',  n='a=2.0 pm=100',clr='green',  ls='dashed', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_2.5_pm_100',  n='a=2.5 pm=100',clr='blue',   ls='dashed', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_3.0_pm_100',  n='a=3.0 pm=100',clr='purple', ls='dashed', p=scratch_gpu,s='run')


    htype = 'output.scream.2D.1hr.AVERAGE.nhours_x1'
    first_file,num_files = -1,None
    time_slice = -1




    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_0',           n='RCE 5m control',     clr='black', ls='solid',  p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_100',  n='RCE 5m zm mvgr:100', clr='red',    ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_10',   n='RCE 5m zm mvgr: 10', clr='orange', ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_6',    n='RCE 5m zm mvgr:  6', clr='green',  ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_4',    n='RCE 5m zm mvgr:  4', clr='cyan',   ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_2',    n='RCE 5m zm mvgr:  2', clr='blue',   ls='solid', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_300.enable_zm_1.mvgr_1',    n='RCE 5m zm mvgr:  1', clr='purple', ls='solid', p=scratch_gpu, s='run')

    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_0',            n='RCE 1m control',     clr='black', ls='dashed',  p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_100',   n='RCE 1m zm mvgr:100', clr='red',    ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_10',    n='RCE 1m zm mvgr: 10', clr='orange', ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_6',     n='RCE 1m zm mvgr:  6', clr='green',  ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_4',     n='RCE 1m zm mvgr:  4', clr='cyan',   ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_2',     n='RCE 1m zm mvgr:  2', clr='blue',   ls='dashed', p=scratch_gpu, s='run')
    # add_case('DP.2026-RCE-00.GPU.NN_01.ne_010.lx_400.dt_60.enable_zm_1.mvgr_1',     n='RCE 1m zm mvgr:  1', clr='purple', ls='dashed', p=scratch_gpu, s='run')

    # htype = 'output.scream.2D.1hr.AVERAGE.nhours_x1'
    # first_file,num_files = 2,3
    # first_file,num_files = -30,None
    # time_slice = -1

#-------------------------------------------------------------------------------
if host=='olcf':
    scratch_frontier = '/lustre/orion/cli115/proj-shared/hannah6/scream_scratch'

#-------------------------------------------------------------------------------
add_var('precip_total_surf_mass_flux', htype=htype, n='precip')
# add_var('VapWaterPath',                htype, n='VapWP')
add_var('LiqWaterPath',                htype, n='LiqWP')
add_var('IceWaterPath',                htype, n='IceWP')
# add_var('LW_flux_up_at_model_top',     htype, n='LW_flux_up_at_model_top')
# add_var('surf_evap',                   htype, n='surf_evap')
add_var('ps',                          htype, n='ps')

# add_var('zm_activity',                   htype)


# tmp_htype = 'output.scream.3D.1hr.AVERAGE'
# add_var('T_mid',           htype, n='T_mid @ k=127',  mlev=127)
# add_var('qv',              htype, n='qv @ k=127',  mlev=127)

# add_var('wind_speed_10m',                 htype, n='wind_speed_10m')
# add_var('surf_sens_flux',                 htype, n='surf_sens_flux')
# add_var('surf_evap',                      htype, n='surf_evap')

#-------------------------------------------------------------------------------
tmp_data_path = 'data_tmp'

fig_file = 'figs_dp/dp.xy.v1.png'

print_stat = True

var_x_case = False
# num_plot_col = 3

recenter_on_peak = True  # Will store (i_shift, j_shift) for each case, determined from var[0]
recenter_shifts = {}  # keyed by case index

#---------------------------------------------------------------------------------------------------
num_case,num_var = len(case),len(var)

#---------------------------------------------------------------------------------------------------
def get_coords(ncol,lx=100):
    nx = int(np.sqrt(ncol))
    dx = lx/nx
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
def arrange_2d_array(data, x_coords, y_coords):
    unique_x = sorted(set(x_coords))
    unique_y = sorted(set(y_coords))
    
    x_index = {value: idx for idx, value in enumerate(unique_x)}
    y_index = {value: idx for idx, value in enumerate(unique_y)}
    
    max_x = len(unique_x)
    max_y = len(unique_y)
    
    array_2d = np.empty((max_x, max_y))
    
    for i, value in enumerate(data):
        x = x_coords[i]
        y = y_coords[i]
        array_2d[x_index[x], y_index[y]] = value
    
    arranged_x_coords = np.array(unique_x)
    arranged_y_coords = np.array(unique_y)
    
    return array_2d, arranged_x_coords, arranged_y_coords
#---------------------------------------------------------------------------------------------------
# Determine figure layout
num_plots = num_var * num_case
# if num_case == 1 or num_var == 1:
if 'num_plot_col' in locals() and num_plot_col is not None:
    nrows = int(np.ceil(num_plots / float(num_plot_col)))
    ncols = min(num_plots, num_plot_col)
else:
    (nrows,ncols) = (num_var,num_case) if var_x_case else (num_case,num_var)

fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)
axes_flat = axes.flatten()

if 'first_file' not in globals(): first_file = None
if 'num_files' not in globals(): num_files = None
#---------------------------------------------------------------------------------------------------
ip = 0
for v in range(num_var):
    print(' '*2+'var: '+hapy.tclr.GREEN+var[v]+hapy.tclr.END)
    #---------------------------------------------------------------------------
    data_list = []
    #---------------------------------------------------------------------------
    for c in range(num_case):
        print(''+' '*4+'case: '+hapy.tclr.CYAN+case[c]+hapy.tclr.END)

        file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/*{htype_list[v]}*'
        file_list = sorted(glob.glob(file_path))

        if first_file is not None: file_list = file_list[first_file:]
        if num_files is not None: file_list = file_list[:num_files]
        #-------------------------------------------------------------------------
        if file_list==[]: print('ERROR: Empty file list:'); print(); print(file_path); exit()
        #-------------------------------------------------------------------------
        ds = xr.open_mfdataset( file_list )

        if 'time_slice' in globals():
            if time_slice is not None:
                ds = ds.isel(time=time_slice)

        data = ds[var[v]].load()

        #-------------------------------------------------------------------------
        if 'time' in data.dims:
            data = data.mean(dim='time')
            # data = data.var(dim='time')
        #-------------------------------------------------------------------------
        if 'lev' in data.dims:
            if mlev_list[v] is not None: 
                data = data.isel(lev=mlev_list[v])
        #-------------------------------------------------------------------------
        # unit conversions    
        if 'precip' in var[v]: data = data*86400.*1e3
        #-------------------------------------------------------------------------
        if print_stat: hapy.print_stat(data,name=var[v],stat='naxsh',indent=' '*4,compact=True)
        #-------------------------------------------------------------------------
        data_list.append( data.values )

    # #---------------------------------------------------------------------------
    # # Recenter data on peak of first variable (for each case independently)
    # # Useful when cloud aggregation occurs at different domain locations across runs
    # if recenter_on_peak:
    #     for c in range(num_case):
    #         if c==1:
    #             xc, yc = get_coords( len(data_list[c]) )
    #             data_2d, x_arr, y_arr = arrange_2d_array(data_list[c], xc, yc)

    #             if v == 0:
    #                 # Find peak location in 2D array and compute shift needed to center it
    #                 peak_idx = np.unravel_index(np.argmax(data_2d), data_2d.shape)
    #                 nx, ny = data_2d.shape
    #                 i_shift = nx//2 - peak_idx[0]
    #                 j_shift = ny//2 - peak_idx[1]
    #                 recenter_shifts[c] = (i_shift, j_shift)

    #             i_shift, j_shift = recenter_shifts[c]
    #             # np.roll wraps the data cyclically, appropriate for doubly-periodic domains
    #             data_list[c] = np.roll(
    #                 arrange_2d_array(data_list[c], xc, yc)[0],
    #                 shift=(i_shift, j_shift), axis=(0, 1)
    #             ).flatten()  # flatten back so arrange_2d_array still works in the plot loop
    #---------------------------------------------------------------------------
    # Recenter data on peak of first variable (for each case independently)
    # Useful when cloud aggregation occurs at different domain locations across runs
    data_2d_list = {}  # cache rolled 2D arrays keyed by case index, used in plot loop below
    if recenter_on_peak:
        for c in range(num_case):
            xc, yc = get_coords(len(data_list[c]))
            data_2d, x_arr, y_arr = arrange_2d_array(data_list[c], xc, yc)

            if v == 0:
                peak_idx = np.unravel_index(np.argmax(data_2d), data_2d.shape)
                nx, ny = data_2d.shape
                i_shift = nx//2 - peak_idx[0]
                j_shift = ny//2 - peak_idx[1]
                recenter_shifts[c] = (i_shift, j_shift)

            i_shift, j_shift = recenter_shifts[c]
            data_2d_list[c] = np.roll(data_2d, shift=(i_shift, j_shift), axis=(0, 1))
    #---------------------------------------------------------------------------
    # Determine color levels per variable
    levels = None
    cmap   = 'viridis'

    # if 'precip' in var[v]:     levels = np.arange(3, 60+3, 3)
    if 'precip' in var[v]:     levels = np.logspace(np.log10(0.1), np.log10(500), 40)
    if var[v]=='VapWaterPath': levels = np.arange(8, 48+2, 2)
    if var[v]=='LiqWaterPath': levels = np.arange(0.1, 3.1+0.1, 0.1)
    if var[v]=='IceWaterPath': levels = np.arange(0.5, 15.5+1, 1)
    if var[v]=='qv':           levels = np.arange(10e-3, 14e-3+0.2e-3, 0.2e-3)
    if var[v]=='T_mid':        levels = np.arange(293, 300, 1)
    if var[v]=='surf_evap':    levels = np.arange(2e-5, 12e-5, 1e-5)

    if levels is None:
        data_min = np.nanmin([np.nanmin(d) for d in data_list])
        data_max = np.nanmax([np.nanmax(d) for d in data_list])
        levels = np.linspace(data_min, data_max, 20)

    norm = mcolors.BoundaryNorm(levels, ncolors=256, extend='both') if levels is not None else None

    #-----------------------------------------------------------------------------
    for c in range(num_case):
        xc, yc = get_coords( len(data_list[c]) )

        if recenter_on_peak:
            data_2d = data_2d_list[c]
            _, x_arr, y_arr = arrange_2d_array(data_list[c], xc, yc)  # coords only
        else:
            data_2d, x_arr, y_arr = arrange_2d_array(data_list[c], xc, yc)

        ip = v*num_case+c if var_x_case else c*num_var+v

        ax = axes_flat[ip]

        im = ax.pcolormesh(x_arr, y_arr, data_2d.T, cmap=cmap, norm=norm, shading='nearest')

        plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)

        # Subtitles: left=case_name, right=var_str (mirrors hs.set_subtitles)
        ax.set_title(var_str[v], loc='right',  fontsize=7)
        ax.set_title(case_name[c], loc='left', fontsize=7)

        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_aspect('equal')

        # ip += 1
#-------------------------------------------------------------------------------
# Hide any unused axes
for i in range(ip+1, len(axes_flat)):
    axes_flat[i].set_visible(False)

plt.tight_layout(w_pad=2, h_pad=3)
os.makedirs(os.path.dirname(fig_file), exist_ok=True)
fig.savefig(fig_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved figure: {fig_file}')
#-------------------------------------------------------------------------------