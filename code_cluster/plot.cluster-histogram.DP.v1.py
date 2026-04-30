import os, glob, copy, numpy as np, xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import hapy; host = hapy.get_host()
#-------------------------------------------------------------------------------
case,case_opts_list = [],[]
def add_case(case_in,**kwargs):
    case.append(case_in)
    case_opts = {}
    for k, val in kwargs.items(): case_opts[k] = val
    case_opts_list.append(case_opts)
#-------------------------------------------------------------------------------
var,var_opts_list = [],[]
def add_var(**kwargs):
    var_opts = {}
    for k, val in kwargs.items(): var_opts[k] = val
    var_opts_list.append(var_opts)
#-------------------------------------------------------------------------------
if host=='nersc':
    scratch_cpu = '/pscratch/sd/w/whannah/scream_scratch/pm-cpu'
    scratch_gpu = '/pscratch/sd/w/whannah/scream_scratch/pm-gpu'

    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_1.0_pm_100', n='a=1.0 pm=100',clr='red',    ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_1.5_pm_100', n='a=1.5 pm=100',clr='orange', ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_2.0_pm_100', n='a=2.0 pm=100',clr='green',  ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_2.5_pm_100', n='a=2.5 pm=100',clr='blue',   ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_3.0_pm_100', n='a=3.0 pm=100',clr='purple', ls='solid', p=scratch_gpu,s='run')

    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_1.0_pm_100',  n='a=1.0 pm=100',clr='red',    ls='dashed', lx=600,ne=67, p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_1.5_pm_100',  n='a=1.5 pm=100',clr='orange', ls='dashed', lx=600,ne=67, p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_2.0_pm_100',  n='a=2.0 pm=100',clr='green',  ls='dashed', lx=600,ne=67, p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_2.5_pm_100',  n='a=2.5 pm=100',clr='blue',   ls='dashed', lx=600,ne=67, p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_3.0_pm_100',  n='a=3.0 pm=100',clr='purple', ls='dashed', lx=600,ne=67, p=scratch_gpu,s='run')

    htype = 'output.scream.2D.1hr.INSTANT.nhours_x1'
    first_file,num_files = 2,None

#-------------------------------------------------------------------------------
if host=='olcf':
    scratch_frontier = '/lustre/orion/cli115/proj-shared/hannah6/scream_scratch'
    add_case('DPSCREAM.RCE-GAUSS-FLUX-TEST-01.ne44.len_400km.DT_60.GFLUX',n='RCE ne44 400km GFLUX',clr='blue',lx=400,ne=44, p=scratch_frontier,s='run')
    htype = 'output.scream.2D.1hr.AVERAGE.nhours_x1'
    first_file,num_files = 0,5
#-------------------------------------------------------------------------------
add_var(var='precip_total_surf_mass_flux', str='precip', htype=htype)
add_var(var='LiqWaterPath',                str='LiqWP',  htype=htype)
add_var(var='IceWaterPath',                str='IceWP',  htype=htype)

#-------------------------------------------------------------------------------
fig_file = 'figs_cluster/cluster-histogram.DP.v1.png'

eps_width    = 0.1
num_plot_col = 1

#-------------------------------------------------------------------------------
num_case,num_var = len(case),len(var_opts_list)

if 'first_file' not in globals(): first_file = None
if 'num_files'  not in globals(): num_files  = None
#---------------------------------------------------------------------------------------------------
def get_coords(ncol,lx):
    nx = int(np.sqrt(ncol)) ; dx = lx/nx ; ne = int(nx/2)
    uxi = np.linspace(0,lx,nx+1)
    uxc = uxi[:-1] + np.diff(uxi)/2
    xc,yc = np.full(ncol,np.nan), np.full(ncol,np.nan)
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
# build figure
nrows = int(np.ceil(num_var / float(num_plot_col)))
ncols = num_plot_col
fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), squeeze=False)
axes_flat = axes.flatten()
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
    var_opts   = var_opts_list[v]
    var_name   = var_opts['var']
    var_str    = var_opts.get('str', var_name)
    var_htype  = var_opts.get('htype', None)
    var_lev    = var_opts.get('lev', None)
    print(' '*2+'var: '+hapy.tclr.GREEN+var_name+hapy.tclr.END)
    ax = axes_flat[v]
    #-----------------------------------------------------------------------------
    hst_list = []
    bin_list = []
    #-----------------------------------------------------------------------------
    for c in range(num_case):
        case_opts = case_opts_list[c]
        case_dir  = case_opts['p']
        case_sub  = case_opts['s']
        case_lx   = case_opts.get('lx', None)
        print(' '*4+'case: '+hapy.tclr.CYAN+case[c]+hapy.tclr.END)
        #-------------------------------------------------------------------------
        file_path = f'{case_dir}/{case[c]}/{case_sub}/*{var_htype}*'
        file_list = sorted(glob.glob(file_path))
        if first_file is not None: file_list = file_list[first_file:]
        if num_files  is not None: file_list = file_list[:num_files]
        if file_list==[]: raise ValueError(f'No files found for path: {file_path}')
        #-------------------------------------------------------------------------
        ds = xr.open_mfdataset( file_list )
        #-------------------------------------------------------------------------
        # # discard first few hours
        # print(' '*4+f'{hapy.tclr.RED}WARNING - skipping first 12 hours{hapy.tclr.END}')
        # ds = ds.isel(time=slice(48,))
        #-------------------------------------------------------------------------
        data = ds[var_name]
        #-------------------------------------------------------------------------
        if 'lev' in data.dims and var_lev is not None:
            if var_lev<0: data = data.isel(lev=var_lev)
            if var_lev>0: data = data.sel(lev=var_lev)
        #-------------------------------------------------------------------------
        # unit conversions
        if 'precip' in var_name: data = data*86400.*1e3
        #-------------------------------------------------------------------------
        if 'area' in ds.variables:
            area = ds['area'].isel(time=0, missing_dims='ignore')
        elif 'area_PG2' in ds.variables:
            area = ds['area_PG2'].isel(time=0, missing_dims='ignore')
        else:
            raise ValueError('area variable not found in dataset?')
        #-------------------------------------------------------------------------
        area_coeff = 1/1e6
        scene_area = np.sum(area.values) * area_coeff

        num_t = len(data['time'])
        xc,yc = get_coords( len(data['ncol']), lx=case_lx )
        #-------------------------------------------------------------------------
        # Calculate mean and std deviation of spatial coords to normalize the data
        xc_mean,xc_std = np.mean(xc), np.std(xc)
        yc_mean,yc_std = np.mean(yc), np.std(yc)
        #-------------------------------------------------------------------------
        max_cluster_cnt = 8000
        cluster_sz = np.zeros([num_t,max_cluster_cnt]) # size
        cluster_yc = np.zeros([num_t,max_cluster_cnt]) # latitude
        cluster_xc = np.zeros([num_t,max_cluster_cnt]) # longitude
        cluster_av = np.zeros([num_t,max_cluster_cnt]) # cluster mean data value
        #-------------------------------------------------------------------------
        for t in np.arange(num_t):
            data_tmp = data.isel(time=t)
            data_tmp.load()
            #-----------------------------------------------------------------------
            threshold_mode,threshold_val = None,None
            if 'precip' in var_name  : threshold_mode='min'; threshold_val = 5
            if var_name=='LiqWaterPath': threshold_mode='min'; threshold_val = 0.5
            if var_name=='IceWaterPath': threshold_mode='min'; threshold_val = 0.5
            if var_name=='T_mid':        threshold_mode='max'; threshold_val = 298
            if threshold_mode is None: raise ValueError(f'threshold_mode not defined for variable: {var_name}')
            if threshold_val  is None: raise ValueError(f'threshold_val not defined for variable: {var_name}')
            #-----------------------------------------------------------------------
            # Create mask
            mask = np.zeros(data_tmp.shape)
            if threshold_mode=='min': mask[ data_tmp.values > threshold_val ] = 1
            if threshold_mode=='max': mask[ data_tmp.values < threshold_val ] = 1
            #-----------------------------------------------------------------------
            # skip cases with fewer than 4 grid boxes above threshold
            if np.sum(mask)<4: continue
            #-----------------------------------------------------------------------
            xc_tmp = ( xc - xc_mean ) / xc_std
            yc_tmp = ( yc - yc_mean ) / yc_std
            xc_tmp_masked = xc_tmp[ mask==1 ]
            yc_tmp_masked = yc_tmp[ mask==1 ]
            #-----------------------------------------------------------------------
            # Create array with location info
            coords_masked = np.zeros([len(xc_tmp_masked),2])
            coords_masked[:,0] = xc_tmp_masked
            coords_masked[:,1] = yc_tmp_masked
            #-----------------------------------------------------------------------
            # Clustering step using DBSCAN
            db = DBSCAN(eps=eps_width, min_samples=2).fit(coords_masked)
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            #-----------------------------------------------------------------------
            data_masked = data_tmp.values[mask==1]
            area_masked = area.values[mask==1]
            for k in set(labels):
                class_member_mask = (labels == k)
                if k == -1:
                    cluster_sz[t,k] = 0.
                    cluster_yc[t,k] = 0.
                    cluster_xc[t,k] = 0.
                    cluster_av[t,k] = 0.
                else:
                    area_label = np.array(area_masked[class_member_mask])
                    data_label = np.array(data_masked[class_member_mask])
                    yc_label   = np.array(yc_tmp_masked[class_member_mask])
                    xc_label   = np.array(xc_tmp_masked[class_member_mask])
                    # Save values for each cloud object and area weight
                    cluster_sz[t,k] = np.sum(area_label)*area_coeff
                    cluster_yc[t,k] = np.sum(area_label*yc_label)  /np.sum(area_label)
                    cluster_xc[t,k] = np.sum(area_label*xc_label)  /np.sum(area_label)
                    cluster_av[t,k] = np.sum(area_label*data_label)/np.sum(area_label)
        #-------------------------------------------------------------------------
        chord_len = np.sqrt( cluster_sz / np.pi )
        bin_edges = 2**np.arange(1,10.1,0.4)
        bin_ctr   = (bin_edges[1:]*bin_edges[:-1])**0.5
        cnt, _    = np.histogram(chord_len,bins=bin_edges)
        bin_width = bin_edges[1:] - bin_edges[:-1]

        cnt = cnt / np.sum(cnt)

        cnt_nrm = cnt / bin_width / scene_area
        #-------------------------------------------------------------------------
        hst_list.append(cnt_nrm * 1e5)
        bin_list.append(bin_ctr)
    #-----------------------------------------------------------------------------
    for c in range(num_case):
        case_opts = case_opts_list[c]
        ax.plot(bin_list[c], hst_list[c],
                color     = case_opts.get('clr', None),
                linestyle = case_opts.get('ls',  '-'),
                linewidth = 2.0,
                label     = case_opts.get('n', case[c]))
    #-----------------------------------------------------------------------------
    ax.set_xscale('log')
    ax.set_xlim(left=None, right=1e2)
    ax.set_xlabel('Log10(Obj Size) [km]', fontsize=10)
    ax.set_ylabel('Count Weighted Histogram', fontsize=10)
    ax.set_title(var_str, loc='right', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)
    if num_case > 1: ax.legend(fontsize=8, loc='best', framealpha=0.6)
#-------------------------------------------------------------------------------
# Hide unused axes
for i in range(num_var, len(axes_flat)):
    axes_flat[i].set_visible(False)

plt.tight_layout(w_pad=2, h_pad=3)
os.makedirs(os.path.dirname(fig_file) or '.', exist_ok=True)
fig.savefig(fig_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved figure: {fig_file}')
#-------------------------------------------------------------------------------
