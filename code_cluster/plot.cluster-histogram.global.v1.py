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

#-------------------------------------------------------------------------------
if host=='olcf':
    tmp_scratch = '/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch'

    recipe14 = 'cfr_1.cfi_0.acc_60.rsc_5.eci_0.1.eri_0.1.mti_7.4e6.acp_100.acq_2.5.acn_1.5.acr_2e-05.isf_1'
    recipe15 = 'cfr_1.cfi_0.acc_60.rsc_5.eci_0.1.eri_0.1.mti_7.4e6.acp_100.acq_2.5.acn_1.5.acr_2e-05.isf_0.9'
    recipe17 = 'cfr_1.cfi_0.acc_60.rsc_5.eci_0.1.eri_0.1.mti_7.4e6.acp_100.acq_2.5.acn_1.5.acr_2e-05.isf_0.95'

    tmp_sub = 'remap_1000x2000'

    add_case(f'SCREAM.2025-PC-01.F2010-SCREAMv1-DYAMOND1.ne256pg2.cfr_1',     n='SCREAM cfrc',clr='black',  p=tmp_scratch,s=tmp_sub)
    add_case(f'SCREAM.2025-PC-00.F2010-SCREAMv1-DYAMOND1.ne256pg2.{recipe14}',n='SCREAM R14', clr='red',    p=tmp_scratch,s=tmp_sub)
    add_case(f'SCREAM.2025-PC-00.F2010-SCREAMv1-DYAMOND1.ne256pg2.{recipe15}',n='SCREAM R15', clr='cyan',   p=tmp_scratch,s=tmp_sub)
    add_case(f'SCREAM.2025-PC-00.F2010-SCREAMv1-DYAMOND1.ne256pg2.{recipe17}',n='SCREAM R17', clr='magenta',p=tmp_scratch,s=tmp_sub)

    first_file,num_files = 30,10
#-------------------------------------------------------------------------------
htype = 'output.scream.2D.1hr.ne256pg2.INSTANT.nhours_x1'

add_var(var='precip_total_surf_mass_flux', str='precip', htype=htype)
# add_var(var='LiqWaterPath',                str='LiqWP',  htype=htype)
# add_var(var='IceWaterPath',                str='IceWP',  htype=htype)

#-------------------------------------------------------------------------------
tmp_data_path   = os.getenv('HOME')+'/Research/E3SM/data_tmp'
tmp_file_prefix = 'cluster-histogram.global'

fig_file = 'figs_cluster/cluster-histogram.global.v1.png'

recalculate  = False
eps_width    = 0.1
num_plot_col = 2

#-------------------------------------------------------------------------------
num_case,num_var = len(case),len(var_opts_list)

if 'first_file' not in globals(): first_file = None
if 'num_files'  not in globals(): num_files  = None
#---------------------------------------------------------------------------------------------------
def get_tmp_file(var_name,case_name_in):
    return f'{tmp_data_path}/{tmp_file_prefix}.tmp.{var_name}.{case_name_in}.nc'
#---------------------------------------------------------------------------------------------------
# build figure
nrows = int(np.ceil(num_var / float(num_plot_col)))
ncols = num_plot_col
fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), squeeze=False)
axes_flat = axes.flatten()
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
    var_opts  = var_opts_list[v]
    var_name  = var_opts['var']
    var_str   = var_opts.get('str', var_name)
    var_htype = var_opts.get('htype', None)
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
        print(' '*4+'case: '+hapy.tclr.CYAN+case[c]+hapy.tclr.END)

        tmp_file = get_tmp_file(var_name,case[c])

        if recalculate :
            #-------------------------------------------------------------------------
            file_path = f'{case_dir}/{case[c]}/{case_sub}/*{var_htype}*'
            file_list = sorted(glob.glob(file_path))
            if file_list==[]: raise ValueError(f'No files found for path: {file_path}')
            if first_file is not None: file_list = file_list[first_file:]
            if num_files  is not None: file_list = file_list[:num_files]
            #-------------------------------------------------------------------------
            ds = xr.open_mfdataset( file_list )
            #-------------------------------------------------------------------------
            data = ds[var_name]
            area = ds['area']
            if 'time' in area.dims: area = area.isel(time=0)
            #-------------------------------------------------------------------------
            lat1,lat2 = -40,40
            data = data.where((data.lat>=lat1) & (data.lat<=lat2),drop=True)
            area = area.where((area.lat>=lat1) & (area.lat<=lat2),drop=True)
            #-------------------------------------------------------------------------
            # unit conversions
            if 'precip' in var_name: data = data*86400.*1e3
            #-------------------------------------------------------------------------
            area_coeff = 6.371e6
            scene_area = np.sum(area.values) * area_coeff

            num_t = len(data['time'])
            xc = np.transpose( np.repeat( data['lon'].values[...,None],len(data['lat']),axis=1) )
            yc =               np.repeat( data['lat'].values[...,None],len(data['lon']),axis=1)
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
                if 'precip' in var_name  : threshold_mode='min'; threshold_val = 1
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
                print(' '*4+f't: {t:04d}  # clusters: {n_clusters_}')
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
            # Write to file
            print(' '*6+f'Writing data to file: {tmp_file}')
            os.makedirs(os.path.dirname(tmp_file) or '.', exist_ok=True)
            tmp_ds = xr.Dataset()
            tmp_ds['cnt_nrm'] = xr.DataArray(cnt_nrm, dims=['bin'])
            tmp_ds['bin_ctr'] = xr.DataArray(bin_ctr, dims=['bin'])
            tmp_ds.to_netcdf(path=tmp_file,mode='w')
        else:
            print(' '*6+f'Reading pre-calculated data from file: {hapy.tclr.MAGENTA}{tmp_file}{hapy.tclr.END}')
            tmp_ds  = xr.open_dataset( tmp_file )
            cnt_nrm = tmp_ds['cnt_nrm'].values
            bin_ctr = tmp_ds['bin_ctr'].values
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
