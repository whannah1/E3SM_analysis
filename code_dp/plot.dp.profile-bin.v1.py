import os, glob, copy, xarray as xr, numpy as np, cmocean
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import hapy
host = hapy.get_host()
#-------------------------------------------------------------------------------
class Stop(Exception): pass
#-------------------------------------------------------------------------------
case,case_opts_list = [],[]
def add_case(case_in,**kwargs):
    case.append(case_in)
    case_opts = {}
    for k, val in kwargs.items(): case_opts[k] = val
    case_opts_list.append(case_opts)
#---------------------------------------------------------------------------------------------------
var,var_opts_list = [],[]
def add_var(**kwargs):
    # var.append(var_name)
    var_opts = {}
    for k, val in kwargs.items(): var_opts[k] = val
    # if 'str' not in var_opts: var_opts['str'] = var_name
    var_opts_list.append(var_opts)
#-------------------------------------------------------------------------------
if host=='nersc':
    scratch_cpu = '/pscratch/sd/w/whannah/scream_scratch/pm-cpu'
    scratch_gpu = '/pscratch/sd/w/whannah/scream_scratch/pm-gpu'

    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_300', n='a=1 pm=300', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_400', n='a=1 pm=400', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_500', n='a=1 pm=500', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_600', n='a=1 pm=600', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_700', n='a=1 pm=700', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_800', n='a=1 pm=800', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_900', n='a=1 pm=900', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_990', n='a=1 pm=990', p=scratch_gpu, s='run')

    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_900', n='a=1 pm=900', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_700', n='a=1 pm=700', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1_pm_500', n='a=1 pm=500', p=scratch_gpu, s='run')

    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_1.5_pm_300', n='a=1.5 pm=300', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_2.5_pm_300', n='a=2.5 pm=300', p=scratch_gpu, s='run')
    # add_case('DP.2026-GATE-IDEAL-02.NN_04.ne67.lx_600km.dt_60.L128v3.1_alpha_2.0_pm_300', n='a=2.0 pm=300', p=scratch_gpu, s='run')

    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_1.0_pm_100', n='a=1.0 pm=100',clr='red',    ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_1.5_pm_100', n='a=1.5 pm=100',clr='orange', ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_2.0_pm_100', n='a=2.0 pm=100',clr='green',  ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_2.5_pm_100', n='a=2.5 pm=100',clr='blue',   ls='solid', p=scratch_gpu,s='run')
    # add_case('DP.2026-GATE-IDEAL-03.NN_01.ne15.lx_600km.dt_300.L128_v3.1_alpha_3.0_pm_100', n='a=3.0 pm=100',clr='purple', ls='solid', p=scratch_gpu,s='run')

    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_1.0_pm_100',  n='a=1.0 pm=100',clr='red',    ls='dashed', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_1.5_pm_100',  n='a=1.5 pm=100',clr='orange', ls='dashed', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_2.0_pm_100',  n='a=2.0 pm=100',clr='green',  ls='dashed', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_2.5_pm_100',  n='a=2.5 pm=100',clr='blue',   ls='dashed', p=scratch_gpu,s='run')
    add_case('DP.2026-GATE-IDEAL-03.NN_04.ne67.lx_600km.dt_60.L128_v3.1_alpha_3.0_pm_100',  n='a=3.0 pm=100',clr='purple', ls='dashed', p=scratch_gpu,s='run')

    # htype = 'output.scream.1D.1hr.AVERAGE.nhours_x1'
    first_file,num_files = 10,None


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
    # first_file,num_files = 2,1
    # first_file,num_files = 20,None

#---------------------------------------------------------------------------------------------------

# add_var(x_var='precip_total_surf_mass_flux',z_var='zm_T_mid_tend', str='zm_T_tend vs total precip', \
#         x_htype='output.scream.2D.1hr.AVERAGE.nhours_x1', \
#         z_htype='output.scream.1D.1hr.AVERAGE.nhours_x1', \
#         xbin_min=0.1,xbin_max=100,use_log=True)

# add_var(x_var='precip_total_surf_mass_flux',z_var='omega', str='omega vs precip', \
#         x_htype='output.scream.2D.1hr.INSTANT.nhours_x1', \
#         z_htype='output.scream.3D.1hr.INSTANT.nhours_x1', \
#         xbin_min=0.1,xbin_max=5000,use_log=True)

add_var(x_var='ps',z_var='omega', str='omega vs ps', \
        x_htype='output.scream.2D.1hr.INSTANT.nhours_x1', \
        z_htype='output.scream.3D.1hr.INSTANT.nhours_x1', \
        xbin_min=1007e2,xbin_max=1012e2,use_log=False)

# add_var(x_var='ps',z_var='qi', str='qi vs ps', \
#         x_htype='output.scream.2D.1hr.INSTANT.nhours_x1', \
#         z_htype='output.scream.3D.1hr.INSTANT.nhours_x1', \
#         xbin_min=1007e2,xbin_max=1012e2,use_log=False)

# add_var(x_var='ps',z_var='T_mid', str='T_mid vs ps', \
#         x_htype='output.scream.2D.1hr.INSTANT.nhours_x1', \
#         z_htype='output.scream.3D.1hr.INSTANT.nhours_x1', \
#         xbin_min=1007e2,xbin_max=1012e2,use_log=False)


# add_var(xvar='','T_mid'           )#,)
# add_var(xvar='','qv'              )#,)
# add_var(xvar='','RelativeHumidity')#,)
# add_var(xvar='','qc'              )#,)
# add_var(xvar='','qr'              )#,)
# add_var(xvar='','qi'              )#,)


#---------------------------------------------------------------------------------------------------

fig_file = 'figs_dp/dp.profile-bin.v1.png'
tmp_file_head = 'data_temp/dp.profile-bin.v2'

recalculate = True

plot_diff   = True
# use_height  = False
print_stats = True
var_x_case = False
# num_plot_col = 2

# ncol_chunk_size = int(100e3)

#---------------------------------------------------------------------------------------------------
num_var,num_case = len(var_opts_list),len(case)
if 'first_file' not in globals(): first_file = None
if 'num_files'  not in globals(): num_files = None
#---------------------------------------------------------------------------------------------------
# set up figure objects
fdx,fdy=20,10;figsize = (fdx*num_case,fdy*num_var) if var_x_case else (fdx*num_var,fdy*num_case)
title_fontsize,lable_fontsize = 25,25
(d1,d2) = (num_var,num_case) if var_x_case else (num_case,num_var)
if 'num_plot_col' in locals():
    (d1,d2) = ( int(np.ceil((num_var*num_case)/float(num_plot_col))), num_plot_col )
fig, axs = plt.subplots( d1, d2, figsize=figsize, squeeze=False )
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
data_list_list,lev_list_list = [],[]
amax_list = [None]*num_case
for v in range(num_var):
    var_opts = var_opts_list[v]
    x_var_name = var_opts['x_var']
    z_var_name = var_opts['z_var']
    var_str = var_opts['str'] if 'str' in var_opts else f'{z_var_name} vs {x_var_name}'
    hapy.print_line()
    print(' '*2+f'var: {hapy.tclr.GREEN}{x_var_name}{hapy.tclr.END} / {hapy.tclr.GREEN}{z_var_name}{hapy.tclr.END}')
    #----------------------------------------------------------------------------
    data_list = []
    bins_list = []
    lev_list = []
    #----------------------------------------------------------------------------
    for c in range(num_case):
        case_opts = case_opts_list[c]
        print('\n'+' '*4+f'case: {hapy.tclr.CYAN}{case[c]}{hapy.tclr.END}')
        # tmp_file = f'{tmp_file_head}.{case[c]}.{var[v]}.f0_{first_file}.nf_{num_files}.nc'
        # print(' '*4+f'case: {hapy.tclr.CYAN}{case[c]}{hapy.tclr.END}  =>  {tmp_file}')
        #-------------------------------------------------------------------------
        if recalculate:
        # if True:
            #-------------------------------------------------------------------
            case_dir = case_opts['p']
            case_sub = case_opts['s']
            x_htype  = var_opts['x_htype']
            z_htype  = var_opts['z_htype']
            #-------------------------------------------------------------------
            def get_file_list(file_path):
                file_list = sorted(glob.glob(file_path))
                if first_file is not None: file_list = file_list[first_file:]
                if num_files  is not None: file_list = file_list[:num_files]
                if file_list==[]:
                    print('ERROR: file list is empty?')
                    print(f'file_path: {file_path}')
                    print(f'file_list: {file_list}')
                    exit()
                return file_list
            #-------------------------------------------------------------------
            x_file_list = get_file_list(f'{case_dir}/{case[c]}/{case_sub}/*{x_htype}*')
            z_file_list = get_file_list(f'{case_dir}/{case[c]}/{case_sub}/*{z_htype}*')
            #-------------------------------------------------------------------
            x_ds = xr.open_mfdataset( x_file_list )
            z_ds = xr.open_mfdataset( z_file_list )
            #-------------------------------------------------------------------
            x_da = x_ds[ var_opts['x_var'] ]
            z_da = z_ds[ var_opts['z_var'] ]
            ilev = z_ds['ilev']
            #-------------------------------------------------------------------
            if 'prec' in var_opts['x_var']: x_da = x_da *86400.*1e3
            #-------------------------------------------------------------------
            def avg_wgt(da,wgt,dim='ncol'):
                return ( (da*wgt).sum(dim=dim,drop=True) / wgt.sum(dim=dim,drop=True) )
                # if 'ncol' in da.dims:
                #     return ( (da*wgt).sum(dim=dim,drop=True) / wgt.sum(dim=dim,drop=True) )
                # else:
                #     return da
            #-------------------------------------------------------------------
            # # if one field has ncol dim and the other doesn't then spatially average
            # if set(x_da.dims) != set(z_da.dims):
            #     if 'ncol' in x_da.dims and len(x_da.ncol)>0: x_da = x_da.isel(ncol=0)
            #     if 'ncol' in z_da.dims and len(z_da.ncol)>0: z_da = z_da.isel(ncol=0)
            #     x_ncol_chk = 'ncol' in x_da.dims and len(x_da.ncol)>0
            #     z_ncol_chk = 'ncol' in z_da.dims and len(z_da.ncol)>0
            #     # if x_ncol_chk and not z_ncol_chk: x_da = ( (x_da*x_ds['area']).sum(dim='ncol') / x_ds['area'].sum(dim='ncol') )
            #     # if z_ncol_chk and not x_ncol_chk: z_da = ( (z_da*z_ds['area']).sum(dim='ncol') / z_ds['area'].sum(dim='ncol') )
            #     if x_ncol_chk and not z_ncol_chk: x_da = avg_wgt(x_da,x_ds['area'])
            #     if z_ncol_chk and not x_ncol_chk: z_da = avg_wgt(z_da,z_ds['area'])
            #-------------------------------------------------------------------
            x_da.load()
            z_da.load()
            #-------------------------------------------------------------------
            if print_stats:
                hapy.print_stat(x_da,name=f'{x_var_name:30}',indent=' '*4)
                hapy.print_stat(z_da,name=f'{z_var_name:30}',indent=' '*4)
            #-------------------------------------------------------------------
            if 'ncol' in x_da.dims and 'ncol' in z_da.dims:
                x_da = x_da.stack(sample=('time','ncol'))
                z_da = z_da.stack(sample=('time','ncol'))
                mean_dim = 'sample'
            else:
                mean_dim = 'time'
            #-------------------------------------------------------------------
            if 'xbin_min' in var_opts:
                xbin_min = var_opts['xbin_min']
            else:
                xbin_min = x_da.where(x_da > 0).min().values

            if 'xbin_max' in var_opts:
                xbin_max = var_opts['xbin_max']
            else:
                xbin_max = x_da.max().values
            
            # xbin_min,xbin_max = x_da.min().values, x_da.max().values
            # xbin_min,xbin_max = x_da.where(x_da > 0).min().values, x_da.max().values
            # if 'prec' in var_opts['x_var']: xbin_min,xbin_max = 0.01,500

            if 'use_log' in var_opts and var_opts['use_log']==True: 
                bins = np.logspace(np.log10(xbin_min), np.log10(xbin_max), 40)
            else:
                bins = np.linspace(xbin_min, xbin_max, 20)

            z_da_binned = z_da.groupby_bins(x_da, bins=bins).mean(dim=mean_dim)
            z_da_binned = z_da_binned.rename({f'{x_var_name}_bins':'bins'})

            xbin_coord = z_da_binned['bins']
            xlft_edges = np.array([b.left  for b in xbin_coord.values])
            xrgt_edges = np.array([b.right for b in xbin_coord.values])
            xbin_edges = np.append(xlft_edges, xrgt_edges[-1])
            #-------------------------------------------------------------------
            # print(); print(z_da_binned)
            # print(); print(xbin_coord)
            # print(); print(z_da_binned['lev'])
            # print(); print(ilev)
            # print(); print(ilev)
            # print()
            # exit()
            #-------------------------------------------------------------------
            # if use_height: 
            #    hght = ???
            #    hght = ( (hght*area).sum(dim='ncol') / area.sum(dim='ncol') )
            #-------------------------------------------------------------------
        #    # Write to file 
        #    if os.path.isfile(tmp_file) : os.remove(tmp_file)
        #    tmp_ds = xr.Dataset()
        #    tmp_ds[var[v]] = data
        #    tmp_ds.to_netcdf(path=tmp_file,mode='w')
        # else:
        #    tmp_ds = xr.open_dataset( tmp_file )
        #    data = tmp_ds[var[v]]
        #-----------------------------------------------------------------------
        # # adjust units
        # if 'unit_fac' in var_opts:
        #     if var_opts['unit_fac']:
        #         z_da_binned = z_da_binned*var_opts['unit_fac']
        #-----------------------------------------------------------------------
        # if print_stats: hapy.print_stat(z_da_binned,compact=True)
        #-----------------------------------------------------------------------
        # print(); print(z_da_binned)
        # print(); print(xbin_edges.shape)
        # print(); print(ilev)
        # print(); exit()
        #-----------------------------------------------------------------------
        data_list.append( z_da_binned.values )
        # bins_list.append( z_da_binned['bins'].values )
        # lev_list.append(  z_da_binned['lev'].values )
        
        # use data for cell edges
        bins_list.append(xbin_edges)
        lev_list.append(ilev)

        # zmid = np.log(data_binned[ 'lev']/1e3) * -6740.
        # lev_list.append( zmid.values )

        # if not use_height: lev_list.append( data['lev'].values )
        # if     use_height: lev_list.append( hght.values )
    #----------------------------------------------------------------------------
    # Create plot

    ip = v
    
    # data_min = np.min([np.nanmin(d) for d in data_list])
    # data_max = np.max([np.nanmax(d) for d in data_list])

    #----------------------------------------------------------------------------
    # print the profiles
    # print()
    # tmp_data = data_list[c][-48,:]
    # for i,d in enumerate(tmp_data):
    #    print(f'{i:3}  {d}')
    # print()

    #----------------------------------------------------------------------------
    # set color bar levels
    clev = None
    # if var[v]=='FRONTGF': clev = np.logspace( -5, -1, num=40)
    # if var[v] in ['PS','sp']: clev = np.linspace( 800e2, 1020e2, num=40)
    # if var[v] in ['PS','sp']: clev = np.arange(600e2,1040e2+2e2,10e2)

    if z_var_name=='omega': clev = np.linspace( -80, 80, num=61)

    #----------------------------------------------------------------------------
    # if var[v] in ['ZM_ENTR_UP','ZM_DETR_UP']: 
    #    for c in range(num_case):
    #       data_list[c] = np.absolute(data_list[c])
    # if var[v]=='ZMDT':
    for c in range(num_case):
        # data_list[c] = np.ma.masked_invalid( data_list[c] )
        data_list[c] = np.ma.masked_where(data_list[c]==0, data_list[c])
    #----------------------------------------------------------------------------
    # set color map
    cmap = 'viridis'
    # cmap = cmocean.cm.rain
    if z_var_name=='omega': cmap = cmocean.cm.curl
    #----------------------------------------------------------------------------
    # calculate common limits for consistent contour levels
    data_min = np.min([np.nanmin(d) for d in data_list])
    data_max = np.max([np.nanmax(d) for d in data_list])

    if z_var_name in ['omega']:
        data_mag_max = max(abs(data_min),abs(data_max))
        data_min = data_mag_max*-1
        data_max = data_mag_max
    #----------------------------------------------------------------------------
    if plot_diff:
        tmp_data = copy.deepcopy(data_list)
        for c in range(num_case): tmp_data[c] = data_list[c] - data_list[0]
        diff_data_max = np.max([np.nanmax(np.absolute(d)) for d in tmp_data])
        diff_data_min = -1*diff_data_max
        for c in range(1,num_case): data_list[c] = data_list[c] - data_list[0]
        # # specify alt color bar limits
        # if z_var_name=='omega': diff_data_min,diff_data_max = -5,5
    #----------------------------------------------------------------------------
    for c in range(num_case):
        #-------------------------------------------------------------------------
        img_kwargs = {}
        # img_kwargs['origin'] = 'lower'
        if plot_diff and c!=0:
           img_kwargs['cmap']   = cmocean.cm.balance
           img_kwargs['vmin']   = diff_data_min
           img_kwargs['vmax']   = diff_data_max
           clev = None
        else:
            img_kwargs['cmap'] = cmap
            img_kwargs['vmin'] = data_min
            img_kwargs['vmax'] = data_max
        if clev is not None:
            img_kwargs['vmin'] = None
            img_kwargs['vmax'] = None
            img_kwargs['norm'] = mcolors.BoundaryNorm(clev, ncolors=256)

        data = data_list[c]
        bins = bins_list[c]
        lev  = lev_list[c]

        # bin_centers = np.array([interval.mid for interval in bins])        
        # bin_left  = np.array([interval.left  for interval in data_binned['precip_bins'].values])
        # bin_right = np.array([interval.right for interval in data_binned['precip_bins'].values])
        # bin_edges = np.append(bin_left, bin_right[-1])  # N+1 edges for N bins

        ax = axs[v,c] if var_x_case else axs[c,v]
        ax.set_title(case_opts_list[c]['n'],        fontsize=title_fontsize, loc='left')
        ax.set_title(var_str,                       fontsize=title_fontsize, loc='right')
        if plot_diff and c!=0:ax.set_title('(diff)',fontsize=title_fontsize, loc='center')
        ax.set_xlabel(var_opts['x_var'],            fontsize=title_fontsize)
        ax.set_ylabel('Reference Pressure Level',   fontsize=title_fontsize)
        ax.tick_params(axis='both', labelsize=title_fontsize)
        ax.invert_yaxis()
        
        if 'use_log' in var_opts and var_opts['use_log']==True: ax.set_xscale('log')

        img = ax.pcolormesh( bins_list[c], lev_list[c], data, **img_kwargs)

        # ax.set_xticks(time_idx)
        # ax.set_xticklabels(time_labels, fontsize=10, rotation=45, ha='right')

        # tick_stride = max(1, len(time_idx) // 20)
        # ax.set_xticks(time_idx[::tick_stride])
        # ax.set_xticklabels(time_labels[::tick_stride], rotation=45, ha='right')

        cbar = fig.colorbar(img, ax=ax, fraction=0.02, orientation='vertical')
        cbar.ax.tick_params(labelsize=lable_fontsize)

#-------------------------------------------------------------------------------
# Add legend

#---------------------------------------------------------------------------------------------------
# Finalize plot
fig.savefig(fig_file, dpi=100, bbox_inches='tight')
plt.close(fig)
print(f'\n{fig_file}\n')
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
