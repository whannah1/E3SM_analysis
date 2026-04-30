import os, numpy as np, xarray as xr, glob
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import hapy; host = hapy.get_host()
#-------------------------------------------------------------------------------
name,case,case_dir,case_sub = [],[],[],[]
clr_list,ls_list = [],[]
def add_case(case_in,n=None,p=None,s=None,c=None,ls=None):
   global name,case,case_dir,case_sub,clr_list,ls_list
   tmp_name = case_in if n is None else n
   case.append(case_in); name.append(tmp_name)
   case_dir.append(p); case_sub.append(s)
   clr_list.append(c); ls_list.append(ls)
#-------------------------------------------------------------------------------
var,lev_list,var_str = [],[],[]
var_opts_list = []
def add_var(var_name,lev=0,s=None,**kwargs):
   var.append(var_name); lev_list.append(lev)
   var_str.append(var_name if s is None else s)
   var_opts = {}
   for k, val in kwargs.items(): var_opts[k] = val
   var_opts_list.append(var_opts)
#-------------------------------------------------------------------------------
fig_file = 'figs_sfc-osc/sfc-osc.histogram.v1.png'
#-------------------------------------------------------------------------------
if host=='nersc':
   
   ### 2026 splitform tests
   tmp_scratch = '/pscratch/sd/w/whannah/scream_scratch/pm-gpu'
   tmp_grid_file = '/pscratch/sd/w/whannah/files_grid/scrip_ne256pg2.nc'
   # add_case('E3SM.2026-splitform-test-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_32.theta_advect_form_1',n='ne256 theta_advect_form_1',p=tmp_scratch,s='run',c='red',ls='-')
   # add_case('E3SM.2026-splitform-test-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_32.theta_advect_form_2',n='ne256 theta_advect_form_2',p=tmp_scratch,s='run',c='blue',ls='--')
   add_case('E3SM.2026-splitform-test-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_32.tms_0.theta_advect_form_1',n='ne256 theta_advect_form_1',p=tmp_scratch,s='run',c='red',ls='-')
   add_case('E3SM.2026-splitform-test-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_32.tms_0.theta_advect_form_2',n='ne256 theta_advect_form_2',p=tmp_scratch,s='run',c='blue',ls='--')
   first_file,num_files = 2,None
   # htype='output.scream.2D.AVERAGE.ndays_x1.'
   # htype='output.scream.2D.AVERAGE.nhours_x3'
   htype='output.scream.2D.MAX.nhours_x3'

   # add_var('ps',          log_bins=True,log_bins_sign='pos')
   # add_var('LiqWaterPath',log_bins=True,log_bins_sign='pos')
   # add_var('IceWaterPath',log_bins=True,log_bins_sign='pos')
   add_var('T_2m',            log_bins=True,log_bins_sign='pos')
   add_var('wind_speed_10m',  log_bins=True,log_bins_sign='pos')

   # add_var('T_2m_btp',            log_bins=True,log_bins_sign='pos')
   # add_var('surf_sens_flux_btp',  log_bins=True,log_bins_sign='pos')
   # add_var('wind_speed_10m_btp',  log_bins=True,log_bins_sign='pos')
   # add_var('qv_at_model_bot_btp', log_bins=True,log_bins_sign='pos')
#-------------------------------------------------------------------------------
if host=='alcf':

   tmp_scratch = '/lus/flare/projects/E3SM_Dec/whannah/scratch'
   add_case('E3SM.2026-osc-test-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_256',n='ne256 dt_phys=10min',p=tmp_scratch,s='run',c='C0',ls='-')
   # add_case('E3SM.2026-osc-test-00.GPU.F2010-SCREAMv1.ne256pg2_ne256pg2.NN_256.NCPL_288',n='ne256 dt_phys=5min',p=tmp_scratch,s='run',c='C1',ls='--')
   first_file,num_files = 5,None

   add_var('T_2m_atm_backtend', s='T2m backtend MAX',     htype='output.scream.2D.MAX.ndays_x1.')
   add_var('T_2m_atm_backtend', s='T2m backtend AVERAGE', htype='output.scream.2D.AVERAGE.ndays_x1.')

#-------------------------------------------------------------------------------

num_bins     = 100
use_density  = True    # normalize histogram to probability density
print_stats  = True

num_plot_col = 2

if 'use_snapshot' not in locals(): use_snapshot,ss_t = False,-1

#---------------------------------------------------------------------------------------------------
if case==[]: raise ValueError('ERROR - case list is empty!')
num_var,num_case = len(var),len(case)

# default color cycle if no color specified
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for c in range(num_case):
   if clr_list[c] is None: clr_list[c] = default_colors[c % len(default_colors)]
   if ls_list[c]  is None: ls_list[c]  = '-'

#---------------------------------------------------------------------------------------------------
# set up figure
nrows = int(np.ceil(num_var / float(num_plot_col)))
ncols = num_plot_col

fig, axs = plt.subplots(nrows, ncols, figsize=(10, 6), squeeze=False)
axs_flat = axs.flatten()
#---------------------------------------------------------------------------------------------------
for v in range(num_var):
   var_opts = var_opts_list[v]
   ax = axs_flat[v]
   hapy.print_line()
   print(' '*2+'var: '+hapy.tclr.MAGENTA+var[v]+hapy.tclr.END)

   log_bins = var_opts.get('log_bins', False)
   #---------------------------------------------------------------------------
   # load all cases first so we can compute shared bin edges
   data_vals_list = []
   for c in range(num_case):
      print(' '*4+'case: '+hapy.tclr.GREEN+case[c]+hapy.tclr.END)
      #-------------------------------------------------------------------------
      htype_tmp = None
      if 'htype' in globals(): htype_tmp = htype
      if 'htype' in var_opts:  htype_tmp = var_opts['htype']
      #-------------------------------------------------------------------------
      file_path = f'{case_dir[c]}/{case[c]}/{case_sub[c]}/*{htype_tmp}*'
      file_list = sorted(glob.glob(file_path))
      if 'first_file' in locals() and first_file is not None: file_list = file_list[first_file:]
      if 'num_files'  in locals() and num_files  is not None: file_list = file_list[:num_files]
      #-------------------------------------------------------------------------
      if file_list==[]: print('ERROR: Empty file list:'); print(); print(file_path); exit()
      #-------------------------------------------------------------------------
      # print();print(' '*6+'file_list:')
      # for f in file_list[:2]:print(' '*8+f'{hapy.tclr.YELLOW}{f}{hapy.tclr.END}')
      # print(' '*8+f'{hapy.tclr.YELLOW}...{hapy.tclr.END}')
      # for f in file_list[-2:]:print(' '*8+f'{hapy.tclr.YELLOW}{f}{hapy.tclr.END}')
      # print()
      #-------------------------------------------------------------------------
      ds   = xr.open_mfdataset(file_list)
      data = ds[var[v]]
      #-------------------------------------------------------------------------
      if 'lev' in data.dims:
         if lev_list[v]<0: data = data.isel(lev=np.absolute(lev_list[v]),drop=True)
         else:             data = data.isel(lev=lev_list[v],drop=True)
      #-------------------------------------------------------------------------
      if 'time' in data.dims:
         hapy.print_time_length(ds,indent=' '*6)
         if use_snapshot:
            data = data.isel(time=ss_t,drop=True)
            print(' '*4+f'{hapy.tclr.RED}WARNING - snapshot mode enabled{hapy.tclr.END}')
         # else: keep all time steps for histogram
      #-------------------------------------------------------------------------
      if print_stats: hapy.print_stat(data,name=var[v],stat='naxsh',indent='    ',compact=True)
      #-------------------------------------------------------------------------
      # flatten to 1D array
      data_vals = data.values.ravel()
      data_vals = data_vals[np.isfinite(data_vals)]
      data_vals_list.append(data_vals)
   #---------------------------------------------------------------------------
   # compute shared x range across all cases for KDE evaluation
   all_vals = np.concatenate(data_vals_list)
   log_bins_sign = var_opts.get('log_bins_sign', 'pos')
   if log_bins and log_bins_sign is None:
      # auto-detect: use whichever side has more values
      log_bins_sign = 'pos' if (all_vals > 0).sum() >= (all_vals < 0).sum() else 'neg'
      print(' '*4+f'log_bins_sign auto-detected: {log_bins_sign}')
   #---------------------------------------------------------------------------
   def _kde_log(d, x_pts):
      """KDE in log space, density transformed back to linear space."""
      kde = gaussian_kde(np.log10(d))
      return kde(np.log10(x_pts)) / (x_pts * np.log(10))
   #---------------------------------------------------------------------------
   if log_bins and log_bins_sign == 'both':
      # split panel into two sub-axes: neg (left) | gap | pos (right)
      from matplotlib.gridspec import GridSpecFromSubplotSpec
      gs_sub  = GridSpecFromSubplotSpec(1, 2, subplot_spec=ax.get_subplotspec(), wspace=0.08)
      ax.remove()
      ax_neg = fig.add_subplot(gs_sub[0])
      ax_pos = fig.add_subplot(gs_sub[1], sharey=ax_neg)
      #--- compute x ranges for each side
      pos_all = all_vals[all_vals > 0];  neg_all = np.abs(all_vals[all_vals < 0])
      x_pos = np.logspace(np.log10(pos_all.min()), np.log10(pos_all.max()), num_bins)
      x_neg = np.logspace(np.log10(neg_all.min()), np.log10(neg_all.max()), num_bins)
      #--- plot each case on both sub-axes
      for c in range(num_case):
         d = data_vals_list[c]
         pkw = dict(color=clr_list[c], linestyle=ls_list[c], linewidth=1.5)
         ax_pos.plot(x_pos, _kde_log(d[d > 0],         x_pos), **pkw, label=name[c])
         ax_neg.plot(x_neg, _kde_log(np.abs(d[d < 0]), x_neg), **pkw)
      #--- log scale + neg-side formatting
      ax_pos.set_xscale('log');  ax_neg.set_xscale('log')
      ax_neg.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'-{x:g}'))
      ax_neg.invert_xaxis()
      #--- hide inner spines and add break marks
      ax_neg.spines['right'].set_visible(False)
      ax_neg.tick_params(right=False)
      ax_pos.spines['left'].set_visible(False)
      ax_pos.tick_params(left=False, labelleft=False)
      d = 0.012
      for spine_ax, xs in [(ax_neg, (1-d, 1+d)), (ax_pos, (-d, +d))]:
         bk2 = dict(transform=spine_ax.transAxes, color='k', clip_on=False, lw=1.5)
         spine_ax.plot(xs, (-d, +d), **bk2)
         spine_ax.plot(xs, (1-d, 1+d), **bk2)
      #--- labels, title, legend
      ax_neg.set_ylabel('Probability density' if use_density else 'Count', fontsize=12)
      ax_neg.set_xlabel(var[v], fontsize=12);  ax_pos.set_xlabel(var[v], fontsize=12)
      ax_neg.set_title(var_str[v], fontsize=14, loc='left')
      ax_neg.grid(True, alpha=0.3);  ax_pos.grid(True, alpha=0.3)
      ax_pos.legend(fontsize=11)
   #---------------------------------------------------------------------------
   else:
      if log_bins:
         ref_vals = all_vals[all_vals > 0] if log_bins_sign=='pos' else np.abs(all_vals[all_vals < 0])
         if ref_vals.size == 0: raise ValueError(f'log_bins=True, log_bins_sign="{log_bins_sign}" but no matching values found for var "{var[v]}"')
         x_pts = np.logspace(np.log10(ref_vals.min()), np.log10(ref_vals.max()), num_bins)
      else:
         x_pts = np.linspace(all_vals.min(), all_vals.max(), num_bins)
      for c in range(num_case):
         d = data_vals_list[c]
         if log_bins:
            d = d[d > 0] if log_bins_sign=='pos' else np.abs(d[d < 0])
            y = _kde_log(d, x_pts)
         else:
            y = gaussian_kde(d)(x_pts)
         ax.plot(x_pts, y, color=clr_list[c], linestyle=ls_list[c], linewidth=1.5, label=name[c])
      if log_bins:
         ax.set_xscale('log')
         if log_bins_sign=='neg':
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'-{x:g}'))
            ax.invert_xaxis()
      ax.set_title(var_str[v], fontsize=14)
      if ax.get_xlabel() == '': ax.set_xlabel(var[v],    fontsize=12)
      if ax.get_ylabel() == '': ax.set_ylabel('Probability density' if use_density else 'Count', fontsize=12)
      ax.legend(fontsize=11)
      ax.grid(True, alpha=0.3)
   #---------------------------------------------------------------------------
   # # use log Y-axis
   # if log_bins and log_bins_sign == 'both':
   #    ax_pos.set_yscale('log')
   #    ax_neg.set_yscale('log')
   # else:
   #    ax.set_yscale('log')
#---------------------------------------------------------------------------------------------------
# Hide unused axes
for i in range(num_var, len(axs_flat)):
  axs_flat[i].set_visible(False)
#---------------------------------------------------------------------------------------------------
# Finalize plot
fig.tight_layout()
fig.savefig(fig_file, dpi=200, bbox_inches='tight')
plt.close(fig)

print(f'\n{fig_file}\n')

#---------------------------------------------------------------------------------------------------
