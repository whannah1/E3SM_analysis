import os, subprocess as sp, numpy as np, xarray as xr, copy, string, dask, glob, cmocean
import hapy_common as hc
import pywt
# from statsmodels.tsa.arima.model import ARIMA
import QBO_diagnostic_methods as QBO_methods
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
#-------------------------------------------------------------------------------
# based on E3SM diagnostics package:
# https://github.com/E3SM-Project/e3sm_diags/blob/main/e3sm_diags/driver/qbo_driver.py
#-------------------------------------------------------------------------------
case_list,case_root,case_sub = [],[],[]
clr_list,dsh_list,mrk_list,msz_list = [],[],[],[]
yr1_list = []
yr2_list = []
gidx_list = []
def add_case(case_in,root=None,n=None,dsh=0,clr='black',mrk=0,msz=0.01,yr1=None,yr2=None,gidx=0):
   global case_list
   case_list.append(case_in)
   clr_list.append(clr) ; dsh_list.append(dsh) ; mrk_list.append(mrk), msz_list.append(msz)
   yr1_list.append(yr1) ; yr2_list.append(yr2)
   gidx_list.append(gidx) # group index for ensemble mean
#-------------------------------------------------------------------------------
# yr1,yr2=1980,2019
yr1,yr2=1985,2014

grp_msz = 200
ens_msz = 10

add_case('ERA5', n='ERA5',clr='black',mrk='.',msz=grp_msz,yr1=yr1,yr2=yr2,gidx=0)

# add_case('v2.LR.amip_0101', n='E3SMv2 AMIP', p='/global/cfs/cdirs/e3smdata/simulations',s='archive/atm/hist') 
# add_case('v3.LR.amip_0101', n='E3SMv3 AMIP', p='/global/cfs/cdirs/m3312/whannah/e3smv3_amip',s='archive/atm/hist')

ens_root = '/lcrc/group/e3sm2/ac.wlin/E3SMv3'
tmp_case_list = []
tmp_case_list.append('v3.LR.historical_0051')
tmp_case_list.append('v3.LR.historical_0091')
tmp_case_list.append('v3.LR.historical_0101')
tmp_case_list.append('v3.LR.historical_0111')
tmp_case_list.append('v3.LR.historical_0121')
tmp_case_list.append('v3.LR.historical_0131')
tmp_case_list.append('v3.LR.historical_0141')
tmp_case_list.append('v3.LR.historical_0151')
tmp_case_list.append('v3.LR.historical_0161')
tmp_case_list.append('v3.LR.historical_0171')
tmp_case_list.append('v3.LR.historical_0181')
tmp_case_list.append('v3.LR.historical_0191')
tmp_case_list.append('v3.LR.historical_0201')
tmp_case_list.append('v3.LR.historical_0211')
tmp_case_list.append('v3.LR.historical_0221')
tmp_case_list.append('v3.LR.historical_0231')
tmp_case_list.append('v3.LR.historical_0241')
tmp_case_list.append('v3.LR.historical_0251')
tmp_case_list.append('v3.LR.historical_0261')
tmp_case_list.append('v3.LR.historical_0271')
tmp_case_list.append('v3.LR.historical_0281')
tmp_case_list.append('v3.LR.historical_0291')
tmp_case_list.append('v3.LR.historical_0301')
tmp_case_list.append('v3.LR.historical_0311')
tmp_case_list.append('v3.LR.historical_0321')

# for n in range(1):
for n in range(2):
   if n==0: yr1,yr2=1985,2014; c,m='blue','.'
   if n==1: yr1,yr2=2021,2050; c,m='red' ,'.'
   
   # for tmp_case in tmp_case_list:
   #    add_case(tmp_case,root=ens_root,clr=c,mrk=4,msz=0.01,yr1=yr1,yr2=yr2,gidx=n+1*2)

   for tmp_case in tmp_case_list:
      add_case(tmp_case,root=ens_root,clr=c,mrk=m,msz=ens_msz,yr1=yr1,yr2=yr2,gidx=n+1)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

pow_spec_lev = np.array([   1.,    2.,    3.,    5.,    7.,   10.,   20.,   30.,   50.,   70.,  100. ])
num_lev = len(pow_spec_lev)

fig_file = 'figs_QBO/QBO.ensemble_wavelet_amp-vs-period.v4.png'

tmp_file_head     = 'data_temp/QBO.wavelet_profile.v1.alt_period' # latlon data

var = ['U']

# plev = 20.
# plev_list = [10.,   20.,   30.,   50.,   70.]
# plev_list = [10.,   20.,   30.,   50.]
plev_list = [20, 50]

# num_plot_col = 1
num_plot_col = len(plev_list)

#---------------------------------------------------------------------------------------------------
# Set up plot resources
#---------------------------------------------------------------------------------------------------
num_var = len(var)
num_plev = len(plev_list)
num_case = len(case_list)


kwargs = {}
fig,axs = plt.subplots( 1, num_plev,
                        subplot_kw=kwargs,
                        constrained_layout=True, 
                        figsize=(10,5) )

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# def get_alpha_sigma2(data_in):
#    ## Fit AR1 model to estimate the autocorrelation (alpha) and variance (sigma2)
#    mod = ARIMA( data_in, order=(1,0,0) )
#    res = mod.fit()
#    return (res.params[1],res.params[2])
#---------------------------------------------------------------------------------------------------
def get_tmp_file(case,var,yr1,yr2):
   return f'{tmp_file_head}.{case}.{var}.{yr1}-{yr2}.nc'
#---------------------------------------------------------------------------------------------------
(gind,gcnt) = np.unique(gidx_list, return_counts=True)
# print(f'gind: {gind}')
# print(f'gcnt: {gcnt}')
# exit()
#---------------------------------------------------------------------------------------------------
# for v in range(num_var):
#    hc.printline()
#    print(f'  var: {hc.tclr.MAGENTA}{var[v]}{hc.tclr.END}')
v = 0
for k in range(num_plev):
   # hc.printline()
   print(f'  plev: {hc.tclr.MAGENTA}{plev_list[k]}{hc.tclr.END}')
   #----------------------------------------------------------------------------
   gper = np.zeros(len(gind))
   gamp = np.zeros(len(gind))
   gclr = [None]*len(gind)
   gmrk = [None]*len(gind)
   gmsz = [None]*len(gind)

   gper_list = [None]*len(gind)
   gamp_list = [None]*len(gind)

   for g in gind:
      gper_list[g] = []
      gamp_list[g] = []

   # wav_power_list, wav_period_list = [],[]
   x_list,y_list = [],[]
   for c in range(num_case):
      tmp_file = get_tmp_file(case_list[c],var[v],yr1_list[c],yr2_list[c])
      # print(f'    case: {hc.tclr.GREEN}{case_list[c]}{hc.tclr.END}  =>  {tmp_file}')
      #-------------------------------------------------------------------------
      tmp_ds = xr.open_dataset( tmp_file, use_cftime=True  )
      per = tmp_ds['period'].values
      lev = tmp_ds['lev'].values
      # wav = np.sqrt( tmp_ds['wavelet_spec'].values )
      # wav = np.sqrt( tmp_ds['wavelet_spec'].sel(lev=[10., 20., 30.]).values )
      wav = np.sqrt( tmp_ds['wavelet_spec'].sel(lev=plev_list[k]).values )
      #-------------------------------------------------------------------------
      # estimate dominant period and max amplitude
      max_ind = np.argmax(wav)
      max_amp = wav[max_ind]
      max_per = per[max_ind]
      # max_ind = np.unravel_index(np.argmax(wav),wav.shape)
      # max_amp = wav[max_ind]
      # max_lev = lev[max_ind[0]]
      # max_per = per[max_ind[1]]
      #-------------------------------------------------------------------------
      # print()
      # print(f'max_ind: {max_ind}')
      # print(f'max_amp: {max_amp}')
      # print(f'max_lev: {max_lev}')
      # print(f'max_per: {max_per}')
      # print()
      # exit()
      #-------------------------------------------------------------------------
      gper[gidx_list[c]] += max_per/gcnt[gidx_list[c]]
      gamp[gidx_list[c]] += max_amp/gcnt[gidx_list[c]]
      gclr[gidx_list[c]] = clr_list[c]
      gmrk[gidx_list[c]] = mrk_list[c]
      gmsz[gidx_list[c]] = msz_list[c]

      gper_list[gidx_list[c]].append(max_per)
      gamp_list[gidx_list[c]].append(max_amp)
      #-------------------------------------------------------------------------
      x_list.append(max_per); y_list.append(max_amp)
      # x_list.append(max_amp); y_list.append(max_per)
   #----------------------------------------------------------------------------
   # x_min = np.min([np.nanmin(d) for d in x_list])
   # x_max = np.max([np.nanmax(d) for d in x_list])
   y_min = np.min([np.nanmin(d) for d in y_list])
   y_max = np.max([np.nanmax(d) for d in y_list])
   # x_mag = np.abs(x_max-x_min)
   y_mag = np.abs(y_max-y_min)
   
   # tres.trYMinF = y_min - y_mag*0.08
   # tres.trYMaxF = y_max + y_mag*0.08
   #----------------------------------------------------------------------------
   for c in range(num_case):
      axs[k].scatter(np.array([1,1])*x_list[c],
                     np.array([1,1])*y_list[c],
                     c=clr_list[c],
                     s=msz_list[c],
                     marker=mrk_list[c])

   #----------------------------------------------------------------------------
   for g in gind:
      axs[k].scatter(np.average(np.array(gper_list[g][:])),
                     np.average(np.array(gamp_list[g][:])),
                     c=gclr[g],
                     s=grp_msz,
                     marker=gmrk[g])
      if g>0:
         axs[k].errorbar(  x=np.average(np.array(gper_list[g][:])),
                           y=np.average(np.array(gamp_list[g][:])),
                           xerr=np.std(np.array(gper_list[g][:])),
                           yerr=np.std(np.array(gamp_list[g][:])),
                           fmt='o',
                           color=gclr[g],
                           ecolor=gclr[g],
                           elinewidth=0.5,
                           capsize=2,
                           markersize=2)

#---------------------------------------------------------------------------------------------------
# Finalize plot

font_props = {'fontsize':12}
for k in range(len(axs)):
   # axs[k].set_aspect(1.0)
   # axs[k].set_box_aspect(1)
   axs[k].set_xlabel('QBO Period [months]', fontsize=16)
   axs[k].set_ylabel('QBO Amplitude',       fontsize=16)
   axs[k].set_title(f'{plev_list[k]} mb',   fontsize=16, loc='right')
   axs[k].set_xlim(22,32)

lgd_elm_list = []
lgd_elm_list.append( Line2D([0], [0], label='1985-2014 ERA5',   color='black', marker='o', markerfacecolor='black', markersize=10, linewidth=0 ) )
lgd_elm_list.append( Line2D([0], [0], label='1985-2014 E3SMv3', color='blue',  marker='o', markerfacecolor='blue',  markersize=10, linewidth=0 ) )
lgd_elm_list.append( Line2D([0], [0], label='2021-2050 E3SMv3', color='red',   marker='o', markerfacecolor='red',   markersize=10, linewidth=0 ) )
axs[0].legend(handles=lgd_elm_list, loc='lower right')


fig.savefig(fig_file, dpi=200, bbox_inches='tight')
plt.close(fig)

print(f'\n{fig_file}\n')

# hc.trim_png(fig_file)
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
