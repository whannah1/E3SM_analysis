import os, numpy as np, xarray as xr, glob, string
import matplotlib, matplotlib.pyplot as plt, matplotlib.ticker as mticker
matplotlib.use('Agg')
import hapy; host = hapy.get_host()
# -------------------------------------------------------------------
# case configuration
# -------------------------------------------------------------------
gscratch = '/pscratch/sd/w/whannah/e3sm_scratch/pm-cpu'
tmp_date_list = []
tmp_date_list.append('1983-01-01') # phase 1 - pi*1/4
# tmp_date_list.append('1993-04-01')
# tmp_date_list.append('2002-07-01')
# tmp_date_list.append('2022-10-01')
# tmp_date_list.append('1986-10-01') # phase 1 - pi*3/4
# tmp_date_list.append('1995-07-01')
# tmp_date_list.append('2000-04-01')
# tmp_date_list.append('2009-01-01')
# tmp_date_list.append('1982-01-01') # phase 1 - pi*5/4
# tmp_date_list.append('1987-04-01')
# tmp_date_list.append('2014-07-01')
# tmp_date_list.append('2021-10-01')
# tmp_date_list.append('1984-10-01') # phase 1 - pi*7/4
# tmp_date_list.append('1994-07-01')
# tmp_date_list.append('2006-04-01')
# tmp_date_list.append('2013-01-01')
# -------------------------------------------------------------------
compset = 'F20TR'
grid    = 'ne30pg2_r05_IcoswISC30E3r5'
ens_id  = '2024-SCIDAC-PCOMP-TEST'
def get_case_name(e,c,h,d): return '.'.join(['E3SM',ens_id,grid,f'EF_{e:0.2f}',f'CF_{c:02.0f}',f'HD_{h:0.2f}',f'{d}'])
# -------------------------------------------------------------------
case = []
gweff_list, cfrac_list, hdpth_list = [], [], []
start_date_list = []
case_dir, case_sub = [], []
case, name = [], []
clr_list, dsh_list, mrk_list = [], [], []
def add_case(e,c,h,d,n=None,p=None,s=None,dsh=0,clr='black',mrk=0):
   gweff_list.append(e)
   cfrac_list.append(c)
   hdpth_list.append(h)
   case.append( get_case_name(e,c,h,d) )
   name.append(n)
   case_dir.append(p); case_sub.append(s)
   dsh_list.append(dsh)
   clr_list.append(clr)
   mrk_list.append(mrk)
   start_date_list.append(d)
# -------------------------------------------------------------------
# build list of cases with all dates
for t, date in enumerate(tmp_date_list):
   add_case(e=0.35,c=10,h=0.50,d=date,dsh=0,clr='red')  # <<< v3 default
   add_case(e=0.12,c=16,h=0.48,d=date,dsh=0,clr='blue') # prev surrogate optimum
   # add_case(e=0.09,c=20,h=0.25,d=date) # no QBO at all
   # add_case(e=0.70,c=21,h=0.31,d=date) # QBO is too fast
# -------------------------------------------------------------------
# variable configuration
# -------------------------------------------------------------------
var, vname = [], []
def add_var(var_in, name=None):
   var.append(var_in)
   vname.append(name)
# -------------------------------------------------------------------
add_var('u',         name='U')
add_var('utendepfd', name='EP Flux Divergence')
add_var('wf',        name='Total Wave Forcing')
# -------------------------------------------------------------------
# plot settings
# -------------------------------------------------------------------
fig_file      = 'figs_QBO/TEM.profile.v1.png'
tmp_file_head = 'data_temp/TEM.profile.v1'

lat1, lat2 = -5, 5

remap_str, search_str = 'remap_90x180', 'h0.tem.'; first_file, num_files = 0, 1
# remap_str, search_str = 'remap_90x180', 'h2.tem.'; first_file, num_files = 0, 365*20

plot_diff = False

p_min =  10
p_max = 100

num_plot_col = len(var)

# dash pattern cycle for ERA5 dates (matplotlib linestyle strings)
_dash_styles = ['-', '--', ':', '-.']

tend_vars = ['utendepfd', 'wf']
# -------------------------------------------------------------------
# helper functions
# -------------------------------------------------------------------
def lat_mean(da, lat_name='lat'):
   cos_lat_rad = np.cos(da[lat_name] * np.pi / 180)
   return (da * cos_lat_rad).mean(dim=lat_name) / cos_lat_rad.mean(dim=lat_name)

def get_data(ds, v_idx, lat_name='lat', lev_name='plev'):
   if var[v_idx] == 'wf':
      data = ds['residual'] + ds['utendepfd']
   else:
      data = ds[var[v_idx]]
   data = data.sel({lat_name: slice(lat1, lat2)})
   data = data.sel({lev_name: slice(200e2, 0)})
   data = data.mean(dim='time')
   data = lat_mean(data, lat_name)
   if var[v_idx] in tend_vars:
      data = data * 86400.
   return data
# -------------------------------------------------------------------
# load model data
# -------------------------------------------------------------------
num_var, num_case = len(var), len(case)
num_date = len(tmp_date_list)

data_list_list, lev_list_list = [], []
for v in range(num_var):
   hapy.print_line()
   print(f'  var: {hapy.tclr.GREEN}{var[v]}{hapy.tclr.END}')
   data_list, lev_list_v = [], []
   for c in range(num_case):
      print(f'    case: {hapy.tclr.CYAN}{case[c]}{hapy.tclr.END}')

      file_path = f'{gscratch}/{case[c]}/data_{remap_str}_tem/*.eam.{search_str}*'
      file_list = sorted(glob.glob(file_path))
      if 'first_file' in locals(): file_list = file_list[first_file:]
      if 'num_files'  in locals(): file_list = file_list[:num_files]

      # manually combine files due to issues with open_mfdataset on Perlmutter
      ds = xr.open_dataset(file_list[0])
      for f in file_list[1:]:
         ds_tmp = xr.open_dataset(f)
         ds = xr.concat([ds, ds_tmp], dim='time')

      data = get_data(ds, v)
      data_list.append(data.values)
      lev_list_v.append(data['plev'].values / 1e2)

   data_list_list.append(data_list)
   lev_list_list.append(lev_list_v)

# -------------------------------------------------------------------
# load ERA5 observations
# -------------------------------------------------------------------
obs_root = '/global/cfs/cdirs/m4310/whannah/ERA5'
hapy.print_line()
print(f'  case: {hapy.tclr.CYAN}ERA5{hapy.tclr.END}')

obs_data_list = []   # obs_data_list[t][v] = (data_values, plev_hpa)
for t, date in enumerate(tmp_date_list):
   print(f'    date: {hapy.tclr.MAGENTA}{date}{hapy.tclr.END}')
   yr = date[0:4]
   mn = date[5:7]
   obs_file = f'{obs_root}/ERA5_monthly_{yr}{mn}_tem.nc'
   obs_ds = xr.open_mfdataset(obs_file)
   obs_ds = obs_ds.rename({'latitude': 'lat', 'level': 'plev'})
   obs_ds = obs_ds.isel(plev=slice(None, None, -1))
   obs_ds['plev'] = obs_ds['plev'] * 1e2
   obs_ds.load()

   obs_v_list = []
   for v in range(num_var):
      print(f'      var: {hapy.tclr.GREEN}{var[v]}{hapy.tclr.END}')
      data = get_data(obs_ds, v)
      obs_v_list.append((data.values, data['plev'].values / 1e2))
   obs_data_list.append(obs_v_list)

hapy.print_line()

# -------------------------------------------------------------------
# build figure
# -------------------------------------------------------------------
nrows = int(np.ceil(num_var / float(num_plot_col)))
ncols = num_plot_col

fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                         figsize=(4.5 * ncols, 6 * nrows),
                         squeeze=False)
axes_flat = axes.flatten()

panel_labels = list(string.ascii_lowercase)

for v in range(num_var):
   ax = axes_flat[v]
   data_list = [d.copy() for d in data_list_list[v]]
   lev_list_v = lev_list_list[v]

   baseline = data_list[0]
   if plot_diff:
      for c in range(num_case):
         data_list[c] = data_list[c] - baseline

   # mask levels outside [p_min, p_max]
   for c in range(num_case):
      for i in range(len(data_list[c])):
         if lev_list_v[c][i] > p_max or lev_list_v[c][i] < p_min:
            data_list[c][i] = np.nan

   # determine x-axis limits
   if var[v] in tend_vars:
      x_min, x_max = -0.6, 0.6
   else:
      x_min = np.min([np.nanmin(d) for d in data_list])
      x_max = np.max([np.nanmax(d) for d in data_list])

   # -------------------------------------------------------------------
   # plot model cases
   for c in range(num_case):
      ax.plot(np.ma.masked_invalid(data_list[c]), lev_list_v[c],
              color=clr_list[c],
              linestyle=_dash_styles[dsh_list[c] % len(_dash_styles)],
              linewidth=2.0)

   # vertical reference line at x=0
   ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')

   # -------------------------------------------------------------------
   # overlay ERA5 observations
   for t in range(num_date):
      obs_data, obs_lev = obs_data_list[t][v]
      obs_lev_masked = obs_lev.copy().astype(float)
      obs_data_masked = obs_data.copy()
      for i in range(len(obs_lev_masked)):
         if obs_lev_masked[i] > p_max or obs_lev_masked[i] < p_min:
            obs_data_masked[i] = np.nan
      ax.plot(np.ma.masked_invalid(obs_data_masked), obs_lev_masked,
              color='black',
              linestyle=_dash_styles[t % len(_dash_styles)],
              linewidth=1.5)

   # -------------------------------------------------------------------
   # axis formatting
   ax.set_xlim(x_min, x_max)
   ax.set_ylim(p_max, p_min)

   if var[v] in tend_vars:
      ax.set_xlabel('[m/s/day]', fontsize=10)
   elif var[v] == 'u':
      ax.set_xlabel('[m/s]', fontsize=10)

   ax.set_ylabel('Pressure [hPa]', fontsize=10)

   lat1_str = f'{lat1}N' if lat1 >= 0 else f'{abs(lat1)}S'
   lat2_str = f'{lat2}N' if lat2 >= 0 else f'{abs(lat2)}S'
   reg_str = f'{lat1_str}:{lat2_str}'

   var_title = vname[v]
   if plot_diff: var_title += ' (diff)'
   ax.set_title(f'({panel_labels[v]})  {var_title}', fontsize=10, loc='left')
   ax.set_title(reg_str, fontsize=10, loc='right')

   ax.tick_params(axis='both', which='both', labelsize=9)
   ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)

# hide any unused axes
for idx in range(num_var, len(axes_flat)):
   axes_flat[idx].set_visible(False)

# -------------------------------------------------------------------
# save figure
# -------------------------------------------------------------------
plt.tight_layout()
plt.savefig(fig_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n{fig_file}\n')
# -------------------------------------------------------------------
