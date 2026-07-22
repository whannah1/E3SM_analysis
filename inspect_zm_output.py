import os, subprocess as sp, numpy as np, xarray as xr, dask, copy, string, cmocean, glob, datetime
import hapy
class clr:END,RED,GREEN,MAGENTA,CYAN = '\033[0m','\033[31m','\033[32m','\033[35m','\033[36m'
#-------------------------------------------------------------------------------

# var = 'T_2m'
# var = 'ps'
# var = 'VapWaterPath'

var = 'zm_T_mid_tend'
# var = 'zm_qv_tend'
# var = 'zm_horiz_winds_tend'
# var = 'zm_detr_qc'
# var = 'zm_detr_qi'

# var = 'mcsp_ds_out'
# var = 'mcsp_dq_out'
# var = 'mcsp_du_out'
# var = 'mcsp_dv_out'
# var = 'mcsp_freq'
# var = 'mcsp_shear'
# var = 'zm_depth'

# var = 'evap_ds_out'
# var = 'evap_dq_out'

var = 'zm_prec'
# var = 'zm_cape'
# var = 'zm_dcape'
# var = 'zm_activity'

ntime = 4
#-------------------------------------------------------------------------------
data_root = '/lustre/orion/cli115/proj-shared/hannah6/e3sm_scratch'

# c1 = 'E3SM.2026-ZM-DEV-01.F2010xx-ZM.ne4pg2.NN_1.zm_cxx'
# c2 = 'E3SM.2026-ZM-DEV-01.F2010xx-ZM.ne4pg2.NN_1.zm_f90'

# c1 = 'E3SM.2026-ZM-DEV-00.F2010xx-ZM.ne30pg2.NN_4.zm_f90.ts_output_1.dcape_1.ull_1'
# c1 = 'E3SM.2026-ZM-DEV-00.F2010xx-ZM.ne30pg2.NN_4.zm_f90.ts_output_1.dcape_0.ull_1'
c1 = 'E3SM.2026-ZM-DEV-00.F2010xx-ZM.ne30pg2.NN_4.zm_f90.ts_output_1.dcape_1.ull_0'

# c2 = 'E3SM.2026-ZM-DEV-00.F2010xx-ZM.ne30pg2.NN_4.zm_cxx.ts_output_1.dcape_1.ull_1'
# c2 = 'E3SM.2026-ZM-DEV-00.F2010xx-ZM.ne30pg2.NN_4.zm_cxx.ts_output_1.dcape_0.ull_1'
# c2 = 'E3SM.2026-ZM-DEV-00.F2010xx-ZM.ne30pg2.NN_4.zm_cxx.ts_output_1.dcape_1.ull_0'
# c2 = 'E3SM.2026-ZM-DEV-00.F2010xx-ZM.ne30pg2.NN_4.zm_cxx.ts_output_1.dcape_0.ull_0'

c2 = 'E3SM.2026-ZM-DEV-01.F2010xx-ZM.ne30pg2.NN_4.zm_cxx.ts_output_1.dcape_1.ull_0'

f1 = f'{data_root}/{c1}/run/output.debug.INSTANT.nsteps_x1.0001-01-01-00000.nc'
f2 = f'{data_root}/{c2}/run/output.debug.INSTANT.nsteps_x1.0001-01-01-00000.nc'

# f1 = f2; print(f'\n{clr.RED}WARNING - only using one file!{clr.END}')
# f2 = f1; print(f'\n{clr.RED}WARNING - only using one file!{clr.END}')

f1_time = datetime.datetime.fromtimestamp( os.path.getmtime(f1) )
f2_time = datetime.datetime.fromtimestamp( os.path.getmtime(f2) )

print()
print(f' f1: {f1}  {f1_time}\n f2: {f2}  {f2_time}')
# print(f' {var}\n')
# exit()

ds1 = xr.open_dataset(f1); da1 = ds1[var]
ds2 = xr.open_dataset(f2); da2 = ds2[var]

fac = None
if var == 'zm_T_mid_tend': fac=86400
if var == 'evap_ds_out'  : fac=86400/1004.64
if var == 'evap_dq_out'  : fac=86400e3
if var == 'mcsp_ds_out'  : fac=86400
if var == 'mcsp_dq_out'  : fac=86400e3
if var == 'zm_qv_tend'   : fac=86400e3
if var == 'zm_prec'      : fac=86400e3
if 'zm_detr' in var      : fac=86400e3

if fac is not None: da1=da1*fac; da2=da2*fac


# hapy.print_stat(da1.isel(time=1),name=f'cxx t=1 {var}')
# hapy.print_stat(da2.isel(time=1),name=f'f90 t=1 {var}'); print()

# hapy.print_stat(da1.isel(time=2),name=f'cxx t=2 {var}')
# hapy.print_stat(da2.isel(time=2),name=f'f90 t=2 {var}'); print()

if 'lev' in da1.dims:
  # da1 = da1.isel(lev=90)
  # da2 = da2.isel(lev=90)
  da1 = da1.sum(dim='lev')
  da2 = da2.sum(dim='lev')

if var=='zm_horiz_winds_tend':
  da1 = da1.isel(dim2=0)
  da2 = da2.isel(dim2=0)

# ncol_max = da1.isel(time=1).argmax(dim=['ncol'])
# print(ncol_max); exit()



#-------------------------------------------------------------------------------

# print(da1.sum().values)
# exit()

print()
print(f'global sum of {var}:')

msg = f''
# msg += ' '*6+f'  {'cxx':>20}  {'f90':>20}\n'
msg += ' '*6+f'  {'f90':>20}  {'cxx':>20}\n'
for t in range(ntime):
  tmp1 = da1.isel(time=t).sum()
  tmp2 = da2.isel(time=t).sum()
  msg += f't: {t:3}  {tmp1:20.8f}  {tmp2:20.8f}'
  if tmp1==tmp2: msg += f'  {clr.GREEN}<<< EQUAL{clr.END}'
  if tmp1!=tmp2: msg += f'  {clr.RED  }<<< NOT EQUAL{clr.END}'
  msg += f'\n'
msg += f'\n'
print(msg)

exit()

#-------------------------------------------------------------------------------
# check single column

print()
print('ncol=17:')

tmp_da1 = da1.isel(ncol=17)
tmp_da2 = da2.isel(ncol=17)
msg = f''
for t in range(ntime):
  tmp1 = tmp_da1.isel(time=t)
  tmp2 = tmp_da2.isel(time=t)
  msg += f't: {t:3}  {tmp1:20.8f}  {tmp2:20.8f}'
  if tmp1==tmp2: msg += f'  {clr.GREEN}<<< EQUAL{clr.END}'
  if tmp1!=tmp2: msg += f'  {clr.RED  }<<< NOT EQUAL{clr.END}'
  msg += f'\n'
msg += f'\n'
print(msg)

#-------------------------------------------------------------------------------
# find which columns diverge at t==2

print()
print('diffs at t=2')
cnt = 0
max_diff = 0
for i in range(len(da1.ncol)):
  tmp1 = da1.isel(time=2,ncol=i)
  tmp2 = da2.isel(time=2,ncol=i)
  if tmp1!=tmp2:
    cnt += 1
    if cnt<=10:
      diff = tmp2 - tmp1
      max_diff = max(diff,max_diff)
      print(f'  diff found at ncol={i:3} - {tmp1:20.8f}  {tmp2:20.8f}  {diff:20.8f}')
    else:
      break
pct = 100.*cnt/len(da1.ncol)
print(); print(f'cnt: {cnt} ({pct:4.2f}%)')
print(); print(f'max_diff: {max_diff:20.8f}')
print()

#-------------------------------------------------------------------------------

