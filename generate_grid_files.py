import os, subprocess as sp, hapy
host = hapy.get_host()
#---------------------------------------------------------------------------------------------------
if host=='llnl' : grid_root = '/p/lustre1/hannah6/files_grid'
if host=='lcrc' : grid_root = '/lcrc/group/e3sm/ac.whannah/files_grid'
if host=='olcf' : grid_root = '/lustre/orion/cli115/proj-shared/hannah6/files_grid'
if host=='nersc': grid_root = '/pscratch/sd/w/whannah/files_grid/'
if host=='alcf' : grid_root = '/lus/flare/projects/E3SM_Dec/whannah/files_grid'
#---------------------------------------------------------------------------------------------------
ne_list = []
ne_list.append(4)
ne_list.append(16)
ne_list.append(30)
ne_list.append(32)
ne_list.append(64)
ne_list.append(120)
ne_list.append(128)
ne_list.append(256)

overwrite = False
#---------------------------------------------------------------------------------------------------
def run_cmd(cmd):
  print(f'\n  {hapy.tclr.GREEN}{cmd}{hapy.tclr.END}\n')
  try:
    sp.check_output(cmd,shell=True,universal_newlines=True)
  except sp.CalledProcessError as error:
    print(error.output)
    exit()
#---------------------------------------------------------------------------------------------------
if 'grid_root' not in locals(): raise ValueError(f'grid_root not defined for host: {host}')
if not os.path.exists(grid_root): os.makedirs(grid_root,exist_ok=True)
#---------------------------------------------------------------------------------------------------
for ne in ne_list:

  exodus_file = f'{grid_root}/ne{ne}.g'
  pg2_file    = f'{grid_root}/ne{ne}pg2.g'
  scrip_file  = f'{grid_root}/ne{ne}pg2_scrip.nc'

  if not overwrite and os.path.exists(scrip_file):
    print(f'\n  skipping ne{ne}pg2 - file exists: {scrip_file}')
    continue

  run_cmd(f'GenerateCSMesh --alt --res {ne} --file {exodus_file}')
  run_cmd(f'GenerateVolumetricMesh --in {exodus_file} --out {pg2_file} --np 2 --uniform')
  run_cmd(f'ConvertMeshToSCRIP --in {pg2_file} --out {scrip_file}')

#---------------------------------------------------------------------------------------------------
