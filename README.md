# E3SM_analysis

This repository is a place for scripts and notes used for analysis of E3SM simulations. This replaces an older collection of scripts largely relying on the PyNGL library for plot generation. In 2025, After 10 years of using pyngl regularly, I finally started running into problem. Luckily, development of the uxarray library finally allows painless plotting of unstructured data on maps. So the scripts here are all reliant on  uxarry, matplotlib, and cartopy.

# Conda Env for uxarray + matplotlib

Many scripts can be used with the E3SM unified enviroment, but if that is not available the environement below should also work.

```shell
conda create --name ux_env --channel conda-forge --yes numpy xarray dask netcdf4 cftime cmocean nco ncview uxarray numba dask cftime scipy scikit-learn matplotlib cartopy selenium firefox geckodriver
```

# Running python in interactive mode

```shell
python -i
exec( open( 'code_clim/plot.clim.map.v1.py' ).read() )
```

# Interactive Batch Job Commands

```shell
# NERSC
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account=????
# OLCF Andes
salloc --time 4:00:00 --account=cli115 --nodes=1 # -p gpu
micromamba activate taos_env
micromamba activate ux_env
```