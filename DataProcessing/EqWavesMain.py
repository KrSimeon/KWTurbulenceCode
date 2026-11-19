import os
# Limit all internal threading
os.environ["OMP_NUM_THREADS"] = "1"      # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1" # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"      # Intel MKL
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr

import numpy as np
import shutil
import xarray as xr
import xrft
import calendar
from datetime import datetime, timedelta
import argparse
import sys
sys.path.append('../')
from functions import HelperFunctions, CalcFunctions
from pathlib import Path
from dask.distributed import Client, LocalCluster
import dask

def main():

    ###Initialize Dask Cluster

    dask.config.set({
    "distributed.worker.memory.target": 0.75,
    "distributed.worker.memory.spill": 0.85,
    "distributed.worker.memory.pause": 0.92,
    "distributed.worker.memory.terminate": 0.98,
    })

    cluster = LocalCluster(n_workers=8, threads_per_worker=1, memory_limit = "42GB", local_directory="/scratch/projects/phase01/rab/B06/PROCESSED_ERA5")


    client = Client(cluster)
    print(client.dashboard_link)

    ml_vars = {'u','v', 't', 'q'}
    B_vars = {'sp', 'z'}

    chunks_3d = { #### Depends on dataset but good chunking absolutely necessary for good performance
    "time": 1,       
    "latitude": 'auto',
    "level": -1,    ### Generally needed
    "longitude": -1 ### Needed for consistent FFT
    }
    chunks_2d = {
    "time": 1,
    "latitude": -1, 
    "longitude": 'auto'
    }

    year = 2017

    #center_date = '2019-06-24T01:00:00.000000000'
    grid_path = '/scratch/era5/grid_era5.nc'
    data_path = '/scratch/era5/'
    out_path = '/scratch/projects/phase01/rab/B06/PROCESSED_ERA5/EqWaves/'

    outweeks = HelperFunctions.create_output_weeks(year, output_days=7)
    grid = HelperFunctions.load_grid(grid_path)

    for week in outweeks:
        print(week['output_start'])

        out_file = (
            out_path
            + f"EQWaves_{str(week['output_start'])[:10]}_"
            f"{str(week['output_end'])[:10]}.zarr"
        )

        if os.path.exists(out_file):
            print(f"Already exists, skipping: {out_file}")
            continue

        filelist_B, filelist_ml = HelperFunctions.create_filelist_center(data_path, week['center_date'], timespan = 96, timestep = 6)

        ds_B = xr.open_mfdataset(filelist_B, combine='by_coords', chunks = chunks_2d)[list(B_vars)]
        ds_ml = xr.open_mfdataset(filelist_ml, combine='by_coords', chunks = chunks_3d)[list(ml_vars)]

        ds = xr.merge([ds_ml, ds_B]).sel(latitude = slice(24,-24))

        ds = ds.coarsen(
        latitude=5,
        longitude=5,
        boundary="trim"
        ).mean()

        ds = CalcFunctions.calc_vertical(ds, grid)

        ds = ds.chunk({'time' : -1, 'level' : 14})

        y0=4
        waves=np.array(['Kelvin','R1'])

        q,r, c_on_g = CalcFunctions.uz_to_qr(ds.u, ds.gph, y0 =y0)

        time_orig = q.time
        lon_orig = q.longitude

        dx = q.longitude.diff("longitude").mean()
        N = q.sizes["longitude"]
        L = N * dx

        kmin=2/L
        kmax=40/L
        fmin = 1 / (30 * 86400)
        fmax = 1 / (2 * 86400)

        qf = xrft.fft(q, dim=['time', 'longitude'])
        rf = xrft.fft(r, dim=['time', 'longitude'])
        vf = xrft.fft(ds.v, dim=['time', 'longitude'])

        u_wave, z_wave, v_wave = CalcFunctions.filt_project_xarray(qf,rf,vf,ds.latitude,y0,waves,fmin,fmax,kmin,kmax,c_on_g)

        ds_final = xr.Dataset({
            "u_wave": u_wave.rename("u_wave"),
            "v_wave": v_wave.rename("v_wave"),
            "z_wave": z_wave.rename("z_wave"),
        })

        ds_final["u_wave"].attrs = {
            "units": "m s-1",
            "long_name": "Wave Zonal Wind",
            "standard_name": "eastward_wind"
        }

        ds_final["v_wave"].attrs = {
            "units": "m s-1",
            "long_name": "Wave Meridional Wind",
            "standard_name": "northward_wind"
        }

        ds_final["z_wave"].attrs = {
            "units": "m",
            "long_name": "Wave Geopotential Height",
            "standard_name": "geopotential_height"
        }

        out_file = (
            out_path
            + f"EQWaves_{str(week['output_start'])[:10]}_"
            f"{str(week['output_end'])[:10]}.zarr"
        )

        ds_final = ds_final.assign_coords({"time": time_orig.values,"longitude": lon_orig.values,})

        ds_final = ds_final.where(
            (ds_final.time >= week["output_start"]) &
            (ds_final.time < week["output_end"]),
            drop=True
        )

        ds_final.to_zarr(out_file, mode="w")

        ds_B.close()
        ds_ml.close()  

if __name__ == "__main__":
    main()
