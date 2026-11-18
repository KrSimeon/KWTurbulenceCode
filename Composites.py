import os
# Limit all internal threading
os.environ["OMP_NUM_THREADS"] = "1"      # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1" # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"      # Intel MKL
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr

import numpy as np
import shutil
import xarray as xr
import calendar
import argparse
import pandas as pd
import sys
sys.path.append('../')
from functions import HelperFunctions, CalcFunctions
from pathlib import Path
from dask.distributed import Client, LocalCluster
import dask
import glob
import logging
from dask.distributed import wait

from dask.array.core import PerformanceWarning

import warnings

warnings.simplefilter("error", category=PerformanceWarning)

def main():
    ###Initialize Dask Cluster

    dask.config.set({
    "distributed.worker.memory.target": 0.75,
    "distributed.worker.memory.spill": 0.85,
    "distributed.worker.memory.pause": 0.92,
    "distributed.worker.memory.terminate": 0.98,
    })

    cluster = LocalCluster(n_workers=6, threads_per_worker=1, memory_limit = "32GB", local_directory="/scratch/projects/phase01/rab/B06/PROCESSED_ERA5")

    client = Client(cluster)
    print(client.dashboard_link)

    log_file = Path("dask_run.log")


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("dask_run.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True)
    args = parser.parse_args()

    year = args.year

    waves_flag = True
    mjo_flag = True

    input_dir = '/scratch/era5/' + year + '/'
    out_path = f'/scratch/projects/phase01/rab/B06/PROCESSED_ERA5/Composites/'
    grid_path = '/scratch/era5/grid_era5.nc'

    months_array = np.arange(1,12+1)
    months = [f"{m:02d}" for m in months_array]

    ml_vars = {'u','v','t','q','cc','ciwc','o3'}
    B_vars = {'sp', 'z'}

    chunks_3d = { #### Depends on dataset but good chunking absolutely necessary for good performance
    "time": 'auto',       
    "latitude": 'auto',
    "level": -1,    ### Generally needed
    "longitude": -1 ### Needed for consistent FFT
    }
    chunks_2d = {
    "time": 'auto',
    "latitude": 'auto', 
    "longitude": -1
    }

    perturbation_variables = {'u','v','t','o3','q','N2','Si2', 'z'}

    for m in months:
        path_month = input_dir + m + '/'

        latitudes = [-7, 0, 7]

        era5_data_month = HelperFunctions.load_data_timestep(path_month, ml_vars, B_vars, chunks_3d, chunks_2d,
                        str(year), m, timestep=6).sel(latitude = latitudes, method = 'nearest').sortby('longitude')

        times = pd.date_range(
            start=f"{year}-{m}-01 00:00",
            periods=era5_data_month.sizes["time"],
            freq="6h"
        )

        ds = era5_data_month.assign_coords(day=times)

        grid = HelperFunctions.load_grid(grid_path)

        ds = CalcFunctions.calc_vertical(ds, grid)
        ds = CalcFunctions.calc_lrt(ds)
        ds = CalcFunctions.calc_Ri(ds)

        monthly_data = ds.groupby('time.month').mean()

        for var in perturbation_variables:
            ds = CalcFunctions.calc_perturbations_composites(ds, monthly_data, var).squeeze(dim = 'month')


        ### Load Wave Data

        if waves_flag == True:
        
            wave_path = '/scratch/projects/phase01/rab/B06/PROCESSED_ERA5/EqWaves/'
            filenames = glob.glob(wave_path + f'EQWaves_*{year}-{m}*')

            wave_data = xr.open_mfdataset(filenames, engine = 'zarr')

            wave_data = wave_data.interp(
                latitude=ds.latitude,
                longitude=ds.longitude
            )

            ds['u_Kelvin'] = wave_data.u_wave.sel(wave = 'Kelvin')
            ds['u_R1'] = wave_data.u_wave.sel(wave = 'R1')
            ds['u_WMRG'] = wave_data.u_wave.sel(wave = 'WMRG')
            ds['u_R2'] = wave_data.u_wave.sel(wave = 'R2')

            ds['v_R1'] = wave_data.v_wave.sel(wave = 'R1')
            ds['v_WMRG'] = wave_data.v_wave.sel(wave = 'WMRG')
            ds['v_R2'] = wave_data.v_wave.sel(wave = 'R2')

            ds['z_Kelvin'] = wave_data.z_wave.sel(wave = 'Kelvin')
            ds['z_R1'] = wave_data.z_wave.sel(wave = 'R1')
            ds['z_WMRG'] = wave_data.z_wave.sel(wave = 'WMRG')
            ds['z_R2'] = wave_data.z_wave.sel(wave = 'R2')

        # if mjo_flag == True:
        
        #     mjo_path = '/scratch/projects/phase01/rab/B06/PROCESSED_ERA5/EqWaves/'
        #     filenames = glob.glob(mjo_path + f'MJO_*{year}-{m}*')

        #     mjo_data = xr.open_mfdataset(filenames, engine = 'zarr')

        #     mjo_data = mjo_data.interp(
        #         latitude=ds.latitude,
        #         longitude=ds.longitude
        #     )

        #     ds['u_mjo'] = mjo_data.u_mjo
        #     ds['v_mjo'] = mjo_data.v_mjo
        #     ds['t_mjo'] = mjo_data.t_mjo
        #     ds['z_mjo'] = mjo_data.z_mjo

        all_composites = []

        logging.info('persisting')
        ds = ds.persist()
        wait(ds)
        logging.info('done')

        lon_slices = [slice(x - 10, x + 10) for x in range(-170, 171, 20)]

        for lat_center in latitudes:
            print(f"Processing latitude: {lat_center}")

            ds_lat = ds.sel(latitude=lat_center, method = 'nearest')

            composites_lat = []

            previous_box_skip = None

            for lon_slice in lon_slices:

                composite, valid_days, lon_center, box_skip, center_method = CalcFunctions.calculate_composite(
                    ds_lat, lon_slice, previous_box_skip=previous_box_skip
                )

                composite = composite.assign(
                    valid_days=valid_days,
                    lon_center=lon_center,
                    center_method = center_method,
                )

                composite = composite.expand_dims({
                    'box_center': [(lon_slice.start + lon_slice.stop) / 2],
                })

                composite = composite.compute()

                composites_lat.append(composite)

                previous_box_skip = box_skip

            composite_full_lat = xr.concat(
                composites_lat,
                dim = 'box_center',
                join='outer'
            )

            composite_full_lat = composite_full_lat.expand_dims({
                'lat_center': [lat_center],
                })

            composite_full_lat = composite_full_lat.chunk({
                "box_center": 1,
                "time": -1,
                "level": -1,
                "longitude": 241
            })
            
            monthly_file = out_path + f'CompositesInversion_{year}{m}_{lat_center}.zarr'

            print('writing output')
            composite_full_lat.to_zarr(monthly_file, mode = 'w')

            del composites_lat
            del composite_full_lat

if __name__ == "__main__":
    main()

