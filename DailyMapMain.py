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

    cluster = LocalCluster(n_workers=8, threads_per_worker=1, memory_limit = "48GB", local_directory="/scratch/projects/phase01/rab/B06/PROCESSED_ERA5")

    client = Client(cluster)
    print(client.dashboard_link)

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True)
    parser.add_argument("--month", required=True)
    args = parser.parse_args()

    year = args.year
    month = args.month

    print(f"Running for {year}-{month}")


    ml_vars = {'u','v', 'w', 't','q', 'o3'}
    B_vars = {'sp', 'z'}

    chunks_3d = { #### Depends on dataset but good chunking absolutely necessary for good performance
    "time": 1,       
    "latitude": -1,
    "level": -1,    ### Generally needed
    "longitude": -1 ### Needed for consistent FFT
    }
    chunks_2d = {
    "time": 1,
    "latitude": -1, 
    "longitude": -1
    }
    
    ### paths
    input_dir = '/scratch/era5/' + year + '/' + month + '/'
    grid_path = '/scratch/era5/grid_era5.nc'
    output_dir = Path('/scratch/projects/phase01/rab/B06/PROCESSED_ERA5')

    full_path = output_dir / year
    full_path.mkdir(parents=True, exist_ok=True)

    num_days = calendar.monthrange(int(year), int(month))[1]
    month_days = list(range(1, num_days + 1))

    chunk_size = 1
    day_chunks = [month_days[i:i + chunk_size] for i in range(0, len(month_days), chunk_size)]

    daily_files = []
    
    for i, days_subset in enumerate(day_chunks, start=1):
        print(f"Processing days {days_subset} (chunk {i}/{len(day_chunks)})")
    
        ds = HelperFunctions.load_data_subset(input_dir, ml_vars, B_vars,chunks_3d, chunks_2d,int(year), int(month), days_subset, timestep = 6)
        grid = HelperFunctions.load_grid(grid_path)
    
        ds = CalcFunctions.calc_vertical(ds, grid)
        ds = CalcFunctions.calc_lrt(ds)
        ds = CalcFunctions.calc_Ri(ds)
        ds = CalcFunctions.calc_Ti(ds)
 
        shear_profile = CalcFunctions.calc_frequency_map(ds, var='Si2', threshold=4e-4)
        Ri_profile     = CalcFunctions.calc_frequency_map(ds, var='Ri', threshold=1, direction='lower')
        Ti_profile     = CalcFunctions.calc_frequency_map(ds, var='Ti1', threshold=17e-7)

        maps = xr.merge([shear_profile, Ri_profile, Ti_profile])

        daily_file = full_path / f'TurbulenceMap_{year}{month}_chunk{i}.zarr'
        maps.to_zarr(daily_file, mode='w')
        daily_files.append(daily_file)
    
    monthly_profiles = xr.open_mfdataset(daily_files, engine = 'zarr', combine='nested', concat_dim = 'time')
    monthly_profiles.to_zarr(full_path / f'TurbulenceMapsTimesteps_{year}{month}.zarr', mode='w')
    monthly_profiles.mean(dim = 'time').to_zarr(full_path / f'TurbulenceMaps_{year}{month}.zarr', mode='w')
    for f in daily_files:
        shutil.rmtree(f)


if __name__ == "__main__":
    main()
    
### variables
