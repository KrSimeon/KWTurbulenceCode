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
from io import StringIO
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

    cluster = LocalCluster(n_workers=12, threads_per_worker=1, memory_limit = "32GB", local_directory="/scratch/projects/phase01/rab/B06/PROCESSED_ERA5")

    client = Client(cluster)
    print(client.dashboard_link)

    grid_path = '/scratch/era5/grid_era5.nc'
    data_path = '/scratch/era5/'
    ml_vars = {'o3', 't', 'q', 'u', 'v'}
    B_vars = {'sp', 'z'}

    path = '/scratch/projects/phase01/rab/B06/PROCESSED_ERA5/Data/'
    filenames_kuala = glob.glob(path + 'KualaLumpur/kuala*_SHADOZV06.dat')
    filenames_nairobi = glob.glob(path + 'Nairobi/nairobi*_SHADOZV06.dat')
    filenames_sancrist = glob.glob(path + 'SanCristobal/sancrist*.dat')

    def read_shadoz(filename):
        metadata = {}

        with open(filename, "r") as f:
            lines = f.readlines()

        # Find the line containing the column names
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith("Time"):
                header_idx = i
                break

            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

        if header_idx is None:
            raise ValueError("Could not find data header.")

        # Read the data table
        data = "".join(lines[header_idx:])

        df = pd.read_csv(
            StringIO(data),
            sep=r"\s+",
            skiprows=[1],        # Skip the units row
            engine="python",
        )

        # Replace missing values
        missing = float(metadata.get("Missing or bad values", 9000))
        df = df.replace(missing, np.nan)

        # Convert to xarray
        ds = xr.Dataset.from_dataframe(df)

        # Store metadata
        ds.attrs.update(metadata)

        if "Launch Date" in metadata:
            launch_day = pd.to_datetime(
                metadata["Launch Date"],
                format="%Y%m%d"
            )

            ds = ds.expand_dims(
                day=[launch_day]
            )

        return ds

    data_sets = []
    for filename in filenames_kuala:
        data = read_shadoz(filename)
        data_sets.append(data)

    data_kuala = xr.concat(data_sets, dim = 'day')

    data_sets = []
    for filename in filenames_nairobi:
        data = read_shadoz(filename)
        data_sets.append(data)

    data_nairobi = xr.concat(data_sets, dim = 'day')

    data_sets = []
    for filename in filenames_sancrist:
        data = read_shadoz(filename)
        data_sets.append(data)

    data_sancrist = xr.concat(data_sets, dim = 'day')

    lat_kuala = data_kuala.GPS_Lat.mean()
    lon_kuala = data_kuala.GPS_Lon.mean()
    lat_nairobi = data_nairobi.GPS_Lat.mean()
    lon_nairobi = data_nairobi.GPS_Lon.mean()
    lat_sancrist = data_sancrist.GPS_Lat.mean()
    lon_sancrist = data_sancrist.GPS_Lon.mean()

    def load_era5_at_location(
        ds,
        lat,
        lon,
        data_path,
        grid,
        ml_vars,
        B_vars,
    ):
        # Find ERA5 files available for each SHADOZ launch date
        original_count = ds.sizes["day"]

        valid_positions = []
        filelist_ml = []
        filelist_B = []

        for i, day in enumerate(ds["day"].data):
            try:
                ml_file, b_file = HelperFunctions.create_era5_filename(
                    data_path, day
                )
            except FileNotFoundError as e:
                print(f"Skipping {pd.Timestamp(day).date()}: {e}")
                continue

            if ml_file is None or b_file is None:
                print(f"Skipping {pd.Timestamp(day).date()}: ERA5 files unavailable")
                continue

            valid_positions.append(i)
            filelist_ml.append(ml_file)
            filelist_B.append(b_file)

        if not valid_positions:
            raise ValueError("No matching ERA5 files found for the SHADOZ launch dates.")

        # Select by position; this also works with duplicate day coordinates
        ds = ds.isel(day=valid_positions)

        print(
            f"Processing {len(valid_positions)} launch dates; "
            f"skipped {original_count - len(valid_positions)} dates."
        )

        def select_location(ds):
            return ds.sel(
                latitude=lat,
                longitude=lon,
                method="nearest",
            )

        ds_ml = xr.open_mfdataset(
            filelist_ml,
            combine="by_coords",
            preprocess=select_location,
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )[list(ml_vars)]

        ds_B = xr.open_mfdataset(
            filelist_B,
            combine="by_coords",
            preprocess=select_location,
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )[list(B_vars)]

        print("Files opened")

        field = xr.merge([ds_ml, ds_B])
        field = CalcFunctions.calc_vertical(field, grid)
        field = CalcFunctions.calc_lrt(field).chunk("auto")
        return field


    grid = HelperFunctions.load_grid(grid_path)  

    field_kuala = load_era5_at_location(
        ds=data_kuala,
        lat=lat_kuala,
        lon=lon_kuala,
        data_path=data_path,
        grid=grid,
        ml_vars=ml_vars,
        B_vars=B_vars,
    )

    field_nairobi = load_era5_at_location(
        ds=data_nairobi,
        lat=lat_nairobi,
        lon=lon_nairobi,
        data_path=data_path,
        grid=grid,
        ml_vars=ml_vars,
        B_vars=B_vars,
    )

    field_sancrist = load_era5_at_location(
        ds=data_sancrist,
        lat=lat_sancrist,
        lon=lon_sancrist,
        data_path=data_path,
        grid=grid,
        ml_vars=ml_vars,
        B_vars=B_vars,
    )

    field_kuala.to_zarr('/scratch/projects/phase01/rab/B06/PROCESSED_ERA5/Data/kuala_era5.zarr', mode = 'w')
    field_nairobi.to_zarr('/scratch/projects/phase01/rab/B06/PROCESSED_ERA5/Data/nairobi_era5.zarr', mode = 'w')
    field_sancrist.to_zarr('/scratch/projects/phase01/rab/B06/PROCESSED_ERA5/Data/sancrist_era5.zarr', mode = 'w')

if __name__ == "__main__":
    main()
