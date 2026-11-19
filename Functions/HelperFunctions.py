#!/uni-mainz.de/homes/sikraus/miniconda3/envs/myenv/bin/python3

import xarray as xr
import glob
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

def load_data_subset(path, ml_vars, B_vars, chunks_3d, chunks_2d, year, month, days, timestep=6):

    hours = [f"{h:02d}" for h in range(0, 24, timestep)]

    ml_files = []
    B_files  = []

    for day in days:
        day_str = f"{day:02d}"
        for hour in hours:
            ml_files.extend(glob.glob(f"{path}ml{year}{month:02d}{day_str}*_{hour}.nc"))
            B_files.extend(glob.glob(f"{path}B{year}{month:02d}{day_str}*_{hour}.nc"))

    if not ml_files or not B_files:
        raise FileNotFoundError("No files found for the selected days and timestep.")

    ds_ml = xr.open_mfdataset(ml_files, chunks=chunks_3d, combine='nested', concat_dim='time')
    ds_B  = xr.open_mfdataset(B_files, chunks=chunks_2d, combine = 'nested', concat_dim = 'time')

    ds_ml_sel = ds_ml[list(ml_vars)]
    ds_B_sel  = ds_B[list(B_vars)]

    ds_all = xr.merge([ds_ml_sel, ds_B_sel])

    return ds_all

def load_data(path, ml_vars, B_vars, chunks_3d, chunks_2d, year, month, day):
    # open datasets lazily

    ds_ml   = xr.open_mfdataset(path + f"ml{year}{month}{day}*.nc", chunks=chunks_3d, combine='by_coords')
    ds_B    = xr.open_mfdataset(path + f"B{year}{month}{day}*.nc", chunks=chunks_2d, combine='by_coords')

    # select only requested variables
    ds_ml_sel   = ds_ml[list(ml_vars)]
    ds_B_sel    = ds_B[list(B_vars)]

    # merge everything into one Dataset
    ds_all = xr.merge([ds_ml_sel, ds_B_sel])

    return ds_all

def load_data_dummy(path, ml_vars, B_vars, chunks_3d, chunks_2d):
    # open datasets lazily
    ds_ml   = xr.open_mfdataset(path + f"ml*.nc", chunks=chunks_3d, combine='by_coords')
    ds_B    = xr.open_mfdataset(path + f"B*.nc", chunks=chunks_2d, combine='by_coords')

    ds_ml_sel   = ds_ml[list(ml_vars)]
    ds_B_sel    = ds_B[list(B_vars)]

    ds_all = xr.merge([ds_ml_sel, ds_B_sel])

    return ds_all

def load_grid(grid_path, chunks = None):
    ds_grid = xr.open_dataset(grid_path, chunks=chunks)
    return ds_grid

def load_data_timestep(path, ml_vars, B_vars, chunks_3d, chunks_2d, year, month, timestep):
    # open datasets lazily
    
    hours = [f"{h:02d}" for h in range(0, 24, timestep)]

    ml_files = []
    B_files = []

    for hour in hours:
        ml_files.extend(sorted(glob.glob(f"{path}ml{year}{month}*_{hour}.nc")))
        B_files.extend(sorted(glob.glob(f"{path}B{year}{month}*_{hour}.nc")))

    ds_ml = xr.open_mfdataset(ml_files, chunks=chunks_3d, combine='nested', concat_dim='time')
    ds_B  = xr.open_mfdataset(B_files,  chunks=chunks_2d, combine='nested', concat_dim='time')

    # select only requested variables
    ds_ml_sel   = ds_ml[list(ml_vars)]
    ds_B_sel    = ds_B[list(B_vars)]

    # merge everything into one Dataset
    ds_all = xr.merge([ds_ml_sel, ds_B_sel])

    return ds_all

lon_widths = {
    "North Pacific 1": 50,  # degrees
    "North Pacific 2": 55,
    "Southern Ocean (Pacific) 1": 20,
    "Southern Ocean (Pacific) 2": 100,
}
combine_map = {
    "North Pacific": ["North Pacific 1", "North Pacific 2"],
    "Southern Ocean (Pacific)": ["Southern Ocean (Pacific) 1", "Southern Ocean (Pacific) 2"]
}
def combine_boxes(ds, lon_widths=lon_widths, combine_map=combine_map):
    for new_box, old_boxes in combine_map.items():
        weighted_sum = sum(ds.sel(box=b) * lon_widths[b] for b in old_boxes) / sum(lon_widths[b] for b in old_boxes)

        ds= ds.drop_sel(box=old_boxes)
        weighted_sum = weighted_sum.expand_dims({"box": [new_box]})
        ds = xr.concat([ds, weighted_sum], dim="box")
    
    return ds

def create_output_weeks(year, output_days=7):
    
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)

    output_weeks = []

    current = start

    while current < end:
        output_start = current
        output_end = min(
            current + timedelta(days=output_days),
            end
        )

        # Midpoint of the output period
        center_date = output_start + (output_end - output_start) / 2

        output_weeks.append({
            "output_start": np.datetime64(output_start),
            "output_end": np.datetime64(output_end),
            "center_date": np.datetime64(center_date),
        })

        current += timedelta(days=output_days)

    return output_weeks

def create_filelist_center(mainpath, date, timespan=96, timestep=6):
    date = datetime.fromisoformat(str(date).replace('Z', '')) if isinstance(date, (str, np.datetime64)) else date

    start_date = date - timedelta(days=timespan / 2)
    end_date = date + timedelta(days=timespan / 2)

    times = []
    current = start_date

    if start_date == end_date:
        times.append(start_date)

    while current < end_date:
        times.append(current)
        current += timedelta(hours=timestep)

    filelist_B = []
    filelist_ml = []

    for t in times:
        month_dir = Path(mainpath) / f"{t.year}" / f"{t.month:02d}"

        filelist_B.append(month_dir / f"B{t:%Y%m%d}_{t:%H}.nc")
        filelist_ml.append(month_dir / f"ml{t:%Y%m%d}_{t:%H}.nc")

    return filelist_B, filelist_ml

def create_filelist(mainpath, year, timespan=96, timestep = 6):

    overlap = timespan // 2  # 48 days

    # Start and end dates needed for analysis
    start_date = datetime(int(year), 1, 1) - timedelta(days=overlap)
    end_date = datetime(int(year), 12, 31) + timedelta(days=overlap)

    # Collect all year/month directories that fall in the required range
    months = set()

    current = datetime(start_date.year, start_date.month, 1)
    while current <= end_date:
        months.add((current.year, current.month))

        # Advance one month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    hours = [f"{h:02d}" for h in range(0, 24, timestep)]

    filelist_B = []
    filelist_ml = []

    for y, m in sorted(months):
        month_dir = Path(mainpath) / f"{y}" / f"{m:02d}"

        if month_dir.exists():
            for h in hours:
                filelist_B.extend(month_dir.glob(f"B*{h}.nc"))
                filelist_ml.extend(month_dir.glob(f"ml*{h}.nc"))

    return sorted(filelist_B), sorted(filelist_ml)

def center_indices(filelist, year):
    idxs = []

    for i, f in enumerate(filelist):
        name = f.stem  # B20151101_00

        date_part, hour = name.split("_")
        date_part = date_part[1:]  # remove B or ml prefix

        file_year = int(date_part[:4])

        if file_year == year and hour == "00":
            idxs.append(i)

    return idxs
    
def load_data_timespan(filelist_B, filelist_ml, ml_vars, B_vars, idx, chunks_2d, chunks_3d, timespan = 96, timestep = 6):
    filerange = int(timespan/2*(24/timestep))
    files_B = filelist_B[idx - filerange:idx + filerange]
    files_ml = filelist_ml[idx - filerange:idx + filerange]

    ds_B = xr.open_mfdataset(files_B, combine='by_coords', chunks = chunks_2d)
    ds_B_sel    = ds_B[list(B_vars)]
    ds_ml = xr.open_mfdataset(files_ml, combine='by_coords', chunks = chunks_3d)
    ds_ml_sel    = ds_ml[list(ml_vars)]

    ds_all = xr.merge([ds_B_sel, ds_ml_sel])

    return ds_all


def create_era5_filename(data_path, day, hour=16):
    day = pd.Timestamp(day)

    month_dir = Path(data_path) / f"{day.year}" / f"{day.month:02d}"
    hour_str = f"{hour:02d}"
    date_str = day.strftime("%Y%m%d")

    ml_files = sorted(month_dir.glob(f"ml*{hour_str}.nc"))
    B_files = sorted(month_dir.glob(f"B*{hour_str}.nc"))

    ml_files = [f for f in ml_files if date_str in f.name]
    B_files = [f for f in B_files if date_str in f.name]

    if len(ml_files) != 1 or len(B_files) != 1:
        return None, None

    return ml_files[0], B_files[0]