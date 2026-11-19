#!/uni-mainz.de/homes/sikraus/miniconda3/envs/myenv/bin/python3

import xarray as xr
import numpy as np
import xrft
import dask.array as da
from scipy import special

### Constants

r_earth =       6371229.
rh2o =          461.5250
rd =            287.0597
g_earth =       9.80665
cp =            1004.7090
Omega =         7.29e-5


### Functions

def calc_vertical(ds, grid, cutoff = 39):

    hyam = grid.hyam.isel(nhym = slice(cutoff,None)).rename(nhym = 'level')
    hybm = grid.hybm.isel(nhym = slice(cutoff,None)).rename(nhym = 'level')
    hyai = grid.hyai.isel(nhyi = slice(cutoff,None)).rename(nhyi = 'interface')
    hybi = grid.hybi.isel(nhyi = slice(cutoff,None)).rename(nhyi = 'interface')

    sp = ds.sp
    sgp = ds.z

    t= ds.t
    q = ds.q

    pm = sp*hybm + hyam
    pi = sp*hybi + hyai

    t_moist = (t * (1 + ((rh2o / rd) - 1) * q)).rename(level = 'interface')

    pi_top = pi.isel(interface = slice (0,-1))
    pi_bottom = pi.isel(interface = slice (1,None))

    alpha = (1 - pi_top*np.log(pi_bottom/pi_top)/(pi_bottom - pi_top)).rename(interface = 'level')

    gph_temp = (rd * t_moist * np.log(pi_bottom/pi_top)).rename(interface = 'level')
    cumsum = gph_temp.isel(level=slice(None, None, -1)).cumsum(dim = 'level').isel(level=slice(None, None, -1))
    gph_i = cumsum.shift(level=-1, fill_value=0)+ sgp
    gph = (gph_i+ alpha*t_moist.rename(interface = 'level'))
    pt = t*((100000/pm)**(rd/cp))
    rho = pm/rd/t_moist.rename(interface = 'level')
    height = gph/g_earth

    ds['pm'] = pm
    ds['gph'] = gph
    ds['pt'] = pt
    ds['rho'] = rho
    ds['height'] = height

    return ds

def calc_lrt_id(dtdz, gamma_lrt = -0.002):
    dtdz_prev = np.concatenate([[-np.inf], dtdz[:-1]])
    crossings = np.where((dtdz > gamma_lrt) & (dtdz_prev < gamma_lrt))[0]

    for idx in crossings:
        if idx <= 30:
         continue
        
        window = dtdz[idx:idx+8]
        if len(window) < 8:
            continue

        averages = np.cumsum(window) / np.arange(1, 9)

        if np.all(averages >= gamma_lrt):
            return idx
    return None
        
def calc_dtdz(t, z):
    return np.gradient(t, z, axis=0)

def calc_lrt(ds, gamma_lrt = -0.002):
    z = ds.height.isel(level = slice(None, None, -1))
    t = ds.t.isel(level = slice(None, None, -1))
    pt = ds.pt.isel(level = slice(None, None, -1))

    dtdz = t.differentiate(coord = 'level')/z.differentiate(coord = 'level')

    lrt_id = xr.apply_ufunc(
    calc_lrt_id,
    dtdz,         
    input_core_dims=[['level']],
    output_core_dims=[[]],
    vectorize=True, 
    dask="parallelized", 
    output_dtypes=[dtdz.dtype],
    )

    lrt_id = lrt_id.compute().fillna(-1).astype(int)

    z_mid = (z.shift(level = -1) + z).isel(level = slice(0, -1)) / 2
    z_before = z_mid.isel(level=lrt_id-1)
    z_after = z_mid.isel(level=lrt_id)
    
    dtdz_before = dtdz.isel(level=lrt_id-1)
    dtdz_after = dtdz.isel(level=lrt_id)

    z_lrt = z_before + (gamma_lrt - dtdz_before) * (z_after - z_before) / (dtdz_after - dtdz_before)

    pt_lrt = xr.apply_ufunc(
    lambda pt_col, z_col, target: np.interp(target, z_col, pt_col, left=np.nan, right=np.nan),
    pt,z,z_lrt,
    input_core_dims=[["level"], ["level"], []],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[pt.dtype],
    )

    trop_rel_z = z - z_lrt

    dz = z.differentiate(coord='level')
    N2 = g_earth / ds.pt * ds.pt.differentiate(coord='level') / dz

    ds['z_lrt'] = z_lrt
    ds['pt_lrt'] = pt_lrt
    ds['trop_rel_z'] = trop_rel_z
    ds['N2'] = N2
    
    return ds

def calc_frequency_map_chunk(variable, trop_rel_z, direction, threshold):

    height_mask = ((trop_rel_z > 0) & (trop_rel_z < 2000))

    variable_masked = np.where(height_mask, variable, np.nan)

    if direction == 'higher':
        mask = (variable_masked > threshold)
    elif direction == 'lower':
        mask = (variable_masked < threshold)
    else:
        raise ValueError("direction must be 'higher' or 'lower'")

    mask = np.any(mask, axis = 0)

    return mask

def calc_frequency_map(ds, var, threshold, direction = 'higher'):

    lat_bin_size = len(ds.latitude)
    lon_bin_size = len(ds.longitude)

    result = xr.apply_ufunc(
        calc_frequency_map_chunk,
        ds[var],
        ds.trop_rel_z,
        direction,
        threshold,
        input_core_dims=[["level", "latitude", "longitude"], 
                     ["level", "latitude", "longitude"], 
                     [], []],
        output_core_dims=[["latitude","longitude"]],
        dask="parallelized",
        vectorize = True,
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"latitude": lat_bin_size, "longitude": lon_bin_size}, "allow_rechunk": True},
    )

    frequency = result.sum(dim="time")/len(ds.time)

    return frequency

def calculate_composite(daily_data, lon_slice, previous_box_skip = None):

    if previous_box_skip is not None:
        daily_data = daily_data.where(~previous_box_skip)

    Ri = daily_data.Ri.where((daily_data.longitude >= lon_slice.start) &  (daily_data.longitude <= lon_slice.stop))
    mask = (Ri >= 0) & (Ri <= 1) & (daily_data.height > daily_data.z_lrt)

    Ri = Ri.where(mask)

    event_mask = Ri.notnull().any(dim="level")

    has_event = event_mask.any(dim="longitude").compute()

    valid_days = has_event

    if has_event.any():
        z_lrt = daily_data.z_lrt.where(
            (daily_data.longitude >= lon_slice.start) &
            (daily_data.longitude <= lon_slice.stop)
        )

        z_lrt_jump = np.abs(z_lrt.shift(longitude=1) - z_lrt)
        z_lrt_jump = z_lrt_jump.where(has_event)
        max_z_lrt_jump = z_lrt_jump.max(dim="longitude", skipna=True).compute()
        jump_lon = z_lrt_jump.idxmax(dim="longitude").compute()
        use_jump = max_z_lrt_jump > 500

        Ri_min = Ri.min(dim="level",skipna=True)
        Ri_lon = Ri_min.idxmin(dim="longitude").compute()
        lon_center = xr.where(use_jump,jump_lon,Ri_lon)

        center_method = xr.where(use_jump,"tropopause_jump","Ri")

        processed_data = daily_data.where(has_event)


    else:
        empty_composite = daily_data.isel(longitude=slice(0, 481)) * np.nan
        empty_composite = empty_composite.assign_coords(
            longitude=np.arange(-60, 60.25, 0.25)
        )
        #empty_composite = empty_composite.assign(height=monthly_data.height)
        lon_center = xr.full_like(valid_days, np.nan, dtype=float)
        center_method = xr.full_like(valid_days, "", dtype="<U20")
        
        return empty_composite, valid_days, lon_center, None, center_method
    
    shift = xr.where(has_event, np.rint(lon_center / 0.25), 0).astype(int)

    rolled = xr.concat(
        [
            processed_data.isel(time=i).roll(
                longitude=-int(shift[i]),
                roll_coords=False
            )
            for i in range(processed_data.sizes["time"])
        ],
        dim="time",
    )

    rolled = rolled.assign_coords(
        longitude=np.arange(-180, 180, 0.25)
    )

    # Select exactly 241 points centered on zero
    center_idx = len(rolled.longitude) // 2
    rolled = rolled.isel(
        longitude=slice(center_idx - 240, center_idx + 241)
    )

    rolled = rolled.assign_coords(
        longitude=np.arange(-60, 60.25, 0.25)
    )

    rolled_Ri = rolled.Ri.where(
    (rolled.Ri <= 1) &
    (rolled.height > rolled.z_lrt)
    )
    event_mask_rolled = rolled_Ri.notnull().any(dim="level")

    # Does the event extend more than 15° east of its center?
    box_skip = event_mask_rolled.sel(
        longitude=slice(10.25, 20)
    ).any(dim="longitude").compute()

    return rolled, valid_days, lon_center, box_skip, center_method

def calc_perturbations_composites(ds, monthly_data, var):
    x_perturbation = ds[var] - monthly_data[var]
    return ds.assign(**{f"{var}_perturbation": x_perturbation})


def calc_Ri(ds):
    ### Calculates Shear, N2 and Richardson number

    u = ds.u
    v = ds.v
    z = ds.height
    pt = ds.pt

    dz = z.differentiate(coord='level')
    Si2u = (u.differentiate(coord = 'level')/dz)**2
    Si2v = (v.differentiate(coord = 'level')/dz)**2
    Si2 = Si2u + Si2v

    N2 = g_earth/pt*pt.differentiate(coord = 'level')/dz

    Ri = N2/Si2

    ds['N2'] = N2
    ds['Si2'] = Si2
    ds['Ri'] = Ri

    return ds

def uz_to_qr(u,z, y0):

    beta=2.3e-11
    
    y0real= 2*np.pi*r_earth*y0/360.0
    ce=2*y0real**2*beta

    q=z/ce+u
    r=z/ce-u
    c_on_g = ce/g_earth

    return q,r, c_on_g

def filt_project_xarray(
    qf, rf, vf, lats, y0, waves,
    fmin, fmax, kmin, kmax, c_on_g
    ):

    print("projecting PCFs")

    f = qf.freq_time

    positive_time = (f >= fmin) & (f <= fmax)
    negative_time = (f <= -fmin) & (f >= -fmax)

    k = qf.freq_longitude

    positive_lon = (k >= kmin) & (k <= kmax)
    negative_lon = (k <= -kmin) & (k >= -kmax)

    eastward_mask = (
        positive_time & negative_lon
    ) | (
        negative_time & positive_lon
    )

    westward_mask = (
        positive_time & positive_lon
    ) | (
        negative_time & negative_lon
    )

    all_wave_mask = eastward_mask | westward_mask

    spi2 = np.sqrt(2 * np.pi)
    dsq = xr.DataArray(
        [spi2, spi2, 2*spi2, 6*spi2],
        dims="mode",
        coords={"mode": np.arange(4)}
    )

    y = lats / y0
    ysq = y**2

    d = np.array([
        np.exp(-ysq / 4.0),
        y * np.exp(-ysq / 4.0),
        (ysq - 1.0) * np.exp(-ysq / 4.0),
        y * (ysq - 3.0) * np.exp(-ysq / 4.0)
    ])

    dlat = abs(lats[0] - lats[1])

    basis = xr.DataArray(
        d,
        dims=["mode", "latitude"],
        coords={
            "mode": np.arange(4),
            "latitude": lats
        }
    )

    qf_all = qf.where(all_wave_mask, 0)
    rf_all = rf.where(all_wave_mask, 0)
    vf_all = vf.where(all_wave_mask, 0)

    qf_east = qf.where(eastward_mask, 0)

    qf_mode = (
        qf_all * basis
    ).sum("latitude") * dlat / (dsq * y0)

    rf_mode = (
        rf_all * basis
    ).sum("latitude") * dlat / (dsq * y0)

    vf_mode = (
        vf_all * basis
    ).sum("latitude") * dlat / (dsq * y0)

    qf_kel = (
        qf_east * basis.sel(mode=0)
    ).sum("latitude") * dlat / (dsq[0] * y0)

    uf_wave = []
    zf_wave = []
    vf_wave = []

    for wave in waves:

        if wave == "Kelvin":

            u_wave = 0.5 * qf_kel * basis.sel(mode=0, drop=True)
            z_wave = 0.5 * qf_kel * basis.sel(mode=0, drop=True) * c_on_g
            v_wave = xr.zeros_like(u_wave)

        elif wave == "WMRG":

            u_wave = 0.5 * qf_mode.sel(mode=1, drop=True) * basis.sel(mode=1, drop=True)
            z_wave = (
                0.5
                * qf_mode.sel(mode=1, drop=True)
                * basis.sel(mode=1, drop=True)
                * c_on_g
            )
            v_wave = vf_mode.sel(mode=0, drop=True) * basis.sel(mode=0, drop=True)

        elif wave == "R1":

            u_wave = (
                0.5
                * (
                    qf_mode.sel(mode=2, drop=True) * basis.sel(mode=2, drop=True)
                    - rf_mode.sel(mode=0, drop=True) * basis.sel(mode=0, drop=True)
                )
            )

            z_wave = (
                0.5
                * (
                    qf_mode.sel(mode=2, drop=True) * basis.sel(mode=2, drop=True)
                    + rf_mode.sel(mode=0, drop=True) * basis.sel(mode=0, drop=True)
                )
                * c_on_g
            )

            v_wave = vf_mode.sel(mode=1, drop=True) * basis.sel(mode=1, drop=True)

        elif wave == "R2":

            u_wave = (
                0.5
                * (
                    qf_mode.sel(mode=3, drop=True) * basis.sel(mode=3, drop=True)
                    - rf_mode.sel(mode=1, drop=True) * basis.sel(mode=1, drop=True)
                )
            )

            z_wave = (
                0.5
                * (
                    qf_mode.sel(mode=3, drop=True) * basis.sel(mode=3, drop=True)
                    + rf_mode.sel(mode=1, drop=True) * basis.sel(mode=1, drop=True)
                )
                * c_on_g
            )

            v_wave = vf_mode.sel(mode=2, drop=True) * basis.sel(mode=2, drop=True)

        else:
            raise ValueError(f"Unknown wave type: {wave}")

        uf_wave.append(
            xrft.ifft(
                u_wave,
                dim=["freq_time", "freq_longitude"]
            ).real
        )

        zf_wave.append(
            xrft.ifft(
                z_wave,
                dim=["freq_time", "freq_longitude"]
            ).real
        )

        vf_wave.append(
            xrft.ifft(
                v_wave,
                dim=["freq_time", "freq_longitude"]
            ).real
        )

    uf_wave = xr.concat(uf_wave, dim="wave")
    zf_wave = xr.concat(zf_wave, dim="wave")
    vf_wave = xr.concat(vf_wave, dim="wave")

    uf_wave = uf_wave.assign_coords(wave=waves)
    zf_wave = zf_wave.assign_coords(wave=waves)
    vf_wave = vf_wave.assign_coords(wave=waves)

    print("PCF projection complete")

    return uf_wave, zf_wave, vf_wave


