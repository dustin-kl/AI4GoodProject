from netCDF4 import Dataset
import numpy as np

fn = './dataset/data-2013-09-27-01-1.nc'
# ds = netcdf.NetCDFFile(fn)
ds = Dataset(fn)

print(ds["data"]["T500"][:])
# print(ds['T500'].description)
# print(type(np.asarray(ds['T500'])))
# print(ds.variables)

# for dim in ds.dimensions.values():
#     print(dim)

# print(ds['T500'][0, :10, :10])


'''
dimensions: lat, lon, time
variables: different features we use (e.g. T500 --> temperature at 500 mbar pressure surface)

lat: latitude
lon: longitude
time

How to access data?      ds['T500'][0, :10, :10]            ds[<feature>][time, lat, lon]

Classes to predict:      0: Background, 1: Tropical Cyclone, 2: Athmospheric river


Features:
    total (vertically integrated) precipitable water
    zonal wind at 850 mbar pressure surface
    meridional wind at 850 mbar pressure surface
    lowest level zonal wind
    lowest model level meridional wind
    reference height humidity
    surface pressure
    sea level pressure
    temperature at 200 mbar pressure surface
    temperature at 500 mbar pressure surface
    total (convective and large-scale) precipitation rate (liq + ice)
    surface temperature (radiative)
    reference height temperature
    geopotential Z at 1000 mbar pressure surface
    geopotential Z at 200 mbar pressure surface
    lowest modal level height
'''