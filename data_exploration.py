import netCDF4 as nc
import numpy as np

fn = '/home/dustin/Documents/Study/3. Master/AI4Good/data/data-1996-06-09-01-1_0.nc'
ds = nc.Dataset(fn)

# print(ds)
print(ds['T500'])
# print(type(np.asarray(ds['T500'])))
# print(ds.variables)

# for dim in ds.dimensions.values():
#     print(dim)

# for var in ds.variables.values():
#     print(var)

# print(ds['T500'][0, :10, :10])


'''
dimensions: lat, lon, time
variables: different features we use (e.g. T500 --> temperature at 500 mbar pressure surface)

lat: latitude
lon: longitude
time

How to access data?      ds['T500'][0, :10, :10]

'''