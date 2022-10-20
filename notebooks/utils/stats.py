import xarray as xr
import numpy as np
import xrft

class Stats():

    def __init__(self, ds, num_samples=3):
        concat_ds = []
        for i in range(num_samples):
            concat_ds.append(ds[i])
        self.ds = xr.concat(concat_ds, dim='time')
        data_vars = []
        for var in ds[0].data_vars:
            print(var)
            if (var == 'LABELS'):
                self.labels = var
            else:
                data_vars.append(var)
        self.data_vars = data_vars

    def get_label_distribution(self):
        bg = self.ds[self.labels].where(self.ds[self.labels] == 0).count(dim='time')
        tc = self.ds[self.labels].where(self.ds[self.labels] == 1).count(dim='time')
        ar = self.ds[self.labels].where(self.ds[self.labels] == 2).count(dim='time')
        return bg, tc, ar

    def get_mean(self, var):
        return self.ds[var].mean(dim='time')

    def get_std(self, var):
        return self.ds[var].std(dim='time')

    def get_min(self, var):
        return self.ds[var].min(dim='time')

    def get_max(self, var):
        return self.ds[var].max(dim='time')

    def get_var(self, var):
        return self.ds[var].var(dim='time')

    def get_mean_grouped_by_label(self, var):
        return self.ds[var].groupby(self.ds[self.labels]).mean()
    
    def get_stats(self, var):
        return {
            'mean': self.get_mean(var).mean().values,
            'std': self.get_std(var).mean().values,
            'min': self.get_min(var).min().values,
            'max': self.get_max(var).max().values,
            'var': self.get_var(var).mean().values,
            'mean_grouped_by_label': self.get_mean_grouped_by_label(var).values,
        }

    def get_corr(self, var1, var2, dim=None):
        if dim is not None:
            return xr.corr(self.ds[var1], self.ds[var2], dim=dim)
        return xr.corr(self.ds[var1], self.ds[var2], dim='time').mean()

    def get_xcorr(self, var1, var2):
        fft1 = xrft.fft(self.ds[var1], dim=['lat', 'lon'], shift=False, detrend='linear')
        fft2 = xrft.fft(self.ds[var2], dim=['lat', 'lon'], shift=False, detrend='linear')
        xcorr = xrft.ifft(fft1 * fft2.conj(), dim=['freq_lat', 'freq_lon'], shift=False, detrend='linear')
        return xcorr

    def get_corr_matrix(self):
        return self.get_corr_matrix_vars(self.data_vars)

    def get_corr_matrix_vars(self, vars):
        corr_matrix = np.zeros((len(vars), len(vars)))
        for i, var1 in enumerate(vars):
            for j, var2 in enumerate(vars):
                corr_matrix[i, j] = self.get_corr(var1, var2)
                corr_matrix[j, i] = corr_matrix[i, j]
        return corr_matrix