import xarray as xr
import numpy as np

class Stats():

    def __init__(self, ds, num_time_steps=3):
        concat_ds = []
        for i in range(num_time_steps):
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


    def get_mean(self, var):
        return self.ds[var].mean(dim='time')

    def get_std(self, var):
        return self.ds[var].std(dim='time')

    def get_min(self, var):
        return self.ds[var].min(dim='time')

    def get_max(self, var):
        return self.ds[var].max(dim='time')

    def get_percentile(self, var, percentile):
        return self.ds[var].quantile(percentile, dim='time')
    
    def get_stats(self, var):
        return {
            'mean': self.get_mean(var),
            'std': self.get_std(var),
           'min': self.get_min(var),
            'max': self.get_max(var),
            'percentile_10': self.get_percentile(var, 0.1),
            'percentile_90': self.get_percentile(var, 0.9)
        }

    def get_mean_grouped_by_label(self, var):
        return self.ds['TMQ'].groupby(self.ds[self.labels]).mean()

    def get_corr(self, var1, var2, dim=None):
        if dim is not None:
            return xr.corr(self.ds[var1], self.ds[var2], dim=dim)
        return xr.corr(self.ds[var1], self.ds[var2])

    def get_corr_matrix(self):
        corr_matrix = np.zeros((len(self.data_vars), len(self.data_vars)))
        for i, var1 in enumerate(self.data_vars):
            for j, var2 in enumerate(self.data_vars):
                corr_matrix[i, j] = self.get_corr(var1, var2)
                corr_matrix[j, i] = corr_matrix[i, j]
        return corr_matrix

    def get_eigenvalues(self, corr_matrix):
        return np.linalg.eigvals(corr_matrix)

    def get_variance(self, var):
        return self.ds[var].var(dim='time')

    def get_num_missing_values(self, var):
        return self.ds[var].isnull().sum(dim='time')