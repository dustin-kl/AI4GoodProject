from torch.utils.data import Dataset, DataLoader
import xarray as xr

class ClimateNetDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = list(data_dir.glob('*.nc'))
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return xr.open_dataset(self.files[idx])

class ClimateNetDataLoader(DataLoader): # before this can be used xarrays need to be transformed into tensors or np
    def __init__(self, data_dir, batch_size=1, shuffle=False, num_workers=0):
        dataset = ClimateNetDataset(data_dir)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)