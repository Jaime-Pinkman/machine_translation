import torch
import numpy as np

from .base_dataset import BaseDataset


class DenseDataset(BaseDataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(self.features[idx]).float()
        label = torch.from_numpy(np.asarray(self.targets[idx])).long()
        return features, label
