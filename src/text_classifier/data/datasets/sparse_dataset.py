import torch
import numpy as np

from .base_dataset import BaseDataset


class SparseDataset(BaseDataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(self.features[idx].toarray()[0]).float()
        label = torch.from_numpy(np.asarray(self.targets[idx])).long()
        return features, label
