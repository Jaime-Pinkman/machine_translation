from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse
import torch


class BaseDataset(ABC):
    def __init__(
        self,
        features: np.ndarray | sparse._dok.dok_matrix,
        targets: np.ndarray | sparse._dok.dok_matrix,
    ):
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return self.features.shape[0]

    @abstractmethod
    def __getitem__(self, idx: int) -> torch.Tensor:
        pass
