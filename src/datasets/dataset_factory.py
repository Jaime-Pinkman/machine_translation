from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse

from .base_dataset import BaseDataset
from .dense_dataset import DenseDataset
from .sparse_dataset import SparseDataset


class BaseDatasetFactory(ABC):
    def __init__(
        self,
        features: np.ndarray | sparse._dok.dok_matrix,
        targets: np.ndarray | sparse._dok.dok_matrix,
    ):
        self.features = features
        self.targets = targets

    @abstractmethod
    def create_dataset(self) -> BaseDataset:
        pass


class SparseDatasetFactory(BaseDatasetFactory):
    def create_dataset(self) -> SparseDataset:
        return SparseDataset(
            features=self.features,
            targets=self.targets,
        )


class DenseDatasetFactory(BaseDatasetFactory):
    def create_dataset(self) -> DenseDataset:
        return DenseDataset(
            features=self.features,
            targets=self.targets,
        )
