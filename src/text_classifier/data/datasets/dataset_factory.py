from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse

from .base_dataset import BaseDataset
from .dense_dataset import DenseDataset
from .sparse_dataset import SparseDataset


class BaseDatasetFactory(ABC):
    @abstractmethod
    def create_dataset(
        self, features: np.ndarray | sparse._dok.dok_matrix, targets: np.ndarray
    ) -> BaseDataset:
        pass


class DenseDatasetFactory(BaseDatasetFactory):
    def create_dataset(self, features: np.ndarray, targets: np.ndarray) -> DenseDataset:
        return DenseDataset(features=features, targets=targets)


class SparseDatasetFactory(BaseDatasetFactory):
    def create_dataset(
        self, features: sparse._dok.dok_matrix, targets: np.ndarray
    ) -> SparseDataset:
        return SparseDataset(features=features, targets=targets)
