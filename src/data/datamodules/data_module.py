from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.data.datasets import BaseDataset


class TrainValDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: BaseDataset,
        val_dataset: BaseDataset,
        batch_size: int = 32,
        num_workers: int = 128,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
