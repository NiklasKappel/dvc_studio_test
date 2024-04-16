# pyright: reportIncompatibleMethodOverride=false

import lightning as L
import torch
from dvclive.lightning import DVCLiveLogger
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.rand(1000, 3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MyDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=7):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        del stage  # unused
        self.train_dataset = MyDataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class MyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch):
        loss = self(batch).mean()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def main():
    model = MyModel()
    data_module = MyDataModule()
    logger = DVCLiveLogger()
    trainer = L.Trainer(
        enable_checkpointing=False,
        log_every_n_steps=1,
        logger=logger,
        max_epochs=1000,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
