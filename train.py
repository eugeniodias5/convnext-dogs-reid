from csv import writer

import torch

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from YtDataModule import YtDataModule
from ConvNext import ConvNext


def write_csv_row(file, row=[]):
    with open(file, "a", newline="") as csv_obj:
        row_writer = writer(csv_obj)
        row_writer.writerow(row)

        csv_obj.close()


def train(
    batch_size,
    dataset_path="yt_db",
    train_split=0.7,
    epochs=10,
    lr=1e-5,
    device="cpu",
    devices=[1],
    max_imgs_per_class=10,
    num_workers=1,
):
    ytDataModule = YtDataModule(
        data_dir=dataset_path,
        train_split=train_split,
        batch_size=batch_size,
        max_imgs_per_class=max_imgs_per_class,
        num_workers=num_workers,
    )
    convNext = ConvNext(lr=lr)

    logger = CSVLogger("logs", name="csv_logs")

    callbacks = [EarlyStopping(monitor="loss")]

    if torch.cuda.device_count() == 0:
        device = "cpu"
        devices = None

    trainer = Trainer(
        logger=logger,
        max_epochs=epochs,
        accelerator=device,
        devices=devices,
        callbacks=callbacks,
    )

    trainer.fit(convNext, ytDataModule)
    trainer.test(convNext, ytDataModule)


if __name__ == "__main__":
    train(
        batch_size=4,
        dataset_path="../../yt_db",
        train_split=0.7,
        epochs=5,
        lr=1e-4,
        device="gpu",
        devices=[1],
        max_imgs_per_class=10,
        num_workers=12,
    )
