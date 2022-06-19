import torchvision
import pytorch_lightning as pl

from torch.utils.data import random_split, DataLoader

from YtDataset import YtDataset

class YtDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers=1, train_split=0.7, max_imgs_per_class=10, min_imgs_per_class=5):
        super().__init__()

        self.path = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_imgs_per_class = max_imgs_per_class
        self.min_imgs_per_class = min_imgs_per_class
        self.train_split = train_split

        # Create transformer
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((230, 230)),
                torchvision.transforms.CenterCrop((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup(self, stage=None):
        # Create a dataset
        dataset = YtDataset(self.path, transform=self.transform, max_imgs_per_class=self.max_imgs_per_class, min_imgs_per_class=self.min_imgs_per_class)

        train_size = int(self.train_split * len(dataset))
        val_size = len(dataset) - train_size

        # Split into train and val
        self.train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        test_size = int(val_size/2)
        val_size -= test_size

        # Split into val and test
        self.val_dataset, self.test_dataset = random_split(val_dataset, [val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
