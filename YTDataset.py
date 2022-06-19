from operator import ne
import os, random

from PIL import Image
from torch.utils.data import Dataset


class YtDataset(Dataset):
    def __init__(self, path, transform=None, max_imgs_per_class=10, min_imgs_per_class=5):
        self.path = path
        self.transform = transform
        self.max_imgs_per_class = max_imgs_per_class
        self.dogs = []

        # Going through the database
        for race in os.listdir(path):
            print(f"Composing race {race}")
            race_path = os.path.join(path, race)
            num_race_dogs = len(os.listdir(race_path))
            # If there is only one individual from a race, it's not possible to
            # do triplet loss with another individual
            if num_race_dogs <= 1:
                print(f"Jumping race {race}")
                continue

            for label in os.listdir(race_path):
                label_path = os.path.join(race_path, label)
                
                dogs = os.listdir(label_path)
                if len(dogs) < min_imgs_per_class:
                    continue

                if len(dogs) > max_imgs_per_class:
                    # Choose a number of max_imgs_per_class
                    dogs = random.choices(dogs, k=max_imgs_per_class)

                for dog in dogs:
                    anchor_path = os.path.join(label_path, dog)

                    # Choose an image that is not the same as the anchor
                    dog = random.choice([dog_el for dog_el in dogs if dog_el != dog])
                    pos_path = os.path.join(label_path, dog)

                    # Choose negative dog from the same race
                    while 1:
                        neg_label = random.choice(os.listdir(race_path))

                        if neg_label != label and len(os.listdir(os.path.join(race_path, neg_label))) > 1:
                            break 

                    neg_path = os.path.join(race_path, neg_label)
                    neg_path = os.path.join(
                        neg_path, random.choice(os.listdir(neg_path))
                    )

                    self.dogs.append(
                        {
                            "anchor": anchor_path,
                            "label": label,
                            "positive": pos_path,
                            "negative": neg_path,
                        }
                    )



    def __len__(self):
        return len(self.dogs)

    def __getitem__(self, idx):
        dog = self.dogs[idx]
        anchor = Image.open(dog["anchor"]).convert("RGB")
        positive = Image.open(dog["positive"]).convert("RGB")
        negative = Image.open(dog["negative"]).convert("RGB")

        if self.transform:
            return (
                dog["label"],
                self.transform(anchor),
                self.transform(positive),
                self.transform(negative),
            )

        return anchor, positive, negative
