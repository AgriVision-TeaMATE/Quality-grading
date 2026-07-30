"""
Dataset and DataLoader setup for the CNN pipeline (Approach B).
"""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TeaParticleDataset(Dataset):
    """Loads particle images + labels from a manifest dataframe filtered to one split."""

    def __init__(self, df, label_encoder, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.label_encoder.transform([row["label"]])[0]
        return image, label


def build_transforms(cfg_dl, split):
    size = tuple(cfg_dl["image_size"])
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if split == "train":
        aug = cfg_dl["augmentation"]
        t = [transforms.Resize(size)]
        if aug.get("horizontal_flip"):
            t.append(transforms.RandomHorizontalFlip())
        if aug.get("vertical_flip"):
            t.append(transforms.RandomVerticalFlip())
        if aug.get("rotation_degrees"):
            t.append(transforms.RandomRotation(aug["rotation_degrees"]))
        if aug.get("color_jitter"):
            cj = aug["color_jitter"]
            t.append(transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj))
        t += [transforms.ToTensor(), transforms.Normalize(imagenet_mean, imagenet_std)]
    else:
        t = [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    return transforms.Compose(t)


def build_dataloaders(manifest_path, cfg_dl, label_encoder):
    df = pd.read_csv(manifest_path)

    loaders = {}
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        transform = build_transforms(cfg_dl, split)
        dataset = TeaParticleDataset(split_df, label_encoder, transform)
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg_dl["batch_size"],
            shuffle=(split == "train"),
            num_workers=0,  # set >0 if running on a machine with multiple CPU cores to spare
            pin_memory=True,
        )
    return loaders
