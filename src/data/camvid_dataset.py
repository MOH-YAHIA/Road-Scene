from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CamVidDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        image_size,
        ignore_index,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.ignore_index = ignore_index

        # Training: Standardization + Augmentation
        self.train_transform_image = transforms.Compose(
            [
                transforms.RandomCrop(
                    image_size
                ),  # Use RandomCrop for Train to add variety
                # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                #  transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Validation: Standardization ONLY
        self.val_transform_image = transforms.Compose(
            [
                transforms.CenterCrop(
                    image_size
                ),  # Use CenterCrop for Val to be consistent
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.train_transform_mask = transforms.RandomCrop(image_size)
        self.val_transform_mask = transforms.CenterCrop(image_size)

        if split not in {"train", "val", "test"}:
            raise ValueError("Invalid split: %s. Use train, val, or test.", split)

        self.image_dir = self.root_dir / split
        self.mask_dir = self.root_dir / f"{split}_labels"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(self.image_dir)

        if not self.mask_dir.is_dir():
            raise FileNotFoundError(self.mask_dir)

        # Read class names and their RGB colors.
        class_df = pd.read_csv(self.root_dir / "class_dict.csv")

        self.classes = class_df["name"].astype(str).tolist()
        self.num_classes = len(self.classes)

        # RGB color -> integer class ID.
        self.rgb_to_id = {}

        for class_id, row in class_df.iterrows():
            rgb = (row["r"], row["g"], row["b"])
            self.rgb_to_id[rgb] = class_id

        # Match each image with its mask by filename.
        image_extensions = {".png"}

        image_files = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        )

        mask_files = sorted(
            path
            for path in self.mask_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        )

        masks_by_stem = {path.stem.removesuffix("_L"): path for path in mask_files}

        self.samples = []

        for image_path in image_files:
            mask_path = masks_by_stem.get(image_path.stem)

            if mask_path is None:
                error = f"No matching mask for image: {image_path.name}"
                raise FileNotFoundError(error)

            self.samples.append((image_path, mask_path))

        if not self.samples:
            error = f"No image-mask pairs found for split '{split}'."
            raise RuntimeError(error)

    def __len__(self):
        return len(self.samples)

    def _encode_mask(self, mask):
        """Convert an RGB mask to a 2D array of class IDs."""
        mask = np.asarray(mask)
        height, width, _ = mask.shape

        # Unknown RGB colors receive ignore_index.
        encoded = np.full(
            (height, width),
            fill_value=self.ignore_index,
            dtype=np.int64,
        )

        for rgb, class_id in self.rgb_to_id.items():
            matches = np.all(mask == rgb, axis=-1)
            encoded[matches] = class_id

        return encoded

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if image.size != mask.size:
            error = f"Image-mask size mismatch: {image_path.name}: {image.size} vs {mask.size}"
            raise ValueError(error)

        seed = np.random.randint(0, 2**16)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        image = (
            self.train_transform_image(image)
            if self.split == "train"
            else self.val_transform_image(image)
        )

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        mask = (
            self.train_transform_mask(mask)
            if self.split == "train"
            else self.val_transform_mask(mask)
        )

        # Image: (C, H, W), float32.
        # Mask: (H, W), int64 class IDs.
        mask = self._encode_mask(mask)
        mask = torch.from_numpy(mask)

        return image, mask


if __name__ == "__main__":
    dataset = CamVidDataset(
        root_dir="dataset/CamVid", split="train", image_size=(224, 224)
    )
    print(len(dataset.samples))
    image, mask = dataset.__getitem__(0)
    # display image and mask
    import matplotlib.pyplot as plt

    plt.imshow(image.permute(1, 2, 0))
    plt.show()
    plt.imshow(mask)
    plt.show()
