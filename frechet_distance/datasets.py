"""Image datasets and dataloader utilities for Frechet distance evaluation.

Consolidates dataset classes previously scattered across eval.py, eval_all_fds.py,
and compute_repr_stats.py.
"""

import os

import torchvision.datasets as tv_datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DATASET_CHOICES = ("imagenet", "cifar10")
CIFAR10_NUM_CLASSES = 10
IMAGENET_NUM_CLASSES = 1000

_DATASET_DEFAULTS = {
    "imagenet": {
        "data_path": "data/imagenet",
        "img_size": 256,
        "num_classes": IMAGENET_NUM_CLASSES,
        "fid_stats_path": "data/fid_stats/guided_diffusion_stats.npz",
        "stats_dir": "data/fid_stats",
    },
    "cifar10": {
        "data_path": "data/cifar10",
        "img_size": 32,
        "num_classes": CIFAR10_NUM_CLASSES,
        "fid_stats_path": "data/fid_stats/cifar10/inception_in32_t299_stats.npz",
        "stats_dir": "data/fid_stats/cifar10",
    },
}


def normalize_dataset_name(dataset: str | None) -> str:
    dataset = (dataset or "imagenet").lower()
    if dataset not in DATASET_CHOICES:
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of {DATASET_CHOICES}")
    return dataset


def dataset_defaults(dataset: str | None) -> dict:
    return dict(_DATASET_DEFAULTS[normalize_dataset_name(dataset)])


def sanitize_model_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def stats_path_for_model(
    name: str,
    img_size: int,
    target_size: int,
    dataset: str | None = "imagenet",
    default_inception_path: str | None = None,
) -> str:
    """Return the expected reference-statistics path for a dataset/repr model."""
    dataset = normalize_dataset_name(dataset)
    if name == "inception" and default_inception_path is not None:
        return default_inception_path

    stats_img_size = 256 if dataset == "imagenet" and img_size == 512 else img_size
    stats_dir = _DATASET_DEFAULTS[dataset]["stats_dir"]
    safe_name = sanitize_model_name(name)
    return os.path.join(stats_dir, f"{safe_name}_in{stats_img_size}_t{target_size}_stats.npz")


def apply_dataset_defaults(args, *, validate: bool = True):
    """Apply dataset-specific defaults to an argparse namespace in-place."""
    dataset = normalize_dataset_name(getattr(args, "dataset", "imagenet"))
    defaults = dataset_defaults(dataset)
    args.dataset = dataset

    for key in ("data_path", "img_size", "num_classes", "fid_stats_path", "output_dir"):
        if not hasattr(args, key):
            continue
        if getattr(args, key) is None:
            if key == "output_dir":
                setattr(args, key, defaults["stats_dir"])
            else:
                setattr(args, key, defaults[key])

    if dataset == "cifar10":
        if validate and hasattr(args, "img_size") and args.img_size != 32:
            raise ValueError(f"CIFAR-10 generation uses img_size=32, got {args.img_size}")
        if validate and hasattr(args, "num_classes") and args.num_classes != CIFAR10_NUM_CLASSES:
            raise ValueError(f"CIFAR-10 uses num_classes=10, got {args.num_classes}")
        if hasattr(args, "class_of_interest"):
            args.class_of_interest = None
        if hasattr(args, "force_class_of_interest"):
            args.force_class_of_interest = False
    return args


def _default_transform(img_size: int):
    """Center-crop to img_size and convert to [0, 1] tensor."""
    from utils.data_util import center_crop_arr

    return transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
        transforms.ToTensor(),
    ])


def build_real_image_dataset(
    dataset: str,
    data_path: str,
    split: str,
    img_size: int,
    download: bool = False,
) -> Dataset:
    """Build a real-image dataset returning ``[3, H, W]`` float tensors in ``[0, 1]``."""
    dataset = normalize_dataset_name(dataset)
    transform = _default_transform(img_size)

    if dataset == "imagenet":
        folder = data_path
        if os.path.basename(os.path.normpath(folder)) != split:
            folder = os.path.join(data_path, split)
        return tv_datasets.ImageFolder(folder, transform=transform)

    if split not in ("train", "test"):
        raise ValueError(f"CIFAR-10 split must be 'train' or 'test', got '{split}'")
    return tv_datasets.CIFAR10(
        root=data_path,
        train=(split == "train"),
        transform=transform,
        download=download,
    )


def _find_images(folder: str) -> list[str]:
    """Find all image files in a flat folder, sorted by name."""
    paths = []
    for filename in os.listdir(folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext in IMAGE_EXTS:
            paths.append(os.path.join(folder, filename))
    paths.sort()

    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")

    return paths


class ImageFolderDataset(Dataset):
    """Flat image folder with center-crop preprocessing to [0, 1].

    Finds all images in *folder* matching common extensions (.png, .jpg, .jpeg, .webp).
    """

    def __init__(self, folder: str, img_size: int = 256, transform=None):
        self.paths = _find_images(folder)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = _default_transform(img_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)


class ImageListDataset(Dataset):
    """Dataset from an explicit list of image paths, with center-crop to [0, 1]."""

    def __init__(self, paths: list[str], img_size: int = 256, transform=None):
        self.paths = paths

        if transform is not None:
            self.transform = transform
        else:
            self.transform = _default_transform(img_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 8,
    distributed: bool = False,
) -> DataLoader:
    """Build a DataLoader with optional DistributedSampler.

    Args:
        dataset: any map-style Dataset.
        batch_size: per-GPU batch size.
        num_workers: data loading workers.
        distributed: if True, wraps with DistributedSampler (shuffle=False).
    """
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
