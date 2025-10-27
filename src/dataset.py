import os, random
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

def build_transforms(img_size=224, use_imagenet_norm: bool = False):
    """Build training and evaluation torchvision transforms.

    Args:
        img_size: Final square image size (H=W) after resizing.
        use_imagenet_norm: If True, append ImageNet mean/std normalization.

    Returns:
        (tf_train, tf_eval): A tuple of composed transforms for training and evaluation.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Training transforms: simple geometric/color augments + tensor conversion
    train_tf_list = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
    ]
    if use_imagenet_norm:
        train_tf_list.append(transforms.Normalize(mean=imagenet_mean, std=imagenet_std))
    tf_train = transforms.Compose(train_tf_list)

    # Evaluation transforms: deterministic resize + tensor conversion
    eval_tf_list = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
    if use_imagenet_norm:
        eval_tf_list.append(transforms.Normalize(mean=imagenet_mean, std=imagenet_std))
    tf_eval = transforms.Compose(eval_tf_list)

    return tf_train, tf_eval

class ImageFolderPairs(Dataset):
    """
    A pair-generating dataset built on top of torchvision.datasets.ImageFolder.

    It yields tuples of (img1, img2, label), where:
      - label = 1 for a positive pair (same identity/class)
      - label = 0 for a negative pair (different identities/classes)

    Expected directory structure (ImageFolder-style):
        root/
          person_a/*.jpg
          person_b/*.jpg
          ...

    This dataset pre-generates a fixed list of index pairs per epoch according to
    the desired positive ratio, then samples images by file index when __getitem__ is called.
    """

    def __init__(self, root: str, transform=None, pairs_per_epoch: int = 50000, pos_ratio: float = 0.5):
        # Root folder containing class-subfolders (ImageFolder format)
        self.root = root
        self.transform = transform

        # Underlying ImageFolder to manage files/classes mapping
        self.ds = datasets.ImageFolder(root)

        # Build class -> [indices of samples] dictionary, keeping only classes
        # with at least 2 images (so we can form positive pairs)
        self.class_to_indices = self._build_class_index()
        self.classes = list(self.class_to_indices.keys())

        # Sampling configuration per epoch
        self.pairs_per_epoch = pairs_per_epoch
        self.pos_ratio = pos_ratio

        # Pre-generate a list of (i, j, label) index tuples for this epoch
        self.pairs = self._make_pairs()

    def _build_class_index(self):
        """Build a mapping: class_name -> list of sample indices.

        Classes with fewer than 2 images are dropped because positive pairs
        cannot be formed for them.
        """
        m = {}
        for idx, (path, label) in enumerate(self.ds.samples):
            cls = self.ds.classes[label]
            m.setdefault(cls, []).append(idx)
        # Drop classes with < 2 images (cannot form positive pairs)
        m = {k: v for k, v in m.items() if len(v) >= 2}
        return m

    def _sample_positive(self) -> Tuple[int, int, int]:
        """Sample a positive pair: two different images from the same class."""
        cls = random.choice(list(self.class_to_indices.keys()))
        idxs = random.sample(self.class_to_indices[cls], 2)
        return idxs[0], idxs[1], 1

    def _sample_negative(self) -> Tuple[int, int, int]:
        """Sample a negative pair: images from two different classes.

        If the dataset has fewer than 2 classes, fall back to two random images,
        which are likely negative pairs.
        """
        if len(self.classes) < 2:
            # Fallback: choose any two distinct images (likely negative)
            i, j = random.sample(range(len(self.ds.samples)), 2)
            return i, j, 0
        c1, c2 = random.sample(self.classes, 2)
        i = random.choice(self.class_to_indices[c1])
        j = random.choice(self.class_to_indices[c2])
        return i, j, 0

    def _make_pairs(self) -> List[Tuple[int, int, int]]:
        """Pre-generate the list of (i, j, label) pairs for one epoch.

        The number of positives is determined by pos_ratio, and the rest are negatives.
        Pairs are then shuffled to mix positive and negative samples.
        """
        pairs = []
        n_pos = int(self.pairs_per_epoch * self.pos_ratio)
        n_neg = self.pairs_per_epoch - n_pos
        for _ in range(n_pos):
            pairs.append(self._sample_positive())
        for _ in range(n_neg):
            pairs.append(self._sample_negative())
        random.shuffle(pairs)
        return pairs

    def __len__(self):
        """Number of pre-generated pairs for this epoch."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Load and return a pair of images along with the binary label.

        Returns:
            (img1, img2, label):
              - img1, img2: tensors after optional transforms
              - label: 1 if positive pair (same class), else 0
        """
        i, j, label = self.pairs[idx]
        path_i, _ = self.ds.samples[i]
        path_j, _ = self.ds.samples[j]

        # Load images with PIL and convert to RGB to ensure 3 channels
        img1 = Image.open(path_i).convert("RGB")
        img2 = Image.open(path_j).convert("RGB")

        # Apply transforms if provided
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label

    def _build_class_index(self):
        """Build a mapping: class_name -> list of sample indices.

        Classes with fewer than 2 images are dropped because positive pairs
        cannot be formed for them.
        """
        m = {}
        for idx, (path, label) in enumerate(self.ds.samples):
            cls = self.ds.classes[label]
            m.setdefault(cls, []).append(idx)
        # Drop classes with < 2 images (cannot form positive pairs)
        m = {k: v for k, v in m.items() if len(v) >= 2}
        return m

    def _sample_positive(self) -> Tuple[int, int, int]:
        """Sample a positive pair: two different images from the same class."""
        cls = random.choice(list(self.class_to_indices.keys()))
        idxs = random.sample(self.class_to_indices[cls], 2)
        return idxs[0], idxs[1], 1

    def _sample_negative(self) -> Tuple[int, int, int]:
        """Sample a negative pair: images from two different classes.

        If the dataset has fewer than 2 classes, fall back to two random images,
        which are likely negative pairs.
        """
        if len(self.classes) < 2:
            # Fallback: choose any two distinct images (likely negative)
            i, j = random.sample(range(len(self.ds.samples)), 2)
            return i, j, 0
        c1, c2 = random.sample(self.classes, 2)
        i = random.choice(self.class_to_indices[c1])
        j = random.choice(self.class_to_indices[c2])
        return i, j, 0

    def _make_pairs(self) -> List[Tuple[int, int, int]]:
        """Pre-generate the list of (i, j, label) pairs for one epoch.

        The number of positives is determined by pos_ratio, and the rest are negatives.
        Pairs are then shuffled to mix positive and negative samples.
        """
        pairs = []
        n_pos = int(self.pairs_per_epoch * self.pos_ratio)
        n_neg = self.pairs_per_epoch - n_pos
        for _ in range(n_pos):
            pairs.append(self._sample_positive())
        for _ in range(n_neg):
            pairs.append(self._sample_negative())
        random.shuffle(pairs)
        return pairs

    def __len__(self):
        """Number of pre-generated pairs for this epoch."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Load and return a pair of images along with the binary label.

        Returns:
            (img1, img2, label):
              - img1, img2: tensors after optional transforms
              - label: 1 if positive pair (same class), else 0
        """
        i, j, label = self.pairs[idx]
        path_i, _ = self.ds.samples[i]
        path_j, _ = self.ds.samples[j]

        # Load images with PIL and convert to RGB to ensure 3 channels
        img1 = Image.open(path_i).convert("RGB")
        img2 = Image.open(path_j).convert("RGB")

        # Apply transforms if provided
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label
