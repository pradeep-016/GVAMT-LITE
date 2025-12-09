"""
dataset.py

Dataset + collation utilities for GVAMT-Lite.

This module provides:
- MNISTWithText: a dataset that pairs MNIST images with synthetic text captions.
- collate_fn: a batching function that pads caption sequences.

Each MNIST digit (0–9) is mapped to a text word ("zero", "one", ...),
wrapped with <sos> and <eos> tokens, forming a minimal multimodal dataset.

This dataset is intentionally lightweight and ideal for multimodal experiments
without requiring external caption corpora.
"""

from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Vocabulary setup
# ---------------------------------------------------------------------------

digit_words = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}

special_tokens = ["<pad>", "<sos>", "<eos>"]
word_list = special_tokens + sorted(set(digit_words.values()))

# Token → index mapping
word2idx = {w: i for i, w in enumerate(word_list)}
idx2word = {i: w for w, i in word2idx.items()}

PAD_IDX = word2idx["<pad>"]
SOS_IDX = word2idx["<sos>"]
EOS_IDX = word2idx["<eos>"]


# ---------------------------------------------------------------------------
# Dataset Definition
# ---------------------------------------------------------------------------

class MNISTWithText(Dataset):
    """
    MNIST digits paired with synthetic textual captions.

    Each sample consists of:
      * A grayscale MNIST image x ∈ [0,1]^{1×28×28}
      * A caption like "<sos> seven <eos>"
      * A numeric label (0–9)

    This dataset enables multimodal learning (image + text)
    without downloading external caption datasets.
    """

    def __init__(self, train: bool = True) -> None:
        """
        Parameters
        ----------
        train : bool
            Whether to load the MNIST training split.
        """
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.mnist = datasets.MNIST(
            root="./data",
            train=train,
            transform=transform,
            download=True,
        )

    def __len__(self) -> int:
        """Returns number of samples in the dataset."""
        return len(self.mnist)

    @staticmethod
    def encode_caption(label: int) -> torch.Tensor:
        """
        Converts a digit label into a caption.

        Example:
            label = 7 → "seven"
            caption = [<sos>, seven, <eos>]

        Returns
        -------
        caption_tokens : torch.Tensor of shape (3,)
        """
        word = digit_words[int(label)]
        tokens = [SOS_IDX, word2idx[word], EOS_IDX]
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx: int):
        """
        Retrieve a single example.

        Parameters
        ----------
        idx : int
            Index into the dataset.

        Returns
        -------
        img : torch.Tensor
            Shape (1, 28, 28)
        caption_tokens : torch.Tensor
            Shape (3,)
        label : int
            Digit label
        """
        img, label = self.mnist[idx]
        caption_tokens = self.encode_caption(label)
        return img, caption_tokens, label


# ---------------------------------------------------------------------------
# Collation Function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
    """
    Custom collation function for DataLoader.

    Performs:
      1. Stacking all images → (B, 1, 28, 28)
      2. Padding caption sequences with PAD_IDX
      3. Computing all true lengths

    Parameters
    ----------
    batch : list of (img, caption, label)

    Returns
    -------
    imgs : torch.Tensor
        Shape (B, 1, 28, 28)
    padded : torch.Tensor
        Padded captions, shape (B, L_max)
    lengths : torch.Tensor
        True caption lengths, shape (B,)
    labels : torch.Tensor
        Digit labels, shape (B,)
    """

    imgs = []
    captions = []
    labels = []
    lengths = []

    for img, cap, lab in batch:
        imgs.append(img)
        captions.append(cap)
        labels.append(lab)
        lengths.append(len(cap))

    max_len = max(lengths)

    # Create padded caption matrix
    padded = torch.full((len(batch), max_len), PAD_IDX, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap

    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return imgs, padded, lengths, labels
