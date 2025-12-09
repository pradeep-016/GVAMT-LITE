"""
utils.py

Utility helpers for GVAMT-Lite.

Provides:
- token mapping exports (word_list, word2idx, idx2word, PAD/SOS/EOS indices)
- model save/load helpers
- token decoding helper for readability
- deterministic seed helper
- device helper

Docstrings use a detailed academic style for clarity in a research project.
"""

from typing import Dict, List
import torch
import random
import numpy as np
import os

# ---------------------------------------------------------------------------
# Token / vocabulary utilities (same mapping as dataset.py)
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

special_tokens: List[str] = ["<pad>", "<sos>", "<eos>"]
word_list: List[str] = special_tokens + sorted(set(digit_words.values()))

# mappings
word2idx: Dict[str, int] = {w: i for i, w in enumerate(word_list)}
idx2word: Dict[int, str] = {i: w for w, i in word2idx.items()}

PAD_IDX: int = word2idx["<pad>"]
SOS_IDX: int = word2idx["<sos>"]
EOS_IDX: int = word2idx["<eos>"]

# ---------------------------------------------------------------------------
# Model persistence helpers
# ---------------------------------------------------------------------------

def save_model(model: torch.nn.Module, path: str) -> None:
    """
    Save the state dictionary of a PyTorch model to disk.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be saved (state_dict is stored).
    path : str
        Destination file path for the saved state_dict. Parent directories
        will be created if they do not exist.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str, device: str = "cpu") -> None:
    """
    Load a saved state_dict into the provided model instance.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance into which the state_dict will be loaded.
    path : str
        Path to the saved state_dict file.
    device : str, optional
        Device string (e.g., "cpu" or "cuda"). The state dict will be mapped
        to this device during loading.
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)


# ---------------------------------------------------------------------------
# Token decoding & display helpers
# ---------------------------------------------------------------------------

def decode_token_sequence(tokens: torch.Tensor) -> str:
    """
    Convert a 1-D tensor of token IDs into a human-readable string.

    The function ignores PAD and SOS tokens and stops at EOS if present.

    Parameters
    ----------
    tokens : torch.Tensor
        1-D tensor of token IDs.

    Returns
    -------
    str
        Human-readable decoded string (words joined by spaces) or "<empty>".
    """
    words = []
    for t in tokens.tolist():
        if t == EOS_IDX:
            break
        if t in idx2word and t not in (PAD_IDX, SOS_IDX, EOS_IDX):
            words.append(idx2word[t])
    return " ".join(words) if len(words) > 0 else "<empty>"


def decode_batch_tokens(batch_tokens: torch.Tensor) -> List[str]:
    """
    Decode a batch of token sequences into a list of strings.

    Parameters
    ----------
    batch_tokens : torch.Tensor
        Tensor of shape (B, L) containing token IDs.

    Returns
    -------
    List[str]
        Decoded strings for each sequence in the batch.
    """
    return [decode_token_sequence(batch_tokens[i]) for i in range(batch_tokens.size(0))]


# ---------------------------------------------------------------------------
# Reproducibility & device helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for torch, numpy and Python's random to improve reproducibility.

    Parameters
    ----------
    seed : int, optional
        Desired random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Return a torch.device object based on availability and user preference.

    Parameters
    ----------
    prefer_cuda : bool, optional
        If True and CUDA is available, returns 'cuda', otherwise 'cpu'.

    Returns
    -------
    torch.device
        Selected device.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
