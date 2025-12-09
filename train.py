"""
train.py

Training entry point for GVAMT-Lite.

This script performs the following steps:

  1. Loads the MNISTWithText dataset (image + synthetic captions).
  2. Constructs the GVAMTLite model.
  3. Runs a standard training loop optimizing the composite multimodal loss.
  4. Periodically saves model checkpoints.

Intended usage:
    python train.py

Hyperparameters can be adjusted at the top of the file or passed into train_main.
"""

import os
import time
import argparse
from typing import Optional

import torch
from torch.utils.data import DataLoader

from dataset import MNISTWithText, collate_fn, word_list
from model import GVAMTLite, multimodal_loss
from utils import save_model

# Default training hyperparameters
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 5
DEFAULT_LR = 1e-3
CHECKPOINT_DIR = "models"


def train_main(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    device: Optional[str] = None,
) -> None:
    """
    Train GVAMT-Lite on MNISTWithText.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate for Adam optimizer.
    device : Optional[str]
        Device string (e.g., "cpu" or "cuda"). If None, auto-selects CUDA when available.
    """
    # Device selection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[train] Using device: {device}")

    # Prepare dataset and dataloader
    train_ds = MNISTWithText(train=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2 if os.name != "nt" else 0,  # avoid issues on Windows
        pin_memory=(device.type == "cuda"),
    )

    # Model, optimizer
    vocab_size = len(word_list)
    model = GVAMTLite(
        vocab_size=vocab_size,
        img_latent_dim=64,
        txt_hidden_dim=64,
        fused_dim=64,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_txt = 0.0
        n_samples = 0

        for batch_idx, (imgs, captions, lengths, labels) in enumerate(train_loader, start=1):
            imgs = imgs.to(device)              # (B, 1, 28, 28)
            captions = captions.to(device)      # (B, L)
            lengths = lengths.to(device)        # (B,)

            optimizer.zero_grad()

            # Forward pass
            img_recon, mu, logvar, logits, captions_tgt = model(imgs, captions, lengths)

            # Compute loss
            loss, recon_loss, kl_loss, txt_loss = multimodal_loss(
                imgs, img_recon, mu, logvar, logits, captions_tgt, lengths
            )

            # Backprop + step
            loss.backward()
            optimizer.step()

            batch_size_actual = imgs.size(0)
            n_samples += batch_size_actual
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_txt += txt_loss.item()

            # Optional: lightweight progress logging per N batches
            if batch_idx % 100 == 0:
                avg_loss = total_loss / n_samples
                print(f"[Epoch {epoch}] Batch {batch_idx} | avg_loss={avg_loss:.4f}")

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / n_samples if n_samples else float("nan")
        avg_recon = total_recon / n_samples if n_samples else float("nan")
        avg_kl = total_kl / n_samples if n_samples else float("nan")
        avg_txt = total_txt / n_samples if n_samples else float("nan")

        print(
            f"Epoch {epoch}/{epochs} finished — "
            f"loss={avg_loss:.4f}, recon={avg_recon:.4f}, kl={avg_kl:.6f}, text={avg_txt:.4f}, "
            f"time={epoch_time:.1f}s"
        )

        # Save checkpoint for this epoch
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"gvamt_lite_epoch{epoch}.pt")
        save_model(model, ckpt_path)
        print(f"[train] Saved checkpoint: {ckpt_path}")


def parse_args():
    """Simple CLI for training parameters."""
    parser = argparse.ArgumentParser(description="Train GVAMT-Lite (image+text multimodal VAE).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g., 'cpu' or 'cuda'.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_main(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
