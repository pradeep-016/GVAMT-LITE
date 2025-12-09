"""
model.py

Core model components for GVAMT-Lite:
- ImageVAEEncoder
- ImageVAEDecoder
- TextEncoder (GRU)
- TextDecoder (GRU)
- GVAMTLite wrapper
- multimodal_loss function

This file implements a lightweight multimodal generative architecture inspired by
the GVAMT model proposed in:

    Pradeep L. (2024), IJARESM – Multimodal Integration in LLMs.

The architecture encodes images and text into separate latent spaces, fuses the
representations, and decodes both modalities jointly.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Image Encoder (VAE)
# ---------------------------------------------------------------------------

class ImageVAEEncoder(nn.Module):
    """
    Convolutional encoder for the image branch of GVAMT-Lite.

    Maps an input image x ∈ [0, 1]^{1×28×28} to the parameters of a Gaussian
    posterior distribution q(z | x) = N(mu, sigma^2).

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the Gaussian latent vector.
    """

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (B,32,14,14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> (B,64,7,7)
            nn.ReLU(),
        )

        self.flat_dim = 64 * 7 * 7
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes a batch of images into latent distribution parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (B, 1, 28, 28).

        Returns
        -------
        mu : torch.Tensor
            Mean of approximate posterior q(z|x), shape (B, latent_dim).
        logvar : torch.Tensor
            Log-variance of q(z|x), shape (B, latent_dim).
        """
        h = self.conv(x)
        h = h.view(x.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


# ---------------------------------------------------------------------------
# Image Decoder (VAE style)
# ---------------------------------------------------------------------------

class ImageVAEDecoder(nn.Module):
    """
    Deconvolutional decoder that reconstructs images from fused latent vectors.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent code used for reconstruction.
    """

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()

        self.flat_dim = 64 * 7 * 7
        self.fc = nn.Linear(latent_dim, self.flat_dim)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> (B,32,14,14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # -> (B,1,28,28)
            nn.Sigmoid(),  # Normalize output to [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a batch of fused latent vectors into reconstructed images.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor of shape (B, latent_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed images of shape (B, 1, 28, 28).
        """
        h = self.fc(z)
        h = h.view(z.size(0), 64, 7, 7)
        x_recon = self.deconv(h)
        return x_recon


# ---------------------------------------------------------------------------
# Text Encoder (GRU)
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """
    GRU-based sentence encoder for digit captions.

    Produces fixed-size vector representations of shape (B, hidden_dim).

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embed_dim : int
        Token embedding dimension.
    hidden_dim : int
        GRU hidden size.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()

        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0   # PAD token assumed to be index 0
        )

        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, captions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Encodes token sequences into fixed-length vectors using GRU.

        Parameters
        ----------
        captions : torch.Tensor
            Integer token IDs of shape (B, L).
        lengths : torch.Tensor
            Actual caption lengths (used for packing), shape (B,).

        Returns
        -------
        torch.Tensor
            Encoded caption vectors of shape (B, hidden_dim).
        """
        embeddings = self.embed(captions)

        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, h_n = self.gru(packed)

        return h_n.squeeze(0)   # (B, hidden_dim)


# ---------------------------------------------------------------------------
# Text Decoder (GRU)
# ---------------------------------------------------------------------------

class TextDecoder(nn.Module):
    """
    GRU-based conditional decoder for text generation.

    Uses teacher forcing during training.

    Parameters
    ----------
    vocab_size : int
        Size of token vocabulary.
    embed_dim : int
        Embedding dimension.
    hidden_dim : int
        GRU hidden size (must match fused_dim).
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, captions_in: torch.Tensor, hidden: torch.Tensor):
        """
        Teacher-forced decoding of captions.

        Parameters
        ----------
        captions_in : torch.Tensor
            Input token IDs excluding <eos>, shape (B, L_in).
        hidden : torch.Tensor
            Initial GRU hidden state derived from fused latent, shape (B, H).

        Returns
        -------
        logits : torch.Tensor
            Next-token logits, shape (B, L_in, vocab_size).
        final_hidden : torch.Tensor
            Final GRU hidden state, shape (B, H).
        """
        embeddings = self.embed(captions_in)
        output, h_n = self.gru(embeddings, hidden.unsqueeze(0))
        logits = self.fc_out(output)
        return logits, h_n.squeeze(0)


# ---------------------------------------------------------------------------
# GVAMT-Lite Wrapper
# ---------------------------------------------------------------------------

class GVAMTLite(nn.Module):
    """
    GVAMT-Lite: a lightweight multimodal generative model.

    - Encodes images using VAE encoder
    - Encodes captions using GRU
    - Fuses the two latents
    - Decodes images and captions jointly

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    img_latent_dim : int
        Image latent size.
    txt_hidden_dim : int
        Text latent size.
    fused_dim : int
        Size of shared multimodal latent.
    """

    def __init__(
        self,
        vocab_size: int,
        img_latent_dim: int = 64,
        txt_hidden_dim: int = 64,
        fused_dim: int = 64,
    ):
        super().__init__()

        self.img_encoder = ImageVAEEncoder(img_latent_dim)
        self.img_decoder = ImageVAEDecoder(fused_dim)

        self.txt_encoder = TextEncoder(vocab_size, embed_dim=32, hidden_dim=txt_hidden_dim)
        self.txt_decoder = TextDecoder(vocab_size, embed_dim=32, hidden_dim=fused_dim)

        self.fusion = nn.Linear(img_latent_dim + txt_hidden_dim, fused_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, imgs: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor):
        """
        Full multimodal forward pass:

        1. Encode image → VAE latent (mu, logvar)
        2. Sample latent z_img
        3. Encode text → z_txt
        4. Fuse both → fused_z
        5. Decode fused_z → reconstructed image
        6. Decode fused_z → caption logits (teacher forcing)

        Returns
        -------
        img_recon : torch.Tensor
            Reconstructed images.
        mu : torch.Tensor
            VAE mean vector.
        logvar : torch.Tensor
            VAE log-variance.
        logits : torch.Tensor
            Predicted token logits.
        captions_tgt : torch.Tensor
            Ground-truth target tokens.
        """
        mu, logvar = self.img_encoder(imgs)
        z_img = self.reparameterize(mu, logvar)
        z_txt = self.txt_encoder(captions, lengths)

        fused = torch.cat([z_img, z_txt], dim=1)
        fused_z = self.fusion(fused)

        img_recon = self.img_decoder(fused_z)

        captions_in = captions[:, :-1]
        captions_tgt = captions[:, 1:]

        logits, _ = self.txt_decoder(captions_in, fused_z)

        return img_recon, mu, logvar, logits, captions_tgt


# ---------------------------------------------------------------------------
# Multimodal Loss Function
# ---------------------------------------------------------------------------

def multimodal_loss(
    imgs: torch.Tensor,
    img_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    logits: torch.Tensor,
    captions_tgt: torch.Tensor,
    lengths: torch.Tensor,
    alpha_img: float = 1.0,
    beta_kl: float = 1e-3,
    gamma_txt: float = 1.0,
):
    """
    Combined loss for image + text generation.

    L_total = α * L_img + β * KL + γ * L_text

    Returns
    -------
    total : torch.Tensor
        Scalar training loss.
    recon_loss : torch.Tensor
    kl_loss : torch.Tensor
    txt_loss : torch.Tensor
    """

    # Image reconstruction
    recon_loss = F.binary_cross_entropy(img_recon, imgs, reduction="sum")

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Text cross-entropy
    B, Lm1, V = logits.shape
    logits_flat = logits.reshape(B * Lm1, V)
    tgt_flat = captions_tgt.reshape(B * Lm1)

    txt_loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=0, reduction="sum")

    total = alpha_img * recon_loss + beta_kl * kl_loss + gamma_txt * txt_loss

    return total, recon_loss, kl_loss, txt_loss
