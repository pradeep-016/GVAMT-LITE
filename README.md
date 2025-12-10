# GVAMT-Lite  
*A lightweight multimodal generative architecture inspired by the GVAMT research model (Pradeep L., IJARESM 2024)*  

GVAMT-Lite is a simplified, training-friendly implementation of the **GVAMT (GAN + VAE + Transformer) multimodal architecture** proposed by **Pradeep L.** in:

> *Multimodal Integration in Large Language Models:  
> Advancing AI with Generative Adversarial Networks, Variational Autoencoders, and Transformers (GVAMT Model)*  
> IJARESM, Volume 12, Issue 4, April 2024.

This backend version is optimized for:
- CPU or **free Google Colab**
- Small datasets (MNIST + synthetic text captions)
- Clear modular code for research and extension  
- Easy migration into more complex multimodal pipelines

---

## Key Features

- **Image Encoder (VAE)**  
  Converts MNIST images into a Gaussian latent space using CNN layers.

- **Text Encoder (GRU)**  
  Encodes a textual caption such as `"seven"` into a fixed-size latent vector.

- **Early Fusion Layer**  
  Merges image and text latent vectors into a shared multimodal embedding.

- **Image Decoder (VAE deconv)**  
  Reconstructs images from the fused latent.

- **Text Decoder (GRU)**  
  Generates captions via teacher-forcing training.

- **End-to-end Multimodal Training**  
  Optimizes image BCE loss + text cross-entropy + KL divergence.

- **Easy to Extend**  
  Replace GRU with a Transformer, or upgrade the dataset to COCO, Flickr8K, or custom domains.

---

## Installation

```bash
git clone https://github.com/<your-username>/GVAMT-LITE.git
cd GVAMT-LITE
pip install -r requirements.txt
````

For CPU-only PyTorch (recommended for laptops):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## Training

Run locally:

```bash
python train.py
```

Or in Google Colab (GPU optional):

* Upload the project
* Run `train.py`
* Checkpoints saved under `models/`

---

## Output Checkpoints

After each epoch, the model saves:

```
models/gvamt_lite_epoch{N}.pt
```

You can later load these using:

```python
from model import GVAMTLite
from utils import load_model

model = GVAMTLite(vocab_size=len(word_list))
load_model(model, "models/gvamt_lite_epoch5.pt")
```

---

## How GVAMT-Lite Relates to the Original GVAMT Model

The original GVAMT architecture (Pradeep, 2024) combines:

* **GANs** for adversarial generation
* **VAEs** for latent representation
* **Transformers** for multimodal alignment

GVAMT-Lite keeps the **core multimodal idea** but simplifies architecture to ensure:

* Fast training without GPU
* Minimal dependencies
* Clear explainability
* Easy prototyping

This implementation can act as a **baseline** for future experiments:

* Adding cross-attention fusion
* Replacing GRU with Transformer encoder/decoder
* Using natural captions (COCO, Flickr8K)
* Multi-image or audio-text extension

---

## Citation

If you use this project in academic work, cite:

```
Pradeep L. (2024).
Multimodal Integration in Large Language Models:
Advancing AI with GANs, VAEs, and Transformers (GVAMT Model).
IJARESM, Volume 12, Issue 4.
```

---

## Author

**Pradeep L.**
AI/ML Research & Product Developer
Creator of GVAMT Architecture (2024)

---

## Future Enhancements (Roadmap)

* [ ] Transformer-based Text Encoder/Decoder
* [ ] Cross-Modal Attention Fusion
* [ ] GAN-based image reconstruction head
* [ ] True multimodal datasets (MS-COCO, Flickr30K)
* [ ] HuggingFace training integration
* [ ] Automatic caption beam search decoding

---

## Contributions

Contributions are welcome—feel free to open issues or submit PRs!
