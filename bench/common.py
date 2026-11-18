"""Shared constants and helpers: the classes, the null label, model loading, the initial noise and hashing.

Every other script imports from here, so all experiments use the same 8 classes in the same order and the same noise per seed.
Importing this module also puts the upstream SiT repository (sit/) on sys.path, so `models` and `transport` can be imported.
"""
import hashlib, sys

import numpy as np
import torch

sys.path.insert(0, "sit")
from models import SiT_models

LABELS = [207, 360, 387, 974, 88, 979, 417, 279]  # sample.py's default class_labels (ImageNet-1k indices)
LABEL_NAMES = ["golden retriever", "otter", "lesser panda", "geyser", "macaw", "valley", "balloon", "arctic fox"]
# SiT was trained with 1000 real classes plus one extra embedding row, index 1000, that stands for "no class". Feeding it gives the
# unconditional prediction f_uncond, which is what classifier-free guidance compares against.
NULL_CLASS = 1000
GUIDED_CHANNELS = 3  # forward_with_cfg guides model_out[:, :3] only; latent channel 3 always takes the conditional output


def sha256(path):
    """SHA-256 of a file, read in 16 MB chunks so multi-GB checkpoints never sit in memory at once. Recorded in manifests."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(a):
    """SHA-256 of an array's raw bytes. Two runs with equal hashes produced bit-identical arrays (same dtype, shape and values)."""
    # ascontiguousarray: tobytes() of a strided view would otherwise depend on memory layout rather than only on the values.
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_model(name, weights, device):
    """Build the SiT architecture `name` (e.g. "SiT-L/2"), load `weights` into it, and return it on `device` in eval mode."""
    # input_size is in latent pixels: 256px images / 8 (VAE downsampling) = 32. learn_sigma=True matches the 256x256
    # checkpoints, which have 8 output channels; SiT.forward() drops the 4 sigma channels before returning.
    model = SiT_models[name](input_size=32, num_classes=1000, learn_sigma=True)
    # Neither branch can run code from the file: safetensors holds only tensors, and weights_only=True refuses pickled objects.
    if weights.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(weights)
    else:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    # strict=True fails on any missing or extra key, so weights for the wrong architecture cannot load silently.
    model.load_state_dict(state, strict=True)
    # eval() matters: in train mode LabelEmbedder randomly replaces labels with the null class.
    return model.to(device).eval()


def initial_noise(seed):
    """CPU generator, so the noise does not depend on the device. Shape (8, 4, 32, 32): one latent per class."""
    # A private generator keeps the noise independent of global RNG state. This is not the noise sample.py makes for the same
    # --seed: sample.py draws from the global RNG after model init has already consumed part of it.
    return torch.randn(len(LABELS), 4, 32, 32, generator=torch.Generator().manual_seed(seed))
