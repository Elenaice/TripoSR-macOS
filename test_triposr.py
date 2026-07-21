import torch

print("Torch:", torch.__version__)
print("MPS:", torch.backends.mps.is_available())

import trimesh
print("trimesh OK")

import rembg
print("rembg OK")

from diffusers import DiffusionPipeline
print("diffusers OK")

from transformers import AutoModel
print("transformers OK")