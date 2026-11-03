"""MiniMax H3 sidecar kernels (BMG, Linux).

Two native extensions are built next to this package when the wheel targets
BMG on Linux:

* ``h3_fused``: ``torch.ops.h3fuse.{norm_rot_quant, rot_quant,
  silu_mul_rot_quant}``, which fuse the activation preprocessing (RMSNorm /
  modulation, ConvRot rotation, INT8 quantization) in front of each H3 INT8
  GEMM.
* ``h3_int8_attn``: ``torch.ops.h3i8attn.{rms_rope_quant, fmha}``, the H3
  attention path with INT8 QK^T and bf16 P*V.

ComfyUI-OmniXPU's H3 adapter loads them through :func:`find_extension`; the
ops are registered by ``torch.ops.load_library`` and have no Python wrapper.
"""

import glob
import os
from importlib.machinery import EXTENSION_SUFFIXES

EXTENSIONS = ("h3_fused", "h3_int8_attn")


def find_extension(name):
    """Return the path of the native ``name`` extension, or ``""``."""
    if name not in EXTENSIONS:
        raise ValueError(f"unknown H3 extension {name!r}; expected one of {EXTENSIONS}")
    here = os.path.dirname(os.path.abspath(__file__))
    for suffix in (*EXTENSION_SUFFIXES, ".so"):
        exact = os.path.join(here, name + suffix)
        if os.path.isfile(exact):
            return exact
        matches = sorted(glob.glob(os.path.join(here, name + "*" + suffix)))
        if matches:
            return matches[0]
    return ""
