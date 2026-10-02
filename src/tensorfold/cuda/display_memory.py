"""Opt-in NVIDIA DRM storage, outside PyTorch's CUDA allocator (experimental, Linux).

No model policy belongs here. The caller selects the card, byte ceiling, shape and dtype.
Keep the returned owner for every CUDA graph's lifetime, or use ``retain_graph``. Tensor
views also own the native allocation. Allocations may only be created before capture.
CPU access to the I/O mapping is deliberately not exposed; transfer through CUDA instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import operator
import sys
from typing import Any


@lru_cache(maxsize=1)
def _extension():
    if sys.platform != "linux":
        raise RuntimeError("Display storage requires Linux NVIDIA DRM")
    from tensorfold.cuda.build import load

    return load(name="tensorfold_display_memory_v1", sources=[str(Path(__file__).with_suffix(".cpp"))],
                with_cuda=True, extra_cflags=["-O2", "-std=c++17"])


@dataclass
class DisplayStorage:
    tensor: Any
    receipt: dict

    def retain_graph(self, graph) -> None:
        """Keep the allocation alive even if all other tensor references disappear before replay."""
        owners = getattr(graph, "_tensorfold_display_storage", None)
        if owners is None:
            owners = []
            graph._tensorfold_display_storage = owners
        owners.append(self)


def empty(shape, *, dtype, device="cuda", card="/dev/dri/card0", max_bytes: int) -> DisplayStorage:
    """Contiguous external CUDA tensor; allocation failures propagate, with no ordinary-memory fallback.

    ``max_bytes`` limits the actual DRM allocation including scanline padding, not just tensor bytes.
    The DRM card and selected CUDA device must refer to the same PCI GPU (checked natively).
    """
    import torch

    shape = tuple(operator.index(n) for n in shape)
    if not shape or any(n <= 0 for n in shape):
        raise ValueError("Display storage needs a nonempty shape with positive dimensions")
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("Display storage is a CUDA tensor")
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    max_bytes = operator.index(max_bytes)
    if size > max_bytes or max_bytes <= 0 or max_bytes > 2**63 - 1:
        raise ValueError(f"Display storage needs {size} bytes, limit is {max_bytes}")
    raw, receipt = _extension().allocate(size, max_bytes, str(card), dev.index)
    tensor = raw.view(dtype).view(shape)
    return DisplayStorage(tensor, dict(receipt))


def zeros(shape, **kwargs) -> DisplayStorage:
    """As ``empty``, initialized by a GPU write on the current CUDA stream."""
    allocation = empty(shape, **kwargs)
    allocation.tensor.zero_()
    return allocation
