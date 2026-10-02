# Experimental external CUDA display storage

`tensorfold.cuda.display_memory` supplies contiguous CUDA tensors backed by an
NVIDIA DRM dumb buffer. It uses the public Linux DRM UAPI and CUDA
`cudaHostRegisterMapped | cudaHostRegisterIoMemory` APIs. This is an independent
Apache-2.0 implementation; no AGPL allocator source is included.

This mechanism is hardware and driver dependent. Linux's dumb-buffer API alone
does not promise firmware-reserved memory or CUDA interoperability. Qualify the
actual machine before counting these bytes as additional capacity. GLM layer
selection and memory accounting belong to the consuming recipe.

```python
import torch
from tensorfold.cuda.display_memory import zeros

owner = zeros((8192, 528), dtype=torch.uint8, device="cuda:0",
              card="/dev/dri/card0", max_bytes=1792 << 20)
cache = owner.tensor
print(owner.receipt)
```

The Linux DRM UAPI headers (`libdrm-dev` or equivalent), PyTorch's extension
compiler and CUDA headers are required. The extension is compiled on first use.
No CUDA allocation, compilation or DRM access occurs when the module is merely
imported. In Docker, pass `--gpus all --device /dev/dri/card0`. The native allocator
checks that the DRM card belongs to the selected CUDA PCI device.

Allocation is explicit, occurs before graph capture, and fails if the card,
mapping, registration, alignment or size check fails. There is no fallback to
ordinary memory. The ceiling covers DRM scanline padding as well as tensor data.
Tensor memory is not reflected in `torch.cuda.memory_allocated()`; include the
reported bytes separately in monitoring.

The tensor's storage deleter owns the CUDA registration, mapping, DRM handle and
file descriptor. Slices and dtype views retain it. After the last reference goes
away, cleanup synchronizes the CUDA device before unregistering and destroying
the allocation, so in-flight work on other streams finishes first. If CUDA is
already shutting down or synchronization fails, cleanup is deferred to process
exit and a diagnostic is printed. Never call `cudaFree` on this pointer.

A captured CUDA graph holds raw addresses. Keep the owner alive throughout the
graph's lifetime or call `owner.retain_graph(graph)` after capture. Callers must
also keep their other graph inputs and outputs alive, as usual. Do not resize or
replace the external cache after capture. Access through CUDA; CPU reads/writes
to this noncoherent I/O mapping are not exposed.

Run `PYTHONPATH=src python tools/check_display_memory.py --mib 1792` on an idle,
headless Spark with modesetting enabled and DRM fbdev disabled. The check writes
and reads every byte, replays a graph after dropping the original Python owner,
checks pointer stability and allocator accounting, and reallocates the full
reservation three times. JSON output records the runtime and allocation sizes.

API references:

- [Linux DRM userland interfaces](https://www.kernel.org/doc/html/latest/gpu/drm-uapi.html)
- [CUDA memory management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html)
