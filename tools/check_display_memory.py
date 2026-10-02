#!/usr/bin/env python3
"""Hardware qualification of external CUDA display storage; no checkpoint required."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from tensorfold.cuda import display_memory


def available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) * 1024


def check(mib, card, cycles):
    size = mib << 20
    torch.cuda.init()
    torch.empty(1, device='cuda')
    gc.collect()
    torch.cuda.empty_cache()
    before = available()
    before_allocated = torch.cuda.memory_allocated()
    owner = display_memory.empty((size,), dtype=torch.uint8, card=card, max_bytes=size)
    receipt = dict(owner.receipt)
    assert owner.tensor.is_cuda and owner.tensor.is_contiguous()
    assert owner.tensor.data_ptr() % 256 == 0
    assert owner.tensor.numel() == size
    assert torch.cuda.memory_allocated() == before_allocated, 'external storage was charged to torch allocator'
    receipt.update(host_available_before=before, host_available_allocated=available())
    # Write and read back EVERY byte, with a pattern that changes at every chunk boundary.
    chunk = min(4 << 20, size)
    pattern = torch.arange(chunk, dtype=torch.int32, device='cuda').to(torch.uint8)
    started = time.monotonic()
    for i, offset in enumerate(range(0, size, chunk)):
        n = min(chunk, size - offset)
        expected = pattern[:n] + i % 251
        owner.tensor[offset:offset + n].copy_(expected)
        assert torch.equal(owner.tensor[offset:offset + n].cpu(), expected.cpu()), offset
    receipt['full_write_readback_seconds'] = time.monotonic() - started
    # Stream ordering and alias ownership, including deletion of the original Python owner.
    alias = owner.tensor[-65536:].view(torch.int32)
    alias.fill_(19)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        alias.add_(1)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    alias.fill_(19)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        alias.add_(1)
    owner.retain_graph(graph)
    ptr = owner.tensor.data_ptr()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(alias.cpu(), torch.full_like(alias.cpu(), 29))
    del owner
    gc.collect()
    graph.replay()
    torch.cuda.synchronize()
    assert alias[0].item() == 30
    assert graph._tensorfold_display_storage[0].tensor.data_ptr() == ptr
    del alias, graph, pattern, expected
    gc.collect()
    torch.cuda.synchronize()
    # Reallocate the full reservation after cleanup; leaks fail by the second allocation.
    for _ in range(cycles - 1):
        owner = display_memory.zeros((size,), dtype=torch.uint8, card=card, max_bytes=size)
        assert owner.tensor[0].item() == 0 and owner.tensor[-1].item() == 0
        del owner
        gc.collect()
    receipt.update(cycles=cycles, graph_replays=11, host_available_after=available(),
                   torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(), passed=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mib', type=int, default=1792)
    parser.add_argument('--card', default='/dev/dri/card0')
    parser.add_argument('--cycles', type=int, default=3)
    args = parser.parse_args()
    if args.mib < 1 or args.cycles < 1:
        parser.error('mib and cycles must be positive')
    print(json.dumps(check(args.mib, args.card, args.cycles), sort_keys=True), flush=True)
