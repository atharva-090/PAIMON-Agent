"""Diagnose where each parameter/buffer actually ended up after dispatch.

Chasing "Tensor.item() cannot be called on meta tensors": something is still a
meta placeholder. Loading is only ~23s, so inspecting directly beats guessing.
"""

from __future__ import annotations

import collections

import torch

from local_model.k2_runtime import RUNTIME

RUNTIME.load()
m = RUNTIME.model

print("\n--- model-level ---")
try:
    print("model.device      :", m.device)
except Exception as exc:
    print("model.device      : RAISED", exc)
print("hf_device_map     :", getattr(m, "hf_device_map", None))
gc = m.generation_config
print("gen eos/pad       :", gc.eos_token_id, "/", gc.pad_token_id)

print("\n--- parameters by device ---")
pd: collections.Counter = collections.Counter()
meta_params = []
for name, p in m.named_parameters():
    pd[str(p.device)] += 1
    if p.device.type == "meta":
        meta_params.append(name)
print(dict(pd))
if meta_params:
    print(f"META params ({len(meta_params)}): {meta_params[:10]}")

print("\n--- buffers by device ---")
bd: collections.Counter = collections.Counter()
meta_bufs = []
for name, b in m.named_buffers():
    bd[str(b.device)] += 1
    if b.device.type == "meta":
        meta_bufs.append(name)
print(dict(bd))
if meta_bufs:
    print(f"META buffers ({len(meta_bufs)}): {meta_bufs[:10]}")

print("\n--- key modules ---")
for name in ("model.embed_tokens", "model.layers.0.self_attn.q_proj",
             "model.norm", "lm_head"):
    try:
        mod = m.get_submodule(name)
        w = getattr(mod, "weight", None)
        print(f"  {name:35s} {type(mod).__name__:22s} "
              f"weight.device={getattr(w, 'device', None)}")
    except Exception as exc:
        print(f"  {name:35s} MISSING ({exc})")

print("\n--- can we make an eos tensor on model.device? ---")
try:
    t = torch.tensor(gc.eos_token_id, device=m.device)
    print("  ok ->", t, t.device)
    print("  .item() ->", t.flatten()[0].item())
except Exception as exc:
    print("  FAILED:", type(exc).__name__, exc)
