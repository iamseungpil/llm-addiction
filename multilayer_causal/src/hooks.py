"""Multi-layer forward hooks. Batch size 1 only (M3/replay protocol).

HF decoder layers return tuples; element 0 is hidden states (B, T, D).

MultiLayerPatcher  — E1: replace last-k positions with cached +G twin values,
                     once per generation (first forward with T>1), per layer.
SubspacePatcher    — E2: same positions/timing, but move only the rank-r
                     projection: h += (h_plus − h) @ V @ V^T.
MultiLayerSteerer  — E3: h += alpha * scale_l * v_l at ALL positions on EVERY
                     forward (prefill + each decode step).
"""
from __future__ import annotations

import torch


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _repack(output, out):
    return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out


class _HookSet:
    def __init__(self):
        self._handles = []

    def install(self, model):
        layers = model.model.layers
        for li in self.layer_indices:
            assert 0 <= li < len(layers), f"layer {li} out of range ({len(layers)})"
            self._handles.append(layers[li].register_forward_hook(self._hook_for(li)))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class MultiLayerPatcher(_HookSet):
    def __init__(self, cached):
        super().__init__()
        self.cached = {int(k): v for k, v in cached.items()}
        self.layer_indices = sorted(self.cached)
        self._fired = {li: 0 for li in self.layer_indices}

    def _hook_for(self, li):
        patch_h = self.cached[li]

        def hook(module, _input, output):
            out = _hidden(output)
            assert out.shape[0] == 1, "batch size must be 1"
            if out.shape[1] > 1 and self._fired[li] == 0:
                self._fired[li] += 1
                k = min(out.shape[1], patch_h.shape[0])
                out[0, -k:, :] = patch_h[-k:].to(out.dtype).to(out.device)
            return _repack(output, out)

        return hook


class SubspacePatcher(_HookSet):
    def __init__(self, cached, bases):
        super().__init__()
        self.cached = {int(k): v for k, v in cached.items()}
        self.bases = {int(k): v for k, v in bases.items()}
        assert set(self.cached) == set(self.bases), "cached/bases layer mismatch"
        self.layer_indices = sorted(self.cached)
        self._fired = {li: 0 for li in self.layer_indices}

    def _hook_for(self, li):
        patch_h, V = self.cached[li], self.bases[li]

        def hook(module, _input, output):
            out = _hidden(output)
            assert out.shape[0] == 1, "batch size must be 1"
            if out.shape[1] > 1 and self._fired[li] == 0:
                self._fired[li] += 1
                k = min(out.shape[1], patch_h.shape[0])
                Vd = V.to(out.dtype).to(out.device)
                delta = patch_h[-k:].to(out.dtype).to(out.device) - out[0, -k:, :]
                out[0, -k:, :] += (delta @ Vd) @ Vd.T
            return _repack(output, out)

        return hook


class MultiLayerSteerer(_HookSet):
    def __init__(self, directions, scales, alpha):
        super().__init__()
        self.directions = {int(k): v for k, v in directions.items()}
        self.scales = {int(k): float(v) for k, v in scales.items()}
        assert set(self.directions) == set(self.scales), "directions/scales mismatch"
        self.layer_indices = sorted(self.directions)
        self.alpha = float(alpha)

    def _hook_for(self, li):
        add = self.alpha * self.scales[li] * self.directions[li]

        def hook(module, _input, output):
            out = _hidden(output)
            out += add.to(out.dtype).to(out.device)
            return _repack(output, out)

        return hook


class PrefillTap(_HookSet):
    """Read-only capture of one layer's LAST-token hidden state on the first
    prefill forward (T > 1). Used as the steer-arm manipulation check
    (runner log_vectors): install AFTER editing hooksets via HookGroup so the
    tap observes the post-intervention value when it shares a layer."""

    def __init__(self, layer):
        super().__init__()
        self.layer_indices = [int(layer)]
        self.hidden = None

    def _hook_for(self, li):
        def hook(module, _input, output):
            out = _hidden(output)
            assert out.shape[0] == 1, "batch size must be 1"
            if out.shape[1] > 1 and self.hidden is None:
                self.hidden = out[0, -1, :].detach().float().cpu().clone()
        return hook


class HookGroup:
    """Install/remove several hooksets as one; registration order = arg order
    (PyTorch fires same-module hooks in registration order)."""

    def __init__(self, *hooksets):
        self.hooksets = [h for h in hooksets if h is not None]

    def install(self, model):
        for h in self.hooksets:
            h.install(model)

    def remove(self):
        for h in self.hooksets:
            h.remove()


def cache_layer_outputs(model, tokenizer, device, prompt_text, layer_indices):
    """One no-grad forward over prompt_text; capture each layer's (T, D) output."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    cached, handles = {}, []
    layers = model.model.layers

    def mk(li):
        def hook(module, _i, output):
            cached[li] = _hidden(output)[0].detach().clone()
        return hook

    for li in layer_indices:
        handles.append(layers[li].register_forward_hook(mk(li)))
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()
    return cached, inputs["input_ids"][0]
