import torch
import torch.nn as nn

from multilayer_causal.src import hooks


class Layer(nn.Module):
    def forward(self, x):
        return (x + 1.0,)          # HF layers return tuples


class Mock(nn.Module):
    """model.model.layers structure, D=8, 4 layers."""

    def __init__(self):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([Layer() for _ in range(4)])
        self.model = inner

    def forward(self, x):
        for l in self.model.layers:
            x = l(x)[0]
        return x


def test_patcher_replaces_suffix_once_on_prefill():
    m = Mock()
    cached = {1: torch.full((3, 8), 9.0), 2: torch.full((5, 8), 7.0)}
    p = hooks.MultiLayerPatcher(cached)
    p.install(m)
    out = m(torch.zeros(1, 5, 8))                  # prefill T=5
    # layer idx2 replaces ALL 5 positions with 7.0, then layer idx3 adds 1 → 8
    assert torch.allclose(out, torch.full((1, 5, 8), 8.0))
    out2 = m(torch.zeros(1, 1, 8))                 # decode step T=1 → no patch
    assert torch.allclose(out2, torch.full((1, 1, 8), 4.0))
    p.remove()
    assert torch.allclose(m(torch.zeros(1, 5, 8)), torch.full((1, 5, 8), 4.0))


def test_subspace_patcher_moves_only_projection():
    m = Mock()
    D, r = 8, 1
    V = torch.zeros(D, r)
    V[0, 0] = 1.0                                   # subspace = e0
    plus = torch.full((4, D), 5.0)
    sp = hooks.SubspacePatcher({1: plus}, {1: V})
    sp.install(m)
    out = m(torch.zeros(1, 4, D))
    # live h at layer idx1 = 2.0; delta=3 on all dims, projected → only dim0 → 5
    # then layers idx2, idx3 add +2 total
    assert torch.allclose(out[0, :, 0], torch.full((4,), 7.0))
    assert torch.allclose(out[0, :, 1:], torch.full((4, D - 1), 4.0))
    sp.remove()


def test_steerer_adds_every_forward_all_positions():
    m = Mock()
    v = torch.zeros(8)
    v[0] = 1.0
    st = hooks.MultiLayerSteerer({0: v, 3: v}, {0: 2.0, 3: 2.0}, alpha=1.5)
    st.install(m)
    out = m(torch.zeros(1, 2, 8))
    assert torch.allclose(out[0, :, 0], torch.full((2,), 4.0 + 2 * 1.5 * 2.0))
    out2 = m(torch.zeros(1, 1, 8))                 # fires again on decode-like pass
    assert torch.allclose(out2[0, :, 0], torch.full((1,), 4.0 + 2 * 1.5 * 2.0))
    st.remove()
