"""LLaMA prompt parity against the §3 corpus generator (w3L gate, RUN_PLAN_W3).

The original llama_gemma_experiment.py builds ONE model-agnostic prompt
template (create_prompt / get_history_text contain no model branching) and
wraps BOTH models with the tokenizer chat template at generation time, so
there is no llama-specific prompt construction to mirror.

The frozen M3'' builder (src/prompts.py) deviates from the original in
exactly four MODEL-AGNOSTIC ways, all of which already applied to every
gemma W1/W2 arm:
  1. no ROLE_INSTRUCTION prefix;
  2. history window = last 3 rounds, two-space indent, no consecutive-loss
     warning (original: last 5, flush-left, warning at >= 3 straight losses);
  3. variable bet menu capped at $100 (original: full balance);
  4. raw prompt at generation (original: apply_chat_template for BOTH models).
test_frozen_parity_50_llama_states proves byte-level that these are the ONLY
differences on 50 real llama catalog states; the gemma twin test proves the
delta set is identical for gemma, i.e. nothing is llama-specific.
"""
import importlib.util
import inspect
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ORIG_PATH = (REPO / "paper_experiments" / "slot_machine_6models" / "src"
             / "llama_gemma_experiment.py")


@pytest.fixture(scope="module")
def orig():
    spec = importlib.util.spec_from_file_location("llama_gemma_orig", ORIG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _states(model, k=50):
    from multilayer_causal.src.states import behavioral_root, load_minusG_states
    cat = behavioral_root() / "slot_machine" / f"{model}_v4_role"
    if not (cat.exists() and any(cat.glob("*.json"))):
        pytest.skip(f"{model} catalog not present locally")
    pool = load_minusG_states(model, n=10 * k)
    picked = [(g, r) for g, r in pool if len(g.get("history", [])) >= r][:k]
    assert len(picked) == k, f"only {len(picked)} usable {model} states"
    return picked


@pytest.fixture(scope="module")
def llama_states():
    return _states("llama", 50)


def _original_prompt(orig, game, round_idx):
    """Drive the REAL create_prompt at the same decision point.

    create_prompt reads nothing from self, so it is called unbound; the game
    object carries the catalog balance and the first round_idx history rounds
    (catalog entries already use the round/bet/result/balance keys that
    get_history_text expects)."""
    g = orig.SlotMachineGame()
    g.balance = int(game["decisions"][round_idx]["balance_before"])
    g.history = [dict(h) for h in game.get("history", [])[:round_idx]]
    return orig.RestartExperiment.create_prompt(
        None, game["bet_type"], game.get("prompt_combo", ""), g)


def _frozen_view(orig_text, role_instruction, balance):
    """Apply the four documented deltas to the ORIGINAL prompt text.

    Any undocumented difference (a llama-specific template change, reordered
    components, etc.) breaks the byte-equality assertion downstream."""
    assert orig_text.startswith(role_instruction)
    body = orig_text[len(role_instruction):]
    head, sep, rest = body.partition("\nGame History:\n")
    if sep:
        hist, sep2, tail = rest.partition("\nChoose one of the following:\n")
        assert sep2, "history block without choice menu"
        lines = [l for l in hist.split("\n") if l.startswith("Round ")]
        body = (head + "\nGame History:\n"
                + "".join(f"  {l}\n" for l in lines[-3:])
                + "\nChoose one of the following:\n" + tail)
    if balance > 100:
        body = body.replace(f"1) Bet (choose $5-${balance})\n",
                            "1) Bet (choose $5-$100)\n", 1)
    return body


def test_original_template_is_model_agnostic(orig):
    cp = inspect.getsource(orig.RestartExperiment.create_prompt)
    gh = inspect.getsource(orig.SlotMachineGame.get_history_text)
    gr = inspect.getsource(orig.RestartExperiment.generate_response)
    for src in (cp, gh, gr):
        # comments may name the models ("both Gemma-IT and LLaMA-Instruct");
        # the CODE must not branch on them
        code = "\n".join(l.split("#", 1)[0] for l in src.splitlines()).lower()
        assert "model_name" not in code and "llama" not in code \
            and "gemma" not in code
    # the only model-dependent wrapping is the tokenizer chat template,
    # applied identically to both instruction-tuned models
    assert "apply_chat_template" in gr


def test_frozen_parity_50_llama_states(orig, llama_states):
    from multilayer_causal.src.prompts import build_prompt
    for game, round_idx in llama_states:
        ours = build_prompt(game, round_idx)
        assert ours is not None
        original = _original_prompt(orig, game, round_idx)
        bal = int(game["decisions"][round_idx]["balance_before"])
        assert _frozen_view(original, orig.ROLE_INSTRUCTION, bal) == ours, \
            f"undocumented prompt divergence (combo={game.get('prompt_combo')}, " \
            f"round_idx={round_idx}, balance={bal})"


def test_frozen_parity_gemma_twin(orig):
    """Same transform, gemma states: the delta set is model-agnostic."""
    from multilayer_causal.src.prompts import build_prompt
    for game, round_idx in _states("gemma", 10):
        ours = build_prompt(game, round_idx)
        original = _original_prompt(orig, game, round_idx)
        bal = int(game["decisions"][round_idx]["balance_before"])
        assert _frozen_view(original, orig.ROLE_INSTRUCTION, bal) == ours
