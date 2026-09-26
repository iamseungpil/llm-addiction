"""W9 IC EXTRACT - config-parity / smoke guard for the LLaMA investment_choice
full-layer phase_a re-extraction node job (amlt/sec4_w9icextract.yaml.template).

CPU-only, no torch / no HF / no model. Asserts the template is internally
consistent and cannot silently do the wrong thing:
  * layer list is exactly 0..31 (32 layers) in BOTH the env knob and the
    shape-check;
  * the ONLY HF write target is the NEW .../llama/checkpoints_w9/ path (never the
    existing count-only checkpoints/ IC file, the dp dump, or any other task ->
    no overwrite of the existing IC data);
  * all FOUR IC constraint catalogs (c10/c30/c50/c70) are staged into one dir;
  * the on-HF existence guard is present (SKIP instead of overwrite);
  * the staleness guard against a non-provenance IC extractor is present, keyed
    to the extract_llama_ic.py provenance marker;
  * paper_axes.LLAMA_IC_HIDDEN points at the checkpoints_w9 re-dump;
  * push_code_to_hf ships extract_llama_ic.py so the node sees the provenance write;
  * amlt YAML traps: description uses ' - ' not ': ', the block parses, and the
    python -c blocks carry only backslash-escaped double quotes.
"""
import re
from pathlib import Path

import yaml

from multilayer_causal.src import paper_axes as pa

MLC = Path(__file__).resolve().parents[1]
TEMPLATE = MLC / "amlt" / "sec4_w9icextract.yaml.template"

IC_DEST = "sae_features_v3/investment_choice/llama/checkpoints_w9"
IC_OLD = "sae_features_v3/investment_choice/llama/checkpoints/phase_a_hidden_states.npz"
IC_CATALOGS = (
    "llama_investment_c10_20260308_112003.json",
    "llama_investment_c30_20260308_135842.json",
    "llama_investment_c50_20260308_151326.json",
    "llama_investment_c70_20260308_172552.json",
)


def _env():
    doc = yaml.safe_load(TEMPLATE.read_text())
    return doc["jobs"][0]["submit_args"]["env"], doc


def test_template_parses_and_header_parity():
    env, doc = _env()
    job = doc["jobs"][0]
    assert job["sku"] == "80G4-H100"
    assert doc["target"]["service"] == "sing"
    assert doc["target"]["name"] == "msrresrchbasicvc"
    # description trap: first line uses ' - ' separators, never ': '
    desc = TEMPLATE.read_text().splitlines()[0]
    assert desc.startswith("description:")
    assert ": " not in desc[len("description:"):]
    # token placeholders wired for render.sh
    assert env["HF_TOKEN"] == "HF_TOKEN_PLACEHOLDER"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "HF_TOKEN_PLACEHOLDER"


def test_layer_list_is_exactly_0_to_31():
    env, _ = _env()
    layers = [int(x) for x in env["MLC_LAYERS"].split(",")]
    assert layers == list(range(32)), layers
    assert len(layers) == 32
    text = TEMPLATE.read_text()
    # the shape/layers assertion in the upload block hard-checks 0..31 too
    assert "list(np.asarray(z[\\\"layers\\\"])) == list(range(32))" in text
    assert "shp[1] == 32 and shp[2] == 4096" in text


def test_upload_target_is_only_checkpoints_w9_never_existing_ic():
    env, _ = _env()
    # the env-declared destination is exactly the NEW checkpoints_w9 dir
    assert env["MLC_IC_DEST"] == IC_DEST
    text = TEMPLATE.read_text()
    # upload_file writes ONLY `dest + "/" + fname`; dest is the single env knob
    # -> no other repo path is ever written.
    assert 'path_in_repo=path_in_repo' in text
    assert 'path_in_repo = dest + \\"/\\" + fname' in text
    # the job NEVER writes to the existing count-only IC file, the dp dump, or
    # any other task namespace (would clobber existing data).
    assert IC_OLD not in text
    assert "hidden_states_dp" not in text
    assert "slot_machine" not in text and "mystery_wheel" not in text
    # in the command body the dest is reached ONLY via the env knob (never a bare
    # repo-path literal), and the write is exactly dest + "/" + fname.
    body = text[text.index("bash -c '"):]
    assert IC_DEST not in body                       # no hard-coded write path
    assert 'dest = os.environ[\\"MLC_IC_DEST\\"]' in body
    # local read dir is the extractor's own checkpoints/ (out/ic); the only
    # remote checkpoints literal is the _w9 variant (existing IC file untouched).
    assert 'sae_features_v3/investment_choice/llama/checkpoints/' not in body
    # the two filenames uploaded are the phase_a pair (npz + metadata json)
    assert 'phase_a_hidden_states.npz' in text and 'phase_a_metadata.json' in text


def test_all_four_ic_catalogs_staged_into_one_dir():
    env, _ = _env()
    rels = [r for r in env["MLC_IC_CATALOGS"].split(",") if r]
    assert len(rels) == 4, rels
    for rel, name in zip(rels, IC_CATALOGS):
        assert rel == "behavioral/investment_choice/v2_role_llama/" + name, rel
    text = TEMPLATE.read_text()
    # staged into ONE clean dir; the staging block asserts exactly 4
    assert 'assert len(rels) == 4' in text
    assert "/scratch/mlc/data/ic" in text
    # extractor invoked with that single data dir and --phase-a-only, layers 0..31
    assert "--data-dir /scratch/mlc/data/ic" in text
    body = text[text.index("bash -c '"):]
    assert body.count("--phase-a-only") == 1
    assert "extract_llama_ic.py --gpu 0 --phase-a-only" in text


def test_existence_guard_refuses_overwrite():
    text = TEMPLATE.read_text()
    # pull the current file list and SKIP (no upload) if the target exists
    assert "list_repo_files(repo" in text
    assert "if path_in_repo in existing:" in text
    assert re.search(r'print\(\\"SKIP\\"', text), "no SKIP print in existence guard"
    # SKIP path must `continue` (not fall through to upload_file)
    guard = text[text.index("if path_in_repo in existing:"):]
    assert "continue" in guard[:200]


def test_staleness_guard_and_ic_only_fetch():
    text = TEMPLATE.read_text()
    # aborts loudly if the fetched IC extractor lacks the additive provenance write
    assert "Additive provenance (W9)" in text
    assert "STALE_EXTRACTOR_NO_PROVENANCE" in text
    # fetches ONLY extract_llama_ic.py (IC catalog carries prompts -> no
    # prompt_reconstruction, and never the SM/MW extractors)
    assert "extract_llama_ic.py" in text
    assert "prompt_reconstruction.py" not in text
    assert "extract_llama_sm.py" not in text and "extract_llama_mw.py" not in text
    assert "sae_v3_analysis/src/" in text
    # the upload block requires the catalog-join provenance keys in the npz
    assert 'game_ids' in text and 'round_nums' in text


def test_python_c_blocks_escape_inner_quotes():
    """Every LITERAL quote inside a `python -c "..."` payload must be
    backslash-escaped (\") or it would prematurely close the arg inside the outer
    `bash -c '...'` and break submission (same rule w8extract follows)."""
    text = TEMPLATE.read_text()
    body = text[text.index("bash -c '"):]
    # at least the four python -c blocks (imports, fetch, catalogs, upload)
    openers = re.findall(r'python -c "', body)
    assert len(openers) >= 4, len(openers)
    assert 'os.environ[\\"HF_TOKEN\\"]' in body
    assert 'os.environ[\\"MLC_HF_REPO\\"]' in body
    assert 'z[\\"hidden_states\\"]' in body
    assert 'repo_type=\\"dataset\\"' in body


def test_paper_axes_points_at_checkpoints_w9():
    assert pa.LLAMA_IC_HIDDEN == IC_DEST + "/phase_a_hidden_states.npz"
    # the existing count-only file is NOT the wired path
    assert pa.LLAMA_IC_HIDDEN != IC_OLD


def test_push_code_ships_ic_extractor():
    src = (MLC / "scripts" / "push_code_to_hf.py").read_text()
    assert '"sae_v3_analysis/src/extract_llama_ic.py"' in src
