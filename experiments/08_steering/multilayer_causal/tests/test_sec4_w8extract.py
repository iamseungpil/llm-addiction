"""W8 EXTRACT - config-parity / smoke guard for the LLaMA SM+MW full-layer
phase_a re-extraction node job (amlt/sec4_w8extract.yaml.template).

CPU-only, no torch / no HF / no model. Asserts the template is internally
consistent and cannot silently do the wrong thing:
  * layer list is exactly 0..31 (32 layers) in BOTH the env knob and the
    shape-check;
  * the ONLY HF write targets are the two new .../llama/checkpoints/ paths
    (never the dp dump, IC files, or any other path -> no overwrite);
  * the on-HF existence guard is present (SKIP instead of overwrite);
  * the staleness guard against a non-provenance extractor is present;
  * amlt YAML traps: description uses ' - ' not ': ', the block parses, and the
    python -c blocks carry only backslash-escaped double quotes.
"""
import re
from pathlib import Path

import yaml

MLC = Path(__file__).resolve().parents[1]
TEMPLATE = MLC / "amlt" / "sec4_w8extract.yaml.template"

SM_DEST = "sae_features_v3/slot_machine/llama/checkpoints"
MW_DEST = "sae_features_v3/mystery_wheel/llama/checkpoints"


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


def test_upload_targets_are_only_the_two_checkpoints_paths():
    env, _ = _env()
    # the env-declared destinations are exactly the two new checkpoints/ dirs
    assert env["MLC_SM_DEST"] == SM_DEST
    assert env["MLC_MW_DEST"] == MW_DEST
    text = TEMPLATE.read_text()
    # upload_file writes ONLY `dest + "/" + fname`; dest is one of the two env
    # knobs -> no other repo path is ever written.
    assert 'path_in_repo=path_in_repo' in text
    assert 'path_in_repo = dest + \\"/\\" + fname' in text
    # the job never writes to the dp dump, the IC namespace, or a bare
    # sae_features_v3 path literal (would risk clobbering existing data).
    assert "hidden_states_dp" not in text
    assert "investment_choice" not in text
    # both write dirs appear as READ-side env values only, each exactly once
    assert text.count(SM_DEST) == 1 and text.count(MW_DEST) == 1
    # the two filenames uploaded are the phase_a pair (npz + metadata json)
    assert 'phase_a_hidden_states.npz' in text and 'phase_a_metadata.json' in text


def test_existence_guard_refuses_overwrite():
    text = TEMPLATE.read_text()
    # pull the current file list and SKIP (no upload) if the target exists
    assert "list_repo_files(repo" in text
    assert "if path_in_repo in existing:" in text
    assert re.search(r'print\(\\"SKIP\\"', text), "no SKIP print in existence guard"
    # SKIP path must `continue` (not fall through to upload_file)
    guard = text[text.index("if path_in_repo in existing:"):]
    assert "continue" in guard[:200]


def test_staleness_guard_and_provenance_write():
    text = TEMPLATE.read_text()
    # aborts loudly if the fetched extractor lacks the additive provenance write
    assert "Additive provenance (W8)" in text
    assert "STALE_EXTRACTOR_NO_PROVENANCE" in text
    # fetches the three reused sae_v3_analysis scripts + the two catalogs
    for name in ("extract_llama_sm.py", "extract_llama_mw.py",
                 "prompt_reconstruction.py"):
        assert name in text
    assert "sae_v3_analysis/src/" in text
    assert "final_llama_20260315_062428.json" in text
    assert "llama_mysterywheel_c30_20260320_092707.json" in text
    # runs phase-A-only for both tasks (exactly two invocations in the command
    # body; the description prose mentions it once more)
    body = text[text.index("bash -c '"):]
    assert body.count("--phase-a-only") == 2


def test_python_c_blocks_escape_inner_quotes():
    """The `python -c "..."` arg uses double quotes as its delimiter (opener
    UNescaped, correct inside the outer `bash -c '...'` single quotes); every
    LITERAL quote inside the python payload must be backslash-escaped (\") or it
    would prematurely close the arg and break submission (same rule the sibling
    w8scan template follows)."""
    text = TEMPLATE.read_text()
    body = text[text.index("bash -c '"):]
    # at least the four python -c blocks (imports, fetch, catalogs, upload)
    openers = re.findall(r'python -c "', body)
    assert len(openers) >= 4, len(openers)
    # inner string literals are escaped, e.g. env access and repo id
    assert 'os.environ[\\"HF_TOKEN\\"]' in body
    assert 'os.environ[\\"MLC_HF_REPO\\"]' in body
    # the shape/upload payload uses escaped inner quotes throughout
    assert 'z[\\"hidden_states\\"]' in body
    assert 'repo_type=\\"dataset\\"' in body
