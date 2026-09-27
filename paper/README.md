# paper/ — code and data behind the paper's figures and tables

The LaTeX source of the NeurIPS 2026 paper lives in its own repo (`LLM_Addiction_NMT_KOR`, also the
Overleaf project). Everything in that repo that LaTeX does not read was moved here on 2026-09-27, so
the Overleaf project holds only what builds the paper and the poster. The paper repo's tag
`pre-cleanup` marks its last commit before the move; this repo's tag `pre-paper-import` marks the
commit before the files arrived.

| Here | What it is | Was in the paper repo at |
|---|---|---|
| `scripts/` | Figure generators (`scripts/figures/`), table generators (`scripts/tables/`), `build_figure_data.py` | `scripts/` |
| `paper_data/` | The values each body figure and table plots, as JSON, plus generated table `.tex` | `paper_data/` |
| `paper_index/` | One provenance manifest per figure or table (source files, schema, load invariant) | `paper_index/` |
| `PAPER_ASSET_MAP.md` | Every printed figure and table → HF file → generating code | `PAPER_ASSET_MAP.md` |
| `NEURIPS_CANONICAL_INDEX.md` | Index of the NeurIPS source and its canonical numbers | `NEURIPS_CANONICAL_INDEX.md` |
| `generate_paper_figures.py`, `build_overview_figure.py`, `regenerate_fig2_fig4.py` | Older top-level figure scripts | repo root |
| `notes/` | May 2026 planning notes and the interim report | `docs/` |
| `images_unused/` | 101 graphics from earlier drafts that the paper no longer includes | `images/` |

## Running a script

The scripts locate the paper through their own path (`Path(__file__).parents[k] / "images"`,
`… / "neurips_content_en"`), exactly as when they sat in the paper repo. Link this folder to a paper
checkout once, then run them from here:

```bash
./link_paper_repo.sh            # expects the paper repo at ../../LLM_Addiction_NMT_KOR
python scripts/figures/fig02_slot_machine.py
```

Figures are written into the paper repo's `images/`. Two files that scripts read stayed there for
that reason: `images/fig_cross_context_write_values.json` (read by `fig04_causal_battery.py`) and
`images/fig2_combined.pdf` (the submitted Figure 2, compared against by `fig02_slot_machine.py`).

The submitted version of the paper is commit `dd2d229` of the paper repo.
