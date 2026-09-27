# Table — `tab:appendix-band-readout`

**Paper location**
Appendix, `neurips_content_en/_appendix_band_readout_table.tex:5`, \input at `appendix.tex:594`

**What the experiment asks**
The causal experiments write into a band of layers; can the same behaviour still be read out at every layer inside that band, or only at the one layer the paper reports?

**HF path(s) of the raw data**
`sae_v3_analysis/results/table1_groupkfold_band_{gemma,llama}.json`, plus `table1_groupkfold_L22.json` for the reference column.

**Code that turns raw data into the printed values**
Upstream generator: HF `sae_v3_analysis/scripts/build_appendix_band_readout_table.py` (wired in).
Regeneration harness: `scripts/tables/appendix_neural_tables.py` -> `paper_data/tables/appendix/A03_band_readout.tex`.

**Corpus vintage**
Canonical. **Band-definition mismatch (recorded as DEFECT 2 by the regeneration script):** this table evaluates the LLaMA IC and MW rows at L14-19, but the paper's own causal protocol names L12-17 (IC) and L16-21 (MW) as those tasks' write bands. The rows re-emitted at the protocol layers are in `paper_data/tables/appendix/A03b_band_readout_protocol_layers.tex`; the differences are small (IC I_BA -0.0021, IC I_EC -0.0018, MW I_LC -0.0087, MW I_BA +0.0004). A03b is a PROPOSED fragment, not \input by any .tex. The printed caption now discloses the mismatch itself, in a `\rev{}` sentence: "The LLaMA rows are all evaluated over L14--19. Two LLaMA tasks are steered over slightly different bands, L12--17 for investment choice and L16--21 for the mystery wheel, and re-running those rows over their own bands moves every value by less than 0.01." Confirmed 2026-08-27 against A03b, whose deltas are −0.002, −0.002, −0.009 and 0.000. The slice is still the one the causal protocol does not name, but it is no longer silent. No `DEPRECATION_WARNING.md` applies.

**Reproduction status**
VERIFIED, 2026-08-25. 91 numeric values in the printed tabular body vs the regenerated fragment; max deviation 0.000. The band-vs-protocol layer mismatch above is a definitional defect, not a numeric one.
