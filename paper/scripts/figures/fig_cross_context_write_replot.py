"""Redraw the appendix alignment bars and condition-writability ladders from the saved values.

``fig_cross_context_write.py`` recomputes both figures from the gated rollouts on the Hugging Face
dataset.  This wrapper draws the same two figures from ``images/fig_cross_context_write_values.json``,
which that script writes, so a style change needs no rollout access.  No number is recomputed.
"""
import inspect
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_cross_context_write as X  # noqa: E402

VALUES = json.loads((HERE.parents[1] / "images" / "fig_cross_context_write_values.json").read_text())

# ladders: the per-condition summaries were saved verbatim
X.summarize = lambda model: VALUES["ladders"][model]

# alignment bars: swap the rollout computation for the saved cosines, keep the drawing code as is
src = inspect.getsource(X.fig_alignment_bars)
start = src.index("    axis_files = {")
end = src.index("    bk_cos = {")
src = (src[:start] + "    beh_cos = dict(VALUES['alignment']['behavioural_L16_21_mean'])\n"
       + "    pairs = [('IC', 'SM'), ('SM', 'MW'), ('IC', 'MW')]\n" + src[end:])
ns = dict(vars(X)); ns["VALUES"] = VALUES
exec(compile(src, "fig_alignment_bars_replot", "exec"), ns)

if __name__ == "__main__":
    X.fig_ladders(panel_letter=None, outname="fig_xctx_ladders_solo.pdf")
    beh, bk = ns["fig_alignment_bars"]()
    assert beh == VALUES["alignment"]["behavioural_L16_21_mean"], beh
    print("redrew fig_xctx_ladders_solo.pdf and fig_axis_alignment.pdf from saved values")
