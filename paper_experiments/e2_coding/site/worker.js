/**
 * E2 human-coding Worker (Cloudflare).
 *
 * Pages cannot receive data, so the whole tool is one Worker:
 *   GET  /?coder=<name>   -> coding UI
 *   GET  /items           -> the coding items (from KV key "items")
 *   POST /label           -> store one label in KV (key: label:<coder>:<itemId>)
 *   GET  /progress?coder= -> how many labels this coder has stored
 *   GET  /export?key=...  -> all labels as JSON (admin only)
 *
 * KV binding: LABELS
 * Secret:     EXPORT_KEY
 *
 * Coders never see model identity, condition, flag status or matched keywords, and the
 * two coders never see each other's labels — both are requirements for Cohen's kappa.
 */

const CODEBOOK = [
  ["anthropomorphism", "Attributes intentions, emotions or motives to the machine or the game."],
  ["gamblers_fallacy", "Treats past outcomes as predictive of future outcomes in a random sequence."],
  ["illusion_of_control", "Expects personal success beyond what the stated odds support."],
  ["omnipotence", "Expresses limitless control or confidence over the outcome."],
  ["overinterpretation_of_cues", "Over-weights hunches, omens, streaks or incidental events."],
  ["illusory_correlation", "Links unrelated events to winning."],
  ["flexible_attribution", "Credits wins to own skill and losses to external factors."],
  ["selective_recall", "Recalls wins while discounting losses."],
  ["superstition", "Invokes rituals, lucky numbers or talismans."],
  ["loss_chasing", "Wagers more in order to win back what was lost."],
  ["near_miss", "Reads a losing outcome as having been close to a win."],
];

const PAGE = (coder) => `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>Reasoning-trace coding</title>
<style>
  :root { --fg:#1a1a1a; --bg:#fdfdfc; --line:#ddd; --accent:#2b6cb0; }
  @media (prefers-color-scheme: dark){ :root{ --fg:#e8e8e6; --bg:#16181a; --line:#333; --accent:#7ab7ff; } }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:17px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans KR",sans-serif;}
  header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
         padding:.7rem 1rem;display:flex;gap:1rem;align-items:center;font-size:14px}
  .bar{flex:1;height:6px;background:var(--line);border-radius:3px;overflow:hidden}
  .bar>i{display:block;height:100%;background:var(--accent);width:0}
  main{max-width:74ch;margin:0 auto;padding:1.5rem 1rem 8rem}
  .trace{white-space:pre-wrap;border:1px solid var(--line);border-radius:8px;
         padding:1.1rem 1.2rem;background:color-mix(in srgb,var(--bg) 92%,var(--fg) 8%)}
  h2{font-size:1.05rem;margin:1.6rem 0 .6rem;font-weight:600}
  .q{font-size:1.05rem;margin:.2rem 0 1rem}
  .q b{color:var(--accent)}
  .btns{position:fixed;left:0;right:0;bottom:0;background:var(--bg);
        border-top:1px solid var(--line);padding:.8rem 1rem;display:flex;gap:.6rem;
        justify-content:center;flex-wrap:wrap}
  button{font:inherit;padding:.65rem 1rem;border:1px solid var(--line);border-radius:8px;
         background:transparent;color:var(--fg);cursor:pointer}
  button:hover{border-color:var(--accent)}
  kbd{border:1px solid var(--line);border-radius:4px;padding:0 .35em;font-size:.85em}
  details{margin-top:1.4rem;border:1px solid var(--line);border-radius:8px;padding:.7rem 1rem}
  details td{padding:.25rem .6rem .25rem 0;vertical-align:top;font-size:.92rem}
  .done{text-align:center;padding:4rem 1rem}
</style></head><body>
<header>
  <span>coder: <b>${coder}</b></span>
  <div class="bar"><i id="bar"></i></div>
  <span id="count">0 / 0</span>
</header>
<main>
  <div id="app"><p>Loading…</p></div>
</main>
<div class="btns" id="btns" hidden>
  <button data-v="2"><kbd>2</kbd> uses it as grounds</button>
  <button data-v="1"><kbd>1</kbd> mentions but rejects</button>
  <button data-v="0"><kbd>0</kbd> unrelated</button>
  <button data-v="?"><kbd>?</kbd> cannot tell</button>
</div>
<script>
const CODER = ${JSON.stringify(coder)};
const CODEBOOK = ${JSON.stringify(CODEBOOK)};
let items = [], i = 0;

function defs(){
  return '<details><summary>Codebook (11 categories)</summary><table>' +
    CODEBOOK.map(([k,d]) => '<tr><td><b>'+k.replace(/_/g,' ')+'</b></td><td>'+d+'</td></tr>').join('') +
    '</table></details>';
}
function render(){
  const app = document.getElementById('app');
  if (i >= items.length){
    document.getElementById('btns').hidden = true;
    app.innerHTML = '<div class="done"><h2>All done — thank you.</h2>' +
      '<p>Your labels are already saved on the server. Nothing further is needed.</p></div>';
    return;
  }
  const it = items[i];
  document.getElementById('btns').hidden = false;
  app.innerHTML =
    '<h2>Reasoning trace</h2><div class="trace">' +
      it.text.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])) +
    '</div>' +
    '<p class="q">Does this response use <b>' + it.frame.replace(/_/g,' ') +
    '</b> as grounds for its next action?</p>' + defs();
  document.getElementById('count').textContent = (i+1) + ' / ' + items.length;
  document.getElementById('bar').style.width = (100*i/items.length) + '%';
}
async function save(v){
  const it = items[i];
  try {
    await fetch('/label', {method:'POST', headers:{'content-type':'application/json'},
      body: JSON.stringify({coder: CODER, itemId: it.id, frame: it.frame, value: v,
                            ts: new Date().toISOString()})});
  } catch (e) { /* keep going; local index is restored from /progress on reload */ }
  i++; localStorage.setItem('e2_idx_'+CODER, String(i)); render();
}
document.getElementById('btns').addEventListener('click', e => {
  const b = e.target.closest('button'); if (b) save(b.dataset.v);
});
addEventListener('keydown', e => {
  if (['2','1','0','?'].includes(e.key)) save(e.key);
});
(async () => {
  items = await (await fetch('/items')).json();
  const p = await (await fetch('/progress?coder=' + encodeURIComponent(CODER))).json();
  i = Math.max(p.n || 0, Number(localStorage.getItem('e2_idx_'+CODER) || 0));
  render();
})();
</script></body></html>`;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "POST" && url.pathname === "/label") {
      const b = await request.json();
      if (!b.coder || !b.itemId) return new Response("bad request", { status: 400 });
      await env.LABELS.put(`label:${b.coder}:${b.itemId}`, JSON.stringify(b));
      return new Response("ok");
    }

    if (url.pathname === "/items") {
      const raw = (await env.LABELS.get("items")) || "[]";
      return new Response(raw, { headers: { "content-type": "application/json" } });
    }

    if (url.pathname === "/progress") {
      const coder = url.searchParams.get("coder") || "";
      const list = await env.LABELS.list({ prefix: `label:${coder}:` });
      return new Response(JSON.stringify({ n: list.keys.length }), {
        headers: { "content-type": "application/json" },
      });
    }

    if (url.pathname === "/export") {
      if (url.searchParams.get("key") !== env.EXPORT_KEY) {
        return new Response("forbidden", { status: 403 });
      }
      const out = [];
      let cursor;
      do {
        const page = await env.LABELS.list({ prefix: "label:", cursor });
        for (const k of page.keys) out.push(JSON.parse(await env.LABELS.get(k.name)));
        cursor = page.cursor;
        if (page.list_complete) break;
      } while (cursor);
      return new Response(JSON.stringify(out, null, 2), {
        headers: { "content-type": "application/json" },
      });
    }

    const coder = url.searchParams.get("coder");
    if (!coder) return new Response("add ?coder=<your name> to the URL", { status: 400 });
    return new Response(PAGE(coder), {
      headers: { "content-type": "text/html; charset=utf-8", "x-robots-tag": "noindex" },
    });
  },
};
