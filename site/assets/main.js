(() => {
  const D = window.PAPER_DATA;
  const NS = "http://www.w3.org/2000/svg";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const el = (tag, attrs = {}, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  };
  const txt = (parent, x, y, s, attrs = {}) => {
    const t = el("text", { x, y, ...attrs }, parent);
    t.textContent = s;
    return t;
  };

  // ---------- reveal on scroll ----------
  const onVisible = (node, fn, threshold = 0.25) => {
    if (!("IntersectionObserver" in window)) return fn();
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) { fn(); io.disconnect(); }
      });
    }, { threshold });
    io.observe(node);
  };
  document.querySelectorAll(".reveal").forEach((n) => onVisible(n, () => n.classList.add("in"), 0.15));

  // ---------- count-up ----------
  document.querySelectorAll("[data-count]").forEach((n) => {
    const target = parseFloat(n.dataset.count);
    const dec = parseInt(n.dataset.dec || "0", 10);
    onVisible(n, () => {
      if (reduce) { n.textContent = target.toFixed(dec); return; }
      const t0 = performance.now();
      const dur = 1400;
      const tick = (t) => {
        const p = Math.min(1, (t - t0) / dur);
        const e = 1 - Math.pow(1 - p, 3);
        n.textContent = (target * e).toFixed(dec);
        if (p < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
  });

  // Charts are laid out in real pixels (viewBox width = rendered width), so
  // text keeps its size on a phone. Each redraws when the width changes.
  const responsive = (fig, height, draw) => {
    const svg = fig.querySelector("svg");
    let lastW = 0;
    const render = () => {
      const W = Math.round(svg.parentNode.clientWidth - 0);
      if (!W || W === lastW) return;
      lastW = W;
      svg.innerHTML = "";
      const H = typeof height === "function" ? height(W) : height;
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      draw(svg, W, H);
    };
    render();
    window.addEventListener("resize", () => { clearTimeout(render.t); render.t = setTimeout(render, 150); });
    return render;
  };

  // ---------- Finding 1: dumbbell ----------
  (() => {
    const fig = document.getElementById("dumbbell");
    const narrow = (W) => W < 520;
    responsive(fig, (W) => (narrow(W) ? 330 : 300), (svg, W) => {
      const n = narrow(W);
      const L = n ? 112 : 170, R = W - (n ? 52 : 60), top = 20, row = n ? 48 : 44;
      const x = (v) => L + (v / 80) * (R - L);
      [0, 20, 40, 60, 80].forEach((v) => {
        el("line", { x1: x(v), x2: x(v), y1: top - 6, y2: top + row * 6 - 14, class: "axis" }, svg);
        txt(svg, x(v), top + row * 6 + 4, v + "%", { "text-anchor": "middle", "font-size": 12, opacity: 0.6 });
      });
      D.slotBankruptcy.forEach((d, i) => {
        const y = top + i * row + 10;
        const g = el("g", {}, svg);
        txt(g, L - 12, y + 5, d.model, { "text-anchor": "end", "font-size": n ? 12 : 14, "font-weight": 600 });
        const bar = el("line", { x1: x(d.fixed), x2: x(d.variable), y1: y, y2: y, stroke: "#d8453a", "stroke-width": 4, "stroke-linecap": "round", class: "grow-x", opacity: 0.45 }, g);
        bar.style.transitionDelay = i * 0.12 + "s";
        el("circle", { cx: x(d.fixed), cy: y, r: n ? 6 : 8, fill: "#5b8def" }, g);
        const dot = el("circle", { cx: x(d.variable), cy: y, r: n ? 7 : 9, fill: "#d8453a", class: "fade" }, g);
        dot.style.transitionDelay = 0.8 + i * 0.12 + "s";
        const lab = txt(g, x(d.variable) + 12, y + 5, d.variable.toFixed(1) + "%", { "font-size": n ? 12 : 13, "font-weight": 700, class: "fade" });
        lab.style.transitionDelay = 0.9 + i * 0.12 + "s";
      });
    });
    onVisible(fig, () => fig.classList.add("in"));
  })();

  // ---------- Finding 4: matched cap ----------
  (() => {
    const fig = document.getElementById("capchart");
    responsive(fig, 240, (svg, W) => {
      const base = 200, h = 160, maxV = 20, left = 44;
      const y = (v) => base - (v / maxV) * h;
      [0, 5, 10, 15, 20].forEach((v) => {
        el("line", { x1: left, x2: W - 10, y1: y(v), y2: y(v), class: "axis" }, svg);
        txt(svg, left - 8, y(v) + 4, v + "%", { "text-anchor": "end", "font-size": 11, opacity: 0.6 });
      });
      const slot = (W - left - 10) / 3;
      const bw = Math.min(28, slot / 4);
      D.matchedCap.forEach((d, i) => {
        const cx = left + slot * (i + 0.5);
        [["fixed", "#5b8def", -bw - 4], ["variable", "#d8453a", 4]].forEach(([k, c, off], j) => {
          const v = d[k];
          const r = el("rect", { x: cx + off, y: y(v), width: bw, height: Math.max(1, base - y(v)), rx: 4, fill: c, class: "grow-y" }, svg);
          r.style.transitionDelay = i * 0.2 + j * 0.1 + "s";
          const t = txt(svg, cx + off + bw / 2, y(v) - 8, (v < 1 && v > 0 ? v.toFixed(1) : v) + "%", { "text-anchor": "middle", "font-size": 12, "font-weight": 700, class: "fade" });
          t.style.transitionDelay = 0.7 + i * 0.2 + "s";
        });
        txt(svg, cx, base + 24, "cap $" + d.cap, { "text-anchor": "middle", "font-size": 13, "font-weight": 600 });
      });
    });
    onVisible(fig, () => fig.classList.add("in"));
  })();

  // ---------- Finding 5: choice ladder ----------
  (() => {
    const fig = document.getElementById("ladder");
    const host = fig.querySelector(".ladder-steps");
    const steps = D.choiceLadder.map((d, i) => {
      const s = document.createElement("div");
      s.className = "lstep";
      s.innerHTML = `<div class="lbar-wrap"><div class="lbar ${d.bankrupt < 10 ? "low" : ""}"><span>${d.bankrupt.toFixed(0)}%</span></div></div>
        <h4>${i + 1}. ${d.arm}</h4><p>bets again after a first loss: ${d.rebet}% · ${d.rounds} rounds · ${d.games} games</p>`;
      host.appendChild(s);
      return { s, bar: s.querySelector(".lbar"), v: d.bankrupt };
    });
    let timers = [];
    const play = () => {
      timers.forEach(clearTimeout);
      steps.forEach((st) => { st.s.classList.remove("lit"); st.bar.style.height = "0"; });
      steps.forEach((st, i) => {
        timers.push(setTimeout(() => {
          st.s.classList.add("lit");
          st.bar.style.height = Math.max(3, st.v) + "%";
        }, reduce ? 0 : 250 + i * 900));
      });
    };
    onVisible(fig, play);
    document.getElementById("ladderReplay").addEventListener("click", play);
  })();

  // ---------- Finding 3: meters ----------
  document.querySelectorAll(".meters").forEach((host) => {
    const rows = JSON.parse(host.dataset.meters);
    rows.forEach(([label, v, cls, shown]) => {
      const m = document.createElement("div");
      m.className = "meter";
      m.innerHTML = `<div class="meter-top"><span>${label}</span><b>${shown ? "~" + shown : "~" + v}%</b></div><div class="meter-track"><div class="meter-fill ${cls}"></div></div>`;
      host.appendChild(m);
      const fill = m.querySelector(".meter-fill");
      onVisible(host, () => { fill.style.width = v + "%"; });
    });
  });

  // ---------- moving target (schematic) ----------
  (() => {
    const fig = document.getElementById("moving");
    const svg = fig.querySelector("svg");
    const W = Math.max(320, Math.round(svg.parentNode.clientWidth) || 720), base = 190;
    svg.setAttribute("viewBox", `0 0 ${W} 220`);
    el("line", { x1: 40, x2: W - 20, y1: base, y2: base, class: "axis" }, svg);
    txt(svg, 40, 208, "rounds →", { "font-size": 11, opacity: 0.6 });
    const goal = el("line", { x1: 40, x2: W - 20, y1: 120, y2: 120, stroke: "#d8453a", "stroke-width": 2, "stroke-dasharray": "6 5" }, svg);
    const goalLab = txt(svg, W - 56, 112, "self-set goal", { "text-anchor": "end", "font-size": 12, fill: "#d8453a", "font-weight": 600 });
    const flag = el("path", { d: "", fill: "#d8453a" }, svg);
    const path = el("polyline", { points: "", fill: "none", stroke: "#e3a92f", "stroke-width": 3, "stroke-linejoin": "round" }, svg);
    const head = el("circle", { r: 6, fill: "#e3a92f" }, svg);
    const bust = txt(svg, 0, 0, "", { "font-size": 13, "font-weight": 700, fill: "#d8453a" });
    // A deterministic, illustrative walk in dollars: the balance climbs toward
    // the goal, the goal is raised each time it is nearly reached, and the
    // losses that follow take the balance to zero.
    const walk = [100, 108, 104, 112, 118, 122, 116, 124, 132, 128, 138, 146, 150, 142, 130, 136, 118, 104, 112, 90, 74, 82, 60, 44, 52, 30, 14, 0];
    const goalAt = (i) => (i < 5 ? 120 : i < 11 ? 150 : 180);
    const toY = (b) => base - b * 0.85;
    const toX = (k) => 40 + (k / (walk.length - 1)) * (W - 110);
    txt(svg, 34, toY(100) + 4, "$100", { "text-anchor": "end", "font-size": 11, opacity: 0.6 });
    el("line", { x1: 40, x2: W - 20, y1: toY(100), y2: toY(100), class: "axis", "stroke-dasharray": "2 4" }, svg);
    let i = 0, pts = [], running = false;
    const setGoal = (g) => {
      const gy = toY(g);
      goal.setAttribute("y1", gy); goal.setAttribute("y2", gy);
      goalLab.setAttribute("y", gy - 8);
      goalLab.textContent = `self-set goal: $${g}`;
      const fx = W - 24;
      flag.setAttribute("d", `M${fx},${gy} v-26 l-18,7 l18,7 z`);
    };
    const step = () => {
      if (i >= walk.length) {
        bust.setAttribute("x", toX(walk.length - 1) - 70);
        bust.setAttribute("y", base - 10);
        bust.textContent = "bankrupt";
        setTimeout(reset, 2600);
        return;
      }
      pts.push(`${toX(i)},${toY(walk[i])}`);
      path.setAttribute("points", pts.join(" "));
      head.setAttribute("cx", toX(i)); head.setAttribute("cy", toY(walk[i]));
      setGoal(goalAt(i));
      i++;
      setTimeout(step, 230);
    };
    const reset = () => {
      i = 0; pts = []; bust.textContent = "";
      if (running) step();
    };
    setGoal(120);
    if (reduce) {
      pts = walk.map((b, k) => `${toX(k)},${toY(b)}`);
      path.setAttribute("points", pts.join(" "));
      setGoal(180);
      return;
    }
    onVisible(fig, () => { running = true; step(); });
  })();

  // ---------- Finding 9: steering ----------
  (() => {
    const S = D.steering;
    const state = { model: "gemma", dir: "behaviour", dose: 0 };
    const svg = document.getElementById("steerPlot");
    const out = document.getElementById("ratioOut");
    const doseOut = document.getElementById("doseOut");
    const note = document.getElementById("steerNote");
    const chips = document.getElementById("chips");
    const slider = document.getElementById("dose");
    const X0 = 50, X1 = 440, Y0 = 220, Y1 = 20, maxY = 0.3;
    const x = (d) => X0 + ((d + 3) / 6) * (X1 - X0);
    const y = (v) => Y0 - (v / maxY) * (Y0 - Y1);

    const draw = () => {
      svg.innerHTML = "";
      const m = S[state.model];
      [0, 0.1, 0.2, 0.3].forEach((v) => {
        el("line", { x1: X0, x2: X1, y1: y(v), y2: y(v), class: "axis" }, svg);
        txt(svg, X0 - 8, y(v) + 4, v.toFixed(1), { "text-anchor": "end", "font-size": 11, opacity: 0.6 });
      });
      S.doses.forEach((d) => txt(svg, x(d), Y0 + 18, (d > 0 ? "+" : "") + d, { "text-anchor": "middle", "font-size": 11, opacity: 0.6 }));
      txt(svg, (X0 + X1) / 2, Y0 + 36, "dose (steps of 3% of the median hidden-state norm)", { "text-anchor": "middle", "font-size": 11, opacity: 0.7 });
      // random-direction band(s)
      Object.entries(m.band).forEach(([d, [lo, hi]]) => {
        el("rect", { x: x(+d) - 12, y: y(hi), width: 24, height: y(lo) - y(hi), fill: "#ffffff", opacity: 0.12, rx: 4 }, svg);
      });
      const other = state.dir === "behaviour" ? "readout" : "behaviour";
      el("polyline", { points: m[other].map((v, k) => `${x(S.doses[k])},${y(v)}`).join(" "), fill: "none", stroke: "#ffffff", "stroke-opacity": 0.25, "stroke-width": 2, "stroke-dasharray": "4 4" }, svg);
      el("polyline", { points: m[state.dir].map((v, k) => `${x(S.doses[k])},${y(v)}`).join(" "), fill: "none", stroke: "#e3a92f", "stroke-width": 3 }, svg);
      m[state.dir].forEach((v, k) => el("circle", { cx: x(S.doses[k]), cy: y(v), r: 4, fill: "#e3a92f" }, svg));
      const k = S.doses.indexOf(state.dose);
      const v = m[state.dir][k];
      el("circle", { cx: x(state.dose), cy: y(v), r: 10, fill: "none", stroke: "#fff", "stroke-width": 2 }, svg);
      txt(svg, X1, Y1 - 4, "shaded: random directions (mean ± 2 SD)", { "text-anchor": "end", "font-size": 10.5, opacity: 0.6 });
      out.textContent = v.toFixed(3);
      doseOut.textContent = (state.dose > 0 ? "+" : "") + state.dose;
      renderChips(v);
      note.textContent = state.dir === "behaviour"
        ? `${state.model === "gemma" ? "Gemma" : "LLaMA"} ${m.layers}: betting rises with the dose from ${m.behaviour[0].toFixed(3)} to ${m.behaviour[6].toFixed(3)}. Projecting the direction out lowers betting by ${Math.abs(m.removal).toFixed(3)}.`
        : `The sparse-feature readout direction on ${state.model === "gemma" ? "Gemma" : "LLaMA"}: the bet stays inside the random-direction band, and removing it leaves the bet at baseline. It reports the risk without moving it.`;
    };
    const renderChips = (v) => {
      const n = Math.round((v / maxY) * 18) + 1;
      while (chips.children.length < 19) {
        const c = document.createElement("div");
        c.className = "chip";
        c.style.bottom = chips.children.length * 7 + "px";
        chips.appendChild(c);
      }
      [...chips.children].forEach((c, k) => { c.style.opacity = k < n ? 1 : 0; });
    };
    const seg = (id, key) => {
      document.getElementById(id).addEventListener("click", (e) => {
        const b = e.target.closest("button"); if (!b) return;
        [...b.parentNode.children].forEach((c) => c.classList.toggle("on", c === b));
        state[key] = b.dataset.v; draw();
      });
    };
    seg("steerModel", "model"); seg("steerDir", "dir");
    slider.addEventListener("input", () => { state.dose = parseInt(slider.value, 10); draw(); });
    draw();
    // Sweep the dial once when the card comes into view.
    onVisible(document.getElementById("steer"), () => {
      if (reduce) return;
      const seq = [-3, -2, -1, 0, 1, 2, 3, 2, 1, 0];
      seq.forEach((d, k) => setTimeout(() => {
        if (slider.dataset.touched) return;
        slider.value = d; state.dose = d; draw();
      }, 400 + k * 420));
    }, 0.4);
    slider.addEventListener("pointerdown", () => { slider.dataset.touched = "1"; });
  })();

  // ---------- slot machine you can play ----------
  (() => {
    const SYM = ["◆", "●", "▲", "★", "♣"];
    const reels = [...document.querySelectorAll(".reel")];
    const bal = document.getElementById("bal"), roundsEl = document.getElementById("rounds"), last = document.getElementById("last");
    const msg = document.getElementById("slotMsg"), betRow = document.getElementById("betRow"), bet = document.getElementById("bet"), betOut = document.getElementById("betOut");
    const spinBtn = document.getElementById("spin"), stopBtn = document.getElementById("stop");
    const spark = document.querySelector("#spark polyline");
    let mode = "fixed", balance = 100, rounds = 0, hist = [100], over = false, busy = false;

    const render = () => {
      bal.textContent = "$" + balance;
      roundsEl.textContent = rounds;
      const maxB = Math.max(200, ...hist);
      spark.setAttribute("points", hist.map((b, k) => `${(k / Math.max(1, hist.length - 1)) * 300},${58 - (b / maxB) * 56}`).join(" "));
      spinBtn.disabled = over || busy || balance <= 0;
      stopBtn.disabled = over || busy;
      if (mode === "variable") {
        bet.max = Math.min(100, balance) || 5;
        if (+bet.value > balance) bet.value = Math.max(5, Math.floor(balance / 5) * 5);
        betOut.textContent = "$" + bet.value;
      }
    };
    const reset = () => {
      balance = 100; rounds = 0; hist = [100]; over = false; last.textContent = "–";
      msg.textContent = "Expected value per spin: −10% of the bet.";
      reels.forEach((r) => r.classList.remove("win"));
      render();
    };
    document.querySelector(".mode").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      mode = b.dataset.mode;
      document.querySelectorAll(".mode button").forEach((c) => { c.classList.toggle("on", c === b); c.setAttribute("aria-checked", c === b); });
      betRow.hidden = mode !== "variable";
      reset();
    });
    bet.addEventListener("input", () => { betOut.textContent = "$" + bet.value; });
    spinBtn.addEventListener("click", () => {
      const wager = Math.min(balance, mode === "fixed" ? 10 : +bet.value);
      if (wager <= 0 || over) return;
      busy = true; render();
      const win = Math.random() < 0.3;
      reels.forEach((r) => { r.classList.remove("win"); r.classList.add("spin"); });
      setTimeout(() => {
        const s = win ? SYM[Math.floor(Math.random() * SYM.length)] : null;
        reels.forEach((r, k) => {
          r.classList.remove("spin");
          r.firstElementChild.textContent = win ? s : SYM[(k * 2 + Math.floor(Math.random() * 5)) % SYM.length];
          if (win) r.classList.add("win");
        });
        if (!win && reels.every((r) => r.firstElementChild.textContent === reels[0].firstElementChild.textContent)) {
          reels[2].firstElementChild.textContent = SYM[(SYM.indexOf(reels[0].firstElementChild.textContent) + 1) % SYM.length];
        }
        balance += win ? wager * 2 : -wager;
        rounds += 1; hist.push(balance);
        last.textContent = win ? "+$" + wager * 2 : "−$" + wager;
        busy = false;
        if (balance <= 0) {
          over = true;
          msg.textContent = mode === "variable"
            ? `Bankrupt after ${rounds} rounds. LLaMA-3.1-8B ended 72% of its variable-betting games like this.`
            : `Bankrupt after ${rounds} rounds. Under fixed betting the models almost never got here (0–3.1%).`;
        } else {
          msg.textContent = win ? "A win. The odds have not changed." : "A loss. Stopping now is still the best move.";
        }
        render();
      }, reduce ? 0 : 650);
    });
    stopBtn.addEventListener("click", () => {
      over = true;
      const diff = balance - 100;
      msg.textContent = `You stopped with $${balance} (${diff >= 0 ? "+" : "−"}$${Math.abs(diff)}) after ${rounds} rounds.`;
      render();
    });
    document.getElementById("reset").addEventListener("click", reset);
    render();
  })();

  // ---------- BibTeX copy ----------
  document.getElementById("copyBib").addEventListener("click", async (e) => {
    try {
      await navigator.clipboard.writeText(document.getElementById("bib").textContent);
      e.target.textContent = "Copied";
      setTimeout(() => (e.target.textContent = "Copy"), 1500);
    } catch (_) { /* clipboard blocked: the text stays selectable */ }
  });
})();
