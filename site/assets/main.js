(() => {
  const D = window.PAPER_DATA;
  const G = window.GAMES;
  const NS = "http://www.w3.org/2000/svg";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const money = (v) => (v < 0 ? "−$" : "$") + Math.abs(v);

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

  // ---------- run once when scrolled into view ----------
  const onVisible = (node, fn, threshold = 0.25) => {
    if (!("IntersectionObserver" in window)) return fn();
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) { fn(); io.disconnect(); }
      });
    }, { threshold });
    io.observe(node);
  };

  // ---------- count-up (ends on the number written in the HTML) ----------
  const countUp = (n, from = 0) => {
    const target = parseFloat(n.dataset.count);
    const dec = parseInt(n.dataset.dec || "0", 10);
    if (reduce) { n.textContent = target.toFixed(dec); return; }
    const t0 = performance.now();
    const dur = 1100;
    const tick = (t) => {
      const p = Math.max(0, Math.min(1, (t - t0) / dur));
      n.textContent = (from + (target - from) * p).toFixed(dec);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };

  // =====================================================================
  // Scene 1: two recorded games, replayed round by round
  // =====================================================================
  (() => {
    const duel = $("#duel");
    const playBtn = $("#playGames");
    const scrub = $("#scrub");
    const scrubOut = $("#scrubOut");
    const tryBtn = $("#tryIt");
    const ROUND_MS = 900;
    const BAR_MAX = 150;
    const games = { fixed: G.fixed, variable: G.variable };
    const maxR = Math.max(G.fixed.steps.length, G.variable.steps.length);
    scrub.max = maxR;

    $("#gameSource").textContent =
      `Recorded game · ${G.variable.model} · prompt ${G.variable.promptCombo}, game #${G.variable.repetition}`;

    const machines = {};
    $$(".machine", duel).forEach((root) => {
      const arm = root.dataset.arm;
      machines[arm] = {
        arm, root,
        bubble: $(".bubble", root), round: $(".b-round", root), text: $(".b-text", root),
        reels: $$(".reel", root), flash: $(".flash", root), delta: $(".delta", root),
        stack: $(".stack", root), bar: $(".bal-bar i", root), mark: $(".start-mark", root),
        bal: $(".bal", root), betNow: $(".bet-now", root), end: $(".m-end", root),
        ctl: $(".play-ctl", root), betRow: $(".bet-row", root),
        betIn: $(".bet-row input", root), betOut: $(".bet-row output", root),
        spin: $(".spin", root), stop: $(".stop", root),
      };
    });

    // ---- drawing helpers shared by replay and free play ----
    const SYM = ["7", "$", "*"];
    const setReels = (m, win, seed) => {
      m.reels.forEach((r, k) => {
        const s = win ? "7" : SYM[(seed + k * (seed % 2 ? 1 : 2)) % 3];
        const span = r.firstElementChild;
        span.textContent = s;
        span.className = s === "7" ? "r7" : s === "$" ? "rd" : "";
      });
      if (!win && m.reels.every((r) => r.textContent === m.reels[0].textContent)) {
        const span = m.reels[2].firstElementChild;
        span.textContent = "*"; span.className = "";
      }
    };
    const setStack = (m, bet) => {
      const n = bet ? Math.min(20, Math.max(1, Math.round(bet / 5))) : 0;
      while (m.stack.children.length < 20) {
        const c = document.createElement("img");
        c.src = "assets/img/chip.png"; c.alt = ""; c.width = 35; c.height = 24;
        m.stack.appendChild(c);
      }
      const step = m.stack.clientWidth < 50 ? 5 : 7;
      [...m.stack.children].forEach((c, k) => {
        c.style.bottom = k * step + "px";
        c.style.display = k < n ? "block" : "none";
      });
    };
    const setBalance = (m, v, max) => {
      m.bal.textContent = money(v);
      m.bar.style.width = Math.min(100, (v / max) * 100) + "%";
      m.bar.classList.toggle("low", v < 30);
      m.mark.style.left = `calc(${(100 / max) * 100}% - 2px)`;
    };
    const say = (m, label, text, sys = false) => {
      m.round.textContent = label;
      m.text.textContent = text;
      m.bubble.classList.toggle("sys", sys);
      if (!reduce) { m.bubble.classList.remove("pop"); void m.bubble.offsetWidth; m.bubble.classList.add("pop"); }
    };
    const spinFx = (m, win, amount, seed, done) => {
      if (reduce) { setReels(m, win, seed); done(); return; }
      m.reels.forEach((r) => r.classList.add("spin"));
      setTimeout(() => {
        m.reels.forEach((r) => r.classList.remove("spin"));
        setReels(m, win, seed);
        m.flash.className = "flash " + (win ? "win" : "loss");
        m.delta.className = "delta";
        void m.delta.offsetWidth;
        m.delta.textContent = win ? "+" + money(amount) : money(-amount);
        m.delta.className = "delta show " + (win ? "win" : "loss");
        setTimeout(() => (m.flash.className = "flash"), 520);
        done();
      }, 380);
    };

    // ---- the recorded replay ----
    let r = 0, timer = null, mode = "replay", gen = 0;
    const endText = (g) =>
      g.outcome === "bankruptcy"
        ? `Bankrupt after ${g.rounds} rounds`
        : `Stopped after ${g.rounds} rounds with ${money(g.finalBalance)}`;

    const renderArm = (m, g, round, animate) => {
      const steps = g.steps;
      const k = Math.min(round, steps.length);
      if (k === 0) {
        say(m, "START", "Starts with $100.", true);
        setBalance(m, 100, BAR_MAX);
        setStack(m, 0);
        m.betNow.textContent = "";
        m.end.textContent = ""; m.end.className = "m-end";
        return;
      }
      const s = steps[k - 1];
      const fresh = round <= steps.length;
      if (!fresh && animate) return; // this game is already over
      const label = s.action === "stop" ? `ROUND ${s.round} · STOPS` : `ROUND ${s.round} · BETS ${money(s.bet)}`;
      say(m, label, s.quote);
      setStack(m, s.action === "bet" ? s.bet : 0);
      m.betNow.textContent = s.action === "bet" ? `BET ${money(s.bet)}` : "";
      const mine = gen;
      const finish = () => {
        if (mine !== gen || mode !== "replay") return; // a newer render or free play took over
        setBalance(m, s.balanceAfter, BAR_MAX);
        const over = k === steps.length;
        m.end.textContent = over ? endText(g) : "";
        m.end.className = "m-end" + (over ? (g.outcome === "bankruptcy" ? " bust" : " stop") : "");
      };
      if (animate && s.action === "bet") {
        setBalance(m, s.balanceBefore, BAR_MAX);
        spinFx(m, s.result === "W", s.result === "W" ? s.bet * 2 : s.bet, s.round, finish);
      } else {
        if (s.action === "bet") setReels(m, s.result === "W", s.round);
        finish();
      }
    };
    const render = (animate) => {
      gen += 1;
      scrub.value = r;
      scrubOut.textContent = r;
      for (const arm of ["fixed", "variable"]) renderArm(machines[arm], games[arm], r, animate);
      if (r === maxR) fillGrids();
    };
    const stopPlay = () => {
      clearInterval(timer); timer = null;
      playBtn.textContent = r >= maxR ? "▶ Replay" : "▶ Play";
    };
    const play = () => {
      if (timer) { stopPlay(); return; }
      if (reduce) { r = maxR; render(false); stopPlay(); return; }
      if (r >= maxR) { r = 0; render(false); }
      playBtn.textContent = "❚❚ Pause";
      const tick = () => {
        r += 1; render(true);
        if (r >= maxR) stopPlay();
      };
      tick();
      timer = setInterval(tick, ROUND_MS);
    };
    playBtn.addEventListener("click", play);
    scrub.addEventListener("input", () => { stopPlay(); r = parseInt(scrub.value, 10); render(false); stopPlay(); });

    // Keep the bubbles from jumping: reserve the height of the longest quote.
    const sizeBubbles = () => {
      if (mode !== "replay") return;
      let h = 0;
      for (const arm of ["fixed", "variable"]) {
        const m = machines[arm];
        m.bubble.style.minHeight = "0";
        for (const s of games[arm].steps) {
          m.round.textContent = "ROUND 13 · BETS $25";
          m.text.textContent = s.quote;
          h = Math.max(h, m.bubble.offsetHeight);
        }
      }
      for (const arm of ["fixed", "variable"]) machines[arm].bubble.style.minHeight = h + "px";
      render(false);
    };
    (document.fonts ? document.fonts.ready : Promise.resolve()).then(sizeBubbles);
    let rt;
    window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(sizeBubbles, 150); });

    // ---- dot grids: share of 1,600 games that went bankrupt ----
    let filled = false;
    const grids = $$(".grid-card");
    grids.forEach((card) => {
      const dots = $(".dots", card);
      for (let i = 0; i < 100; i++) {
        const d = document.createElement("span");
        d.className = "dot";
        // fill from the bottom row up, left to right
        d.style.gridRow = String(10 - Math.floor(i / 10));
        d.style.gridColumn = String((i % 10) + 1);
        d.appendChild(document.createElement("i"));
        dots.appendChild(d);
      }
    });
    // Fill the grids when they come into view, whether or not the replay was
    // watched, so the page never shows a placeholder 0.0% as if it were data.
    grids.forEach((card) => onVisible(card, fillGrids, 0.35));
    function fillGrids() {
      if (filled) return;
      filled = true;
      grids.forEach((card) => {
        const pct = parseFloat(card.dataset.pct);
        const cells = $$(".dot i", card);
        cells.forEach((c, i) => {
          const frac = Math.max(0, Math.min(1, pct - i));
          if (!frac) return;
          const set = () => (c.style.height = frac * 100 + "%");
          if (reduce) set(); else setTimeout(set, 200 + i * 14);
        });
        countUp($("[data-count]", card));
      });
    }

    // ---- free play: the same machine, your own spins ----
    const free = {};
    const freeReset = (m) => {
      const st = (free[m.arm] = { balance: 100, rounds: 0, peak: 100, over: false, busy: false });
      m.betRow.hidden = m.arm !== "variable";
      m.betIn.value = 10;
      say(m, "YOUR TURN", "Spin, or stop and keep the $100.", true);
      m.end.textContent = ""; m.end.className = "m-end";
      freeRender(m, st);
    };
    const freeRender = (m, st) => {
      const bet = m.arm === "fixed" ? 10 : +m.betIn.value;
      if (m.arm === "variable") {
        m.betIn.max = Math.max(5, st.balance);
        if (+m.betIn.value > st.balance) m.betIn.value = Math.max(5, Math.floor(st.balance / 5) * 5);
        m.betOut.textContent = money(+m.betIn.value);
      }
      setBalance(m, st.balance, Math.max(200, st.peak));
      setStack(m, st.over ? 0 : Math.min(bet, st.balance));
      m.betNow.textContent = st.over ? "" : `BET ${money(Math.min(bet, st.balance))}`;
      m.spin.disabled = st.over || st.busy || st.balance <= 0;
      m.stop.disabled = st.over || st.busy;
    };
    Object.values(machines).forEach((m) => {
      m.betIn.addEventListener("input", () => freeRender(m, free[m.arm]));
      m.spin.addEventListener("click", () => {
        const st = free[m.arm];
        const wager = Math.min(st.balance, m.arm === "fixed" ? 10 : +m.betIn.value);
        if (wager <= 0 || st.over) return;
        st.busy = true; freeRender(m, st);
        const win = Math.random() < 0.3;
        spinFx(m, win, win ? wager * 2 : wager, st.rounds + 1, () => {
          st.balance += win ? wager * 2 : -wager;
          st.rounds += 1; st.peak = Math.max(st.peak, st.balance); st.busy = false;
          if (st.balance <= 0) {
            st.over = true;
            say(m, `ROUND ${st.rounds}`, "Broke.", true);
            m.end.textContent = `Bankrupt after ${st.rounds} rounds`; m.end.className = "m-end bust";
          } else {
            say(m, `ROUND ${st.rounds}`, win ? "A win. The odds have not changed." : "A loss. Stopping is still the best move.", true);
          }
          freeRender(m, st);
        });
      });
      m.stop.addEventListener("click", () => {
        const st = free[m.arm];
        st.over = true;
        say(m, "STOP", `You keep ${money(st.balance)}.`, true);
        m.end.textContent = `Stopped after ${st.rounds} rounds with ${money(st.balance)}`; m.end.className = "m-end stop";
        freeRender(m, st);
      });
    });
    tryBtn.addEventListener("click", () => {
      stopPlay();
      mode = mode === "replay" ? "free" : "replay";
      gen += 1;
      const isFree = mode === "free";
      tryBtn.setAttribute("aria-pressed", String(isFree));
      tryBtn.textContent = isFree ? "Back to the recorded games" : "Try it yourself";
      $(".replay-ctl").style.visibility = isFree ? "hidden" : "visible";
      $("#gameSource").style.visibility = isFree ? "hidden" : "visible";
      Object.values(machines).forEach((m) => {
        m.ctl.hidden = !isFree;
        if (isFree) freeReset(m);
      });
      if (!isFree) render(false);
    });

    render(false);
    onVisible(duel, () => { if (mode === "replay" && r === 0) play(); }, 0.45);
  })();

  // =====================================================================
  // Scene 2a: the rule ladder
  // =====================================================================
  (() => {
    const fig = $("#ladder");
    const list = $("#ladderSteps");
    const num = $("#ladderNum");
    const fill = $("#ladderFill");
    const signV = $(".bs-v", fig);
    const signCap = $(".bs-cap", fig);
    const labels = [
      "The game sets $70",
      "The model names its bet once",
      "It may change the bet every round",
      "…and the cap rises to $100",
    ];
    const sign = [
      ["$70", "set by the game", false],
      ["$?", "named once", false],
      ["$?", "every round · max $70", true],
      ["$?", "every round · max $100", true],
    ];
    const items = D.choiceLadder.map((d, i) => {
      const li = document.createElement("li");
      li.innerHTML = `<b>${d.bankrupt.toFixed(0)}%</b>${labels[i]}`;
      list.appendChild(li);
      return li;
    });
    let timers = [], shown = 0;
    const show = (i) => {
      const v = D.choiceLadder[i].bankrupt;
      items.forEach((li, k) => { li.classList.toggle("lit", k <= i); li.classList.toggle("now", k === i); });
      signV.textContent = sign[i][0];
      signV.classList.toggle("shuffle", sign[i][2]);
      signCap.textContent = sign[i][1];
      num.dataset.count = v; num.dataset.dec = "0";
      countUp(num, shown);
      shown = v;
      fill.style.width = v + "%";
      fill.classList.toggle("hot", v > 50);
      num.parentNode.classList.toggle("hot", v > 50);
    };
    const play = () => {
      timers.forEach(clearTimeout); timers = [];
      if (reduce) { show(D.choiceLadder.length - 1); items.forEach((li) => li.classList.add("lit")); return; }
      shown = 0; fill.style.width = "0";
      D.choiceLadder.forEach((_, i) => timers.push(setTimeout(() => show(i), 300 + i * 1700)));
    };
    onVisible(fig, play, 0.4);
    $("#ladderReplay").addEventListener("click", play);
  })();

  // =====================================================================
  // Scene 2b: the moving goal (illustration, not a recorded game)
  // =====================================================================
  (() => {
    const host = $("#goal .stairs-inner");
    const N = 6;
    let steps = [], robot, flag, t = [], geo;
    const build = () => {
      host.innerHTML = "";
      const W = host.clientWidth, H = host.clientHeight;
      // column width from the box; each coin stack adds half a column of height
      const col = Math.floor(Math.min((W - 40) / N, (H - 16) / 5.4, 64));
      const coinH = Math.round((col * 28) / 32);
      const rise = Math.round(col * 0.5);
      const x0 = Math.round((W - col * N) / 2);
      geo = { col, rise, coinH, x0, w: W };
      steps = [];
      for (let i = 0; i < N; i++) {
        const s = document.createElement("div");
        s.className = "step";
        s.style.left = x0 + i * col + "px";
        s.style.width = col + "px";
        s.style.height = i * rise + coinH + "px";
        for (let k = 0; k <= i; k++) {
          const c = document.createElement("img");
          c.src = "assets/img/coins.png"; c.alt = ""; c.width = 32; c.height = 28;
          c.style.bottom = k * rise + "px";
          s.appendChild(c);
        }
        host.appendChild(s);
        steps.push(s);
      }
      robot = document.createElement("img");
      robot.src = "assets/img/robot.png"; robot.alt = ""; robot.className = "climber";
      robot.style.width = col + "px";
      flag = document.createElement("img");
      flag.src = "assets/img/flag.png"; flag.alt = ""; flag.className = "pole";
      flag.style.width = Math.round(col * 0.62) + "px";
      host.append(robot, flag);
    };
    const place = (node, i, w) => {
      const { col, rise, coinH, x0 } = geo;
      node.style.left = x0 + i * col + (col - w) / 2 + "px";
      node.style.bottom = i * rise + coinH - Math.round(col * 0.12) + "px";
    };
    const setFlag = (i) => {
      steps.forEach((s, k) => s.classList.toggle("hidden", k > i));
      place(flag, i, Math.round(geo.col * 0.62));
    };
    const setRobot = (i) => place(robot, i, geo.col);
    // climb: robot steps up; each time it is one step below the flag, the flag moves two steps higher
    const seq = [["f", 2], ["r", 0], ["r", 1], ["f", 4], ["r", 2], ["r", 3], ["f", 5], ["r", 4]];
    const run = () => {
      t.forEach(clearTimeout); t = [];
      if (reduce) { setFlag(5); setRobot(4); return; }
      seq.forEach(([k, i], n) => t.push(setTimeout(() => (k === "f" ? setFlag(i) : setRobot(i)), n * 750)));
      t.push(setTimeout(run, seq.length * 750 + 1800));
    };
    const init = () => { build(); setFlag(2); setRobot(0); };
    init();
    onVisible($("#goal"), run, 0.3);
    let rt;
    window.addEventListener("resize", () => {
      clearTimeout(rt);
      rt = setTimeout(() => { if (host.clientWidth !== geo.w) { init(); run(); } }, 200);
    });
  })();

  // =====================================================================
  // Scene 3: the steering dial (Figure 4 dose ladders, kept exactly)
  // =====================================================================
  (() => {
    const S = D.steering;
    const state = { model: "gemma", dir: "behaviour", dose: 0 };
    const svg = $("#steerPlot");
    const out = $("#ratioOut");
    const doseOut = $("#doseOut");
    const note = $("#steerNote");
    const chips = $("#chips");
    const knob = $("#knob");
    const slider = $("#dose");
    const maxY = 0.3;
    const name = () => (state.model === "gemma" ? "Gemma" : "LLaMA");

    const draw = () => {
      const W = Math.max(280, Math.round(svg.parentNode.clientWidth));
      const H = 220;
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      svg.innerHTML = "";
      const X0 = 40, X1 = W - 12, Y0 = H - 42, Y1 = 14;
      const x = (d) => X0 + ((d + 3) / 6) * (X1 - X0);
      const y = (v) => Y0 - (v / maxY) * (Y0 - Y1);
      const m = S[state.model];
      [0, 0.1, 0.2, 0.3].forEach((v) => {
        el("rect", { x: X0, y: Math.round(y(v)), width: X1 - X0, height: 2, fill: "#1a2a4b" }, svg);
        txt(svg, X0 - 8, y(v) + 4, v.toFixed(1), { "text-anchor": "end", "font-size": 12 });
      });
      S.doses.forEach((d) => txt(svg, x(d), Y0 + 20, (d > 0 ? "+" : d < 0 ? "−" : "") + Math.abs(d), { "text-anchor": "middle", "font-size": 12 }));
      txt(svg, (X0 + X1) / 2, Y0 + 38, "dose (steps of 3% of the hidden state's size)", { "text-anchor": "middle", "font-size": 12 });
      Object.entries(m.band).forEach(([d, [lo, hi]]) => {
        el("rect", { x: x(+d) - 12, y: y(hi), width: 24, height: y(lo) - y(hi), fill: "#b3bdd1", opacity: 0.18, "shape-rendering": "crispEdges" }, svg);
      });
      txt(svg, X1, Y1 - 2, "shaded: random directions", { "text-anchor": "end", "font-size": 12 });
      const other = state.dir === "behaviour" ? "readout" : "behaviour";
      el("polyline", { points: m[other].map((v, k) => `${x(S.doses[k])},${y(v)}`).join(" "), fill: "none", stroke: "#b3bdd1", "stroke-opacity": 0.35, "stroke-width": 2, "stroke-dasharray": "4 4" }, svg);
      el("polyline", { points: m[state.dir].map((v, k) => `${x(S.doses[k])},${y(v)}`).join(" "), fill: "none", stroke: "#f3c14b", "stroke-width": 3 }, svg);
      m[state.dir].forEach((v, k) => el("rect", { x: x(S.doses[k]) - 4, y: y(v) - 4, width: 8, height: 8, fill: "#f3c14b", "shape-rendering": "crispEdges" }, svg));
      const k = S.doses.indexOf(state.dose);
      const v = m[state.dir][k];
      el("rect", { x: x(state.dose) - 9, y: y(v) - 9, width: 18, height: 18, fill: "none", stroke: "#f4efe3", "stroke-width": 3, "shape-rendering": "crispEdges" }, svg);
      out.textContent = v.toFixed(3);
      doseOut.textContent = (state.dose > 0 ? "+" : state.dose < 0 ? "−" : "") + Math.abs(state.dose);
      knob.style.transform = `rotate(${state.dose * 40}deg)`;
      renderChips(v);
      note.textContent = state.dir === "behaviour"
        ? `${name()}: ${m.behaviour[0].toFixed(3)} at −3, ${m.behaviour[6].toFixed(3)} at +3. Removing the direction lowers betting by ${Math.abs(m.removal).toFixed(3)}.`
        : `${name()}: the readout direction stays inside the random band. It reads risk but does not move the bet.`;
    };
    const renderChips = (v) => {
      const n = Math.round((v / maxY) * 14) + 1;
      while (chips.children.length < 15) {
        const c = document.createElement("img");
        c.src = "assets/img/chip.png"; c.alt = ""; c.width = 35; c.height = 24;
        chips.appendChild(c);
      }
      const step = chips.clientWidth < 70 ? 9 : 11;
      [...chips.children].forEach((c, k) => { c.style.bottom = k * step + "px"; c.style.opacity = k < n ? 1 : 0; });
    };
    const seg = (id, key) => {
      $(id).addEventListener("click", (e) => {
        const b = e.target.closest("button"); if (!b) return;
        [...b.parentNode.children].forEach((c) => { c.classList.toggle("on", c === b); c.setAttribute("aria-pressed", String(c === b)); });
        state[key] = b.dataset.v; draw();
      });
    };
    seg("#steerModel", "model"); seg("#steerDir", "dir");
    slider.addEventListener("input", () => { slider.dataset.touched = "1"; state.dose = parseInt(slider.value, 10); draw(); });
    draw();
    let rt;
    window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(draw, 150); });
    // Sweep the dial once when it comes into view.
    onVisible($("#steer"), () => {
      if (reduce) return;
      [-1, -2, -3, -2, -1, 0, 1, 2, 3, 2, 1, 0].forEach((d, k) => setTimeout(() => {
        if (slider.dataset.touched) return;
        slider.value = d; state.dose = d; draw();
      }, 400 + k * 380));
    }, 0.4);
  })();

  // ---------- BibTeX copy ----------
  $("#copyBib").addEventListener("click", async (e) => {
    try {
      await navigator.clipboard.writeText($("#bib").textContent);
      e.target.textContent = "Copied";
      setTimeout(() => (e.target.textContent = "Copy"), 1500);
    } catch (_) { /* clipboard blocked: the text stays selectable */ }
  });
})();
