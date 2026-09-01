/* token_journey/js/app.js
 * Main controller: current-step state, camera/narrative/real-panel sync,
 * the right-side step rail, autoplay, and keyboard navigation.
 */
(function () {
  "use strict";
  const TJ = (window.TJ = window.TJ || {});

  const state = {
    index: 0,
    playing: false,
    timer: null,
    attention: "gqa",
    positional: "rope",
    steps: null,
  };

  function variantKey() {
    return state.attention === "mla" ? "mla_rope" : `gqa_${state.positional}`;
  }

  function updateVariantButtons() {
    document.querySelectorAll("#controls .ctrl-group").forEach((group) => {
      const key = group.dataset.key;
      group.querySelectorAll("button").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.value === state[key]);
      });
    });
  }

  function switchVariant(key) {
    stopAutoplay();
    TJ.store.set(key);
    const data = TJ.store.data();
    state.steps = TJ.buildSteps(data);
    TJ.diagram.mount("#diagram", data);
    TJ.camera.init("#diagram");
    updateVariantButtons();
    buildRail();
    goToStep(Math.min(state.index, state.steps.length - 1));
  }

  function chapterGroups() {
    const groups = [];
    for (const step of state.steps) {
      let g = groups[groups.length - 1];
      if (!g || g.chapter !== step.chapter) {
        g = { chapter: step.chapter, steps: [] };
        groups.push(g);
      }
      g.steps.push(step);
    }
    return groups;
  }

  function buildRail() {
    const rail = document.getElementById("rail-steps");
    const groups = chapterGroups();
    let globalIndex = 0;
    const html = [];
    for (const g of groups) {
      html.push(`<div class="rail-chapter">${g.chapter}</div>`);
      for (const step of g.steps) {
        html.push(`<button class="rail-step" data-index="${globalIndex}"><span class="rail-num">${globalIndex + 1}</span><span class="rail-title">${step.title}</span></button>`);
        globalIndex++;
      }
    }
    rail.innerHTML = html.join("");
    rail.querySelectorAll(".rail-step").forEach((btn) => {
      btn.addEventListener("click", () => {
        stopAutoplay();
        goToStep(parseInt(btn.dataset.index, 10));
      });
    });
  }

  function renderNarrative(step) {
    const card = document.querySelector(".narrative-card");
    card.classList.remove("step-enter");
    document.getElementById("step-chapter").textContent = step.chapter;
    document.getElementById("step-title").textContent = step.title;
    document.getElementById("step-narrative").textContent = step.narrative;

    const formulaEl = document.getElementById("step-formula");
    if (step.formula) {
      formulaEl.style.display = "";
      katex.render(step.formula, formulaEl, { throwOnError: false, displayMode: true });
    } else {
      formulaEl.style.display = "none";
      formulaEl.innerHTML = "";
    }

    const realEl = document.getElementById("step-real");
    if (step.real) {
      realEl.style.display = "";
      TJ.renderToken.renderReal(step.real, realEl);
    } else {
      realEl.style.display = "none";
      realEl.innerHTML = "";
    }

    TJ.matrixViz.stopAll();
    const panel = document.getElementById("deepdive-panel");
    const btn = document.getElementById("deepdive-btn");
    panel.style.display = "none";
    panel.innerHTML = "";
    btn.setAttribute("aria-expanded", "false");
    if (TJ.deepDive[step.id]) {
      btn.style.display = "";
      panel.style.display = "";
      TJ.deepDive[step.id](panel);
      btn.textContent = "\u2191 Close math view";
      btn.setAttribute("aria-expanded", "true");
    } else {
      btn.style.display = "none";
    }
    void card.offsetWidth;
    card.classList.add("step-enter");
  }

  function updateRailActive() {
    document.querySelectorAll(".rail-step").forEach((btn) => {
      const active = parseInt(btn.dataset.index, 10) === state.index;
      btn.classList.toggle("active", active);
      if (active) btn.setAttribute("aria-current", "step");
      else btn.removeAttribute("aria-current");
    });
    const active = document.querySelector(".rail-step.active");
    if (active) active.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
    document.getElementById("progress-label").textContent = `Step ${state.index + 1} of ${state.steps.length}`;
    document.getElementById("progress-fill").style.width = `${((state.index + 1) / state.steps.length) * 100}%`;
  }

  function goToStep(i) {
    state.index = Math.max(0, Math.min(state.steps.length - 1, i));
    const step = state.steps[state.index];
    TJ.camera.goTo(step.focus);
    TJ.renderToken.syncExpertHighlight();
    renderNarrative(step);
    updateRailActive();
    document.getElementById("prev-btn").disabled = state.index === 0;
    document.getElementById("next-btn").disabled = state.index === state.steps.length - 1;
  }

  function next() {
    if (state.index >= state.steps.length - 1) {
      stopAutoplay();
      return;
    }
    goToStep(state.index + 1);
  }
  function prev() {
    goToStep(state.index - 1);
  }

  function startAutoplay() {
    if (state.index >= state.steps.length - 1) goToStep(0);
    state.playing = true;
    document.getElementById("play-btn").textContent = "\u23f8 Pause";
    document.getElementById("play-btn").classList.add("playing");
    state.timer = setInterval(() => {
      if (state.index >= state.steps.length - 1) {
        stopAutoplay();
        return;
      }
      next();
    }, 5000);
  }
  function stopAutoplay() {
    state.playing = false;
    document.getElementById("play-btn").textContent = "\u25b6 Play";
    document.getElementById("play-btn").classList.remove("playing");
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
  }
  function toggleAutoplay() {
    if (state.playing) stopAutoplay();
    else startAutoplay();
  }

  function wireControls() {
    document.getElementById("next-btn").addEventListener("click", () => {
      stopAutoplay();
      next();
    });
    document.getElementById("prev-btn").addEventListener("click", () => {
      stopAutoplay();
      prev();
    });
    document.getElementById("play-btn").addEventListener("click", toggleAutoplay);
    document.getElementById("deepdive-btn").addEventListener("click", () => {
      const panel = document.getElementById("deepdive-panel");
      const button = document.getElementById("deepdive-btn");
      const step = state.steps[state.index];
      const showing = panel.style.display !== "none";
      if (showing) {
        TJ.matrixViz.stopAll();
        panel.style.display = "none";
        panel.innerHTML = "";
        button.textContent = "\uD83D\uDD0D Zoom into the math";
        button.setAttribute("aria-expanded", "false");
      } else {
        stopAutoplay();
        panel.style.display = "";
        TJ.deepDive[step.id](panel);
        button.textContent = "\u2191 Close math view";
        button.setAttribute("aria-expanded", "true");
        panel.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    });

    document.querySelectorAll("#controls .ctrl-group").forEach((group) => {
      const key = group.dataset.key;
      group.querySelectorAll("button").forEach((btn) => {
        btn.addEventListener("click", () => {
          if (key === "attention") {
            state.attention = btn.dataset.value;
            state.positional = "rope";
          } else {
            state.positional = btn.dataset.value;
            state.attention = "gqa";
          }
          switchVariant(variantKey());
        });
      });
    });

    window.addEventListener("keydown", (e) => {
      if (e.target.closest("button, a, input, select, textarea") || e.target.isContentEditable) return;
      if (e.key === "ArrowRight" || e.key === " ") {
        e.preventDefault();
        stopAutoplay();
        next();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        stopAutoplay();
        prev();
      }
    });
  }

  function init() {
    const data = TJ.store.data();
    state.steps = TJ.buildSteps(data);
    TJ.diagram.mount("#diagram", data);
    TJ.camera.init("#diagram");
    buildRail();
    wireControls();
    updateVariantButtons();
    goToStep(0);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
