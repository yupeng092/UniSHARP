"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const jobId = document.body.dataset.jobId;
  const status = document.querySelector("#job-status");
  const panel = document.querySelector("#result-panel");
  const failure = document.querySelector("#failure-panel");
  const navigator = document.querySelector("#view-navigator");
  const mainImage = document.querySelector("#main-view");
  const label = document.querySelector("#view-label");
  const thumbs = document.querySelector("#thumbnail-bar");
  const names = { left: "Left", right: "Right", up: "Up", down: "Down", upper_left: "Upper left", upper_right: "Upper right", lower_left: "Lower left", lower_right: "Lower right", forward: "Forward", back: "Back" };
  let views = new Map();
  const phrase = { queued: "Queued — waiting for the CPU worker", inference: "Inferring 3D scene…", rendering: "Rendering 10 viewpoints…" };
  function select(name) {
    const view = views.get(name); if (!view) return;
    navigator.dataset.currentView = name; mainImage.src = view.url; label.textContent = `${names[name] || name} view`;
    thumbs.querySelectorAll("button").forEach((button) => button.classList.toggle("is-selected", button.dataset.view === name));
  }
  function showResult(result) {
    views = new Map(result.views.map((view) => [view.name, view]));
    thumbs.replaceChildren(...result.views.map((view) => {
      const button = document.createElement("button"); button.type = "button"; button.dataset.view = view.name; button.title = `${names[view.name] || view.name} view`;
      const image = document.createElement("img"); image.src = view.url; image.alt = `${names[view.name] || view.name} rendered view`; button.append(image); button.addEventListener("click", () => select(view.name)); return button;
    }));
    document.querySelector("#download-gif").href = result.gif; document.querySelector("#source-image").href = result.source;
    panel.hidden = false; status.textContent = "Complete — drag the image or select a viewpoint.";
    select(views.has("back") ? "back" : result.views[0].name);
    window.UniSharpViews.attachViewNavigator(navigator, select);
  }
  async function poll() {
    try {
      const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
      if (!response.ok) throw new Error("Unable to read render status.");
      const job = await response.json();
      if (job.phase === "complete") { showResult(job.result); return; }
      if (job.phase === "failed") { status.textContent = "Render failed"; document.querySelector("#failure-message").textContent = job.error || "The local renderer stopped unexpectedly."; failure.hidden = false; return; }
      status.textContent = phrase[job.phase] || "Preparing render job…"; window.setTimeout(poll, 2000);
    } catch (exception) { status.textContent = exception.message; window.setTimeout(poll, 4000); }
  }
  poll();
});
