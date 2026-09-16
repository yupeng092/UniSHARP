"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("#upload-form");
  const input = document.querySelector("#image");
  const dropZone = document.querySelector("#drop-zone");
  const selected = document.querySelector("#selected-file");
  const error = document.querySelector("#upload-error");
  const button = document.querySelector("#submit-button");
  const showFile = (file) => { selected.textContent = file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB` : "Maximum upload size: 15 MB"; };
  input.addEventListener("change", () => showFile(input.files[0]));
  ["dragenter", "dragover"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove("is-over"); }));
  dropZone.addEventListener("drop", (event) => { input.files = event.dataTransfer.files; showFile(input.files[0]); });
  form.addEventListener("submit", async (event) => {
    event.preventDefault(); error.hidden = true;
    if (!input.files.length) { error.textContent = "Choose an image first."; error.hidden = false; return; }
    button.disabled = true; button.textContent = "Queuing render…";
    try {
      const response = await fetch("/api/jobs", { method: "POST", body: new FormData(form) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Unable to create render job.");
      window.location.assign(data.job_url);
    } catch (exception) {
      error.textContent = exception.message; error.hidden = false;
      button.disabled = false; button.textContent = "Generate views →";
    }
  });
});
