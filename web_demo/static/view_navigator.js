"use strict";

(function attachModule(globalScope) {
  function viewForDrag(dx, dy, currentName, threshold = 32) {
    const horizontal = Math.abs(dx);
    const vertical = Math.abs(dy);
    if (Math.max(horizontal, vertical) < threshold) {
      return currentName;
    }
    const ratio = vertical === 0 ? Infinity : horizontal / vertical;
    if (horizontal >= threshold && vertical >= threshold && ratio >= 0.5 && ratio <= 2) {
      const prefix = dy < 0 ? "upper" : "lower";
      return `${prefix}_${dx < 0 ? "left" : "right"}`;
    }
    if (horizontal >= vertical) {
      return dx < 0 ? "left" : "right";
    }
    return dy < 0 ? "up" : "down";
  }

  function attachViewNavigator(element, onSelect) {
    let start = null;
    element.addEventListener("pointerdown", (event) => {
      start = { x: event.clientX, y: event.clientY };
      element.setPointerCapture(event.pointerId);
      element.classList.add("is-dragging");
    });
    element.addEventListener("pointerup", (event) => {
      if (start === null) return;
      const next = viewForDrag(event.clientX - start.x, event.clientY - start.y, element.dataset.currentView || "left");
      start = null;
      element.classList.remove("is-dragging");
      if (next !== element.dataset.currentView) onSelect(next);
    });
    element.addEventListener("pointercancel", () => {
      start = null;
      element.classList.remove("is-dragging");
    });
  }

  const api = { viewForDrag, attachViewNavigator };
  globalScope.UniSharpViews = api;
  if (typeof module !== "undefined") module.exports = api;
})(typeof window === "undefined" ? globalThis : window);
