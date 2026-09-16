import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { viewForDrag } = require("../../web_demo/static/view_navigator.js");

test("horizontal drag chooses the matching side", () => {
  assert.equal(viewForDrag(80, 5, "up"), "right");
  assert.equal(viewForDrag(-80, 5, "up"), "left");
});

test("dominant diagonal drag chooses a diagonal render", () => {
  assert.equal(viewForDrag(70, -70, "left"), "upper_right");
  assert.equal(viewForDrag(-70, 70, "right"), "lower_left");
});

test("short drag preserves the current named view", () => {
  assert.equal(viewForDrag(8, 5, "back"), "back");
});
