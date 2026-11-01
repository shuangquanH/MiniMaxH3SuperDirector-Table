import test from "node:test";
import assert from "node:assert/strict";
import { buildReferencePlan, LIMITS } from "./reference_plan.mjs";

const clip = (id, start, duration, extra = {}) => ({
  id, file: `${id}.mp4`, name: id, start, duration, trimStart: 0,
  hasAudio: false, referenceMode: "edit", ...extra,
});
const base = (overrides = {}) => ({
  selection: { start: 0, duration: 5.875 },
  videoAudioEnabled: false,
  videoClips: [], images: [], audios: [], ...overrides,
});

test("single clip inside the selection becomes Video 1", () => {
  const plan = buildReferencePlan(base({ videoClips: [clip("a", 0, 31.37)] }));
  assert.deepEqual(plan.videos.map(v => v.ordinal), [1]);
  assert.equal(plan.videos[0].duration, 5.875);
  assert.equal(plan.videos[0].sourceStart, 0);
  assert.equal(plan.nominal.videos, 1);
  assert.equal(plan.overflow.videos, 0);
});

test("long overlaps split into 15 s windows and cap at three", () => {
  const plan = buildReferencePlan(base({
    selection: { start: 0, duration: 50 },
    videoClips: [clip("a", 0, 60)],
  }));
  assert.deepEqual(plan.videos.map(v => v.duration), [15, 15, 15]);
  assert.equal(plan.nominal.videos, 4);
  assert.equal(plan.overflow.videos, 1);
  assert.deepEqual(plan.videos.map(v => v.sourceStart), [0, 15, 30]);
});

test("video ordinals follow start time, not insertion order", () => {
  const plan = buildReferencePlan(base({
    selection: { start: 0, duration: 6 },
    videoClips: [clip("late", 2, 5), clip("early", 0, 5)],
  }));
  assert.deepEqual(plan.videos.map(v => v.clipId), ["early", "late"]);
  assert.deepEqual(plan.videos.map(v => v.ordinal), [1, 2]);
});

test("source time honours the clip's trim start", () => {
  const plan = buildReferencePlan(base({
    selection: { start: 4, duration: 1 },
    videoClips: [clip("a", 2, 10, { trimStart: 3 })],
  }));
  assert.equal(plan.videos[0].sourceStart, 5);
  assert.equal(plan.videos[0].timelineStart, 4);
});

test("video soundtracks precede standalone audio, capped at three", () => {
  const audios = ["a1", "a2", "a3", "a4"].map((id, index) => ({
    id, name: id, file: `${id}.wav`, audioMode: "reference", duration: 5,
  }));
  const plan = buildReferencePlan(base({
    videoAudioEnabled: true,
    videoClips: [clip("v", 0, 5, { hasAudio: true })],
    audios,
  }));
  assert.deepEqual(plan.audios.map(a => [a.ordinal, a.kind, a.name]),
    [[1, "soundtrack", "v"], [2, "standalone", "a1"], [3, "standalone", "a2"],
     [4, "standalone", "a3"]]);
  assert.equal(plan.nominal.audios, 5);
  assert.equal(plan.overflow.audios, 1);
});

test("one video and two standalone audios shift when its soundtrack is disabled", () => {
  const audios = ["a1", "a2"].map(id => ({
    id, name: id, file: `${id}.wav`, audioMode: "reference", duration: 5,
  }));
  const videoClips = [clip("v", 0, 5, { hasAudio: true })];
  const on = buildReferencePlan(base({ videoAudioEnabled: true, videoClips, audios }));
  assert.deepEqual(on.audios.map(a => [a.ordinal, a.kind]), [
    [1, "soundtrack"], [2, "standalone"], [3, "standalone"],
  ]);
  const off = buildReferencePlan(base({ videoAudioEnabled: false, videoClips, audios }));
  assert.deepEqual(off.audios.map(a => [a.ordinal, a.kind]), [
    [1, "standalone"], [2, "standalone"],
  ]);
});

test("legacy kept-original audio follows the video soundtrack", () => {
  const locked = { id: "l", name: "l", file: "l.wav", audioMode: "locked", duration: 5 };
  const on = buildReferencePlan(base({
    videoAudioEnabled: true,
    videoClips: [clip("v", 0, 5, { hasAudio: true })],
    audios: [locked],
  }));
  assert.deepEqual(on.audios.map(a => a.kind), ["soundtrack", "standalone"]);
  const off = buildReferencePlan(base({
    videoAudioEnabled: false,
    videoClips: [clip("v", 0, 5, { hasAudio: true })],
    audios: [locked],
  }));
  assert.deepEqual(off.audios.map(a => a.kind), ["standalone"]);
});

test("pictures keep timeline order and overflow past nine", () => {
  const images = Array.from({ length: 12 }, (_, index) => ({ id: `p${index}`, file: `p${index}.png`, name: `p${index}` }));
  const plan = buildReferencePlan(base({ images }));
  assert.equal(plan.pictures.length, LIMITS.pictures);
  assert.deepEqual(plan.pictures.map(p => p.ordinal), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert.equal(plan.nominal.pictures, 12);
  assert.equal(plan.overflow.pictures, 3);
});

test("guide pieces stay independent of the video cap", () => {
  const plan = buildReferencePlan(base({
    selection: { start: 0, duration: 50 },
    videoClips: [clip("a", 0, 60, { referenceMode: "guide" })],
  }));
  assert.equal(plan.videos.length, 3);
  assert.equal(plan.nominal.videos, 4);
  assert.equal(plan.guidePieces.length, 1);
  assert.equal(plan.guidePieces[0].frames, 1200);
});

test("boundary mode anchors two frames and gaps report context", () => {
  const boundary = buildReferencePlan(base({
    videoClips: [clip("a", 0, 5, { referenceMode: "boundary" })],
  }));
  assert.deepEqual(boundary.guidePieces.map(p => p.frames), [2]);
  const gap = buildReferencePlan(base({
    selection: { start: 8, duration: 2 },
    videoClips: [clip("left", 0, 5), clip("right", 12, 5)],
  }));
  assert.equal(gap.guidePieces.length, 0);
  assert.equal(gap.gapGuideCount, 2);
});
