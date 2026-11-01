// Reference numbering shared by every UI surface; no DOM or app imports so it
// can be exercised without a browser.  The rules mirror the backend planner:
//
//   pictures: timeline order, capped at 9
//   videos:   clips sorted by start time, only the part overlapping the cyan
//             generation range, split into <=15 s windows, capped at 3
//   audios:   the soundtrack of every included video first, in video order,
//             then standalone assets capped at 3.  This matches the official
//             MiniMaxH3ReferenceToVideo presentation order.  A legacy "locked"
//             flag is ignored: this copy has no locked-audio sampler, so the
//             asset is treated as a normal reference (matching the backend).
//
// Keeping this in one place is what stops <Video N> chips, clip badges, the
// prompt health check and the upload limits from disagreeing.
export const FPS = 24;
export const MIN_VIDEO_SECONDS = 5 / FPS;
export const MAX_VIDEO_SECONDS = 15;
export const LIMITS = { pictures: 9, videos: 3, audios: 3 };

const clipStart = clip => Math.max(0, Number(clip?.start) || 0);
const clipEnd = clip => clipStart(clip) + Math.max(0, Number(clip?.duration) || 0);
const clipHasAudio = clip => clip?.hasAudio !== false;
const sourceTime = (clip, timelineSecond) =>
  Math.max(0, (Number(clip?.trimStart) || 0) + timelineSecond - clipStart(clip));

export function buildReferencePlan(state) {
  const selection = state?.selection || {};
  const start = Math.max(0, Number(selection.start) || 0);
  const duration = Math.max(MIN_VIDEO_SECONDS, Number(selection.duration) || 5);
  const end = start + duration;
  const videoAudioEnabled = state?.videoAudioEnabled !== false;
  const clips = (Array.isArray(state?.videoClips) ? [...state.videoClips] : [])
    .filter(clip => clip && clip.file)
    .sort((a, b) => clipStart(a) - clipStart(b));

  const allImages = Array.isArray(state?.images) ? state.images : [];
  const pictures = allImages.slice(0, LIMITS.pictures).map((asset, index) => ({
    ordinal: index + 1,
    id: asset.id,
    file: asset.file,
    name: asset.name || asset.file,
  }));

  const videos = [];
  const videoCounts = {};
  let nominalVideos = 0;
  for (const clip of clips) {
    const overlapStart = Math.max(start, clipStart(clip));
    const overlapEnd = Math.min(end, clipEnd(clip));
    const overlap = overlapEnd - overlapStart;
    if (overlap <= 0) continue;
    const overlapSourceStart = sourceTime(clip, overlapStart);
    let remaining = overlap;
    let offset = 0;
    let count = 0;
    while (remaining >= MIN_VIDEO_SECONDS) {
      const piece = Math.min(MAX_VIDEO_SECONDS, remaining);
      count += 1;
      nominalVideos += 1;
      if (videos.length < LIMITS.videos) {
        videos.push({
          ordinal: videos.length + 1,
          clipId: clip.id,
          file: clip.file,
          name: clip.name || clip.file,
          sourceStart: overlapSourceStart + offset,
          timelineStart: overlapStart + offset,
          duration: piece,
          hasAudio: videoAudioEnabled && clipHasAudio(clip),
        });
      }
      offset += piece;
      remaining -= piece;
    }
    if (count) videoCounts[clip.id] = (videoCounts[clip.id] || 0) + count;
  }

  const allAudios = Array.isArray(state?.audios) ? state.audios : [];
  const standalone = allAudios.filter(asset => asset && asset.file);
  // Official order: paired video soundtracks first, standalone audio after.
  const audios = [];
  for (const video of videos) {
    if (!video.hasAudio) continue;
    audios.push({
      ordinal: audios.length + 1,
      kind: "soundtrack",
      clipId: video.clipId,
      file: video.file,
      name: video.name,
      sourceStart: video.sourceStart,
      duration: video.duration,
    });
  }
  standalone.slice(0, LIMITS.audios).forEach(asset => {
    audios.push({
      ordinal: audios.length + 1,
      kind: "standalone",
      id: asset.id,
      file: asset.file,
      name: asset.name || asset.file,
    });
  });
  const pairedAudios = audios.filter(item => item.kind === "soundtrack").length;
  const nominal = {
    pictures: allImages.length,
    videos: nominalVideos,
    audios: standalone.length + pairedAudios,
  };
  const overflow = {
    pictures: Math.max(0, nominal.pictures - LIMITS.pictures),
    videos: Math.max(0, nominalVideos - LIMITS.videos),
    audios: Math.max(0, standalone.length - LIMITS.audios),
  };

  // Native guides use the same overlap geometry but are independent of the
  // three-video reference cap, so they stay a separate list.
  const guidePieces = [];
  let hasOverlap = false;
  for (const clip of clips) {
    const overlapStart = Math.max(start, clipStart(clip));
    const overlapEnd = Math.min(end, clipEnd(clip));
    const overlap = overlapEnd - overlapStart;
    if (overlap <= 0) continue;
    hasOverlap = true;
    const guideFrames = Math.max(1, Math.round(overlap * FPS));
    const mode = clip.referenceMode || "guide";
    if (mode === "guide") {
      guidePieces.push({
        clipId: clip.id, frames: guideFrames, mode,
        hasAudio: videoAudioEnabled && clipHasAudio(clip),
      });
    } else if (mode === "boundary") {
      guidePieces.push({ clipId: clip.id, frames: Math.min(2, guideFrames), mode, hasAudio: false });
    }
  }
  let gapGuideCount = 0;
  if (!hasOverlap) {
    const epsilon = 1 / FPS;
    if (clips.some(clip => clipEnd(clip) <= start + epsilon)) gapGuideCount++;
    if (clips.some(clip => clipStart(clip) >= end - epsilon)) gapGuideCount++;
  }

  return {
    pictures, videos, audios, videoCounts, guidePieces, gapGuideCount,
    nominal, overflow, limits: LIMITS,
  };
}
