"""Backend for the MiniMax H3 Timeline Director node.

The UI stores an edit decision list (EDL) in ``timeline_data``.  At queue time
this module resolves the current generation selection into H3 references and
native ``MiniMaxH3AddGuide`` keyframe guides:

* only the portions overlapping the generated range become reference videos;
* those same overlapping portions may become fixed guide clips;
* a gap between two clips uses native first/last single-frame guides;
* standalone image and audio assets are forwarded as references.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Iterable

import av
import numpy as np
import torch
from aiohttp import web
from PIL import Image, ImageOps

import comfy.sd
import comfy.utils
import folder_paths
import node_helpers
from comfy_api.latest import io
from comfy_extras import nodes_minimax_h3 as h3_nodes
from server import PromptServer


log = logging.getLogger("MiniMaxH3TimelineDirector")
FPS = 24.0
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
MAX_REF_VIDEO_SECONDS = 15.0
MIN_REF_VIDEO_SECONDS = 5.0 / FPS
# Same ceiling the panel uses (5 + 17*211); longer requests are clamped instead
# of building a latent far bigger than any H3 sampler accepts.
MAX_H3_FRAMES = 3592
UPLOAD_SUBDIR = "minimax_h3_timeline_director"
PREVIEW_SUBDIR = f"{UPLOAD_SUBDIR}/preview_proxies"
PREVIEW_MAX_WIDTH = 480
PREVIEW_MAX_HEIGHT = 270
PREVIEW_FPS = 12
MAX_HTTP_MEDIA_BYTES = 512 * 1024 * 1024
MAX_HTTP_MEDIA_SECONDS = 30 * 60.0
MAX_PREVIEW_SOURCE_SECONDS = 15 * 60.0
MAX_HTTP_MEDIA_DIMENSION = 16384
MAX_HTTP_MEDIA_PIXELS = 8192 * 8192
MAX_PROMPT_PREVIEW_CHARS = 200_000
MAX_PREVIEW_CACHE_BYTES = 2 * 1024 * 1024 * 1024
MAX_PREVIEW_CACHE_FILES = 256
MEDIA_INFO_CONCURRENCY = 2
PREVIEW_CONCURRENCY = 1
MEDIA_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"},
    "audio": {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"},
    "video": {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg"},
}
_MEDIA_INFO_SEMAPHORE = asyncio.Semaphore(MEDIA_INFO_CONCURRENCY)
_PREVIEW_SEMAPHORE = asyncio.Semaphore(PREVIEW_CONCURRENCY)
# Private helpers from comfy_extras.nodes_minimax_h3 that this node relies on.
# Checked before any model is loaded so an incompatible ComfyUI build fails with
# a clear message instead of an AttributeError deep inside encoding.
_H3_REQUIRED_HELPERS = (
    "MiniMaxH3AddGuide",
    "_empty_av_latent",
    "_resize",
    "_encode_ref_audio",
    "adapt_canvas",
    "CANVAS_MULTIPLE",
    "REF_IMAGE_SHORT_EDGE",
    "FPS",
)


def _require_h3_helpers() -> None:
    missing = [name for name in _H3_REQUIRED_HELPERS if not hasattr(h3_nodes, name)]
    if missing:
        raise RuntimeError(
            "This ComfyUI build is missing the MiniMax H3 helper(s) %s; "
            "the director needs the build containing PR #15439." % ", ".join(missing)
        )

def _upload_root() -> Path:
    root = Path(folder_paths.get_input_directory()).resolve() / UPLOAD_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_name(value: str) -> str:
    name = Path(str(value or "media")).name
    stem = re.sub(r"[^\w.()\-\u4e00-\u9fff]+", "_", name, flags=re.UNICODE)
    return (stem or "media")[:180]


def _safe_input_path(relative_name: str) -> Path:
    root = Path(folder_paths.get_input_directory()).resolve()
    candidate = (root / str(relative_name or "").replace("/", os.sep)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Media path is outside ComfyUI input directory") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"Reference media not found: {relative_name}")
    return candidate


def _safe_uploaded_path(relative_name: str, kind: str | None = None) -> Path:
    """Resolve a browser-facing media path inside this plugin's upload area.

    Node execution intentionally continues to accept existing files anywhere
    below ComfyUI's input directory.  HTTP helpers are narrower: a remote
    caller may only inspect files uploaded through this plugin's fixed
    subfolder, and may never feed a generated preview back into the helpers.
    """

    candidate = _safe_input_path(relative_name)
    upload_root = _upload_root()
    try:
        relative = candidate.relative_to(upload_root)
    except ValueError as exc:
        raise ValueError("Media is not in the Timeline Director upload folder") from exc
    if len(relative.parts) != 1:
        raise ValueError("Only direct uploads can be inspected or previewed")
    if kind is not None and kind not in MEDIA_EXTENSIONS:
        raise ValueError("Unsupported media type")
    if kind and candidate.suffix.lower() not in MEDIA_EXTENSIONS[kind]:
        raise ValueError(f"File extension is not supported for {kind} media")
    return candidate


def _relative_input(path: Path) -> str:
    root = Path(folder_paths.get_input_directory()).resolve()
    return path.resolve().relative_to(root).as_posix()


def _preview_root() -> Path:
    root = Path(folder_paths.get_input_directory()).resolve() / Path(PREVIEW_SUBDIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _validate_http_media_file(path: Path) -> None:
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("Media file is empty")
    if size > MAX_HTTP_MEDIA_BYTES:
        raise ValueError("Media file exceeds the 512 MiB safety limit")


def _validate_http_media_info(
    info: dict[str, Any], kind: str, *, max_duration: float = MAX_HTTP_MEDIA_SECONDS
) -> None:
    duration = max(0.0, _float(info.get("duration")))
    width = max(0, int(_float(info.get("width"))))
    height = max(0, int(_float(info.get("height"))))
    if kind in {"audio", "video"} and duration <= 0:
        raise ValueError("Media duration could not be determined safely")
    if duration > max_duration:
        raise ValueError(f"Media duration exceeds the {int(max_duration // 60)} minute safety limit")
    if width > MAX_HTTP_MEDIA_DIMENSION or height > MAX_HTTP_MEDIA_DIMENSION:
        raise ValueError("Media dimensions exceed the safety limit")
    if width and height and width * height > MAX_HTTP_MEDIA_PIXELS:
        raise ValueError("Media pixel count exceeds the safety limit")
    if kind == "video" and not info.get("hasVideo"):
        raise ValueError("Uploaded file does not contain a video track")
    if kind == "audio" and not info.get("hasAudio"):
        raise ValueError("Uploaded file does not contain an audio track")


def _uploaded_media_info(path: Path, kind: str) -> dict[str, Any]:
    """Validate one official ComfyUI upload and return bounded UI metadata."""

    _validate_http_media_file(path)
    if kind == "image":
        # verify() must run directly after open; reading EXIF first (which loads
        # PNG metadata) makes Pillow reject the verify call. Validate the file
        # first, then reopen it for the size/orientation metadata.
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            # _load_image() applies EXIF orientation before encoding; report the
            # same orientation here so the panel's "submitted size" is accurate.
            try:
                orientation = image.getexif().get(0x0112, 1)
            except Exception:
                orientation = 1
            if orientation in (5, 6, 7, 8):
                width, height = height, width
            info = {
                "filename": _relative_input(path),
                "name": path.name,
                "duration": 0.0,
                "width": int(width),
                "height": int(height),
                "fps": 0.0,
                "hasVideo": False,
                "hasAudio": False,
                "peaks": [],
            }
            _validate_http_media_info(info, kind)
        return info

    info = _media_info(path, include_peaks=False)
    _validate_http_media_info(info, kind)
    if info["hasAudio"]:
        info["peaks"] = _audio_peaks(path)
    return info


def _prune_preview_cache(keep: Path) -> None:
    """Bound the disk used by generated monitoring proxies."""

    files = sorted(
        (item for item in _preview_root().glob("*.mp4") if item.is_file()),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    total = sum(item.stat().st_size for item in files)
    for index, item in enumerate(files):
        if item == keep:
            continue
        if index < MAX_PREVIEW_CACHE_FILES and total <= MAX_PREVIEW_CACHE_BYTES:
            continue
        size = item.stat().st_size
        item.unlink(missing_ok=True)
        total = max(0, total - size)


def _ensure_preview_proxy(source: Path) -> str:
    """Create a cached, silent low-resolution MP4 for timeline monitoring."""

    _validate_http_media_file(source)
    stat = source.stat()
    fingerprint = hashlib.sha1(
        f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|v1".encode("utf-8")
    ).hexdigest()[:16]
    destination = _preview_root() / f"{_safe_name(source.stem)[:80]}_{fingerprint}.mp4"
    if destination.is_file() and destination.stat().st_size > 1024:
        return _relative_input(destination)

    # Probe only when a proxy actually has to be built; the fingerprint above
    # already covers the source's identity, so a cache hit needs no decode.
    source_info = _media_info(source, include_peaks=False)
    _validate_http_media_info(source_info, "video", max_duration=MAX_PREVIEW_SOURCE_SECONDS)

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ComfyUI environment does not provide ffmpeg for low-resolution preview") from exc

    temporary = destination.with_name(f".{destination.stem}_{uuid.uuid4().hex}.tmp.mp4")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-map", "0:v:0", "-an", "-vf",
        (
            f"fps={PREVIEW_FPS},scale={PREVIEW_MAX_WIDTH}:{PREVIEW_MAX_HEIGHT}:"
            "force_original_aspect_ratio=decrease:force_divisible_by=2"
        ),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False,
            creationflags=creation_flags, timeout=180,
        )
        if completed.returncode != 0 or not temporary.is_file():
            detail = (completed.stderr or completed.stdout or "unknown ffmpeg error").strip()[-1200:]
            raise RuntimeError(f"Low-resolution preview creation failed: {detail}")
        os.replace(temporary, destination)
        _prune_preview_cache(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return _relative_input(destination)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _media_info(path: Path, include_peaks: bool = True) -> dict[str, Any]:
    info: dict[str, Any] = {
        "filename": _relative_input(path),
        "name": path.name,
        "duration": 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "hasVideo": False,
        "hasAudio": False,
        "peaks": [],
    }
    with av.open(str(path)) as container:
        if container.duration is not None:
            info["duration"] = max(0.0, float(container.duration / av.time_base))
        if container.streams.video:
            stream = container.streams.video[0]
            info.update(
                hasVideo=True,
                width=int(stream.width or stream.codec_context.width or 0),
                height=int(stream.height or stream.codec_context.height or 0),
                fps=float(stream.average_rate or stream.guessed_rate or 0),
            )
            if not info["duration"] and stream.duration is not None and stream.time_base:
                info["duration"] = float(stream.duration * stream.time_base)
        if container.streams.audio:
            info["hasAudio"] = True
            stream = container.streams.audio[0]
            if not info["duration"] and stream.duration is not None and stream.time_base:
                info["duration"] = float(stream.duration * stream.time_base)
    if include_peaks and info["hasAudio"]:
        try:
            info["peaks"] = _audio_peaks(path)
        except Exception as exc:
            log.debug("Waveform extraction failed for %s: %s", path, exc)
    return info


def _decode_audio(path: Path, start: float = 0.0, duration: float | None = None) -> dict[str, Any] | None:
    """Decode an audio interval to ComfyUI AUDIO at 44.1 kHz.

    Seeking starts slightly early so compressed sources do not lose samples at
    a keyframe/packet boundary.  The decoded waveform is then sliced using the
    first decoded timestamp.
    """

    target_rate = 44100
    start = max(0.0, float(start))
    end = None if duration is None else start + max(0.0, float(duration))
    chunks: list[np.ndarray] = []
    first_time: float | None = None
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        seek_to = max(0.0, start - 1.0)
        if seek_to > 0:
            container.seek(int(seek_to * av.time_base), backward=True)
        layout = "stereo" if getattr(stream.codec_context, "channels", 1) > 1 else "mono"
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=target_rate)
        stop_after = None if end is None else end + 1.0
        for frame in container.decode(stream):
            frame_time = float(frame.time) if frame.time is not None else seek_to
            if first_time is None:
                first_time = frame_time
            if stop_after is not None and frame_time > stop_after:
                break
            converted = resampler.resample(frame)
            if not isinstance(converted, list):
                converted = [converted]
            for out in converted:
                arr = out.to_ndarray()
                if arr.ndim == 1:
                    arr = arr[None, :]
                chunks.append(arr.astype(np.float32, copy=False))
        flushed = resampler.resample(None)
        if not isinstance(flushed, list):
            flushed = [flushed] if flushed is not None else []
        for out in flushed:
            arr = out.to_ndarray()
            if arr.ndim == 1:
                arr = arr[None, :]
            chunks.append(arr.astype(np.float32, copy=False))
    if not chunks:
        return None
    waveform = np.concatenate(chunks, axis=1)
    decoded_start = max(0.0, first_time or 0.0)
    begin = max(0, int(round((start - decoded_start) * target_rate)))
    finish = waveform.shape[1] if end is None else min(
        waveform.shape[1], int(round((end - decoded_start) * target_rate))
    )
    if finish <= begin:
        return None
    tensor = torch.from_numpy(np.ascontiguousarray(waveform[:, begin:finish])).unsqueeze(0)
    return {"waveform": tensor, "sample_rate": target_rate}


def _audio_mode(asset: dict[str, Any]) -> str:
    """Normalize persisted audio-card behavior without breaking older workflows."""

    return "locked" if str(asset.get("audioMode") or "").lower() == "locked" else "reference"


def _audio_peaks(path: Path, count: int = 180) -> list[float]:
    """Build a fixed-size waveform preview without retaining the full soundtrack."""

    preview_rate = 8000
    peaks = np.zeros(count, dtype=np.float32)
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return []
        stream = container.streams.audio[0]
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        if not duration and stream.duration is not None and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        # Unknown-duration streams are rare for uploaded files.  Give them a
        # conservative ten-minute preview span while keeping memory constant.
        total_samples = max(count, int(math.ceil(max(duration, 600.0 if not duration else duration) * preview_rate)))
        bin_size = max(1, int(math.ceil(total_samples / count)))
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=preview_rate)
        position = 0

        def consume(array: np.ndarray) -> None:
            nonlocal position
            values = np.abs(array.reshape(-1).astype(np.float32, copy=False))
            offset = 0
            while offset < values.size:
                bucket = min(count - 1, position // bin_size)
                take = min(values.size - offset, bin_size - (position % bin_size))
                if take > 0:
                    peaks[bucket] = max(peaks[bucket], float(values[offset : offset + take].max(initial=0.0)))
                position += take
                offset += take

        for frame in container.decode(stream):
            converted = resampler.resample(frame)
            if not isinstance(converted, list):
                converted = [converted]
            for out in converted:
                consume(out.to_ndarray())
        flushed = resampler.resample(None)
        if not isinstance(flushed, list):
            flushed = [flushed] if flushed is not None else []
        for out in flushed:
            consume(out.to_ndarray())
    return [float(value) for value in np.clip(peaks, 0.0, 1.0)]


def _target_size(width: Any = None, height: Any = None) -> tuple[int, int] | None:
    """Validate a generation canvas size used to bound decoded reference media."""

    target_width, target_height = int(_float(width)), int(_float(height))
    if target_width < 1 or target_height < 1:
        return None
    return target_width, target_height


def _reference_decode_size(
    width: Any, height: Any, limit_width: Any, limit_height: Any
) -> tuple[int, int]:
    """Clamp a decode canvas to what the H3 encoder will actually consume.

    Reference videos are re-encoded onto the H3 canvas and guide frames are
    resized onto the generation canvas, so decoding above that size only spends
    memory.  The clamp is uniform (aspect-preserving) and never upscales.
    """

    source_width = max(2, int(round(_float(width, 2.0))))
    source_height = max(2, int(round(_float(height, 2.0))))
    limit_width = max(2, int(round(_float(limit_width, 2.0))))
    limit_height = max(2, int(round(_float(limit_height, 2.0))))
    scale = min(1.0, limit_width / source_width, limit_height / source_height)
    if scale >= 1.0:
        return source_width, source_height
    # Even dimensions keep the aspect close and avoid odd sizes for encoders.
    return (
        max(2, int(round(source_width * scale / 2)) * 2),
        max(2, int(round(source_height * scale / 2)) * 2),
    )


def _fit_image(image: Image.Image, size: tuple[int, int], *, video: bool = False) -> Image.Image:
    """Aspect-fill a decoded video frame into its decode canvas without distortion."""

    method = Image.Resampling.BILINEAR if video else Image.Resampling.LANCZOS
    return ImageOps.fit(image.convert("RGB"), size, method=method, centering=(0.5, 0.5))


def _decode_video(
    path: Path,
    start: float,
    duration: float,
    fps: float = FPS,
    target_width: Any = None,
    target_height: Any = None,
) -> torch.Tensor | None:
    start = max(0.0, float(start))
    duration = max(0.0, float(duration))
    if duration <= 0:
        return None
    target_count = max(1, int(round(duration * fps)))
    target_times = [start + i / fps for i in range(target_count)]
    decoded: list[tuple[float, np.ndarray]] = []
    target_size = _target_size(target_width, target_height)
    with av.open(str(path)) as container:
        if not container.streams.video:
            return None
        stream = container.streams.video[0]
        seek_to = max(0.0, start - 1.0)
        if stream.time_base:
            container.seek(int(seek_to / float(stream.time_base)), stream=stream, backward=True)
        else:
            container.seek(int(seek_to * av.time_base), backward=True)
        limit = start + duration + 1.0 / fps
        fallback_time = seek_to
        source_fps = float(stream.average_rate or stream.guessed_rate or fps)
        for index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.time) if frame.time is not None else fallback_time + index / source_fps
            if timestamp < start - 1.0 / fps:
                continue
            if timestamp > limit:
                break
            if target_size:
                # Resize each decoded frame immediately.  This is deliberately
                # done before frames are retained, so a 4K/8K source never
                # accumulates a full-resolution reference tensor in RAM/VRAM.
                array = np.asarray(_fit_image(frame.to_image(), target_size, video=True), dtype=np.uint8)
            else:
                array = frame.to_ndarray(format="rgb24")
            decoded.append((timestamp, np.ascontiguousarray(array)))
    if not decoded:
        return None
    frames: list[np.ndarray] = []
    cursor = 0
    for target in target_times:
        while cursor + 1 < len(decoded) and abs(decoded[cursor + 1][0] - target) <= abs(decoded[cursor][0] - target):
            cursor += 1
        frames.append(decoded[cursor][1])
    # astype allocates the float32 copy; normalize it in place instead of
    # allocating a second full-size temporary for the division.
    array = np.stack(frames).astype(np.float32)
    array /= 255.0
    return torch.from_numpy(array)


def _decode_frame(
    path: Path, second: float, target_width: Any = None, target_height: Any = None
) -> torch.Tensor | None:
    frames = _decode_video(
        path, max(0.0, second), 1.0 / FPS, FPS, target_width, target_height
    )
    return None if frames is None else frames[:1]


def _ref_image_target_size(
    image_width: int, image_height: int, width: int, height: int, mode: str
) -> tuple[int, int]:
    """Resolve a reference picture's encoded size on H3's 32 px grid.

    ``match`` scales the picture down to the generation's pixel area,
    ``max`` keeps the original resolution (downscaling only above the 2048
    short edge), and ``average`` sits halfway between those two targets.
    """

    image_width = max(1, int(image_width))
    image_height = max(1, int(image_height))
    multiple = h3_nodes.CANVAS_MULTIPLE

    def scaled(scale: float) -> tuple[int, int]:
        return (
            max(multiple, round(image_width * scale / multiple) * multiple),
            max(multiple, round(image_height * scale / multiple) * multiple),
        )

    match_scale = min(
        1.0, math.sqrt(max(1, int(width)) * max(1, int(height)) / (image_width * image_height))
    )
    max_scale = min(1.0, h3_nodes.REF_IMAGE_SHORT_EDGE / min(image_width, image_height))
    match_size = scaled(match_scale)
    max_size = scaled(max_scale)
    if mode == "match":
        return match_size
    if mode == "average":
        return (
            max(multiple, round(((match_size[0] + max_size[0]) / 2) / multiple) * multiple),
            max(multiple, round(((match_size[1] + max_size[1]) / 2) / multiple) * multiple),
        )
    return max_size


def _load_image(path: Path) -> torch.Tensor:
    """Load a reference picture at its native size.

    ``<Picture N>`` is usually an appearance or composition anchor, so H3 must
    see the whole frame; the encoder scales it per ``ref_image_size`` later.
    Cropping it to the generation canvas here discarded most of a landscape
    reference before the model ever saw it.
    """

    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)


def _aligned_h3_length(seconds: float) -> int:
    """Return the nearest H3-valid 5 + 17*n frame count.

    Capped at the same 3592-frame ceiling the panel enforces, so a hand-edited
    timeline cannot ask for a latent longer than the node can produce.
    """

    requested = max(5.0 / FPS, float(seconds)) * FPS
    n = max(0, round((requested - 5.0) / 17.0))
    return min(MAX_H3_FRAMES, int(5 + 17 * n))


def _windows(start: float, duration: float, limit: float = MAX_REF_VIDEO_SECONDS) -> Iterable[tuple[float, float]]:
    remaining = max(0.0, duration)
    cursor = start
    while remaining >= MIN_REF_VIDEO_SECONDS:
        piece = min(limit, remaining)
        yield cursor, piece
        cursor += piece
        remaining -= piece


def _clip_end(clip: dict[str, Any]) -> float:
    return _float(clip.get("start")) + max(0.0, _float(clip.get("duration")))


def _source_time(clip: dict[str, Any], timeline_second: float) -> float:
    return max(0.0, _float(clip.get("trimStart")) + timeline_second - _float(clip.get("start")))


def _reference_mode(clip: dict[str, Any]) -> str:
    mode = str(clip.get("referenceMode") or "guide")
    return mode if mode in {"guide", "edit", "boundary"} else "guide"


def _reference_size(
    clip: dict[str, Any], fallback_width: Any, fallback_height: Any
) -> tuple[int, int]:
    """Resolve one clip's user-selected reference canvas on H3's 32px grid."""

    def dimension(key: str, fallback: Any) -> int:
        value = int(_float(clip.get(key)))
        if value < h3_nodes.CANVAS_MULTIPLE:
            value = max(h3_nodes.CANVAS_MULTIPLE, int(_float(fallback)))
        value = round(value / h3_nodes.CANVAS_MULTIPLE) * h3_nodes.CANVAS_MULTIPLE
        return max(h3_nodes.CANVAS_MULTIPLE, min(16384, value))

    return dimension("referenceWidth", fallback_width), dimension("referenceHeight", fallback_height)


def _video_reference_specs(
    timeline: dict[str, Any], target_width: Any = None, target_height: Any = None
) -> list[dict[str, Any]]:
    """Describe prompt-addressable videos without decoding their frames.

    This is shared by the material planner and the H3 encoder, so ``Video N``
    exposed to a prompt rewriter is guaranteed to be the same source interval
    later encoded as ``ref_video_N``.  A clip contributes only its intersection
    with the cyan generation selection; frames outside that intersection never
    enter prompt rewriting, reference encoding, or per-clip guides.
    """

    clips = [c for c in timeline.get("videoClips", []) if isinstance(c, dict) and c.get("file")]
    selection = timeline.get("selection") or {}
    selection_start = max(0.0, _float(selection.get("start")))
    selection_duration = max(MIN_REF_VIDEO_SECONDS, _float(selection.get("duration"), 5.0))
    selection_end = selection_start + selection_duration
    audio_enabled = timeline.get("videoAudioEnabled", True) is not False
    specs: list[dict[str, Any]] = []
    truncated = False

    for clip in sorted(clips, key=lambda c: _float(c.get("start"))):
        clip_start, clip_end = _float(clip.get("start")), _clip_end(clip)
        overlap_start = max(selection_start, clip_start)
        overlap_end = min(selection_end, clip_end)
        if overlap_end - overlap_start <= 0:
            continue

        overlap_source_start = _source_time(clip, overlap_start)
        reference_width, reference_height = _reference_size(
            clip,
            target_width if target_width is not None else 1344,
            target_height if target_height is not None else 768,
        )
        for source_start, piece_duration in _windows(
            overlap_source_start, overlap_end - overlap_start
        ):
            if len(specs) >= MAX_REF_VIDEOS:
                truncated = True
                break
            timeline_piece_start = overlap_start + source_start - overlap_source_start
            specs.append({
                "file": str(clip["file"]),
                "name": str(clip.get("name") or Path(str(clip["file"])).name),
                "clip_id": clip.get("id"),
                "timeline_start": timeline_piece_start,
                "timeline_end": timeline_piece_start + piece_duration,
                "source_start": source_start,
                "duration": piece_duration,
                "has_audio": audio_enabled and clip.get("hasAudio", True) is not False,
                "reference_width": reference_width,
                "reference_height": reference_height,
            })
        if truncated:
            break
    if truncated:
        log.warning(
            "Timeline has more video reference windows than H3 accepts; only the first %d are sent.",
            MAX_REF_VIDEOS,
        )
    return specs


def _normalize_segment_config(timeline: dict[str, Any]) -> dict[str, Any]:
    """Keep the segment count, active index and mode in a usable state."""

    config = timeline.get("segmentConfig")
    if not isinstance(config, dict):
        return timeline
    count = min(64, max(1, int(_float(config.get("count"), 0))))
    raw_segments = config.get("segments")
    segments = raw_segments if isinstance(raw_segments, list) else []
    config["count"] = count
    config["activeIndex"] = min(max(0, int(_float(config.get("activeIndex"), 0))), count - 1)
    config["mode"] = "timeline"
    config["segments"] = [
        segments[index] if index < len(segments) and isinstance(segments[index], dict) else {}
        for index in range(count)
    ]
    return timeline


def _create_timeline_plan(
    timeline_data: str, width: Any, height: Any,
) -> dict[str, Any]:
    """Turn the panel's timeline JSON into the plan the encoder consumes."""

    timeline = _normalize_segment_config(_parse_timeline(timeline_data))
    target_width, target_height = int(width), int(height)
    selection = timeline.setdefault("selection", {})
    saved_start = max(0.0, _float(selection.get("start")))
    saved_duration = max(MIN_REF_VIDEO_SECONDS, _float(selection.get("duration"), 5.0))
    start, seconds = saved_start, saved_duration

    # In timeline mode the active segment window is the source of truth.  The
    # panel keeps selection in sync while a user edits, but API clients and
    # workflows saved by older builds can carry a stale selection.
    config = timeline.get("segmentConfig") or {}
    segments = config.get("segments") or []
    active = int(_float(config.get("activeIndex"), 0))
    if 0 <= active < len(segments):
        segment = segments[active] if isinstance(segments[active], dict) else {}
        start_frame = int(_float(segment.get("startFrame")))
        end_frame = int(_float(segment.get("endFrame")))
        if start_frame >= 0 and end_frame > start_frame:
            start = start_frame / FPS
            seconds = (end_frame - start_frame) / FPS
            if (abs(start - saved_start) * FPS > 0.5
                    or abs(seconds - saved_duration) * FPS > 0.5):
                log.warning(
                    "Using active segment %d (%d-%d frames) instead of the saved selection "
                    "(%.3fs-%.3fs).",
                    active + 1, start_frame, end_frame,
                    saved_start, saved_start + saved_duration,
                )

    selection["start"] = max(0.0, start)
    selection["duration"] = min(
        MAX_H3_FRAMES / FPS, max(MIN_REF_VIDEO_SECONDS, seconds)
    )
    seconds = selection["duration"]
    return {
        "type": "MINIMAX_H3_TIMELINE_PLAN",
        "version": 1,
        "timeline": timeline,
        "width": target_width,
        "height": target_height,
        "length": _aligned_h3_length(seconds),
    }


def _require_timeline_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("type") != "MINIMAX_H3_TIMELINE_PLAN":
        raise ValueError("Input is not a valid MiniMax H3 timeline material plan.")
    if not isinstance(plan.get("timeline"), dict):
        raise ValueError("The material plan has no timeline data.")
    return plan


def _pick_gap_guides(clips: list[dict[str, Any]], selection_start: float, selection_end: float) -> list[tuple[dict[str, Any], float, int, str]]:
    """Choose still guides only when the selection is a true empty gap."""

    if not clips:
        return []
    epsilon = 1.0 / FPS
    ordered = sorted(clips, key=lambda clip: (_float(clip.get("start")), _clip_end(clip)))
    if any(min(selection_end, _clip_end(c)) - max(selection_start, _float(c.get("start"))) > epsilon for c in ordered):
        return []
    result: list[tuple[dict[str, Any], float, int, str]] = []
    left = [c for c in ordered if _clip_end(c) <= selection_start + epsilon]
    if left:
        clip = max(left, key=_clip_end)
        result.append((clip, max(_float(clip.get("start")), _clip_end(clip) - epsilon), 0, "gap-start"))
    right = [c for c in ordered if _float(c.get("start")) >= selection_end - epsilon]
    if right:
        clip = min(right, key=lambda c: _float(c.get("start")))
        result.append((clip, _float(clip.get("start")), -1, "gap-end"))
    return result


def _valid_guide_frame_count(frame_count: int) -> int:
    """Mirror MiniMaxH3AddGuide's legal 1 or 17*k+5 frame rule."""

    frame_count = max(1, int(frame_count))
    if frame_count < 5:
        return 1
    return frame_count - ((frame_count - 5) % 17)


def _guide_frame_slices(frame_count: int) -> list[tuple[int, int]]:
    """Split every source frame into legal native guides without dropping tails."""

    result: list[tuple[int, int]] = []
    offset = 0
    remaining = max(0, int(frame_count))
    while remaining:
        count = _valid_guide_frame_count(remaining)
        result.append((offset, count))
        offset += count
        remaining -= count
    return result


def _parse_timeline(value: str) -> dict[str, Any]:
    try:
        data = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Timeline data is invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Timeline data must be a JSON object")
    return data


def _build_references(
    timeline: dict[str, Any], target_width: Any = None, target_height: Any = None,
    target_length: int | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    clips = [c for c in timeline.get("videoClips", []) if isinstance(c, dict) and c.get("file")]
    selection = timeline.get("selection") or {}
    selection_start = max(0.0, _float(selection.get("start")))
    selection_duration = max(MIN_REF_VIDEO_SECONDS, _float(selection.get("duration"), 5.0))
    selection_end = selection_start + selection_duration
    target_length = int(target_length or _aligned_h3_length(selection_duration))

    ref_images: dict[str, torch.Tensor] = {}
    ref_audios: dict[str, dict[str, Any]] = {}
    guides: list[dict[str, Any]] = []
    video_audio_enabled = timeline.get("videoAudioEnabled", True) is not False

    # User-uploaded bins own their visible ordinal space. Native guides are not
    # prompt-labelled Picture references and therefore never change ordinals.
    image_assets = [
        asset for asset in timeline.get("images", [])
        if isinstance(asset, dict) and asset.get("file")
    ]
    for asset in image_assets[:MAX_REF_IMAGES]:
        ref_images[f"ref_image_{len(ref_images)}"] = _load_image(
            _safe_input_path(str(asset["file"]))
        )
    if len(image_assets) > MAX_REF_IMAGES:
        log.warning(
            "Timeline has %d reference images; only the first %d are sent to H3.",
            len(image_assets), MAX_REF_IMAGES,
        )

    # The planning specs are shared with the prompt rewriter, so Video/Audio
    # ordinals cannot drift.  Frames are decoded in the encoder instead: a
    # 15-second reference is hundreds of full-canvas float32 frames, and
    # decoding every reference up front kept them all resident at once.
    video_specs = _video_reference_specs(timeline, target_width, target_height)

    # A clip crossing the generated range is split deliberately: the overlap is
    # a hard guide at its target position; only the source context outside the
    # range is a prompt-addressable Video reference.
    for clip in sorted(clips, key=lambda c: _float(c.get("start"))):
        clip_start, clip_end = _float(clip.get("start")), _clip_end(clip)
        overlap_start = max(selection_start, _float(clip.get("start")))
        overlap_end = min(selection_end, clip_end)
        overlap_duration = overlap_end - overlap_start
        if overlap_duration <= 0:
            continue
        path = _safe_input_path(str(clip["file"]))
        mode = _reference_mode(clip)
        reference_width, reference_height = _reference_size(clip, target_width, target_height)
        # Native guides are resized onto the generation canvas by Add Guide, so
        # decoding above that size would only cost memory.
        guide_width, guide_height = _reference_decode_size(
            reference_width, reference_height, target_width, target_height
        )

        if mode in {"edit", "boundary"}:
            if mode == "boundary":
                first_idx = max(0, round((overlap_start - selection_start) * FPS))
                last_idx = min(target_length - 1, max(first_idx, round((overlap_end - selection_start) * FPS) - 1))
                if abs(overlap_start - selection_start) <= 0.5 / FPS:
                    first_idx = 0
                if abs(overlap_end - selection_end) <= 0.5 / FPS:
                    last_idx = target_length - 1
                first_frame = _decode_frame(
                    path, _source_time(clip, overlap_start), reference_width, reference_height
                )
                if first_frame is not None:
                    guides.append({
                        "image": first_frame, "audio": None, "frame_idx": first_idx,
                        "kind": "boundary-start", "clip_id": clip.get("id"), "source_frames": 1,
                    })
                if last_idx != first_idx:
                    last_frame = _decode_frame(
                        path, _source_time(clip, max(overlap_start, overlap_end - 1.0 / FPS)),
                        reference_width, reference_height,
                    )
                    if last_frame is not None:
                        guides.append({
                            "image": last_frame, "audio": None, "frame_idx": last_idx,
                            "kind": "boundary-end", "clip_id": clip.get("id"), "source_frames": 1,
                        })
            continue

        total_frames = min(max(1, round(overlap_duration * FPS)), target_length)
        guide_start_frame = max(0, round((overlap_start - selection_start) * FPS))
        if abs(overlap_start - selection_start) <= 0.5 / FPS:
            guide_start_frame = 0
        elif abs(overlap_end - selection_end) <= 0.5 / FPS:
            guide_start_frame = max(0, target_length - total_frames)
        guide_start_frame = min(guide_start_frame, max(0, target_length - total_frames))

        # Decode long overlaps in bounded windows. This preserves the existing
        # resolution protection without ever materializing a 150-second tensor.
        decoded_frames = 0
        while decoded_frames < total_frames:
            requested_frames = min(
                round(MAX_REF_VIDEO_SECONDS * FPS), total_frames - decoded_frames
            )
            frames = _decode_video(
                path, _source_time(clip, overlap_start) + decoded_frames / FPS,
                requested_frames / FPS, FPS, guide_width, guide_height,
            )
            if frames is None or not frames.shape[0]:
                break
            chunk_frames = min(int(frames.shape[0]), requested_frames)
            frames = frames[:chunk_frames]
            for slice_offset, slice_count in _guide_frame_slices(chunk_frames):
                guide_audio = None
                if video_audio_enabled and clip.get("hasAudio", True):
                    guide_audio = _decode_audio(
                        path, _source_time(clip, overlap_start) + (decoded_frames + slice_offset) / FPS,
                        slice_count / FPS,
                    )
                guides.append({
                    "image": frames[slice_offset:slice_offset + slice_count],
                    "audio": guide_audio,
                    "frame_idx": guide_start_frame + decoded_frames + slice_offset,
                    "kind": "video", "clip_id": clip.get("id"),
                    "source_frames": slice_count,
                })
            decoded_frames += chunk_frames
            if chunk_frames < requested_frames:
                break

    # A true gap has no video reference. Its nearest visual context is anchored
    # as target first/last frames with the official Add Guide mechanism.
    for clip, timeline_second, frame_idx, kind in _pick_gap_guides(clips, selection_start, selection_end):
        reference_width, reference_height = _reference_size(clip, target_width, target_height)
        frame = _decode_frame(
            _safe_input_path(str(clip["file"])), _source_time(clip, timeline_second),
            reference_width, reference_height,
        )
        if frame is not None:
            guides.append({"image": frame, "audio": None, "frame_idx": 0 if frame_idx == 0 else target_length - 1, "kind": kind})

    # Standalone audio is encoded after the paired video soundtracks; this list
    # covers only the standalone part and is capped at MAX_REF_AUDIOS.
    # This copy has no locked-audio sampler, so a legacy "keep original" asset is
    # used as an ordinary reference (matching the panel migration) instead of
    # being silently dropped.
    audio_assets = [
        asset for asset in timeline.get("audios", [])
        if isinstance(asset, dict) and asset.get("file")
    ]
    locked_assets = [asset for asset in audio_assets if _audio_mode(asset) == "locked"]
    if locked_assets:
        log.warning(
            "%d kept-original audio asset(s) are not supported by this copy; using them as reference audio.",
            len(locked_assets),
        )
    for asset in audio_assets[:MAX_REF_AUDIOS]:
        start = max(0.0, _float(asset.get("trimStart")))
        duration = _float(asset.get("duration")) or None
        audio = _decode_audio(_safe_input_path(str(asset["file"])), start, duration)
        if audio is not None:
            ref_audios[f"ref_audio_{len(ref_audios)}"] = audio
    if len(audio_assets) > MAX_REF_AUDIOS:
        log.warning(
            "Timeline has %d standalone reference audios; only the first %d are sent to H3.",
            len(audio_assets), MAX_REF_AUDIOS,
        )

    return ref_images, video_specs, ref_audios, guides


def _apply_h3_guides(conditioning, latent, vae, audio_vae, guides: list[dict[str, Any]]):
    """Apply timeline guides through ComfyUI's native, versioned H3 node."""

    if not guides:
        return conditioning
    if not hasattr(h3_nodes, "MiniMaxH3AddGuide"):
        raise RuntimeError("This ComfyUI build has no MiniMaxH3AddGuide; update to a build containing PR #15439.")
    for guide in guides:
        try:
            conditioning = h3_nodes.MiniMaxH3AddGuide.execute(
                conditioning, latent, int(guide["frame_idx"]), vae=vae,
                audio_vae=audio_vae, image=guide.get("image"), audio=guide.get("audio"),
            )[0]
        except Exception as exc:
            raise RuntimeError(
                "Add Guide failed for %s at frame %s: %s"
                % (guide.get("kind", "guide"), guide.get("frame_idx"), exc)
            ) from exc
    return conditioning


def _execute_h3_independent_first(
    clip, vae, audio_vae, prompt: str, width: int, height: int, length: int,
    ref_image_size: str, ref_images: dict[str, torch.Tensor],
    video_specs: list[dict[str, Any]], ref_audios: dict[str, dict[str, Any]],
) -> tuple[Any, Any]:
    """H3 Ref2VA in the official presentation order.

    References enter the tokenizer as images, then each video's soundtrack (when
    present) immediately before that video, then standalone audio.  This matches
    MiniMaxH3ReferenceToVideo, so <Audio N> numbering is stable against the
    official guide.

    Reference videos are decoded and encoded one spec at a time, so only one
    clip's frames are resident; decoding them all up front held several
    full-canvas float32 clips in memory at once.
    """

    latent, frame_count = h3_nodes._empty_av_latent(width, height, length)
    ref_items: list[dict[str, Any]] = []
    ref_blocks: list[dict[str, Any]] = []

    for image in (ref_images or {}).values():
        if image is None:
            continue
        image_height, image_width = image.shape[1], image.shape[2]
        target_width, target_height = _ref_image_target_size(
            image_width, image_height, width, height, ref_image_size
        )
        resized = h3_nodes._resize(image[:1], target_width, target_height, "disabled")
        encoded = vae.encode(resized)
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({
            "kind": "image", "latent_h": target_height // 16,
            "latent_w": target_width // 16, "latent": encoded,
        })

    for spec in video_specs or []:
        reference_width = int(_float(spec.get("reference_width"), width))
        reference_height = int(_float(spec.get("reference_height"), height))
        limit_width, limit_height = h3_nodes.adapt_canvas(reference_width, reference_height)
        decode_width, decode_height = _reference_decode_size(
            reference_width, reference_height, limit_width, limit_height
        )
        path = _safe_input_path(str(spec["file"]))
        video_frames = _decode_video(
            path, spec["source_start"], spec["duration"], FPS, decode_width, decode_height,
        )
        if video_frames is None or video_frames.shape[0] < 5:
            continue
        # The soundtrack is only decoded for a video that survived decoding, so
        # a failed clip cannot shift the <Audio N> ordinals.
        soundtrack = None
        if spec.get("has_audio"):
            soundtrack = _decode_audio(path, spec["source_start"], spec["duration"])
        video_height, video_width = video_frames.shape[1], video_frames.shape[2]
        canvas_width, canvas_height = h3_nodes.adapt_canvas(video_width, video_height)
        if video_width * video_height < canvas_width * canvas_height:
            canvas_width = max(
                h3_nodes.CANVAS_MULTIPLE,
                round(video_width / h3_nodes.CANVAS_MULTIPLE) * h3_nodes.CANVAS_MULTIPLE,
            )
            canvas_height = max(
                h3_nodes.CANVAS_MULTIPLE,
                round(video_height / h3_nodes.CANVAS_MULTIPLE) * h3_nodes.CANVAS_MULTIPLE,
            )
        frames = h3_nodes._resize(video_frames, canvas_width, canvas_height, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        frame_total = frames.shape[0]
        if frame_total < 5:
            raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
        while frame_total % 17 != 5:
            frame_total -= 1
        frames = frames[:frame_total]
        video_latent = vae.encode(frames)
        audio_latent, ref_audio_t = (None, 0)
        if soundtrack is not None:
            audio_latent, ref_audio_t = h3_nodes._encode_ref_audio(audio_vae, soundtrack)
            ref_items.append({"type": "audio"})
        sample_indices = list(range(0, frames.shape[0], h3_nodes.FPS // 2))
        qwen_frames = frames[sample_indices]
        ref_items.append({
            "type": "video", "data": qwen_frames,
            "timestamps": [index / 2.0 for index in range(len(sample_indices))],
        })
        ref_blocks.append({
            "kind": "video_audio" if ref_audio_t else "video",
            "latent_t": video_latent.shape[2], "latent_h": canvas_height // 16,
            "latent_w": canvas_width // 16, "ref_audio_t": ref_audio_t,
            "latent": video_latent, "audio_latent": audio_latent,
        })
        # Only the latent and the 2 fps Qwen frames are needed from here on.
        del video_frames, frames

    # Standalone audio follows the paired video soundtracks, as in the official
    # MiniMaxH3ReferenceToVideo presentation order.
    for audio in (ref_audios or {}).values():
        if audio is None:
            continue
        audio_latent, ref_audio_t = h3_nodes._encode_ref_audio(audio_vae, audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if ref_blocks:
        conditioning = node_helpers.conditioning_set_values(
            conditioning, {"minimax_refs": ref_blocks}
        )
    return conditioning, latent


async def _acquire_http_slot(semaphore: asyncio.Semaphore) -> bool:
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=2.0)
        return True
    except asyncio.TimeoutError:
        return False


async def _request_json(request: web.Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise ValueError("Request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    return payload


@PromptServer.instance.routes.post("/h3_director_custom/media_info")
async def media_info(request: web.Request) -> web.Response:
    acquired = False
    try:
        payload = await _request_json(request)
        kind = str(payload.get("kind") or "").lower()
        if kind not in MEDIA_EXTENSIONS:
            raise ValueError("Media type must be image, audio, or video")
        path = _safe_uploaded_path(str(payload.get("filename") or ""), kind)
        acquired = await _acquire_http_slot(_MEDIA_INFO_SEMAPHORE)
        if not acquired:
            return web.json_response({"error": "Media inspection is busy; try again shortly"}, status=429)
        info = await asyncio.get_running_loop().run_in_executor(
            None, _uploaded_media_info, path, kind
        )
        return web.json_response(info)
    except FileNotFoundError:
        return web.json_response({"error": "Uploaded media was not found"}, status=404)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception:
        log.exception("Media inspection failed")
        return web.json_response({"error": "Cannot inspect the uploaded media"}, status=400)
    finally:
        if acquired:
            _MEDIA_INFO_SEMAPHORE.release()


@PromptServer.instance.routes.post("/h3_director_custom/preview_proxy")
async def preview_proxy(request: web.Request) -> web.Response:
    """Return (and lazily create) the cached low-resolution monitoring proxy."""

    acquired = False
    try:
        payload = await _request_json(request)
        path = _safe_uploaded_path(str(payload.get("filename") or ""), "video")
        acquired = await _acquire_http_slot(_PREVIEW_SEMAPHORE)
        if not acquired:
            return web.json_response({"error": "Preview generation is busy; try again shortly"}, status=429)
        proxy = await asyncio.get_running_loop().run_in_executor(None, _ensure_preview_proxy, path)
        return web.json_response({
            "proxy": proxy,
            "width": PREVIEW_MAX_WIDTH,
            "height": PREVIEW_MAX_HEIGHT,
            "fps": PREVIEW_FPS,
        })
    except FileNotFoundError:
        return web.json_response({"error": "Uploaded video was not found"}, status=404)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception:
        log.exception("Preview proxy failed")
        return web.json_response({"error": "Cannot create the preview video"}, status=400)
    finally:
        if acquired:
            _PREVIEW_SEMAPHORE.release()


@PromptServer.instance.routes.post("/h3_director_custom/normalize_prompt")
async def normalize_prompt_preview(request: web.Request) -> web.Response:
    """Return the exact prompt text the text encoder will receive.

    The panel's preview button calls this instead of re-implementing the cleanup
    in JavaScript, so the preview cannot drift from the backend.  No model is
    loaded and no media is touched.
    """

    try:
        payload = await _request_json(request)
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if len(prompt) > MAX_PROMPT_PREVIEW_CHARS:
            raise ValueError("prompt is too long")
        normalized = await asyncio.get_running_loop().run_in_executor(
            None, _normalize_prompt, prompt
        )
        return web.json_response({"prompt": normalized})
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception:
        log.exception("Prompt preview failed")
        return web.json_response({"error": "Cannot normalize the prompt"}, status=400)


def _encode_timeline_plan(
    plan: dict[str, Any], clip, vae, audio_vae, prompt: str, ref_image_size: str,
) -> tuple[Any, Any]:
    plan = _require_timeline_plan(plan)
    timeline = plan["timeline"]
    target_width, target_height = int(plan["width"]), int(plan["height"])
    length = int(plan["length"])
    ref_images, video_specs, ref_audios, guides = _build_references(
        timeline, target_width, target_height, length
    )
    if not (ref_images or video_specs or ref_audios or guides):
        log.info(
            "No reference materials in the timeline plan; encoding MiniMax H3 as pure text-to-video."
        )
    log.info(
        "Encoding planned H3 refs at %dx%d: %d images, %d videos, %d standalone audios, %d native guides; %d output frames",
        target_width, target_height, len(ref_images), len(video_specs),
        len(ref_audios), len(guides), length,
    )
    conditioning, latent = _execute_h3_independent_first(
        clip, vae, audio_vae, prompt, target_width, target_height, length,
        ref_image_size, ref_images, video_specs, ref_audios,
    )
    conditioning = _apply_h3_guides(conditioning, latent, vae, audio_vae, guides)
    return conditioning, latent


_MODEL_CACHE: dict[str, tuple[str, Any]] = {}


def _combo_files(kind: str, *prefer: str) -> tuple[list[str], str]:
    files = list(folder_paths.get_filename_list(kind)) or ["none"]
    for needle in prefer:
        for name in files:
            if needle.lower() in name.lower():
                return files, name
    return files, files[0]


def _cached_model(slot: str, name: str):
    entry = _MODEL_CACHE.get(slot)
    return entry[1] if entry is not None and entry[0] == name else None


def _store_model(slot: str, name: str, model):
    # One slot per role. Replacing the tuple drops the previous model, so a node
    # that hops between files cannot keep every encoder/VAE it ever touched alive.
    _MODEL_CACHE[slot] = (name, model)
    return model


def _load_clip_model(clip_name: str):
    """Load the H3 text encoder once per filename and reuse it across runs."""
    cached = _cached_model("clip", clip_name)
    if cached is not None:
        return cached
    path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    clip_type = getattr(comfy.sd.CLIPType, "MINIMAX", comfy.sd.CLIPType.STABLE_DIFFUSION)
    clip = comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type,
        model_options={},
    )
    return _store_model("clip", clip_name, clip)


def _load_vae_model(vae_name: str, slot: str):
    """Load a VAE once per role; ComfyUI still manages moving it on and off the GPU."""
    cached = _cached_model(slot, vae_name)
    if cached is not None:
        return cached
    path = folder_paths.get_full_path_or_raise("vae", vae_name)
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    return _store_model(slot, vae_name, vae)


# ---- Prompt hygiene ---------------------------------------------------------
# The text encoder sees the prompt string verbatim, so missing spaces around tags
# and after punctuation end up in the tokens ("heels.<Picture 1>", "< image 1 >",
# "1,then"). The string is cleaned once per run; the widget text stays untouched.
_PROMPT_TAG_RE = re.compile(
    r"<[^<>\n]{1,200}>|\[[^\[\]\n]{1,200}\]|\(\s*S\d+(?:\s*,\s*S\d+)*\s*\)", re.IGNORECASE
)
_PROMPT_PUNCTUATION = ".,:;!?"
_PROMPT_OPENERS = "([{（【《"
_PROMPT_QUOTES = "\"'“‘"
_PROMPT_CLOSERS = "\"')]}>,.;:!?」』】”’。，、；：！？"


def _clean_tag(tag: str) -> str:
    """`< image 1 >` -> `<image 1>`, `[ Chinese ]` -> `[Chinese]`."""
    inner = re.sub(r"[ \t]{2,}", " ", tag[1:-1].strip())
    return f"{tag[0]}{inner}{tag[-1]}"


def _tidy_prompt(text: str) -> str:
    """Trim line-end whitespace without touching the line breaks the user typed."""

    lines = [line.rstrip() for line in text.replace("\r", "").split("\n")]
    return "\n".join(lines).strip("\n")


def _space_after_punctuation(text: str, tail: str) -> str:
    """Insert the missing space after `,`/`.`/`:` when a word follows."""
    out: list[str] = []
    for index, char in enumerate(text):
        out.append(char)
        if char not in _PROMPT_PUNCTUATION:
            continue
        following = text[index + 1] if index + 1 < len(text) else tail
        # Timecodes, decimals and thousands separators ("00:02.000", "1,000")
        # are numbers, not sentence punctuation.
        if not following or not (following.isalpha() or following in _PROMPT_OPENERS):
            continue
        # Initialisms ("U.S.A.", "e.g.") keep their dots glued to the letters.
        if char == "." and index > 0 and text[index - 1].isalpha():
            if index < 2 or not text[index - 2].isalpha():
                continue
        out.append(" ")
    return "".join(out)


def _seam_space(previous: str, previous_is_tag: bool, current: str, current_is_tag: bool) -> bool:
    """Whether a space belongs between two adjacent pieces."""
    if not previous or not current or previous[-1].isspace() or current[0].isspace():
        return False
    # "<d>" swallows the space that follows it and "</d>" the one before it, so
    # the marked dialogue stays "<d>[English] text</d>".
    if previous_is_tag and previous.lower() == "<d>":
        return False
    if current_is_tag and current.lower() == "</d>":
        return False
    if current_is_tag:
        return previous[-1] not in _PROMPT_OPENERS and previous[-1] not in _PROMPT_QUOTES
    if previous_is_tag:
        return current[0] not in _PROMPT_CLOSERS
    return False


def _normalize_prompt(prompt: str) -> str:
    """Clean the prompt that the text encoder sees.

    Only tag and punctuation spacing is normalized; the line breaks typed in
    the editor are preserved as-is.
    """
    text = str(prompt or "")
    if not text.strip():
        return text
    pieces: list[tuple[bool, str]] = []
    cursor = 0
    for match in _PROMPT_TAG_RE.finditer(text):
        if match.start() > cursor:
            pieces.append((False, text[cursor:match.start()]))
        pieces.append((True, _clean_tag(match.group(0))))
        cursor = match.end()
    if cursor < len(text):
        pieces.append((False, text[cursor:]))

    out = ""
    previous = ""
    previous_is_tag = False
    in_dialogue = False
    for index, (is_tag, value) in enumerate(pieces):
        if is_tag:
            lowered = value.lower()
            if lowered == "<d>":
                in_dialogue = True
            elif lowered == "</d>":
                in_dialogue = False
        else:
            # Dialogue is quoted verbatim, so its spacing is left exactly as typed.
            if not in_dialogue:
                value = value.replace("\u00a0", " ").replace("\u3000", " ")
                value = re.sub(r"[ \t]{2,}", " ", value)
            tail = pieces[index + 1][1][:1] if index + 1 < len(pieces) else ""
            value = _space_after_punctuation(value, tail)
        if value and out and _seam_space(previous, previous_is_tag, value, is_tag):
            out += " "
        out += value
        previous = value
        previous_is_tag = is_tag
    return _tidy_prompt(out)


class MiniMaxH3TimelineDirector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        clip_files, clip_default = _combo_files("text_encoders", "minimax", "qwen3vl")
        vae_files, video_vae_default = _combo_files("vae", "minimax_h3_video", "minimax_h3")
        _, audio_vae_default = _combo_files("vae", "minimax_h3_audio")
        if "audio" not in audio_vae_default.lower():
            audio_vae_default = next((n for n in vae_files if "audio" in n.lower()), vae_files[0])
        return io.Schema(
            node_id="MiniMaxH3TimelineDirectorCustom",
            display_name="MiniMaxH3导演台",
            description=(
                "Assemble H3 references on an editable timeline. Overlapping video can use native Add Guide, "
                "editable reference, or boundary-only mode; gaps automatically anchor their boundary frames."
            ),
            category="model/conditioning/minimax",
            inputs=[
                io.Combo.Input("clip_name", options=clip_files, default=clip_default,
                               tooltip="MiniMax H3 text encoder. Loaded inside this node; no CLIPLoader needed."),
                io.Combo.Input("video_vae_name", options=vae_files, default=video_vae_default,
                               tooltip="Video VAE for reference encoding and latent decoding."),
                io.Combo.Input("audio_vae_name", options=vae_files, default=audio_vae_default,
                               tooltip="Audio VAE for reference soundtracks and the generated audio stream."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Combo.Input(
                    "ref_image_size", options=["match", "average", "max"], default="match",
                    tooltip=(
                        "Reference image sizing. match: scale down to the generation pixel area; "
                        "average: midway between match and max; "
                        "max: original resolution, downscaled only above the 2048 short edge."
                    ),
                ),
                io.String.Input("timeline_data", default="", multiline=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
                io.Vae.Output(display_name="video_vae",
                              tooltip="The video VAE loaded by this node, exposed for other nodes."),
                io.Vae.Output(display_name="audio_vae",
                              tooltip="The audio VAE loaded by this node, exposed for other nodes."),
                io.Int.Output(display_name="segment_index",
                              tooltip="1-based number of the segment currently selected in the panel "
                                      "(\"第 N 段\"). Use it to label or route per-segment output."),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip_name,
        video_vae_name,
        audio_vae_name,
        prompt,
        width,
        height,
        ref_image_size="match",
        timeline_data="",
    ) -> io.NodeOutput:
        _require_h3_helpers()
        # Cheap validation first: an invalid prompt or timeline should fail
        # before the 15 GB text encoder and the VAEs are loaded.
        prompt = _normalize_prompt(prompt)
        plan = _create_timeline_plan(timeline_data, width, height)
        segment_config = plan["timeline"].get("segmentConfig") or {}
        segment_count = max(1, int(_float(segment_config.get("count"), 1)))
        segment_index = min(max(1, int(_float(segment_config.get("activeIndex"), 0)) + 1), segment_count)

        clip = _load_clip_model(clip_name)
        vae = _load_vae_model(video_vae_name, "vae_video")
        audio_vae = vae if audio_vae_name == video_vae_name else _load_vae_model(audio_vae_name, "vae_audio")
        conditioning, latent = _encode_timeline_plan(
            plan, clip, vae, audio_vae, prompt, ref_image_size
        )
        return io.NodeOutput(conditioning, latent, vae, audio_vae, segment_index)
