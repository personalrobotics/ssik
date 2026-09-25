"""Encode the interactive-demo tour PNG sequence into shippable assets.

Reads a capture directory produced by::

    python examples/05_viser_interactive_ik.py --tour --tour-record-dir DIR

…which is a flat ``frame_NNNNN.png`` sequence plus a ``_manifest.json``
giving the per-arm frame ranges. Emits:

- ``docs/assets/demo_tour.mp4`` -- full tour, H.264, 30 fps, 1280x720.
- ``docs/assets/per_arm/<module>.gif`` -- looping per-arm GIF, 256px tall,
  30 fps, cropped to the arm. Sized for README / docs embedding.

GIFs are auto-cropped to the content: the capture frame is 16:9 with the
arm occupying 7-32% of it, so an un-cropped encode is mostly whitespace.
The box is the union of the non-background pixels over the arm's whole
frame range, so nothing the arm reaches during the take is cut off.
``--no-crop`` restores the full frame.

Both pipelines use ffmpeg's ``palettegen`` + ``paletteuse`` for the GIF
encode so colors don't band, plus libx264 ``crf=22`` for the MP4. Both
are idempotent — re-running overwrites existing outputs.

Usage::

    python tools/encode_demo_assets.py /tmp/ssik_tour_frames
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MP4_OUT = REPO_ROOT / "docs" / "assets" / "demo_tour.mp4"
GIF_DIR = REPO_ROOT / "docs" / "assets" / "per_arm"
MONTAGE_OUT = REPO_ROOT / "docs" / "assets" / "demo_tour_montage.gif"
# The self-motion sweep is a single hero asset rather than one of the
# per-arm roster GIFs, so it lands beside the montage instead of in per_arm/.
SELF_MOTION_KEY = "self_motion"
SELF_MOTION_OUT = REPO_ROOT / "docs" / "assets" / "self_motion.gif"

# Output GIFs are normalized on HEIGHT, not width. The capture frame is
# 16:9 with the arm occupying a small, arm-shaped part of it -- a fixed
# output width either leaves the whitespace in (what the un-cropped
# encode did: 7-32% of the frame was arm) or, once cropped, stretches a
# tall narrow arm like the Rizon 4 into a 1000px-tall README card. Fixing
# the height instead gives every arm the same apparent size in the README
# while each card stays exactly as wide as its arm.
#
# Height and palette size are jointly a page-weight budget. Cropping
# removes the static white background that GIF's inter-frame compression
# was living on, so a tight crop costs bytes: the eight README arms are
# 6.3 MB un-cropped, 17.5 MB cropped at 320px/256 colors, and 7.6 MB at
# these settings. 64 colors is not a visible compromise on flat-shaded
# renders -- 64, 128 and 256 are indistinguishable at 2x on the JACO 2's
# shading -- because the whole frame is one hue against white.
GIF_HEIGHT = 256
GIF_MAX_COLORS = 64
# Whitespace kept around the content box, as a fraction of its long side.
CROP_PAD_FRAC = 0.04
# The render background is white. Anything darker than this is content.
# The threshold is deliberately loose: it is measured 6px from the box a
# strict 250 finds, so encoder noise on the background cannot move it.
CROP_BACKGROUND_LEVEL = 244
# A row/column counts as content only if this many of its pixels are ink,
# which discards isolated speckle from a lossy intermediate.
CROP_MIN_INK_PIXELS = 2
# GIF stays at the source capture rate (30 fps). Earlier versions reduced
# to 15 fps via the ``fps`` filter, but ``-frames:v N`` is an OUTPUT-count
# limit applied AFTER the filter -- so the 30→15 downsample made ffmpeg
# consume 2N input PNGs to satisfy N output frames, slurping the next
# arm's first N/2 frames into the GIF. The fix is to stay at 30 fps so
# input and output counts match. GIFs are slightly larger but bounded by
# the correct frame range.
GIF_FPS = 30


def _run(cmd: list[str]) -> None:
    """Run a subprocess and surface its stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("FAILED:", " ".join(cmd), file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        sys.exit(1)


def content_box(frames: list[Path]) -> tuple[int, int, int, int]:
    """Union bounding box of the non-background pixels over ``frames``.

    Returns ``(x, y, w, h)`` in source pixels. Accumulates per-column and
    per-row ink counts as a running max rather than materializing the union
    mask, so memory is flat in the frame count.

    ffmpeg's own ``cropdetect`` is the obvious tool here and is wrong for
    this job: with ``reset=0`` it still settled on the first frame's box,
    which on the Rizon 4 range is 254 of the 476 rows the arm actually
    sweeps. The union has to be taken over every frame of the range.
    """
    try:
        import imageio.v3 as iio
        import numpy as np
    except ImportError:  # pragma: no cover - dev-machine ergonomics
        print(
            "auto-crop needs imageio and numpy: pip install 'ssik[demo]'\n"
            "(or pass --no-crop to encode the full frame)",
            file=sys.stderr,
        )
        sys.exit(1)

    cols = rows = None
    for frame in frames:
        ink = np.asarray(iio.imread(frame))[:, :, :3].min(axis=2) < CROP_BACKGROUND_LEVEL
        per_col, per_row = ink.sum(axis=0), ink.sum(axis=1)
        cols = per_col if cols is None else np.maximum(cols, per_col)
        rows = per_row if rows is None else np.maximum(rows, per_row)

    xs = np.nonzero(cols >= CROP_MIN_INK_PIXELS)[0]
    ys = np.nonzero(rows >= CROP_MIN_INK_PIXELS)[0]
    if not len(xs) or not len(ys):
        print(f"no content found in {len(frames)} frames -- all background?", file=sys.stderr)
        sys.exit(1)
    return int(xs[0]), int(ys[0]), int(xs[-1] - xs[0] + 1), int(ys[-1] - ys[0] + 1)


def scale_filter(frames: list[Path], crop: bool) -> str:
    """Build the ffmpeg filter chain that turns a source frame into one
    ``GIF_HEIGHT``-tall output frame.

    With ``crop``, the chain is crop -> (pad) -> scale: the content box of
    the whole range plus a margin, white-padded on any side where that box
    runs off the source (arms that touch the frame edge), then scaled. The
    same chain must feed both the ``palettegen`` and ``paletteuse`` passes,
    so the output dimensions are computed here rather than left to ``-1``.
    """
    import imageio.v3 as iio

    src_h, src_w = iio.imread(frames[0]).shape[:2]
    if not crop:
        return f"scale=-2:{GIF_HEIGHT}:flags=lanczos"

    x, y, w, h = content_box(frames)
    pad = round(CROP_PAD_FRAC * max(w, h))
    x, y, w, h = x - pad, y - pad, w + 2 * pad, h + 2 * pad
    # The padded box may hang off the source on any side. Crop what is
    # there, then white-pad back to the full box so the margin is even.
    cx, cy = max(x, 0), max(y, 0)
    cw, ch = min(x + w, src_w) - cx, min(y + h, src_h) - cy
    chain = f"crop={cw}:{ch}:{cx}:{cy}"
    if (cw, ch) != (w, h):
        chain += f",pad={w}:{h}:{cx - x}:{cy - y}:white"
    out_w = round(w * GIF_HEIGHT / h)
    return f"{chain},scale={out_w + out_w % 2}:{GIF_HEIGHT}:flags=lanczos"


def encode_full_mp4(capture_dir: Path) -> None:
    MP4_OUT.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "30",
            "-i",
            str(capture_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            str(MP4_OUT),
        ]
    )
    print(f"  mp4: {MP4_OUT.relative_to(REPO_ROOT)}  ({MP4_OUT.stat().st_size / 1e6:.2f} MB)")


def encode_per_arm_gif(
    capture_dir: Path,
    module: str,
    start: int,
    end_exclusive: int,
    out: Path | None = None,
    crop: bool = True,
) -> None:
    """Build one looping GIF from frames [start, end_exclusive).

    Uses ffmpeg's two-pass palette workflow (``palettegen`` then
    ``paletteuse``) so the GIF retains color fidelity at the cost of an
    extra ffmpeg invocation. Single-pass GIF encoding produces visibly
    banded reds against the white background.

    ``out`` overrides the default ``per_arm/<module>.gif`` destination, for
    captures that are not one of the roster arms. ``crop`` frames the GIF
    on this arm's own content box; see ``scale_filter``.
    """
    out = out if out is not None else GIF_DIR / f"{module}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_frames = end_exclusive - start
    frames = [capture_dir / f"frame_{i:05d}.png" for i in range(start, end_exclusive)]
    chain = scale_filter(frames, crop)
    palette = capture_dir / f"_palette_{module}.png"
    # Pass 1: generate optimized palette from this arm's frame range.
    # Note ``fps`` filter is intentionally absent -- see GIF_FPS comment.
    _run(
        [
            "ffmpeg",
            "-y",
            "-start_number",
            str(start),
            "-i",
            str(capture_dir / "frame_%05d.png"),
            "-frames:v",
            str(n_frames),
            "-vf",
            f"{chain},palettegen=stats_mode=diff:max_colors={GIF_MAX_COLORS}",
            str(palette),
        ]
    )
    # Pass 2: render the GIF using that palette, dithered, looping.
    _run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(GIF_FPS),
            "-start_number",
            str(start),
            "-i",
            str(capture_dir / "frame_%05d.png"),
            "-i",
            str(palette),
            "-frames:v",
            str(n_frames),
            "-lavfi",
            f"{chain}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
            "-loop",
            "0",
            str(out),
        ]
    )
    palette.unlink(missing_ok=True)
    print(f"  gif: {out.relative_to(REPO_ROOT)}  ({out.stat().st_size / 1e6:.2f} MB)")


def encode_montage_gif(
    capture_dir: Path,
    manifest: dict,
    frames_per_arm: int = 45,
    crop: bool = True,
) -> None:
    """Stitch ``frames_per_arm`` mid-range frames from each arm into one
    looping GIF. Useful for a single-attachment social post that
    showcases the whole roster without requiring 8 separate uploads.

    Frames are drawn from the middle of each arm's range to skip the
    just-settled home pose (visually static) and the late-motion frames
    (more likely to hit unreachable poses and get skipped).

    The crop here is one box shared by every arm, unlike the per-arm GIFs:
    the montage is a single image sequence, so a per-arm box would make
    the arms jump in scale as the montage cuts between them.
    """
    import shutil
    import tempfile

    MONTAGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ssik_montage_") as tmpdir:
        staging = Path(tmpdir)
        idx = 0
        for rng in manifest["frame_ranges"].values():
            start = int(rng["start"])
            end = int(rng["end_exclusive"])
            available = end - start
            take = min(frames_per_arm, available)
            offset = start + (available - take) // 2
            for k in range(take):
                src = capture_dir / f"frame_{offset + k:05d}.png"
                dst = staging / f"frame_{idx:05d}.png"
                shutil.copy(src, dst)
                idx += 1
        n_frames = idx
        chain = scale_filter([staging / f"frame_{i:05d}.png" for i in range(n_frames)], crop)
        palette = staging / "_palette.png"
        _run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(staging / "frame_%05d.png"),
                "-frames:v",
                str(n_frames),
                "-vf",
                f"{chain},palettegen=stats_mode=diff:max_colors={GIF_MAX_COLORS}",
                str(palette),
            ]
        )
        _run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(GIF_FPS),
                "-i",
                str(staging / "frame_%05d.png"),
                "-i",
                str(palette),
                "-frames:v",
                str(n_frames),
                "-lavfi",
                f"{chain}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
                "-loop",
                "0",
                str(MONTAGE_OUT),
            ]
        )
    print(
        f"  gif: {MONTAGE_OUT.relative_to(REPO_ROOT)}  "
        f"({MONTAGE_OUT.stat().st_size / 1e6:.2f} MB, {n_frames} frames @ {GIF_FPS}fps)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "capture_dir",
        type=Path,
        help="Directory of frame_NNNNN.png + _manifest.json from the tour.",
    )
    parser.add_argument(
        "--skip-mp4",
        action="store_true",
        help="Skip the combined MP4 encode (only emit per-arm GIFs).",
    )
    parser.add_argument(
        "--skip-gifs",
        action="store_true",
        help="Skip the per-arm GIF encode (only emit the MP4).",
    )
    parser.add_argument(
        "--montage",
        action="store_true",
        help="Emit a single docs/assets/demo_tour_montage.gif stitching "
        "~1.5s of each arm. Useful as a single-attachment social post.",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Encode the full capture frame instead of cropping to the arm.",
    )
    parser.add_argument(
        "--montage-frames",
        type=int,
        default=45,
        help="Frames per arm in the montage (default 45 = 1.5s @ 30fps).",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not on PATH; install with: brew install ffmpeg", file=sys.stderr)
        sys.exit(1)

    capture_dir = args.capture_dir.expanduser().resolve()
    if not capture_dir.exists():
        print(f"capture dir does not exist: {capture_dir}", file=sys.stderr)
        sys.exit(1)

    manifest_path = capture_dir / "_manifest.json"
    if not manifest_path.exists():
        print(f"missing _manifest.json in {capture_dir}", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text())

    if not args.skip_mp4:
        encode_full_mp4(capture_dir)

    if not args.skip_gifs:
        for module, rng in manifest["frame_ranges"].items():
            encode_per_arm_gif(
                capture_dir,
                module,
                int(rng["start"]),
                int(rng["end_exclusive"]),
                out=SELF_MOTION_OUT if module == SELF_MOTION_KEY else None,
                crop=not args.no_crop,
            )

    if args.montage:
        encode_montage_gif(
            capture_dir,
            manifest,
            frames_per_arm=args.montage_frames,
            crop=not args.no_crop,
        )


if __name__ == "__main__":
    main()
