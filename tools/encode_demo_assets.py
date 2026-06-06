"""Encode the interactive-demo tour PNG sequence into shippable assets.

Reads a capture directory produced by::

    python examples/05_viser_interactive_ik.py --tour --tour-record-dir DIR

…which is a flat ``frame_NNNNN.png`` sequence plus a ``_manifest.json``
giving the per-arm frame ranges. Emits:

- ``docs/assets/demo_tour.mp4`` — full tour, H.264, 30 fps, 1280×720.
- ``docs/assets/per_arm/<module>.gif`` — looping per-arm GIF, 480px wide,
  15 fps. Sized for README / docs embedding.

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

GIF_WIDTH = 480
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


def encode_full_mp4(capture_dir: Path) -> None:
    MP4_OUT.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", str(capture_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        str(MP4_OUT),
    ])
    print(f"  mp4: {MP4_OUT.relative_to(REPO_ROOT)}  ({MP4_OUT.stat().st_size / 1e6:.2f} MB)")


def encode_per_arm_gif(capture_dir: Path, module: str, start: int, end_exclusive: int) -> None:
    """Build one looping GIF from frames [start, end_exclusive).

    Uses ffmpeg's two-pass palette workflow (``palettegen`` then
    ``paletteuse``) so the GIF retains color fidelity at the cost of an
    extra ffmpeg invocation. Single-pass GIF encoding produces visibly
    banded reds against the white background.
    """
    GIF_DIR.mkdir(parents=True, exist_ok=True)
    out = GIF_DIR / f"{module}.gif"
    n_frames = end_exclusive - start
    palette = capture_dir / f"_palette_{module}.png"
    # Pass 1: generate optimized palette from this arm's frame range.
    # Note ``fps`` filter is intentionally absent -- see GIF_FPS comment.
    _run([
        "ffmpeg", "-y",
        "-start_number", str(start),
        "-i", str(capture_dir / "frame_%05d.png"),
        "-frames:v", str(n_frames),
        "-vf", f"scale={GIF_WIDTH}:-1:flags=lanczos,palettegen=stats_mode=diff",
        str(palette),
    ])
    # Pass 2: render the GIF using that palette, dithered, looping.
    _run([
        "ffmpeg", "-y",
        "-framerate", str(GIF_FPS),
        "-start_number", str(start),
        "-i", str(capture_dir / "frame_%05d.png"),
        "-i", str(palette),
        "-frames:v", str(n_frames),
        "-lavfi",
        f"scale={GIF_WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
        "-loop", "0",
        str(out),
    ])
    palette.unlink(missing_ok=True)
    print(f"  gif: {out.relative_to(REPO_ROOT)}  ({out.stat().st_size / 1e6:.2f} MB)")


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
            )


if __name__ == "__main__":
    main()
