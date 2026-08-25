# Math2MD

Math2MD turns math lecture videos, board photos, scanned pages, and PDFs into one
structured Markdown course bundle. It combines the temporal context of Video2MD with
the high-resolution formula and diagram recognition of Board2MD.

## What the integration adds

- One command for mixed video, image, PDF, and directory inputs.
- A canonical `problems/` plus `index.md` bundle with provenance and quality metadata.
- Video transcript context, scene frames, and selected geometry/combinatorics screenshots.
- Board/PDF layout recognition, page provenance, cross-validation, and lecture synthesis.
- Semantic board-to-video alignment before fusion.
- Multi-round Markdown-only fusion with no-loss and unsupported-addition audits.
- Deterministic preservation of formulas, image links, source timestamps, and raw segments.

## Requirements

- Python 3.10 or newer.
- macOS on Apple Silicon for the local `mlx_whisper` transcription path.
- `ffmpeg` for video/audio/frame processing.
- Poppler (`pdftoppm`) for PDF inputs.
- A local OpenAI-compatible text-and-vision endpoint. The default endpoint is
  `http://127.0.0.1:8085/v1` and the default model path is
  `$HOME/.omlx/models/Qwen3.8-27B-MLX-Mixed-4.95bpw`.

Install system and Python dependencies:

```bash
brew install ffmpeg poppler
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install mlx-whisper
```

Install local command links:

```bash
./install.sh
```

If your local server exposes a different model identifier, configure it explicitly:

```bash
export QWEN_BASE_URL=http://127.0.0.1:8085/v1
export QWEN_MODEL=/path/or/model-id
```

The local endpoint is keyless by default. `QWEN_API_KEY` is supported for authenticated
OpenAI-compatible endpoints. Math2MD does not silently fall back to a cloud vision model.

## Usage

```bash
math2md lesson.mp4 board_photos/ handout.pdf \
  --out-root course_mathmd \
  --profile verified \
  --interval 60 \
  --fuse auto \
  --fuse-rounds 3
```

Preview routing without running media processing or the model:

```bash
math2md inputs/ --out-root course_mathmd --dry-run
```

The final course entry is `course_mathmd/course/index.md`. Existing complete source
bundles are reused unless `--force` is supplied.

The original commands remain available:

```bash
board2md board.pdf --profile verified --bundle-dir board_bundle -o board.md
video2md lesson.mp4 --out lesson_bundle --interval 60
```

## Tests

```bash
python3 -m py_compile mathmd_shared/*.py math2md/*.py board2md/board2md_core.py video2md_review/video2md
python3 -m unittest discover -s mathmd_shared/tests -p 'test_*.py' -v
python3 -m unittest discover -s math2md/tests -p 'test_*.py' -v
python3 -m unittest discover -s board2md/tests -p 'test_*.py' -v
python3 video2md_review/test_video2md_offline.py -v
python3 video2md_review/test_video2md_semantic_fusion.py -v
```

## Privacy and copyright

By default, recognition and fusion use the configured local endpoint. Generated manifests
contain source paths, so inspect outputs before sharing them. Do not commit downloaded
videos, private classroom recordings, transcripts, screenshots, or copyrighted course
materials unless you have permission to redistribute them.

## License

Math2MD is released under the [MIT License](LICENSE).
