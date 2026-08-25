# video2md

`video2md` turns a math lecture video into per-problem Markdown with transcript context, frame understanding, and selected screenshots. It uses the local Qwen3.8-27B MLX endpoint at `http://127.0.0.1:8085/v1` by default.

## Basic Usage

```bash
video2md lesson.mp4 --out lesson_mathmd --interval 60
```

The normal pipeline extracts audio, transcribes it with `mlx_whisper`, extracts fixed/scene frames, asks Qwen to understand the frames, segments the lecture, writes one Markdown file per problem, injects useful geometry/combinatorics screenshots, and semantically fuses adjacent fragments.

## Semantic Fusion

Semantic fusion is the default for new CLI runs:

```bash
video2md lesson.mp4 \
  --out lesson_mathmd \
  --fuse-problems auto \
  --fuse-engine semantic \
  --fuse-rounds 3 \
  --fusion-threshold 0.72
```

The fusion stage:

1. Reads only adjacent generated Markdown files.
2. Uses local Qwen to distinguish a continuous problem/proof from merely similar independent examples.
3. Merges related units and removes semantic repetition.
4. Runs a no-loss audit.
5. Deterministically preserves formulas, source timestamps, screenshot links, and original segment names.
6. Falls back to a lossless side-by-side merge if the audit detects content not supported by either source fragment.
7. Stops when a round finds no sufficiently related adjacent pair.

Use the deterministic legacy rules engine when no model call is desired:

```bash
video2md --fuse-only --out lesson_mathmd --fuse-engine rules --fuse-rounds 3
```

Run semantic fusion again on an existing output:

```bash
video2md --fuse-only \
  --out lesson_mathmd \
  --fuse-problems auto \
  --fuse-engine semantic \
  --fuse-rounds 3
```

## Output Bundle

Each output is also a canonical MathMD bundle:

- `problems/`: final problem Markdown and selected screenshot assets
- `index.md`: navigable content index
- `mathmd_manifest.json`: source/model/hash metadata
- `quality_report.json`: Markdown and image-link checks
- `chunks.json`: retrieval-ready chunks
- `fusion_manifest.json`: pair scores, reasons, and round results
- `problems_raw_segments/`: original pre-fusion files, with prior versions retained under `history/` when needed

## Local Qwen Override

```bash
video2md lesson.mp4 \
  --base-url http://127.0.0.1:8085/v1 \
  --model "$HOME/.omlx/models/Qwen3.8-27B-MLX-Mixed-4.95bpw"
```

The local endpoint does not require an API key. `QWEN_API_KEY` remains supported for compatible authenticated endpoints.

## Tests

```bash
python3 tools/video2md_review/test_video2md_offline.py
python3 tools/video2md_review/test_video2md_semantic_fusion.py
```
