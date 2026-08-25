# math2md

`math2md` is the unified entry point for `video2md` and `board2md`. It accepts any mixture of lecture videos, board photos, scanned pages, PDFs, or directories and produces one cross-source course bundle.

The integration is deliberately deeper than command chaining:

- Video contributes temporal segmentation, speech context, scene frames, and screenshot selection.
- Board/PDF processing contributes high-resolution page reading, layout understanding, cross-validation, and lecture synthesis.
- Both write one canonical MathMD schema with source provenance and quality metadata.
- Each source bundle performs local adjacent fusion first.
- A cross-source alignment pass pairs matching board/PDF and video units, moves each accepted pair next to each other, and records every decision.
- A second fusion can then combine a video explanation with its matching board/PDF problem while refusing unrelated examples.
- Every semantic merge receives a separate no-loss audit and deterministic formula/image/source safeguards; if the audit detects model-invented content, MathMD falls back to a lossless side-by-side merge.

## One Command

```bash
math2md lesson.mp4 board_photos/ handout.pdf \
  --out-root course_mathmd \
  --profile verified \
  --interval 60 \
  --fuse auto \
  --fuse-rounds 3
```

Directories are scanned recursively by default. Incomplete `.part` downloads and split-format files such as `.f137.mp4` are ignored.

Preview routing without processing media or calling Qwen:

```bash
math2md lesson_folder/ --out-root course_mathmd --dry-run
```

## Useful Modes

Fast draft:

```bash
math2md inputs/ --out-root draft --profile quick --fuse-engine rules
```

Verified board/PDF recognition plus semantic fusion:

```bash
math2md inputs/ \
  --out-root checked_course \
  --profile verified \
  --fuse-engine semantic \
  --fusion-threshold 0.72
```

Teaching-handout synthesis for board material:

```bash
math2md inputs/ --out-root lecture_course --profile lecture
```

Disable all fusion while retaining canonical bundles:

```bash
math2md inputs/ --out-root raw_course --fuse never
```

Completed source bundles are reused on reruns. Pass `--force` to regenerate them.

## Output Layout

```text
course_mathmd/
├── index.md
├── run_report.json
├── board/
│   ├── board_notes.md
│   ├── problems/
│   └── mathmd_manifest.json
├── videos/
│   └── 001_lesson/
│       ├── problems/
│       └── mathmd_manifest.json
└── course/
    ├── index.md
    ├── problems/
    ├── mathmd_manifest.json
    ├── quality_report.json
    ├── chunks.json
    ├── alignment_manifest.json
    └── fusion_manifest.json
```

The final deliverable is `course/index.md`. `alignment_manifest.json` records candidate scores, accepted board/video pairs, and the resulting order. `run_report.json` records routing, completed bundles, failures, elapsed time, alignment, and fusion results.

## Tests

```bash
python3 -m unittest discover -s tools/math2md/tests -p 'test_*.py' -v
```
