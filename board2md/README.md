# board2md

`board2md` converts board photos, folders of board images, or PDF pages into Markdown with a vision model. It is tuned for math board/problem extraction and defaults to the local Qwen3.8-27B MLX model at 127.0.0.1:8085.

## Location

- Source: `tools/board2md/`
- CLI entry: `tools/board2md/board2md`
- Core module: `tools/board2md/board2md_core.py`
- Local command target: `~/.local/bin/board2md`

## Basic Usage

```bash
board2md IMG1 [IMG2 ...] -o output.md
board2md ~/Pictures/board*.jpg -o board.md
board2md ~/Downloads/board.pdf -o board.md
```

Dry-run expands input paths without calling the API or converting PDFs:

```bash
board2md ~/Downloads/board.pdf --dry-run
```

## Batch 1: Control And Debug

Process selected PDF pages:

```bash
board2md board.pdf --page-range 1,3-5,9 -o board.md
```

Render PDF pages at a custom DPI:

```bash
board2md board.pdf --render-dpi 220 -o board.md
```

Choose a built-in prompt mode:

```bash
board2md board.png --mode problem -o problem.md
board2md board.png --mode full -o full.md
board2md board.png --mode raw -o raw.md
```

Use a custom prompt:

```bash
board2md board.png --prompt-file prompts/board_problem.md -o problem.md
```

Save per-page raw output, copied image, and metadata:

```bash
board2md board.pdf --debug-dir board_debug -o board.md
```

Add Obsidian source callouts and write per-page atomic notes:

```bash
board2md board.pdf --obsidian --atomic-dir board_atomic -o board.md
```

## Batch 2: Recovery And Provider Options

Continue when one page fails:

```bash
board2md board.pdf --continue-on-error --debug-dir board_debug -o board.md
```

Reuse existing debug raw outputs instead of calling the API again:

```bash
board2md board.pdf --debug-dir board_debug --resume -o board.md
```

Set retry count:

```bash
board2md board.pdf --retries 1 -o board.md
```

Override provider-compatible API settings:

```bash
board2md board.png \
  --base-url http://127.0.0.1:8085/v1 \
  --model "$HOME/.omlx/models/Qwen3.8-27B-MLX-Mixed-4.95bpw" \
  --api-key-env QWEN_API_KEY \
  -o board.md
```

## Batch 3: Folder And Image Workflow

Process a folder of supported files:

```bash
board2md ~/Pictures/boards -o boards.md
```

Recursively scan subfolders:

```bash
board2md ~/Pictures/boards --recursive -o boards.md
```

Shrink images before sending them to the API, using macOS `sips`:

```bash
board2md ~/Pictures/boards --max-side 1800 -o boards.md
```

## Cross Validation

Use balanced cross-validation when the board content is important or fragmented:

```bash
board2md board.pdf \
  --cross-validate balanced \
  --debug-dir board_debug \
  --verification-report board_verification.json \
  -o board.md
```

Balanced mode runs two independent recognition passes, then asks a judge pass to merge agreements and flag conflicts. It writes:

- `*.pass-a.md` and `*.pass-b.md`: independent recognitions
- `*.judge.json`: page-level judge result
- `verification.json`: summary report, also written to `--verification-report` when provided

The final Markdown uses the judge's merged content. Uncertain content belongs in the audit fields, not silently in the final body.

## Lecture Synthesis

Use lecture synthesis when several board images contain scattered notes that should become a clean handout:

```bash
board2md board1.png board2.png board3.png \
  --cross-validate balanced \
  --lecture-synthesis teaching \
  --lecture-audit lecture_audit.md \
  --debug-dir lecture_debug \
  -o lecture.md
```

Teaching synthesis asks the model to connect fragments into a formal math handout. The final `lecture.md` is kept clean: no OCR chatter, no “板书体面”, no “老师口述”, and no source reconstruction language. Source provenance, uncertain links, and fragment previews go into `lecture_audit.md` and `lecture_debug/lecture_plan.json`.

## Batch 4: Profiles, Quality Gates, And RAG Outputs

Use a preset profile instead of remembering every flag:

```bash
board2md board.pdf --profile quick -o draft.md
board2md board.pdf --profile verified --debug-dir board_debug -o checked.md
board2md board1.png board2.png --profile lecture --lecture-audit audit.md -o lecture.md
```

Profiles are conservative defaults:

- `quick`: one recognition pass and fewer retries, for rough drafts
- `verified`: balanced cross-validation, for important pages
- `lecture`: balanced cross-validation, whiteboard-aware prompt, and teaching synthesis
- `custom`: no preset changes

Tell the prompt what kind of material it is:

```bash
board2md board.png --note-type whiteboard -o lecture.md
board2md scan.png --note-type paper -o scan.md
```

`whiteboard` strengthens spatial-layout reconstruction: arrows, separated clusters, side notes, erased or inserted material, and the likely human teaching order.

Write a quality report and fail the run if the final Markdown still contains process/provenance language:

```bash
board2md board.pdf \
  --profile lecture \
  --quality-report quality.json \
  --fail-on-quality \
  -o lecture.md
```

The quality report currently checks for common final-handout problems such as OCR/source chatter, unbalanced dollar delimiters, and fenced model output.

Emit structured outputs for downstream search, RAG, or later review:

```bash
board2md board.pdf \
  --output-format both \
  --chunks-output chunks.json \
  --run-report run.json \
  -o board.md
```

This writes:

- `board.md`: normal Markdown
- `board.json`: Markdown plus source map, verification report, quality report, and lecture plan when available
- `chunks.json`: heading-based semantic chunks, skipping board2md wrapper metadata
- `run.json`: item count, failure count, elapsed time, selected profile, and estimated model calls

Protect large accidental runs and reuse successful OCR calls:

```bash
board2md big_folder --recursive --max-pages 30 -o boards.md
board2md big_folder --recursive --max-pages 30 --force -o boards.md
board2md board.pdf --cache-dir .board2md-cache -o board.md
```

`--cache-dir` is for the ordinary single-pass recognition path. Cross-validation still writes auditable pass outputs under `--debug-dir`.

## MathMD Bundle And Semantic Fusion

`board2md` can now emit the same canonical content bundle as `video2md`. The source image for every board/PDF page is copied into the bundle and linked from its problem Markdown, so geometry diagrams, combinatorial diagrams, tables, and annotated board layouts remain available after transcription.

```bash
board2md board.pdf \
  --profile verified \
  --bundle-dir board_bundle \
  --bundle-fuse auto \
  --bundle-fuse-engine semantic \
  --bundle-fuse-rounds 3 \
  -o board_bundle/board_notes.md
```

The bundle contains:

- `problems/`: one Markdown content unit per page/image, plus copied assets
- `index.md`: navigable content index
- `mathmd_manifest.json`: source provenance, hashes, model, and schema metadata
- `quality_report.json`: Markdown and missing-image checks
- `chunks.json`: heading-level retrieval chunks
- `fusion_manifest.json`: pair decisions and fusion rounds when fusion is enabled
- `problems_raw_segments/`: pre-fusion backups

Semantic fusion uses only the generated Markdown when deciding whether adjacent units share one problem or proof. It then performs a separate no-loss audit, deterministically restores any missing formulas, source markers, image links, and original-segment references, and falls back to a lossless side-by-side merge if the model introduced unsupported content.

For mixed video, image, and PDF material, prefer the unified `math2md` command described in [the Math2MD guide](../math2md/README.md).

## Environment

Optional by default:

```bash
QWEN_API_KEY=...
```

Optional compatibility variables:

```bash
QWEN_BASE_URL=http://127.0.0.1:8085/v1
QWEN_MODEL="$HOME/.omlx/models/Qwen3.8-27B-MLX-Mixed-4.95bpw"
```

PDF input requires `pdftoppm` from Poppler:

```bash
brew install poppler
```

## Tests

The tests use only the Python standard library:

```bash
python3 -B -m unittest discover -s tools/board2md/tests -v
```
