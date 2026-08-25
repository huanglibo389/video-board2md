# mathmd_shared

Internal shared infrastructure for `board2md`, `video2md`, and `math2md`.

It provides:

- One keyless OpenAI-compatible Qwen client with retries, text, and multi-image requests.
- Canonical Markdown cleanup, Obsidian math normalization, quality checks, and chunks.
- A common `mathmd-bundle` schema for board/PDF/video outputs.
- Asset copying and image-link validation.
- Idempotent bundle merging with cross-source provenance.
- Candidate-limited semantic alignment that places matching board/PDF and video units next to each other before fusion.
- Multi-round semantic adjacent fusion with no-loss and unsupported-addition audits, safe fallback, and raw-segment history.

This package is intentionally internal; end users should invoke `board2md`, `video2md`, or `math2md`.
