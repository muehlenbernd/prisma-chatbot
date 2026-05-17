# Architecture

Design decisions and rationale for prisma-chatbot. This document captures
*why* the system is built the way it is, not just *what* it does. See
[`CLAUDE.md`](CLAUDE.md) for the higher-level project framing and
[`ROADMAP.md`](ROADMAP.md) for the deployment plan.

## System overview

> **TODO:** High-level diagram or description — user → Gradio UI → single
> LLM call (dual-role prompt) → response + evaluation → UI.

## Key design decisions

### Dual-role prompt, single LLM call per turn

> **TODO:** Rationale — one call keeps latency and cost predictable, and
> keeps response and evaluation grounded in the same context. Trade-off:
> prompt is more complex than two separate calls would be.

### Structured JSON output

> **TODO:** Rationale — JSON with `response` (string) and `evaluation`
> (object of six attribute scores 1–7) makes parsing and display
> deterministic. Trade-off: the model occasionally produces malformed
> output and needs validation/repair.

### Six evaluation attributes

> **TODO:** Document the attributes (competent, knowledgeable,
> well-prepared, helpful, likeable, pedantic) and why they are chosen to
> match the CMCL/EMNLP study — the demo is a faithful artifact of the
> research, not a redesigned version of it.

### Llama 3.3 70B Instruct via HF Inference API

> **TODO:** Rationale — hosted inference removes deployment complexity for
> a public demo; a 70B-class instruct model is needed for reliable
> structured output and persona adherence. Trade-off: dependency on HF
> endpoint availability and rate limits.

### Gradio on Hugging Face Spaces

> **TODO:** Rationale — lowest-friction path to a public, shareable
> artifact; integrates naturally with HF Inference; the research audience
> is already familiar with the platform.

## Module responsibilities

> **TODO:** Expand each line below with a short description once the
> module is implemented; link back to the relevant design decisions above.

- `src/config.py` — tunables and constants
- `src/prompt.py` — dual-role system prompt construction
- `src/inference.py` — HF Inference API client wrapper
- `src/evaluation.py` — score parsing, validation, display formatting
- `app.py` — Gradio UI assembly and event wiring

## Open design questions

See the "Open Questions" section in [`CLAUDE.md`](CLAUDE.md).
