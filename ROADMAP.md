# Roadmap

Deployment plan and milestones for prisma-chatbot. Living document — items
move from *Planned* to *In progress* to *Done* as the project evolves.

## Milestone 1 — Scaffolding

> **Done:** Repo skeleton, docs, dependency manifest, env template,
> gitignore. No chatbot logic yet.

- [x] Repo created, license added
- [x] CLAUDE.md, README, ARCHITECTURE, ROADMAP drafts
- [x] `src/`, `tests/`, `assets/` directories with placeholder modules
- [x] `requirements.txt`, `.gitignore`, `.env.example`

## Milestone 2 — Minimal end-to-end loop

> **Done:** Get a single message in / response + evaluation out working
> locally, even with a rough prompt.

- [x] Implement `src/config.py` (model id, attributes, turn cap)
- [x] Implement `src/prompt.py` (v1 dual-role prompt)
- [x] Implement `src/inference.py` (inference client wrapper; originally HF,
      now Groq — see Milestone 4b)
- [x] Implement `src/evaluation.py` (JSON parsing + validation)
- [x] Implement `app.py` (minimal Gradio UI)
- [x] First pytest tests for parsing/validation

## Milestone 3 — Prompt and UX iteration

> **Done:** Refine Prisma's voice, evaluation display, and the impressions
> panel.

- [x] Refine system prompt for voice consistency and structured-output
      reliability
- [x] Evaluation display: **both** numeric score and verbal intensifier
      (e.g. *quite polite (5/7)*)
- [x] Update cadence: **always-visible** impressions panel, refreshed each
      turn
- [x] Graphical impressions: colored bar cells per attribute + trajectory
      plot across turns
- [x] Per-turn impression navigation (dropdown + chat message highlight)
- [x] Error handling: failed turns surfaced as toasts, not fake assistant
      messages; user text kept in input for retry
- [x] Dark theme, custom CSS, header/footer, embedded figures

## Milestone 4 — Public deployment

> **Done:** Shipped to a Hugging Face Space and linked from the README.

- [x] Hugging Face Space configuration
- [x] Per-session turn cap (12 turns)
- [x] Public URL added to README and papers
- [x] Mobile-responsive layout (footer and impressions panel)

## Milestone 4b — Post-launch hardening

> **Done:** Keep the live demo stable after initial launch.

- [x] Migrate inference backend from Hugging Face Inference API to Groq
      (same model, faster generation, no monthly credit cliff)
- [x] Auto-retry on `EvaluationParseError` (up to 5 attempts per turn) to
      mask stochastic JSON drift under `json_object` mode
- [x] Gradio 6.x compatibility fixes (theme/css on `launch()`, Chatbot API)

## Milestone 4c — Analytics

> **Done:** Lightweight, privacy-friendly usage analytics via Goatcounter.

- [x] Goatcounter script injected into page `<head>` via Gradio's `head`
      parameter on `demo.launch()`
- [x] Dedicated PRISMA site at `prisma-rolandm.goatcounter.com` (separate
      from the main personal site — keeps demo traffic as a clean standalone
      signal, useful for measuring TDS/LinkedIn post impact)

## Milestone 5 — Stretch ideas

> **Planned:** Explicitly non-blocking; consider only after the demo is live
> and stable.

- [ ] About panel copy (`assets/about.md`)
- [x] Rate limiting (per-IP daily session cap + Groq project-level daily budget) — done in Milestone 6
- [ ] Light usage analytics (anonymous, aggregate)
- [ ] Downloadable conversation + evaluation transcript
- [ ] Linguistic feature highlighting (which words/choices shifted scores)

## Milestone 6 — Model migration & cost control

> **Done:** Forced migration from deprecated `llama-3.3-70b-versatile` to
> `openai/gpt-oss-120b`; structured-output reliability fix; rate limiting.

- [x] Model id updated to `openai/gpt-oss-120b`
- [x] Evaluation calls switched from `json_object` to `json_schema` strict mode
      — eliminates `EvaluationParseError` at the generation level
- [x] Retry logic overhauled: one defensive transport retry only (no more 5×
      parse-error loop)
- [x] `CapacityError` added; Groq 429 (daily limit hit) surfaced as a friendly
      toast rather than a raw error
- [x] Per-IP daily session cap implemented in `app.py` via `gr.Request`
- [x] Groq project-level rate limits confirmed set in the Groq console
      (10,000 RPD / 3,000,000 TPD for `openai/gpt-oss-120b`)
- [x] Pytest suite updated (42 tests, all passing)
- [x] Docs updated: `ROADMAP.md`, `CLAUDE.md`, `README.md`, `ARCHITECTURE.md`


## Future work (v2+)

### Attribute customization
- User selects up to 6 attributes from a curated extended list (~15–20
  dimensions covering social, cognitive, affective, and pragmatic
  perception)
- Extended list includes: pushy, knowledgeable, well-prepared, pedantic,
  helpful, arrogant, warm, evasive, confident, anxious, etc. (final set
  TBD)
- v1's default set remains as the "quick start" option

### Compare models (deferred — separate future update)
- Same conversation, side-by-side perceptions from different models
- Intentionally **not** part of Milestone 6; planned as a separately-
  announced update with its own scope and announcement.
