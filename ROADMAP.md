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
- [x] Implement `src/inference.py` (HF Inference client wrapper)
- [x] Implement `src/evaluation.py` (JSON parsing + validation)
- [x] Implement `app.py` (minimal Gradio UI)
- [x] First pytest tests for parsing/validation

## Milestone 3 — Prompt and UX iteration

> **In progress:** Refine Prisma's voice, evaluation display, and the "check what
> I think" affordance.

- [x] Refine system prompt for voice consistency and structured-output
      reliability
- [x] Decide evaluation display: numeric, verbal, or both
- [x] Decide update cadence: live each turn vs. on-request
- [ ] About panel copy (`assets/about.md`)

## Milestone 4 — Public deployment

> **Planned:** Ship to a Hugging Face Space and link from the README and
> papers.

- [ ] Hugging Face Space configuration
- [ ] Rate limiting / per-session turn cap
- [ ] Public URL added to README and papers
- [ ] Light usage analytics (anonymous, aggregate)

## Milestone 5 — Stretch ideas

> **Planned:** Explicitly non-blocking; consider only after the demo is live
> and stable.

- [ ] "Compare models" mode
- [ ] Downloadable conversation + evaluation transcript
- [ ] Linguistic feature highlighting (which words/choices shifted scores)


## Future work (v2+)

### Attribute customization
- User selects up to 6 attributes from a curated extended list (~15–20 
  dimensions covering social, cognitive, affective, and pragmatic 
  perception)
- Extended list includes: pushy, knowledgeable, well-prepared, pedantic, 
  helpful, arrogant, warm, evasive, confident, anxious, etc. (final set TBD)
- v1's default set remains as the "quick start" option