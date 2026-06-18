# CLAUDE.md

## Project: prisma-chatbot

A conversational AI demo featuring **Prisma** — a chatbot that responds to
users while simultaneously evaluating them on social/pragmatic dimensions.
Built as a research-facing artifact accompanying published work on LLM social
perception (CMCL 2026; EMNLP 2026, under review).

**PRISMA** stands for *Pragmatic Real-time Inference of Social Meaning in
Agents*.

**Tagline:** "Have you ever wondered what your chatbot thinks about you?"

**Live demo:** [HF Space link — to be added]
**Research papers:** [CMCL link — to be added] | [EMNLP link — pending]

## Project Goals

1. **Research dissemination** — make the social-perception findings tangible
   and interactive for both NLP researchers and general audiences.
2. **Portfolio artifact** — serve as a public, polished demonstration of
   applied LLM/NLP engineering for industry job applications.
3. **Conversation starter** — generate discussion about how LLMs perceive
   speakers, not just respond to them.

## Owner Context

The maintainer is a theoretical/computational linguist transitioning into
AI/tech roles. Code should be clean, readable, and well-documented — both
because this is a public artifact and because the author values clarity over
cleverness. Industry-standard practices (typing, docstrings, modular design)
are preferred over research-code shortcuts.

## Bot Persona: Prisma

The chatbot introduces herself as Prisma on the first turn. Suggested opening
(refine later):

> "Hi, I'm Prisma. I'll chat with you — and while we talk, I'll also form
> impressions of you based on how you write. You can check what I think at
> any time."

**Voice:** lightly curious and observational. Helpful and competent as an
assistant, but with a subtle awareness that she's also paying attention to
*how* the user writes, not just *what* they ask. Never roleplay-heavy, never
clinical or diagnostic, never sycophantic. The personality should be carried
mostly by the name, the intro, and small observational touches — not by
constant character performance.

## Architecture

**Frontend:** Gradio app deployed on Hugging Face Spaces.
**Backend:** Single LLM call per turn, dual-role prompt (response + evaluation).
**Model:** GPT-OSS 120B (`openai/gpt-oss-120b`) via Groq.
**Output format:** Structured JSON with `response` (string) and `evaluation`
(object with six attribute scores 1–7), enforced via Groq's `json_schema`
strict mode — constrained decoding guarantees schema-compliant output.

**Evaluation dimensions (v1 default set):** competent, likeable, 
considerate, polite, formal, demanding. v1 ships with this fixed 
general-purpose set chosen to produce meaningful variance across diverse 
conversational inputs. v2 will allow users to pick up to six attributes 
from an extended list (candidates include: pushy, knowledgeable, 
well-prepared, helpful, pedantic, arrogant, warm, evasive — to be 
finalized). The CMCL/EMNLP-specific attributes (pedantic, well-prepared) 
move to the extended list since they only activate around precision-
related stimuli.

**Key design property:** evaluations update *dynamically* across the
conversation. This reflects the research finding that social meaning is
constructed turn by turn, not fixed by a single utterance. The "mirror"
metaphor and PRISMA acronym both lean into this real-time aspect.

## Tech Stack

- Python 3.11+
- Gradio (UI framework)
- `groq` (Groq Inference API client)
- `python-dotenv` (local secrets)
- `pytest` (testing)

Keep dependencies minimal. Add new ones only when clearly justified.

## Code Style

- Type hints on all function signatures.
- Docstrings (Google or NumPy style) on public functions and classes.
- Module-level docstrings explaining purpose.
- Prefer pure functions and small modules over large stateful classes.
- Black for formatting, Ruff for linting.
- No emojis in code or comments.

## Repository Structure

```
prisma-chatbot/
├── README.md              # Public project description
├── CLAUDE.md              # This file — instructions for AI assistants
├── ARCHITECTURE.md        # Design decisions and rationale
├── ROADMAP.md             # Deployment plan and milestones
├── app.py                 # Gradio app entry point (HF Space reads this)
├── src/
│   ├── __init__.py
│   ├── prompt.py          # System prompt construction (Prisma persona + dual-role)
│   ├── inference.py       # HF Inference API client wrapper
│   ├── evaluation.py      # Score parsing, validation, display formatting
│   └── config.py          # Settings, constants, rate limits
├── tests/
│   └── ...                # Pytest-based unit tests
├── assets/
│   └── about.md           # Research background copy for UI
├── requirements.txt
├── .env.example
├── .gitignore
└── LICENSE
```

## Development Workflow

- **Claude Code** is used as project orchestrator: structure decisions,
  cross-file refactoring, documentation, planning, code review.
- **Cursor Agent** handles focused feature implementation and UI iteration.
- All non-trivial changes go through a feature branch and PR review (even if
  solo) — useful both for hygiene and as portfolio evidence of workflow.

## Research Context (for AI assistants)

The project builds on the maintainer's published work investigating whether
LLMs evaluate speakers based on linguistic choices the way humans do — for
example, whether saying "I'll be there at 7:03" vs. "around 7" influences
perceived competence, pedantry, etc. Prisma makes this research thesis
interactive: the model's social perception of the user is surfaced rather
than hidden, and updates as the conversation evolves.

When suggesting features, prompt designs, or UI choices, prefer those that
align with or showcase this research framing. Avoid generic "AI assistant"
patterns that obscure the social-perception angle.

## What This Project Is Not

- Not a production chatbot — it is a research demo with a specific thesis.
- Not a generic LLM wrapper — the dual-role evaluation is the point.
- Not a psychological assessment tool — the evaluation is playful, not
  diagnostic. UI copy should reflect this clearly.

## Naming Disambiguation

"Prisma" is also the name of a well-known TypeScript ORM and an older photo
app. This project is unrelated to both. The repo is intentionally named
`prisma-chatbot` (not `prisma`) to make the distinction clear in searches
and project listings. When referring to the bot, "Prisma" is fine in
user-facing copy; in code comments and docs, prefer "the bot" or "PRISMA"
(acronym form) where ambiguity could arise.

## Open Questions

(Use this section to flag design decisions still being deliberated.)

- Should evaluation scores update live after each turn, or only on user request?
- Numeric (1–7) vs. verbal score display, or both?
- Per-session turn cap value (10? 15?).
- Should there be a "compare models" mode in v2?
