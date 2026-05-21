# Architecture

Design decisions and rationale for prisma-chatbot. This document captures
*why* the system is built the way it is, not just *what* it does. See
[`README.md`](README.md) for a user-facing overview and
[`ROADMAP.md`](ROADMAP.md) for the deployment plan and milestones.

## System overview

Each user turn flows through a single, linear path. The Gradio app
(`app.py`) appends the user message to the running history held in
`gr.State`, prepended with the dual-role system prompt built once at
startup by `src.prompt.build_system_prompt`. The full history is handed
to `PrismaInferenceClient.generate`, which calls Llama 3.3 70B Instruct
via the Hugging Face Inference API with `response_format={"type":
"json_object"}` and returns the raw JSON content. `parse_model_output`
strips any stray markdown fences, validates the schema (string
`response`, integer scores per attribute in `[1, 7]`), and returns a
`ParsedTurn`. The app then appends the assistant response to the chat
display, records the evaluation in state, and re-renders the
impressions panel (colored bar cells) and the trajectory plot. There
is exactly one model call per user turn.

## Key design decisions

### Dual-role prompt, single LLM call per turn

The model is asked to produce both the conversational reply and the
evaluation in a single structured response. Two reasons drive this. The
first is operational: one network call keeps per-turn latency and cost
predictable, and avoids the engineering of stitching two calls together
under partial failure. The second is semantic: the response and the
evaluation are grounded in exactly the same context window, so the
evaluation reflects the model's actual current impression rather than a
separately-prompted second-pass judgment. The trade-off is that the
prompt is more elaborate than two narrowly-scoped calls would be, and
the JSON contract has to be enforced both at the API boundary
(`response_format`) and in the parser. This is accepted because the
research thesis is precisely that the model's response and its
perception of the user are two facets of one act of interpretation, and
collapsing them into a single call mirrors that framing.

### Structured JSON output

The model returns a JSON object with a string `response` field and an
`evaluation` object mapping each attribute name to an integer score in
`[1, 7]`. This makes downstream rendering deterministic: the chat
display reads `response` directly, and the impressions panel and
trajectory plot iterate over `evaluation` without any natural-language
parsing. JSON output is enforced both by the prompt and by passing
`response_format={"type": "json_object"}` to the Inference API —
belt-and-suspenders, because Llama 3.3 70B occasionally drifts toward
conversational preamble before the JSON when relying on prompt
instructions alone. Two defensive choices in `parse_model_output`
deserve a note. Markdown code fences are stripped if present, because
some model snapshots wrap structured output in ``` blocks despite
instructions otherwise. And because `bool` is a subclass of `int` in
Python, the validator rejects `True`/`False` explicitly — without that
check, a model returning `true` for a score would silently pass type
validation and be coerced to `1`.

### Six evaluation attributes

The v1 attribute set is *competent, likeable, considerate, polite,
formal, demanding* — defined once in `src/config.DEFAULT_ATTRIBUTES`
and consumed by the prompt builder, the parser, and the UI renderer.
The set is intentionally a **general-purpose** one, chosen to produce
meaningful variance across the diverse, unconstrained conversational
inputs a public demo will receive. It is **not** identical to the
attribute set used in the CMCL 2026 / EMNLP 2026 studies: the
research-specific dimensions there (notably *pedantic* and
*well-prepared*) are highly informative around precision-related
stimuli but produce flat scores in casual chat, which would make the
live demo look broken. Those research attributes, alongside other
candidates (*pushy, knowledgeable, helpful, arrogant, warm, evasive*),
are slated for the v2 user-selectable extended list. The demo's
research framing is therefore *methodological* — surfacing the model's
ongoing social perception of the user — rather than a literal
replication of the paper's stimuli.

### Llama 3.3 70B Instruct via HF Inference API

Hosted inference on Hugging Face was chosen over self-hosting and over
proprietary APIs (OpenAI, Anthropic) for three reasons. First, the
deployment surface is minimal: no GPU provisioning, no model serving,
no separate auth domain — the Space and the model live on the same
platform and a single `HF_TOKEN` covers both. Second, the audience that
arrives via the research papers is already familiar with the Hugging
Face platform and trusts it, which removes a friction point that a
custom-hosted endpoint or a third-party key requirement would
introduce. Third, a 70B-class instruct model is empirically the
threshold at which the structured JSON contract holds reliably across
varied conversational inputs and the dual-role persona is maintained
without prompt drift; smaller open-weight models tend to break the
schema or leak the evaluation rationale into the reply. The trade-off
is dependency on HF endpoint availability and the (low) per-call rate
limits applied to public Spaces, which the per-session turn cap
(`SESSION_TURN_CAP = 12`) helps absorb.

### Gradio on Hugging Face Spaces

Gradio is the lowest-friction path to a public, shareable artifact: a
single `app.py` declares the UI, the deployment is `git push`, and the
HF Space provides the public URL and HTTPS termination. It also
integrates natively with the HF Inference API used for the model call.
The cost is limited UI flexibility compared to a custom React frontend,
which is accepted because the demo's value is in the interaction
itself, not in bespoke visual design.

## Module responsibilities

- `src/config.py` — single source of truth for the v1 attribute set,
  the model ID, decoding parameters, the session turn cap, and the
  attribute-to-color mapping. Centralizing these means the prompt, the
  parser, and the UI cannot drift out of sync with each other.
- `src/prompt.py` — builds the dual-role system prompt from the
  attribute list at module import time. Templated rather than hardcoded
  so that the v2 selectable attributes (see above) plug in without
  touching the inference layer.
- `src/inference.py` — thin wrapper around `huggingface_hub.
  InferenceClient`. Forces `response_format={"type": "json_object"}`
  uniformly, distinguishes API errors (`InferenceError`) from parse
  errors (`EvaluationParseError`) so the app layer can react
  differently, and validates the response envelope before handing the
  content to the parser.
- `src/evaluation.py` — parses the JSON, validates the schema against
  the expected attribute list, and formats scores for display. Owns
  the intensifier scale (`1 → "not at all"`, ..., `7 → "extremely"`)
  that pairs verbal labels with numeric scores in the UI.
- `app.py` — Gradio Blocks assembly, theme/CSS, event wiring, and the
  rendering of the impressions panel (HTML bar cells) and trajectory
  plot (matplotlib). On parse or inference failure the user's message
  is rolled back from state and surfaced as a `gr.Warning` toast rather
  than as a fake assistant turn, so retries send clean history.

## Open design questions

- **Single-call vs. two-call architecture if the attribute set grows.**
  At six attributes the dual-role prompt is comfortable; if v2 lets
  users pick from a longer extended list, JSON-output reliability and
  attention to the response itself may degrade enough to justify
  splitting into two calls.
- **Whether to expose decoding controls (temperature) as a user
  setting.** Currently fixed at `0.7` in config. Exposing it would
  let visitors probe how stochasticity affects the evaluation, which
  aligns with the research framing — but adds a knob that most
  visitors won't understand.
- **Behavior on non-English input.** The persona prompt is English and
  the attribute labels are English; the model handles other languages
  reasonably but the social perception it produces has not been
  validated outside English. The honest disposition for v1 is to
  accept it silently; a more careful v2 might detect and either
  surface a disclaimer or refuse.
