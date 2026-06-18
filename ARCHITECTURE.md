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
to `PrismaInferenceClient.generate`, which calls GPT-OSS 120B
(`openai/gpt-oss-120b`) via Groq with a strict `json_schema`
`response_format` and returns the raw JSON content. `parse_model_output`
validates the schema (string `response`, integer scores per attribute in
`[1, 7]`) and returns a `ParsedTurn`. The app then appends the assistant
response to the chat display, records the evaluation in state, and
re-renders the impressions panel (colored bar cells) and the trajectory
plot. There is exactly one model call per user turn.

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
parsing.

JSON output is enforced via Groq's **strict `json_schema` mode**
(`response_format={"type": "json_schema", "json_schema": {"strict": true,
...}}`). Constrained decoding makes schema-non-conformant output
structurally impossible — unlike the earlier `json_object` mode, which
guaranteed only syntactic JSON validity and produced an `EvaluationParseError`
at ≈17% frequency on `llama-3.3-70b-versatile` at temperature 0.7. The
strict schema is built dynamically from `DEFAULT_ATTRIBUTES` by
`_build_response_schema` in `src/inference.py` and cached on the client
at construction time. All objects in the schema set
`additionalProperties: false` and list every property in `required`, as
required by Groq's strict mode spec.

`parse_model_output` in `src/evaluation.py` is retained as a belt-and-
suspenders validator; it catches the `bool`-as-`int` edge case
(Python's `isinstance(True, int)` is `True`) that the JSON schema
integer type alone cannot exclude. Markdown fence-stripping is also kept
for defensive completeness, though strict mode makes stray fences
unlikely.

With parse failures now structurally impossible, `generate()` retries
only on transport-level `InferenceError` (timeouts, 5xx) — one defensive
retry, not five. `EvaluationParseError` propagates immediately (an
unexpected failure worth surfacing). A dedicated `CapacityError` is
raised when Groq returns HTTP 429 (project daily limit hit); the app
layer catches it and shows a "daily capacity reached" toast rather than
the generic error message.

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

### GPT-OSS 120B via Groq

Hosted inference on Groq was chosen over self-hosting and over
proprietary APIs (OpenAI, Anthropic) for two main reasons. First, the
deployment surface is minimal: no GPU provisioning, no model serving,
no separate auth domain — a single `GROQ_API_KEY` covers everything,
which keeps the Space configuration to one secret. Second, a large
instruct model is empirically the threshold at which the structured
JSON contract holds reliably across varied conversational inputs and
the dual-role persona is maintained without prompt drift; smaller
open-weight models tend to break the schema or leak the evaluation
rationale into the reply. A third, Groq-specific consideration is
latency: the LPU hardware produces noticeably faster generation
(~250–400 tok/sec vs. typical hosted-API ~30–50 tok/sec on the same
model class), which keeps per-turn latency low enough for the running-
evaluation paradigm to feel responsive — the impressions panel updates
shortly after the reply lands, rather than after a perceptible wait.

The current model is `openai/gpt-oss-120b`. It was adopted on
2026-06-18 following Groq's deprecation notice for
`llama-3.3-70b-versatile` (effective immediately; decommission date
August 16, 2026). Crucially, `gpt-oss-120b` is among the Groq model
families that support strict `json_schema` mode, which resolves the
≈17% `EvaluationParseError` rate observed on the previous model under
`json_object` mode. An earlier version of the demo used the Hugging
Face Inference API; migration to Groq was prompted by free-tier credit
depletion on HF.

### Rate limiting

Two complementary layers protect against traffic spikes:

1. **Per-session turn cap** (`SESSION_TURN_CAP = 12` in `src/config.py`).
   Already present since Milestone 4. Bounds per-session token usage.

2. **Per-IP daily session cap** (`MAX_SESSIONS_PER_IP_PER_DAY`, default 10,
   overridable via env var). Implemented in `app.py` using Gradio's
   `gr.Request` to read the client IP. An in-memory `defaultdict` keyed by
   IP tracks sessions started today; the counter resets when the date
   advances. This prevents a single visitor or bot from consuming the entire
   Groq daily budget before other users can access the demo.

3. **Groq project-level daily limits** (primary budget backstop). Configured
   in the Groq console for `openai/gpt-oss-120b`: 10,000 requests/day and
   3,000,000 tokens/day. At ~1,800 tokens/turn this covers roughly 1,650
   turns/day while bounding worst-case spend to ~$1.80/day. When this ceiling
   is hit, Groq returns HTTP 429; `PrismaInferenceClient._call_once` catches
   `APIStatusError` with `status_code == 429` and raises `CapacityError`,
   which `app.py` surfaces as a "daily capacity reached" toast.

### Gradio on Hugging Face Spaces

Gradio is the lowest-friction path to a public, shareable artifact: a
single `app.py` declares the UI, the deployment is `git push`, and the
HF Space provides the public URL and HTTPS termination. The model call
goes out to Groq (via the `groq` SDK), which is independent of the
Space's hosting platform — the Space is purely the runtime and the UI
shell. The cost is limited UI flexibility compared to a custom React
frontend, which is accepted because the demo's value is in the
interaction itself, not in bespoke visual design.

## Module responsibilities

- `src/config.py` — single source of truth for the v1 attribute set,
  the model ID, decoding parameters, the session turn cap, and the
  attribute-to-color mapping. Centralizing these means the prompt, the
  parser, and the UI cannot drift out of sync with each other.
- `src/prompt.py` — builds the dual-role system prompt from the
  attribute list at module import time. Templated rather than hardcoded
  so that the v2 selectable attributes (see above) plug in without
  touching the inference layer.
- `src/inference.py` — thin wrapper around the `groq` SDK client.
  Builds and caches the strict `json_schema` response_format dict from
  `DEFAULT_ATTRIBUTES`; distinguishes transport errors (`InferenceError`),
  parse failures (`EvaluationParseError`), and daily-cap hits
  (`CapacityError`) so the app layer can react differently; validates the
  response envelope before handing the content to the parser.
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
