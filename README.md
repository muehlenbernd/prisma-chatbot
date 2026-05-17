# prisma-chatbot

> Have you ever wondered what your chatbot thinks about you?

**Prisma** (*Pragmatic Real-time Inference of Social Meaning in Agents*) is a
conversational AI demo that responds to users while simultaneously evaluating
them on social/pragmatic dimensions. It accompanies published research on
LLM social perception (CMCL 2026; EMNLP 2026, under review).

> **TODO:** Replace with a short hero paragraph and screenshot once the demo
> is live.

## Live demo

> **TODO:** Add Hugging Face Space link.

## What it does

> **TODO:** 2–3 sentence description of the dual-role design — Prisma
> responds in conversation while producing a structured evaluation of the
> user across six attributes (competent, knowledgeable, well-prepared,
> helpful, likeable, pedantic). Evaluation updates turn by turn.

## Research context

> **TODO:** Link to CMCL and EMNLP papers; one paragraph on the research
> thesis (do LLMs evaluate speakers based on linguistic choices the way
> humans do?).

## Local development

> **TODO:** Flesh out once `app.py` and the `src/` modules are implemented.

```bash
# Clone, create a virtualenv, install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy env template and add your HF token
cp .env.example .env

# Run the Gradio app locally
python app.py
```

## Project structure

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for design decisions and
[`ROADMAP.md`](ROADMAP.md) for the deployment plan.

## Contributing

This is a personal research project and public portfolio artifact. Issues
are welcome — feel free to open one if you spot a bug, have a question
about the research, or want to suggest a feature. Pull requests are by
invitation only; please open an issue first to discuss.

## License

See [`LICENSE`](LICENSE) (MIT).
