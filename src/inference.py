"""Hugging Face Inference API client wrapper.

Thin wrapper around `huggingface_hub`'s inference client that issues a
single LLM call per turn and returns the raw model output. Keeps API
concerns (auth, model selection, retries) isolated from prompt and
evaluation logic.

Implementation pending — scaffolding only.
"""
