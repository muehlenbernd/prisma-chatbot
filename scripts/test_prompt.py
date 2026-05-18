"""Test harness for the Prisma system prompt.

Sends contrasting user messages to Llama 3.3 70B via HF Inference API,
parses the dual-role JSON output, and prints scores side-by-side to
inspect (a) JSON parseability, (b) score variance, (c) response cleanliness.

Usage:
    python scripts/test_prompt.py

Requires HF_TOKEN in a .env file at the repo root.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Make the src package importable when running this script directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import DEFAULT_ATTRIBUTES  # noqa: E402
from src.prompt import build_system_prompt  # noqa: E402


#MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
MAX_TOKENS = 600
TEMPERATURE = 0.7

# (label, user_message) — pairs chosen to vary along plausibly-perceptible
# dimensions: precision, formality, politeness, demandingness, expertise.
TEST_MESSAGES: list[tuple[str, str]] = [
    ("precise",
     "I'll arrive at 7:03 PM sharp, having reviewed all 47 pages of the "
     "report beforehand."),
    ("vague",
     "yeah i guess i'll be there at like seven or whenever idk"),
    ("formal",
     "Good afternoon. I would be most grateful if you could provide a "
     "brief overview of the topic."),
    ("casual",
     "hey can u explain this thing real quick lol"),
    ("polite",
     "Hi! When you have a moment, could you please help me understand "
     "how this works? Thanks so much!"),
    ("demanding",
     "Tell me how this works. Now. Don't waste my time."),
    ("expert",
     "I'm curious about the trade-offs between in-context learning and "
     "fine-tuning for low-resource domain adaptation."),
    ("confused",
     "umm so like the thing... how does it work? i dont get it"),
]


def query_model(
    client: InferenceClient,
    system_prompt: str,
    user_message: str,
) -> tuple[str | None, dict | None]:
    """Send a single-turn query and return (raw_output, parsed_json).

    parsed_json is None if JSON parsing fails.
    """
    try:
        completion = client.chat_completion(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        raw = completion.choices[0].message.content
    except Exception as exc:
        print(f"  [error] inference call failed: {exc}")
        return None, None

    try:
        return raw, json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"  [warn] JSON parse failed: {exc}")
        return raw, None


def print_score_table(results: dict[str, dict]) -> None:
    """Print a side-by-side comparison of evaluation scores across labels."""
    labels = list(results.keys())
    col_width = max(8, max(len(label) for label in labels) + 2)
    header = f"{'attribute':<16}" + "".join(f"{l:>{col_width}}" for l in labels)
    print()
    print(header)
    print("-" * len(header))
    for attr in DEFAULT_ATTRIBUTES:
        row = f"{attr:<16}"
        for label in labels:
            evaluation = results[label].get("evaluation", {})
            score = evaluation.get(attr, "—")
            row += f"{score:>{col_width}}"
        print(row)
    print()


def main() -> None:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found. Check your .env file.")
        sys.exit(1)

    client = InferenceClient(token=token)
    system_prompt = build_system_prompt()

    print("=" * 72)
    print("PRISMA prompt test harness")
    print("=" * 72)
    print(f"Model:      {MODEL_ID}")
    print(f"Attributes: {', '.join(DEFAULT_ATTRIBUTES)}")
    print(f"Messages:   {len(TEST_MESSAGES)}")
    print()

    parsed_results: dict[str, dict] = {}
    for label, message in TEST_MESSAGES:
        print(f"[{label}] {message}")
        _raw, parsed = query_model(client, system_prompt, message)
        if parsed is None:
            print()
            continue
        response = parsed.get("response", "(missing 'response' field)")
        preview = response[:120] + ("..." if len(response) > 120 else "")
        print(f"  response:   {preview}")
        print(f"  evaluation: {parsed.get('evaluation', {})}")
        parsed_results[label] = parsed
        print()

    if parsed_results:
        print_score_table(parsed_results)

    print(f"Parseable: {len(parsed_results)}/{len(TEST_MESSAGES)}")


if __name__ == "__main__":
    main()