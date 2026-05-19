"""Gradio app entry point for PRISMA.

Wires prompt construction, inference, evaluation parsing, and an
always-visible impressions panel (bar-style colored cells plus a trajectory
plot) into a Gradio Blocks interface with a custom dark theme.

State held in ``gr.State``:
    {
        "history": list[dict],     # OpenAI-format messages (system + chat)
        "evaluations": list[dict], # one per assistant turn
        "turn_count": int,         # completed user turns
    }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib

matplotlib.use("Agg")  # non-interactive backend, required for server-side use
import matplotlib.pyplot as plt  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from src.config import (  # noqa: E402
    ATTRIBUTE_COLORS,
    DEFAULT_ATTRIBUTES,
    MAX_SCORE,
    SESSION_TURN_CAP,
)
from src.evaluation import (  # noqa: E402
    INTENSIFIER_SCALE,
    EvaluationParseError,
)
from src.inference import (  # noqa: E402
    InferenceError,
    PrismaInferenceClient,
)
from src.prompt import build_system_prompt  # noqa: E402


# ---------------------------------------------------------------------------
# One-time setup
# ---------------------------------------------------------------------------

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN not found. Set it in .env at the repo root "
        "(see .env.example)."
    )

CLIENT = PrismaInferenceClient(token=HF_TOKEN)
SYSTEM_PROMPT = build_system_prompt()

# Load the hero figure inline so it renders without needing Gradio's file
# server. If the file is missing, the header just won't show an image.
FIGURE_PATH = Path(__file__).parent / "assets" / "prisma-figure.svg"
FIGURE_SVG = FIGURE_PATH.read_text() if FIGURE_PATH.exists() else ""


# ---------------------------------------------------------------------------
# Theme & CSS
# ---------------------------------------------------------------------------

THEME = gr.themes.Base(
    primary_hue="violet",
    neutral_hue="slate",
).set(
    body_background_fill="#0f0f1a",
    body_background_fill_dark="#0f0f1a",
    block_background_fill="#1a1a2e",
    block_background_fill_dark="#1a1a2e",
    body_text_color="#e5e7eb",
    body_text_color_dark="#e5e7eb",
    border_color_primary="#2a2a44",
    border_color_primary_dark="#2a2a44",
    input_background_fill="#1a1a2e",
    input_background_fill_dark="#1a1a2e",
)

CUSTOM_CSS = """
#prisma-header {
    text-align: center;
    padding: 1rem 0 2rem 0;
}
#prisma-header svg {
    max-width: 640px;
    width: 100%;
    height: auto;
    display: block;
    margin: 0 auto 1rem auto;
}
#prisma-header h1 {
    font-size: 2.75rem;
    margin: 0.5rem 0 0.25rem 0;
    letter-spacing: 0.05em;
}
#prisma-header .tagline {
    font-size: 1.35rem;
    font-style: italic;
    color: #9ca3af;
    margin: 0.25rem 0 0.5rem 0;
}
#prisma-header .description {
    font-size: 1.1rem;
    color: #cbd5e1;
    max-width: 720px;
    margin: 0.5rem auto;
    line-height: 1.5;
}
#impressions-panel {
    flex: 0 0 360px !important;
    max-width: 360px !important;
    min-width: 360px !important;
}
#impressions-panel h3 {
    font-size: 1.4rem;
    margin: 0 0 0.75rem 0;
}
.impressions-header {
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0.5rem 0 0.75rem 0;
    color: #e5e7eb;
}
.impression-row {
    padding: 0.55rem 0.85rem;
    margin: 0.35rem 0;
    border-radius: 6px;
    color: #ffffff;
    font-weight: 500;
    font-size: 0.95rem;
    white-space: nowrap;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.55);
    letter-spacing: 0.01em;
}
.impressions-empty {
    font-style: italic;
    color: #9ca3af;
    padding: 0.5rem 0;
}
"""


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def initial_state() -> dict[str, Any]:
    """Return a fresh conversation state for a new session."""
    return {
        "history": [{"role": "system", "content": SYSTEM_PROMPT}],
        "evaluations": [],
        "turn_count": 0,
    }


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def chat_step(
    user_message: str,
    chat_display: list[dict[str, str]],
    state: dict[str, Any],
):
    """Process one user turn: call the model, update state and UI.

    Returns updates for (chatbot, state, msg_in, turn_dropdown).
    """
    user_message = (user_message or "").strip()
    if not user_message:
        return chat_display, state, "", gr.Dropdown()

    # Session cap reached — refuse further requests.
    if state["turn_count"] >= SESSION_TURN_CAP:
        notice = (
            f"Session complete — Prisma has chatted with you for "
            f"{SESSION_TURN_CAP} turns. Refresh the page to start over."
        )
        chat_display = chat_display + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": notice},
        ]
        return chat_display, state, "", gr.Dropdown()

    state["history"].append({"role": "user", "content": user_message})

    try:
        parsed = CLIENT.generate(state["history"])
        assistant_text = parsed.response
        state["history"].append(
            {"role": "assistant", "content": assistant_text}
        )
        state["evaluations"].append(parsed.evaluation)
        state["turn_count"] += 1
    except (InferenceError, EvaluationParseError) as exc:
        # Roll back the unanswered user message so retries send clean history.
        state["history"].pop()
        assistant_text = (
            "Something went wrong on Prisma's end — please try again. "
            f"(Details: {exc})"
        )

    chat_display = chat_display + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_text},
    ]

    n_evals = len(state["evaluations"])
    if n_evals > 0:
        choices = [(f"Turn {i + 1}", i) for i in range(n_evals)]
        dropdown_update = gr.Dropdown(choices=choices, value=n_evals - 1)
    else:
        dropdown_update = gr.Dropdown(choices=[], value=None)

    return chat_display, state, "", dropdown_update


# ---------------------------------------------------------------------------
# Impressions rendering
# ---------------------------------------------------------------------------

def render_impression(state: dict[str, Any], turn_index: int | None) -> str:
    """Build HTML for the impressions panel: header + colored bar cells.

    Each row uses a linear-gradient background that fills up to (score/MAX)
    of the row's width with the attribute's saturated color, then continues
    with the same color at low alpha for the remainder. This doubles the
    text label as a per-attribute bar plot.
    """
    evaluations = state.get("evaluations", [])
    if not evaluations:
        return (
            '<div class="impressions-empty">'
            "No impressions yet — say something to Prisma."
            "</div>"
        )

    if turn_index is None or turn_index < 0 or turn_index >= len(evaluations):
        turn_index = len(evaluations) - 1

    evaluation = evaluations[turn_index]
    header = (
        f'<div class="impressions-header">After turn {turn_index + 1}:</div>'
    )

    rows: list[str] = []
    for attr in DEFAULT_ATTRIBUTES:
        score = evaluation[attr]
        color = ATTRIBUTE_COLORS[attr]
        intensifier = INTENSIFIER_SCALE[score]
        pct = (score / MAX_SCORE) * 100
        # Two-stop linear gradient: saturated up to `pct`, then ~20% alpha.
        # `{color}33` appends 0x33 (~20%) alpha to the hex color.
        gradient = (
            f"linear-gradient(to right, "
            f"{color} 0%, {color} {pct:.1f}%, "
            f"{color}33 {pct:.1f}%, {color}33 100%)"
        )
        rows.append(
            f'<div class="impression-row" style="background: {gradient};">'
            f"{intensifier} {attr} ({score}/{MAX_SCORE})"
            f"</div>"
        )

    return header + "\n" + "\n".join(rows)


def render_trajectory(state: dict[str, Any]):
    """Render a line plot of scores per attribute across turns.

    Colors match the bar cells so the rating list above acts as the legend.
    """
    evaluations = state.get("evaluations", [])

    fig, ax = plt.subplots(figsize=(5, 3), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    if not evaluations:
        ax.text(
            0.5,
            0.5,
            "No data yet",
            ha="center",
            va="center",
            color="#9ca3af",
            fontsize=12,
            fontstyle="italic",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        return fig

    turns = list(range(1, len(evaluations) + 1))
    for attr in DEFAULT_ATTRIBUTES:
        scores = [e[attr] for e in evaluations]
        ax.plot(
            turns,
            scores,
            color=ATTRIBUTE_COLORS[attr],
            marker="o",
            linewidth=2,
            markersize=5,
        )

    ax.set_xlabel("Turn", color="#e5e7eb")
    ax.set_ylabel("Score", color="#e5e7eb")
    ax.set_ylim(0.5, 7.5)
    ax.set_yticks(range(1, MAX_SCORE + 1))
    ax.set_xticks(turns)
    ax.tick_params(colors="#e5e7eb")
    ax.grid(True, alpha=0.15, color="#9ca3af")
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("bottom", "left"):
        ax.spines[spine_name].set_color("#9ca3af")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="PRISMA") as demo:

    gr.HTML(
        f"""
<div id="prisma-header">
  {FIGURE_SVG}
  <h1>PRISMA</h1>
  <p class="tagline">Have you ever wondered what your chatbot thinks about you?</p>
  <p class="description">
    Chat with Prisma. She'll respond — and form impressions of you based on
    how you write. Her view appears alongside the chat and updates after
    every turn. Scroll back through the dropdown to see how her view shifted.
  </p>
</div>
"""
    )

    state = gr.State(initial_state())

    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                label="Chat with Prisma",
                height=480,
            )
            with gr.Row():
                msg_in = gr.Textbox(
                    placeholder="Say something to Prisma...",
                    show_label=False,
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Column(scale=0, min_width=360, elem_id="impressions-panel"):
            gr.Markdown("### Prisma's impressions of you")
            turn_dropdown = gr.Dropdown(
                choices=[],
                label="View impression after which turn?",
                interactive=True,
            )
            impressions_html = gr.HTML(
                value=(
                    '<div class="impressions-empty">'
                    "No impressions yet — say something to Prisma."
                    "</div>"
                ),
            )
            trajectory_plot = gr.Plot(value=render_trajectory({}), label=None)

    # Same submit handler for Enter-key and Send button.
    for trigger in (send_btn.click, msg_in.submit):
        trigger(
            chat_step,
            inputs=[msg_in, chatbot, state],
            outputs=[chatbot, state, msg_in, turn_dropdown],
        ).then(
            render_impression,
            inputs=[state, turn_dropdown],
            outputs=impressions_html,
        ).then(
            render_trajectory,
            inputs=state,
            outputs=trajectory_plot,
        )

    turn_dropdown.change(
        render_impression,
        inputs=[state, turn_dropdown],
        outputs=impressions_html,
    )


if __name__ == "__main__":
    demo.launch()
