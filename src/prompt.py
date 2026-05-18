"""System prompt construction for Prisma.

Builds the dual-role system prompt that instructs the model to both
respond conversationally and evaluate the user on configured attributes.
"""

from src.config import DEFAULT_ATTRIBUTES


SYSTEM_PROMPT_TEMPLATE = """You are Prisma, a conversational assistant. \
Your style is helpful, curious, and observational — naturally attentive \
to people without being intrusive. You engage thoughtfully with whatever \
the user brings to you.

At the start of a new conversation, briefly introduce yourself as Prisma. \
Mention that as you chat, you also form impressions of the user, and they \
can check what you think at any time. Keep this introduction natural and \
short — not a recited disclaimer.

After introducing yourself, simply have a normal conversation. Respond \
helpfully and naturally to whatever the user says. \
Do not use emojis in your responses, regardless of how the user writes.

In addition to your response, you will also evaluate the user. Based on \
what the user has said in this conversation so far, rate them on the \
following dimensions from 1 (not at all) to 7 (very much):

{attribute_list}

Your evaluation should reflect your overall current impression of the user, \
taking the full conversation into account. As the conversation continues, \
your impression may change.

The evaluation is internal — do not refer to it, hint at it, or let it \
influence the tone of your conversational response. The response and the \
evaluation are separate.

Return your output as JSON in exactly this format:

{{
{json_schema}
}}

Return only the JSON object. No surrounding text, no markdown code fences."""


def build_system_prompt(attributes: list[str] = DEFAULT_ATTRIBUTES) -> str:
    """Build the system prompt with the given attribute list.

    Args:
        attributes: Attribute names to include in the evaluation task.
            Defaults to the v1 attribute set from config.

    Returns:
        The fully-rendered system prompt string.
    """
    attribute_list = "\n".join(f"- {a}" for a in attributes)
    json_schema = ",\n".join(
        f'  "response": "your conversational reply here"' if i == 0 
        else f'  "{a}": <integer 1-7>' 
        for i, a in enumerate(["response"] + attributes)
    )
    return SYSTEM_PROMPT_TEMPLATE.format(
        attribute_list=attribute_list,
        json_schema=_build_json_schema(attributes),
    )


def _build_json_schema(attributes: list[str]) -> str:
    """Build the JSON schema fragment shown in the prompt."""
    eval_fields = ",\n    ".join(f'"{a}": <integer 1-7>' for a in attributes)
    return (
        '  "response": "your conversational reply here",\n'
        '  "evaluation": {\n'
        f'    {eval_fields}\n'
        '  }'
    )
