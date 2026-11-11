"""Shared role-history constructor for fact V2 production and capacity checks."""

from collections.abc import Sequence


def build_fact_v2_message_history(
    system_prompt: str,
    initial_user_message: str,
    program_messages: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Build the ordered system and user messages used by the V2 loop."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_message},
    ]
    messages.extend({"role": "user", "content": item} for item in program_messages)
    return messages
