"""Grounded answer generation."""

from __future__ import annotations

import os

from .retrieval import Hit

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-haiku-4-5")
NO_ANSWER = "I don't have enough information to answer that."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered context passages provided.
- Cite the passages you used with their ids in square brackets, e.g. [c012].
- If the passages don't contain the answer, reply exactly: "{NO_ANSWER}"
- Do not use outside knowledge. Be concise and direct."""


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(f"[{h.chunk_id}]\n{h.text}" for h in hits)
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(client, question: str, hits: list[Hit], model: str = ANSWER_MODEL) -> str:
    if not hits:
        return NO_ANSWER
    msg = client.messages.create(
        model=model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
    )
    text = next(b.text for b in msg.content if b.type == "text").strip()
    # Models sometimes explain a refusal after the exact phrase; normalize it so
    # refusals can be counted reliably in the eval.
    return NO_ANSWER if text.startswith(NO_ANSWER) else text
