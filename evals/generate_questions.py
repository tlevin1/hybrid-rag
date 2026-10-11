"""Generate the evaluation question set once and save it.

For a fixed sample of chunks, Claude writes two questions answerable from that
chunk alone:

- direct: may reuse the passage's wording. Questions generated from a passage
  tend to echo it, which flatters keyword search; this is the usual cookbook setup.
- paraphrased: must avoid the passage's distinctive words, so retrieval has to
  match meaning. This is where vector search should earn its place.

Each question keeps its source chunk id, so retrieval can also be scored exactly
(hit rate, MRR) without an LLM judge. The output is committed so every run
scores the same questions.

    python -m evals.generate_questions            # writes evals/questions.jsonl
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from hybrid_rag.corpus import load_chunks

OUT = Path(__file__).with_name("questions.jsonl")
MODEL = "claude-haiku-4-5"
N_CHUNKS = 40
SEED = 0

PROMPT = """Here is a passage from an essay:

<passage>
{text}
</passage>

Write two quiz questions that can be answered from this passage alone, about a
specific fact in it (not about the passage itself, e.g. never "what does the
passage say about...").

1. "direct": a natural question; it may use the passage's own words.
2. "paraphrased": asks about a DIFFERENT fact from question 1, and must NOT
   reuse the passage's distinctive words (names, rare nouns, technical terms).
   Describe things in other words, as someone who hasn't read it would.

Also give a short reference answer for each, taken from the passage.

Return only JSON:
{{"direct": {{"question": "...", "answer": "..."}}, "paraphrased": {{"question": "...", "answer": "..."}}}}"""


def generate(client: anthropic.Anthropic, chunk) -> list[dict]:
    msg = client.messages.create(
        model=MODEL, max_tokens=600, messages=[{"role": "user", "content": PROMPT.format(text=chunk.text)}]
    )
    text = next(b.text for b in msg.content if b.type == "text").strip()
    text = text[text.find("{") : text.rfind("}") + 1]  # tolerate a code fence
    data = json.loads(text)
    return [
        {"source_chunk": chunk.id, "style": style, "question": data[style]["question"], "reference": data[style]["answer"]}
        for style in ("direct", "paraphrased")
    ]


def main() -> None:
    chunks = load_chunks()
    sample = sorted(random.Random(SEED).sample(chunks, N_CHUNKS), key=lambda c: c.id)
    client = anthropic.Anthropic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [r for rs in pool.map(lambda c: generate(client, c), sample) for r in rs]
    with OUT.open("w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": f"q{i:03d}", **r}) + "\n")
    print(f"Wrote {len(rows)} questions from {len(sample)} chunks to {OUT}")


if __name__ == "__main__":
    main()
