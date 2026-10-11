# Hybrid RAG

[![tests](https://github.com/tlevin1/hybrid-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/tlevin1/hybrid-rag/actions/workflows/tests.yml)

Retrieval-augmented question answering with three interchangeable retrievers —
**BM25 keyword search, Chroma vector search, and a hybrid of the two** — plus a
**cross-encoder reranker fine-tuned in PyTorch**, and an evaluation that
measures what each stage does to answer quality, using Arize Phoenix evaluators.

**Result:** hybrid retrieval answers **78.75%** of questions correctly (63 of 80) versus
72.5% for BM25 and 70.0% for vector search alone. The gap is widest on
questions phrased differently from the source text: **67.5% vs 52.5% and 60%**.
Hallucination was not the failure mode: answers were faithful to their
retrieved context 96.9–98.4% of the time. When retrieval missed, the model said
"I don't have enough information" instead of making something up.

**Adding a reranker** lifts correct answers to **86.25%** with an off-the-shelf
cross-encoder and **88.75%** with one fine-tuned on this corpus (~1 minute on
a laptop GPU). Fine-tuning's gain shows up where the retrievers struggled,
paraphrased questions, and holds across three random seeds
([details](#reranking-a-cross-encoder-fine-tuned-in-pytorch)).

This is a rebuild of my earlier [rag-chatbot](https://github.com/tlevin1/rag-chatbot),
which used hand-written TF-IDF over a small FAQ.

## How it works

```
question ─┬─► BM25 (keyword)   ─ top 20 ─┐                                 optional
          │                              ├─► reciprocal rank fusion ─► top 20 ─► cross-encoder ─► top 4 ─► Claude ─► answer
          └─► Chroma (vectors) ─ top 20 ─┘                                  reranker                       + [chunk] citations
```

- **BM25** (`hybrid_rag/retrieval.py`), written by hand: term frequency
  saturates (`k1 = 1.5`) so repeating a word helps less and less, and long
  chunks are length-normalized (`b = 0.75`). Strong when the question uses the
  document's own words, names and rare terms.
- **Vector search** with Chroma and its default embedding model
  (all-MiniLM-L6-v2 via ONNX Runtime, so no PyTorch). Matches meaning when the
  wording differs.
- **Reciprocal rank fusion** merges the two *rankings*:
  `score = Σ 1 / (60 + rank)`. BM25 scores are unbounded and cosine similarity
  sits in [-1, 1], so adding raw scores would let one retriever dominate. RRF
  needs no normalization or tuning and rewards passages both retrievers agree on.
  Every source in the API response reports its rank in each retriever, so you
  can see why it was chosen.
- **Grounded generation** (Claude Haiku 4.5): answer only from the passages,
  cite them as `[c012]`, or reply exactly "I don't have enough information to
  answer that."

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What did the author work on before college?", "mode": "hybrid"}'
```

```json
{
  "answer": "Before college the author worked on two main things outside of school: writing and programming [c000]. ...",
  "mode": "hybrid",
  "sources": [{"chunk_id": "c000", "score": 0.0328, "ranks": {"bm25": 1, "vector": 2}, "text": "..."}]
}
```

## Evaluation

**Corpus:** Paul Graham's essay "What I Worked On" (the one used in the
LlamaIndex and Phoenix RAG cookbooks), split into 123 chunks of ~800 characters.
It's downloaded and checksum-verified rather than committed, since it's his copyright.

**Questions** (`evals/questions.jsonl`, generated once and committed so every
run scores the same set): for 40 sampled chunks, Claude wrote two questions
answerable from that chunk alone, plus a reference answer.

- **Direct:** may reuse the passage's wording. Questions generated from a
  passage tend to echo it, which flatters keyword search. This is the usual
  cookbook setup.
- **Paraphrased:** must avoid the passage's distinctive words, so retrieval
  has to match meaning. Example: *"When was the book 'On Lisp' published?"*
  versus *"During which phase of his academic career did the author write most
  of a programming language instructional book?"*

### 1. Retrieval, scored exactly (no LLM)

Each question knows the chunk it came from, so retrieval can be checked
directly: is that chunk ranked first, and is it in the top 4 that Claude sees?

| Questions | Retriever | hit@1 | hit@4 | MRR@20 |
|---|---|---|---|---|
| Direct (40) | BM25 | **0.825** | **0.925** | **0.879** |
| | Vector | 0.575 | 0.825 | 0.681 |
| | Hybrid | 0.700 | 0.900 | 0.806 |
| Paraphrased (40) | BM25 | 0.300 | 0.550 | 0.420 |
| | Vector | 0.275 | 0.600 | 0.430 |
| | Hybrid | **0.350** | **0.700** | **0.507** |
| All (80) | BM25 | **0.562** | 0.738 | 0.649 |
| | Vector | 0.425 | 0.713 | 0.556 |
| | Hybrid | 0.525 | **0.800** | **0.656** |

### 2. Answer quality, judged with Phoenix

Every question was answered once per retriever (240 answers), then judged with
Claude Haiku 4.5 through Arize Phoenix:

- **Document relevance** (Phoenix `DocumentRelevanceEvaluator`) on each of the
  4 retrieved passages → precision@4, NDCG@4, and hit (any relevant passage).
- **Correctness** against the saved reference answer (a custom judge built with
  Phoenix's `create_classifier`). A refusal counts as not correct.
- **Faithfulness** (Phoenix `FaithfulnessEvaluator`): is the answer supported by
  the retrieved passages, i.e. not hallucinated? Refusals are excluded.

| Questions | Retriever | Correct | Refused | Faithful | Hit | NDCG@4 | Precision@4 |
|---|---|---|---|---|---|---|---|
| Direct (40) | BM25 | **92.5%** | 5.0% | 100% | 0.950 | 0.912 | 0.294 |
| | Vector | 80.0% | 15.0% | 100% | 0.850 | 0.764 | 0.287 |
| | Hybrid | 90.0% | 7.5% | 97.3% | 0.925 | 0.846 | 0.294 |
| Paraphrased (40) | BM25 | 52.5% | 42.5% | 95.7% | 0.600 | 0.505 | 0.206 |
| | Vector | 60.0% | 37.5% | 96.0% | 0.750 | 0.613 | 0.287 |
| | Hybrid | **67.5%** | **30.0%** | 96.4% | **0.825** | **0.654** | 0.281 |
| **All (80)** | BM25 | 72.5% | 23.75% | 98.4% | 0.775 | 0.709 | 0.250 |
| | Vector | 70.0% | 26.3% | 98.3% | 0.800 | 0.688 | 0.287 |
| | **Hybrid** | **78.75%** | **18.75%** | 96.9% | **0.875** | **0.750** | **0.287** |

### What the numbers say

- **Hybrid is the best default, not the best everywhere.** On direct questions
  BM25 is slightly ahead (92.5% vs 90% correct), and it is still the best at
  putting the source chunk at rank 1. Hybrid wins because the two retrievers
  miss *different* questions, so fusing them recovers most of each one's misses.
- **Wording mismatch is the real problem.** Every retriever drops 20–40 points
  from direct to paraphrased questions. Vector search alone helps less than
  expected (52.5% → 60%), and fusion helps more (→ 67.5%).
- **Failures show up as refusals, not hallucinations.** Faithfulness stays at
  95.7–100% everywhere; what changes between retrievers is how often the model
  says it doesn't know. Of the 55 refusals across all runs, 43 happened when
  retrieval missed (the correct behaviour) and 12 happened even though a
  relevant passage was retrieved: the model being over-cautious, a generation
  problem rather than a retrieval one.
- **Precision@4 stays near 0.29 by construction:** most questions have one
  relevant passage among the four retrieved.
- **Sample size is modest:** 40 questions per style, so a 0.1 difference is 4
  questions. Treat the direction as solid and the exact figures as approximate.

## Reranking: a cross-encoder fine-tuned in PyTorch

The retrievers encode the question and each passage separately, so they never
compare them word by word. A **cross-encoder** reads the question and a passage
*together* through a transformer and outputs one relevance score: much more
accurate, too slow for the whole corpus. So hybrid search narrows 123 chunks to
20 candidates and the cross-encoder reorders those (`hybrid_rag/rerank.py`).
It can only reorder what retrieval found: the source passage is among hybrid's
20 candidates for 76 of 80 eval questions, so 95% is the ceiling.

Three versions were compared on the same eval questions:

1. **No reranker** (hybrid only).
2. **Pretrained:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (22.7M parameters,
   trained on web search queries), used as is.
3. **Fine-tuned:** the same model trained on this essay (`training/`).

### Training (`training/train.py`)

- **Data without eval leakage:** 166 questions written (by the same prompt as
  the eval set) from the 83 chunks that *no* eval question uses. Train and
  validation are split **by chunk** (136 / 30 questions), so validation passages
  are unseen.
- **Hard negatives:** for each question, the 7 passages hybrid search ranks
  highest that aren't the answer. The source's neighboring chunks are excluded:
  chunks overlap by 100 characters, so a neighbor may contain the answer, and
  labelling it irrelevant would teach the model something false.
- **Listwise loss:** each question's 8 passages are scored together and trained
  with cross-entropy over "which one is correct", which optimizes the ranking
  directly instead of judging each pair in isolation.
- AdamW (lr 2e-5, weight decay 0.01), 10% linear warmup, gradient clipping,
  4 epochs, batches of 8 questions (64 pairs). **About 1.2 minutes on Apple MPS.**
- **Early stopping on the real task:** each epoch reranks the validation
  questions' actual hybrid candidates and measures MRR. Epoch 0 is the
  untrained model, so if fine-tuning never beat it, the pretrained model would
  be the one saved. Validation MRR went 0.806 → 0.836 (epoch 3).

### Passage ranking (no LLM)

| Questions | Reranker | hit@1 | hit@4 | MRR@20 |
|---|---|---|---|---|
| Direct (40) | none (hybrid) | 0.700 | 0.900 | 0.806 |
| | pretrained | **0.900** | **1.000** | **0.946** |
| | fine-tuned | **0.900** | **1.000** | **0.946** |
| Paraphrased (40) | none (hybrid) | 0.350 | 0.700 | 0.507 |
| | pretrained | 0.525 | 0.725 | 0.630 |
| | fine-tuned | **0.625** | **0.775** | **0.709** |
| All (80) | none (hybrid) | 0.525 | 0.800 | 0.656 |
| | pretrained | 0.713 | 0.863 | 0.788 |
| | fine-tuned | **0.762** | **0.887** | **0.827** |

**Is fine-tuning's gain real?** Compared question by question with the
pretrained model (`evals/reranker_eval.py`), across three training seeds:

| Paraphrased questions | Source moved up | Moved down | Unchanged | Sign test | MRR gain, 95% CI |
|---|---|---|---|---|---|
| Seed 0 | 12 | 1 | 27 | p = 0.003 | +0.014 to +0.150 |
| Seed 1 | 13 | 1 | 26 | p = 0.002 | +0.013 to +0.136 |
| Seed 2 | 13 | 1 | 26 | p = 0.002 | +0.013 to +0.136 |

On direct questions it changes nothing (39–40 of 40 unchanged on every seed):
those were already solved. All three seeds reach the same hit@1 and hit@4, so
the result isn't a lucky run.

### Answer quality with reranking (Phoenix-judged)

| Questions | Reranker | Correct | Refused | Faithful | Hit | NDCG@4 | Precision@4 |
|---|---|---|---|---|---|---|---|
| Direct (40) | none (hybrid) | 90.0% | 7.5% | 97.3% | 0.925 | 0.846 | 0.294 |
| | pretrained | **100%** | 0% | 100% | 1.000 | 0.988 | 0.338 |
| | fine-tuned | **100%** | 0% | 100% | 1.000 | 0.982 | 0.350 |
| Paraphrased (40) | none (hybrid) | 67.5% | 30.0% | 96.4% | 0.825 | 0.654 | 0.281 |
| | pretrained | 72.5% | 25.0% | 100% | 0.825 | 0.767 | 0.350 |
| | fine-tuned | **77.5%** | **17.5%** | 97.0% | **0.875** | **0.801** | **0.381** |
| **All (80)** | none (hybrid) | 78.75% | 18.75% | 96.9% | 0.875 | 0.750 | 0.287 |
| | pretrained | 86.25% | 12.5% | 100% | 0.9125 | 0.878 | 0.344 |
| | **fine-tuned** | **88.75%** | **8.75%** | 98.6% | **0.9375** | **0.892** | **0.366** |

Counted per answer: hybrid → fine-tuned reranker fixes 10 answers and breaks 2
(sign test p = 0.04). Pretrained → fine-tuned fixes 2 and breaks none, **too few
to be significant on their own**: fine-tuning's gain is clear at the ranking
level, and most of the end-to-end improvement comes from adding a reranker at all.

### Reranking takeaways

- **A pretrained cross-encoder is the biggest single win** in this project:
  rank-1 accuracy 0.525 → 0.713, and refusals drop by a third. It's cheap to add
  and transfers well from web search to an essay.
- **Fine-tuning helps where the vocabulary is the problem.** It improves
  paraphrased questions and leaves the already-solved direct ones alone.
- **The limit is now retrieval, not ranking:** 4 of 80 eval questions never
  reach the reranker. Query rewriting or more candidates would be the next lever.
- **Caveats:** one essay, 80 eval questions, and the training pool includes the
  eval chunks' neighbors, which share 100 characters of overlap with them.
  Fine-tuned weights aren't committed (≈90 MB); `python -m training.train`
  reproduces them in about a minute.

### Choices in the eval, and why

- **Reference-based correctness instead of Phoenix's built-in correctness
  judge.** The built-in `CorrectnessEvaluator` has no answer key and judges
  from its own knowledge. In a smoke test it marked one of three near-identical,
  verbatim-quoted answers incorrect, which is noise larger than the differences
  being measured.
- **Scores computed from evaluator outputs directly**, not queried back out of
  Phoenix traces, so results don't depend on a running Phoenix server.
- **All judges in one event loop.** The Phoenix LLM wrapper's async client is
  bound to its first event loop; a separate `asyncio.run()` per evaluator caused
  connection errors and retries.
- **Saved answers store chunk ids, not passage text**, so the repo doesn't
  republish the essay. `--reuse-answers` rebuilds the text from the downloaded copy.

## Run it

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt          # app + tests
pip install -r requirements-eval.txt         # Phoenix, for the answer-quality eval

python -m pytest -q                          # 36 tests: offline, no model download, no API key

export ANTHROPIC_API_KEY=...
uvicorn hybrid_rag.app:app --reload          # downloads the essay and the embedding model on first run

python -m evals.retrieval_eval               # free: exact retrieval metrics
python -m evals.phoenix_eval                 # ~6 min, about $1–3 on Haiku: answers + Phoenix judges
python -m evals.phoenix_eval --reuse-answers # re-judge the saved answers only
python -m evals.generate_questions           # regenerate the question set (changes every result)

# Reranker (PyTorch)
pip install -r requirements-train.txt
python -m training.data                      # training questions (~$0.10; committed already)
python -m training.train                     # ~1-2 min on Apple MPS / a GPU; writes models/reranker-ft
python -m evals.retrieval_eval --rerank      # free: ranking metrics incl. both rerankers
python -m training.train --seed 1 --out models/reranker-ft-seed1 --report reports/training-seed1.json
python -m training.train --seed 2 --out models/reranker-ft-seed2 --report reports/training-seed2.json
python -m evals.reranker_eval                # pretrained vs each fine-tuned seed: paired tests
python -m evals.phoenix_eval --modes hybrid+ms-marco hybrid+finetuned   # answer quality, ~$1
```

## Layout

```
hybrid_rag/
  corpus.py      download (checksum-pinned) and chunking
  retrieval.py   BM25, Chroma vector index, reciprocal rank fusion, Retriever
  rerank.py      cross-encoder reranking (PyTorch / transformers)
  generate.py    grounded answers with citations
  app.py         FastAPI: POST /ask {question, mode, top_k}
evals/
  generate_questions.py   direct + paraphrased question set
  retrieval_eval.py       hit@k, MRR against each question's source chunk
  phoenix_eval.py         answers per retriever + Phoenix judges
  metrics.py              NDCG, refusal handling, summaries
  reranker_eval.py        pretrained vs fine-tuned: per seed, sign test, bootstrap CI
training/
  data.py        leakage-free questions, chunk-level split, hard negatives
  train.py       listwise fine-tuning with early stopping on validation MRR
reports/                  every number above, plus per-answer scores
```

The tests use a small deterministic embedding in place of the real model, a
fake LLM in place of Claude, and a tiny randomly initialized BERT for the
training loop, so CI needs no downloads or API key.
