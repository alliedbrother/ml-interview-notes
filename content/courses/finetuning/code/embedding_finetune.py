"""Embedding fine-tuning lab for module 16 (Multimodal Fine-Tuning).

Purpose
    --smoke : a CPU-only, pure-torch rehearsal of a Sentence Transformers run.
              A tiny bag-of-embeddings encoder (EmbeddingBag mean -> Linear) is
              trained on a synthetic "domain retrieval" corpus in which queries
              use different surface tokens (synonyms) from the documents they
              should retrieve, so the random-init model is near chance.
              Phase 1: MultipleNegativesRankingLoss (in-batch negatives, scale 20)
                       wrapped in a Matryoshka sum over dims (64, 32, 16).
              Phase 2: mine one hard negative per pair with the phase-1 model
                       (top-ranked non-positive with an absolute margin below the
                       positive, the false-negative filter), then train on
                       (anchor, positive, negative) triplets with the same loss.
              The run asserts that NDCG@10 on held-out queries rises from below
              0.35 to above 0.80, and that 16-dim truncated embeddings still
              score above 0.70.
    --full  : lab L16.3 with Sentence Transformers v6: BAAI/bge-small-en-v1.5
              on philschmid/finanical-rag-embedding-dataset (question, context)
              pairs, mine_hard_negatives, MNRL inside MatryoshkaLoss, NDCG@10
              before and after with InformationRetrievalEvaluator at 384 and
              64 dims. --no-hard-negatives trains on pairs only (the ablation).

Module      content/courses/finetuning/16-multimodal-and-embedding-fine-tuning.md
Hardware    smoke: any CPU, < 60 s.  full: T0 (free T4 16 GB, fp16), ~10 minutes.
Versions    smoke: Python 3.11+, torch 2.x (no other dependency).
            full (as of October 2026): sentence-transformers[train] 6.1.0,
            transformers 5.19, datasets, accelerate.

Run (from the repository root)
    python content/courses/finetuning/code/embedding_finetune.py --smoke
    python content/courses/finetuning/code/embedding_finetune.py --full
    python content/courses/finetuning/code/embedding_finetune.py --full --no-hard-negatives
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Synthetic corpus: 20 topic families x 2 sibling documents
# --------------------------------------------------------------------------

N_FAMILIES, DOCS_PER_FAMILY = 20, 2
N_DOCS = N_FAMILIES * DOCS_PER_FAMILY
TOPIC_TOKENS, KEY_TOKENS, NOISE_POOL = 3, 4, 30


def build_corpus(g: torch.Generator):
    """Documents use 'doc-side' token ids; queries use disjoint 'synonym' ids."""
    nid = 0

    def fresh(n):
        nonlocal nid
        out = list(range(nid, nid + n))
        nid += n
        return out

    noise = fresh(NOISE_POOL)
    topic = [fresh(TOPIC_TOKENS) for _ in range(N_FAMILIES)]
    key = [fresh(KEY_TOKENS) for _ in range(N_DOCS)]
    topic_syn = [fresh(TOPIC_TOKENS) for _ in range(N_FAMILIES)]
    key_syn = [fresh(KEY_TOKENS) for _ in range(N_DOCS)]
    docs = []
    for d in range(N_DOCS):
        f = d // DOCS_PER_FAMILY
        nz = torch.randint(0, NOISE_POOL, (3,), generator=g).tolist()
        docs.append(topic[f] + key[d] + [noise[i] for i in nz])
    return {"docs": docs, "noise": noise, "topic_syn": topic_syn, "key_syn": key_syn, "vocab": nid}


def sample_query(c, d: int, g: torch.Generator) -> list[int]:
    f = d // DOCS_PER_FAMILY
    k = torch.randperm(KEY_TOKENS, generator=g)[:2].tolist()
    t = int(torch.randint(0, TOPIC_TOKENS, (1,), generator=g))
    nz = torch.randint(0, NOISE_POOL, (2,), generator=g).tolist()
    return [c["key_syn"][d][i] for i in k] + [c["topic_syn"][f][t]] + [c["noise"][i] for i in nz]


def relevance(q_doc: int, d: int) -> int:
    if d == q_doc:
        return 2  # the exact document
    if d // DOCS_PER_FAMILY == q_doc // DOCS_PER_FAMILY:
        return 1  # its sibling: same topic, partially useful
    return 0


# --------------------------------------------------------------------------
# Encoder, losses, metric
# --------------------------------------------------------------------------


class BagEncoder(nn.Module):
    def __init__(self, vocab: int, dim: int = 64):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab, dim, mode="mean")
        self.proj = nn.Linear(dim, dim)

    def forward(self, seqs: list[list[int]]) -> torch.Tensor:
        flat = torch.tensor([t for s in seqs for t in s])
        offsets = torch.tensor([0] + [len(s) for s in seqs[:-1]]).cumsum(0)
        return self.proj(self.emb(flat, offsets))  # (B, dim)


def mnrl(q: torch.Tensor, d: torch.Tensor, scale: float = 20.0) -> torch.Tensor:
    """InfoNCE over cosine similarities. d holds B positives, then optional extra negatives."""
    sims = F.normalize(q, dim=-1) @ F.normalize(d, dim=-1).T  # (B, B or 2B)
    return F.cross_entropy(scale * sims, torch.arange(q.shape[0]))


def matryoshka_mnrl(q, d, dims=(64, 32, 16), weights=(1.0, 1.0, 1.0)) -> torch.Tensor:
    return sum(w * mnrl(q[:, :k], d[:, :k]) for k, w in zip(dims, weights))


def ndcg_at_k(rels_in_rank_order: list[int], all_rels: list[int], k: int = 10) -> float:
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels_in_rank_order[:k]))
    ideal = sorted(all_rels, reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg


@torch.no_grad()
def evaluate(model, c, queries, dim: int | None = None) -> float:
    q = model([s for s, _ in queries])
    d = model(c["docs"])
    if dim is not None:
        q, d = q[:, :dim], d[:, :dim]
    scores = F.normalize(q, dim=-1) @ F.normalize(d, dim=-1).T  # (Q, N_DOCS)
    out = []
    for i, (_, target) in enumerate(queries):
        order = scores[i].argsort(descending=True).tolist()
        all_rels = [relevance(target, j) for j in range(N_DOCS)]
        out.append(ndcg_at_k([all_rels[j] for j in order], all_rels))
    return sum(out) / len(out)


@torch.no_grad()
def mine_hard_negatives(model, c, pairs, margin: float = 0.05):
    """One negative per pair: the best-scoring non-positive whose score is at least
    `margin` below the positive's (skip candidates that may be unlabelled positives)."""
    q = F.normalize(model([s for s, _ in pairs]), dim=-1)
    d = F.normalize(model(c["docs"]), dim=-1)
    scores = q @ d.T
    triplets, skipped = [], 0
    for i, (seq, pos) in enumerate(pairs):
        for j in scores[i].argsort(descending=True).tolist():
            if j == pos:
                continue
            if scores[i, j] > scores[i, pos] - margin:
                skipped += 1  # too close to the positive: possible false negative
                continue
            triplets.append((seq, pos, j))
            break
    return triplets, skipped


def batches(items, bs: int, g: torch.Generator):
    """No-duplicates sampler: a document appears at most once per batch,
    so no in-batch 'negative' is secretly a positive."""
    order = torch.randperm(len(items), generator=g).tolist()
    pending, batch, seen = list(order), [], set()
    while pending:
        rest = []
        for i in pending:
            pos = items[i][1]
            if pos in seen or len(batch) == bs:
                rest.append(i)
                continue
            batch.append(items[i])
            seen.add(pos)
        if batch:
            yield batch
        batch, seen, pending = [], set(), rest


def run_smoke() -> None:
    t0 = time.time()
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    c = build_corpus(g)
    train = [(sample_query(c, d, g), d) for d in range(N_DOCS) for _ in range(12)]
    test = [(sample_query(c, d, g), d) for d in range(N_DOCS) for _ in range(3)]
    model = BagEncoder(c["vocab"])
    before = evaluate(model, c, test)
    print(f"NDCG@10 before training: {before:.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    for epoch in range(6):  # phase 1: pairs, in-batch negatives only
        for b in batches(train, 32, g):
            loss = matryoshka_mnrl(model([s for s, _ in b]), model([c["docs"][d] for _, d in b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
    phase1 = evaluate(model, c, test)
    print(f"NDCG@10 after MNRL (pairs):          {phase1:.3f}")

    triplets, skipped = mine_hard_negatives(model, c, train)
    print(f"mined {len(triplets)} hard negatives, skipped {skipped} near-positive candidates")
    for epoch in range(4):  # phase 2: (anchor, positive, hard negative)
        for b in batches([(s, p, n) for s, p, n in triplets], 32, g):
            q = model([s for s, _, _ in b])
            docs = model([c["docs"][p] for _, p, _ in b] + [c["docs"][n] for _, _, n in b])  # (2B, dim)
            loss = sum(mnrl(q[:, :k], docs[:, :k]) for k in (64, 32, 16))
            opt.zero_grad()
            loss.backward()
            opt.step()
    after = evaluate(model, c, test)
    after16 = evaluate(model, c, test, dim=16)
    print(f"NDCG@10 after hard negatives:        {after:.3f}  (16-dim truncation: {after16:.3f})")
    assert before < 0.35, before
    assert after > 0.80, after
    assert after16 > 0.70, after16
    print(f"smoke OK in {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------
# Full mode: Sentence Transformers v6 on T0
# --------------------------------------------------------------------------


def run_full(args) -> None:  # pragma: no cover - needs downloads (GPU recommended)
    from datasets import load_dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.sentence_transformer.losses import (
        MatryoshkaLoss,
        MultipleNegativesRankingLoss,
    )
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    from sentence_transformers.util import mine_hard_negatives as st_mine

    ds = load_dataset("philschmid/finanical-rag-embedding-dataset", split="train")
    ds = ds.rename_columns({"question": "anchor", "context": "positive"})
    ds = ds.add_column("id", list(range(len(ds))))
    split = ds.train_test_split(test_size=0.1, seed=42)
    train, test = split["train"], split["test"]

    corpus = {str(i): t for i, t in zip(ds["id"], ds["positive"])}  # retrieve over the full corpus
    queries = {str(i): q for i, q in zip(test["id"], test["anchor"])}
    relevant = {str(i): {str(i)} for i in test["id"]}
    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant, name="fin", ndcg_at_k=[10])
    evaluator64 = InformationRetrievalEvaluator(queries, corpus, relevant, name="fin-64", ndcg_at_k=[10],
                                                truncate_dim=64)  # first 64 of 384 dims

    def ndcg10(ev, m):
        res = ev(m)
        return res[next(k for k in res if k.endswith("ndcg@10"))]

    model = SentenceTransformer(args.model)
    print(f"before: NDCG@10 = {ndcg10(evaluator, model):.4f} (384-d), {ndcg10(evaluator64, model):.4f} (64-d)")

    train = train.select_columns(["anchor", "positive"])
    if not args.no_hard_negatives:
        train = st_mine(
            train, model,
            num_negatives=1, range_min=10, range_max=50,  # skip the top 10: likely unlabelled positives
            relative_margin=0.05, sampling_strategy="top", output_format="triplet",
        )
    dims = [384, 256, 128, 64]
    loss = MatryoshkaLoss(model, MultipleNegativesRankingLoss(model), matryoshka_dims=dims,
                          matryoshka_weights=[1] * len(dims))
    targs = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=64,
        learning_rate=2e-5,
        warmup_steps=0.1,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=train, loss=loss,
                                         evaluator=evaluator)
    trainer.train()
    tag = "pairs only" if args.no_hard_negatives else "with mined negatives"
    print(f"after ({tag}): NDCG@10 = {ndcg10(evaluator, model):.4f} (384-d), "
          f"{ndcg10(evaluator64, model):.4f} (64-d)")
    model.save_pretrained(args.output_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--output-dir", default="bge-small-fin-mnrl")
    ap.add_argument("--no-hard-negatives", action="store_true",
                    help="full mode: skip mining and train MNRL on (anchor, positive) pairs only")
    args = ap.parse_args()
    if args.full:
        run_full(args)
    else:
        run_smoke()


if __name__ == "__main__":
    main()
