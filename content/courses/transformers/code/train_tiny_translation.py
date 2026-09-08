"""Train a tiny English-to-French phrase model on CPU with no downloads.

Run: python content/courses/transformers/code/train_tiny_translation.py
Optional --output-dir persists a weights-only-compatible checkpoint and manifest.
This controlled word-order exercise is not a natural-language quality benchmark.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile

import torch
from torch import nn
import torch.nn.functional as F


PAD, BOS, EOS = 0, 1, 2
SPECIALS = ["<pad>", "<bos>", "<eos>"]


def corpus():
    colors = [("red", "rouge"), ("blue", "bleu"), ("green", "vert")]
    nouns = [("square", "carre"), ("circle", "cercle"), ("triangle", "triangle")]
    train = colors + nouns
    heldout = []
    for i, (color, translated_color) in enumerate(colors):
        for j, (noun, translated_noun) in enumerate(nouns):
            pair = (f"a {color} {noun}", f"un {translated_noun} {translated_color}")
            (heldout if i == j else train).append(pair)
    return train, heldout


def vocabulary(texts):
    return SPECIALS + sorted({word for text in texts for word in text.split()})


def encode(text, vocab, bos=False):
    lookup = {word: i for i, word in enumerate(vocab)}
    if not text.split():
        raise ValueError("Source and target phrases must be nonempty")
    try:
        return ([BOS] if bos else []) + [lookup[word] for word in text.split()] + [EOS]
    except KeyError as exc:
        raise ValueError(f"Word outside the training vocabulary: {exc.args[0]}") from exc


def pad(sequences):
    return nn.utils.rnn.pad_sequence([torch.tensor(row) for row in sequences],
                                    batch_first=True, padding_value=PAD)


def batch(pairs, source_vocab, target_vocab):
    source = pad([encode(src, source_vocab) for src, _ in pairs])
    full_target = pad([encode(tgt, target_vocab, bos=True) for _, tgt in pairs])
    return source, full_target[:, :-1], full_target[:, 1:]


@dataclass
class Config:
    source_size: int
    target_size: int
    width: int = 32
    heads: int = 4
    feedforward: int = 64
    max_length: int = 16


class Translator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.source_embedding = nn.Embedding(config.source_size, config.width, padding_idx=PAD)
        self.target_embedding = nn.Embedding(config.target_size, config.width, padding_idx=PAD)
        self.position = nn.Embedding(config.max_length, config.width)
        self.transformer = nn.Transformer(
            d_model=config.width, nhead=config.heads, num_encoder_layers=1,
            num_decoder_layers=1, dim_feedforward=config.feedforward,
            dropout=0., batch_first=True)
        self.output = nn.Linear(config.width, config.target_size)

    def embed(self, ids, embedding):
        if ids.shape[1] > self.config.max_length:
            raise ValueError("Sequence exceeds the learned position table")
        return embedding(ids) * self.config.width ** 0.5 + self.position(torch.arange(ids.shape[1]))

    def forward(self, source, target_input):
        source_padding, target_padding = source.eq(PAD), target_input.eq(PAD)
        forbidden_future = torch.ones(target_input.shape[1], target_input.shape[1],
                                      dtype=torch.bool).triu(1)
        hidden = self.transformer(
            self.embed(source, self.source_embedding), self.embed(target_input, self.target_embedding),
            tgt_mask=forbidden_future, src_key_padding_mask=source_padding,
            tgt_key_padding_mask=target_padding, memory_key_padding_mask=source_padding)
        return self.output(hidden)

    @torch.no_grad()
    def generate(self, source, max_new_tokens=8):
        if not 1 <= max_new_tokens <= self.config.max_length:
            raise ValueError("Invalid generation limit")
        self.eval()
        prefix = torch.full((len(source), 1), BOS, dtype=torch.long)
        finished = torch.zeros(len(source), dtype=torch.bool)
        for _ in range(max_new_tokens):
            logits = self(source, prefix)[:, -1].clone()
            logits[:, [PAD, BOS]] = float("-inf")
            next_id = logits.argmax(-1).masked_fill(finished, PAD)
            prefix = torch.cat([prefix, next_id[:, None]], dim=1)
            finished |= next_id.eq(EOS)
            if finished.all():
                break
        return prefix[:, 1:]


def loss_for(logits, labels):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=PAD)


def step(model, optimizer, tensors):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = loss_for(model(*tensors[:2]), tensors[2])
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return loss.item()


def terminated(ids):
    result = []
    for token in ids:
        if token == PAD:
            continue
        result.append(token)
        if token == EOS:
            break
    return result


@torch.no_grad()
def evaluate(model, pairs, source_vocab, target_vocab):
    model.eval()
    source, inputs, labels = batch(pairs, source_vocab, target_vocab)
    logits = model(source, inputs)
    valid = labels.ne(PAD)
    generated = model.generate(source)
    rows = []
    for (src, expected), prediction, target in zip(pairs, generated.tolist(), labels.tolist()):
        ids = terminated(prediction)
        rows.append({"source": src, "target": expected,
                     "prediction": " ".join(target_vocab[i] for i in ids if i != EOS),
                     "ended": EOS in ids, "exact": ids == terminated(target)})
    return {"teacher_forced_loss": loss_for(logits, labels).item(),
            "teacher_forced_token_accuracy": ((logits.argmax(-1) == labels) & valid).sum().item() / valid.sum().item(),
            "autoregressive_exact_match": sum(row["exact"] for row in rows) / len(rows), "rows": rows}


def save_checkpoint(path, model, optimizer, source_vocab, target_vocab, steps):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "course-tiny-translation-v1", "config": asdict(model.config),
                "source_vocab": source_vocab, "target_vocab": target_vocab,
                "steps": steps, "torch_version": str(torch.__version__), "seed": 808,
                "scope": "CPU toy phrase experiment; not a translation quality benchmark"}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state()}, path / "checkpoint.pt")


def load_checkpoint(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") != "course-tiny-translation-v1":
        raise ValueError("Unsupported checkpoint format")
    config = Config(**manifest["config"])
    for name, expected in (("source_vocab", config.source_size), ("target_vocab", config.target_size)):
        vocab = manifest[name]
        if len(vocab) != expected or vocab[:3] != SPECIALS or len(set(vocab)) != len(vocab):
            raise ValueError("Vocabulary and checkpoint geometry disagree")
    state = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = Translator(config)
    model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.015)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"])
    return model, optimizer, manifest


def train(steps=250):
    torch.manual_seed(808)
    torch.set_num_threads(1)
    train_pairs, heldout = corpus()
    source_vocab = vocabulary(src for src, _ in train_pairs)
    target_vocab = vocabulary(tgt for _, tgt in train_pairs)
    tensors = batch(train_pairs, source_vocab, target_vocab)
    model = Translator(Config(len(source_vocab), len(target_vocab)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.015)
    initial = loss_for(model(*tensors[:2]), tensors[2]).item()
    for _ in range(steps):
        step(model, optimizer, tensors)
    results = {"initial_loss": initial,
               "train": evaluate(model, train_pairs, source_vocab, target_vocab),
               "heldout_combinations": evaluate(model, heldout, source_vocab, target_vocab)}
    return model, optimizer, source_vocab, target_vocab, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    model, optimizer, source_vocab, target_vocab, results = train()
    with tempfile.TemporaryDirectory() as temporary:
        directory = args.output_dir or Path(temporary)
        save_checkpoint(directory, model, optimizer, source_vocab, target_vocab, 250)
        restored, _, _ = load_checkpoint(directory)
        restored_results = evaluate(restored, corpus()[1], source_vocab, target_vocab)
        assert restored_results == results["heldout_combinations"]
        if args.output_dir:
            (directory / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print("Disk checkpoint reproduces evaluation. Held-out combinations are not natural-language validation.")


if __name__ == "__main__":
    main()
