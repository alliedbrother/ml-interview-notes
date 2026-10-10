"""Train an MLP, a CNN and a GRU end to end, then transfer the CNN (module 19 lab).

Four parts, all on generated data (no downloads):
  mlp       3-class spirals, MLP 2 -> 64 -> 64 -> 3, AdamW + warmup-cosine,
            train/val/test split, evaluation under inference_mode.
  cnn       procedurally generated 16x16 grayscale shapes (filled square, hollow
            square, disk, plus), 2,000 train / 400 val / 400 test images, each
            split drawn from its own seeded generator; conv-BN-ReLU CNN with
            global average pooling, batched augmentation on the training batch,
            150 AdamW steps at batch 32.
  gru       character-level GRU language model on a generated 30,000-char corpus, 100 steps,
            truncated BPTT with the hidden state detached between chunks,
            gradient clipping, temperature sampling; compared with a unigram baseline.
  transfer  pretrain the CNN on 3 source classes, then adapt to a 4-way target
            task (higher noise, a class the source never saw) with 100 labelled
            images: linear probe (frozen backbone, BN in eval), full fine-tune with
            a 10x smaller backbone learning rate, and a from-scratch baseline.

Run from the repository root:
    python content/courses/pytorch/code/19_classic_models.py [--seed 0] [--which {mlp,cnn,gru,transfer,all}] [--json-out PATH]
"""

import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = 16
SHAPE_NAMES = ["filled_square", "hollow_square", "disk", "plus"]


# ------------------------------------------------------------------ shared pieces
def warmup_cosine(opt, steps, warmup_frac=0.1):
    warmup = max(1, int(steps * warmup_frac))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


@torch.inference_mode()
def accuracy(model, x, y, batch=512):
    was_training = model.training
    model.eval()
    correct = 0
    for i in range(0, x.shape[0], batch):
        correct += int((model(x[i : i + batch]).argmax(-1) == y[i : i + batch]).sum())
    model.train(was_training)
    return correct / x.shape[0]


def count_params(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


# ------------------------------------------------------------------ part 1: MLP
def make_spirals(n_per_class, classes, gen, noise=0.08):
    """Interleaved spirals in 2-d; returns x (N, 2) float32 and y (N,) int64."""
    t = torch.rand(classes, n_per_class, generator=gen)
    r = 0.15 + 0.85 * t
    base = torch.arange(classes).unsqueeze(1) * (2 * math.pi / classes)
    theta = base + 3.0 * t
    x = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
    x = x + noise * torch.randn(x.shape, generator=gen)
    y = torch.arange(classes).repeat_interleave(n_per_class)
    return x.reshape(-1, 2), y


def run_mlp(seed, steps=300, batch=128):
    gen = torch.Generator().manual_seed(seed)
    x, y = make_spirals(400, 3, gen)
    perm = torch.randperm(x.shape[0], generator=gen)
    x, y = x[perm], y[perm]
    n_tr, n_va = 720, 240  # 60 / 20 / 20 split of 1,200 points
    x_tr, y_tr = x[:n_tr], y[:n_tr]
    x_va, y_va = x[n_tr : n_tr + n_va], y[n_tr : n_tr + n_va]
    x_te, y_te = x[n_tr + n_va :], y[n_tr + n_va :]

    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    sched = warmup_cosine(opt, steps)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n_tr, (batch,), generator=gen)
        loss = F.cross_entropy(model(x_tr[idx]), y_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return {
        "val_acc": accuracy(model, x_va, y_va),
        "test_acc": accuracy(model, x_te, y_te),
        "params": count_params(model),
    }


# ------------------------------------------------------------------ procedural images
def make_shapes(n, seed, classes=(0, 1, 2, 3), noise=0.15, size=IMG):
    """n grayscale images (n, 1, size, size) of the given shape classes.

    Deterministic in (n, seed, classes, noise). Labels index into `classes`
    (so a 3-class source task has labels 0..2).
    """
    gen = torch.Generator().manual_seed(seed)
    classes_t = torch.tensor(classes)
    label = torch.randint(0, len(classes), (n,), generator=gen)
    shape = classes_t[label]
    r = 2.5 + 2.0 * torch.rand(n, generator=gen)                       # half-size 2.5..4.5
    lo, hi = 5, size - 5
    cy = torch.randint(lo, hi + 1, (n,), generator=gen).float()
    cx = torch.randint(lo, hi + 1, (n,), generator=gen).float()
    level = 0.6 + 0.4 * torch.rand(n, generator=gen)
    ys, xs = torch.meshgrid(torch.arange(size).float(), torch.arange(size).float(), indexing="ij")
    dy = (ys - cy.view(n, 1, 1)).abs()                                 # (n, size, size)
    dx = (xs - cx.view(n, 1, 1)).abs()
    rr = r.view(n, 1, 1)
    cheb = torch.maximum(dx, dy)
    masks = torch.stack(
        [
            cheb <= rr,                                                # filled square
            (cheb <= rr) & (cheb >= rr - 1.0),                         # hollow square
            dx**2 + dy**2 <= rr**2,                                    # disk
            ((dx <= 0.5) & (dy <= rr)) | ((dy <= 0.5) & (dx <= rr)),   # plus
        ],
        dim=1,
    )                                                                  # (n, 4, size, size)
    mask = masks[torch.arange(n), shape].float()
    img = level.view(n, 1, 1) * mask + noise * torch.randn(n, size, size, generator=gen)
    return img.unsqueeze(1), label


def split_seeds(seed):
    """Disjoint generator seeds per split: train, val, test."""
    return seed * 100 + 1, seed * 100 + 2, seed * 100 + 3


# ------------------------------------------------------------------ augmentation
def random_hflip(x, gen, p=0.5):
    flip = torch.rand(x.shape[0], generator=gen) < p
    return torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)


def random_shift(x, gen, pad=2):
    """Pad by `pad` (zeros) and crop back at a random offset per image."""
    n, _, h, w = x.shape
    padded = F.pad(x, (pad, pad, pad, pad))
    oy = torch.randint(0, 2 * pad + 1, (n,), generator=gen)
    ox = torch.randint(0, 2 * pad + 1, (n,), generator=gen)
    rows = (oy.view(n, 1) + torch.arange(h)).view(n, 1, h, 1).expand(n, x.shape[1], h, w + 2 * pad)
    cols = (ox.view(n, 1) + torch.arange(w)).view(n, 1, 1, w).expand(n, x.shape[1], h, w)
    return padded.gather(2, rows).gather(3, cols)


def gaussian_noise(x, gen, std=0.05):
    return x + std * torch.randn(x.shape, generator=gen)


def random_erase(x, gen, p=0.25, size=4):
    """Zero one size x size square per selected image (cutout)."""
    n, _, h, w = x.shape
    oy = torch.randint(0, h - size + 1, (n, 1, 1, 1), generator=gen)
    ox = torch.randint(0, w - size + 1, (n, 1, 1, 1), generator=gen)
    ys = torch.arange(h).view(1, 1, h, 1)
    xs = torch.arange(w).view(1, 1, 1, w)
    box = (ys >= oy) & (ys < oy + size) & (xs >= ox) & (xs < ox + size)
    chosen = (torch.rand(n, generator=gen) < p).view(n, 1, 1, 1)
    return x.masked_fill(box & chosen, 0.0)


def augment(x, gen):
    return random_erase(gaussian_noise(random_shift(random_hflip(x, gen), gen), gen), gen)


# ------------------------------------------------------------------ part 2: CNN
def conv_bn_relu(c_in, c_out):
    return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU()]


class SmallCNN(nn.Module):
    def __init__(self, num_classes=4, c1=16, c2=32):
        super().__init__()
        self.features = nn.Sequential(
            *conv_bn_relu(1, c1), *conv_bn_relu(c1, c1), nn.MaxPool2d(2),
            *conv_bn_relu(c1, c2), *conv_bn_relu(c2, c2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(c2, num_classes)

    def forward(self, x):
        return self.head(self.features(x))


def cnn_param_formula(num_classes=4, c1=16, c2=32, k=3):
    convs = [(1, c1), (c1, c1), (c1, c2), (c2, c2)]
    conv = sum(ci * co * k * k for ci, co in convs)       # bias=False
    bn = sum(2 * co for _, co in convs)                   # weight + bias (running stats are buffers)
    head = c2 * num_classes + num_classes
    return conv + bn + head


def set_bn_eval(module):
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def train_cnn(model, x, y, steps, gen, lr=3e-3, batch=64, param_groups=None, bn_eval=False, aug=True):
    groups = param_groups if param_groups is not None else [{"params": model.parameters()}]
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=1e-4)
    sched = warmup_cosine(opt, steps)
    model.train()
    if bn_eval:
        set_bn_eval(model)
    for _ in range(steps):
        idx = torch.randint(0, x.shape[0], (batch,), generator=gen)
        xb = augment(x[idx], gen) if aug else x[idx]
        loss = F.cross_entropy(model(xb), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return model


def run_cnn(seed, steps=150, batch=32):
    with torch.backends.nnpack.flags(enabled=False):   # NNPACK conv backward was the bottleneck on the reference Mac
        return _run_cnn(seed, steps, batch)


def _run_cnn(seed, steps, batch):
    s_tr, s_va, s_te = split_seeds(seed)
    x_tr, y_tr = make_shapes(2000, s_tr)
    x_va, y_va = make_shapes(400, s_va)
    x_te, y_te = make_shapes(400, s_te)
    torch.manual_seed(seed)
    model = SmallCNN()
    train_cnn(model, x_tr, y_tr, steps, torch.Generator().manual_seed(seed + 7), batch=batch)
    return {
        "val_acc": accuracy(model, x_va, y_va),
        "test_acc": accuracy(model, x_te, y_te),
        "params": count_params(model),
        "params_formula": cnn_param_formula(),
        "class_counts_train": torch.bincount(y_tr, minlength=4).tolist(),
    }


# ------------------------------------------------------------------ part 3: GRU
FIRST = ["ada", "bo", "cyra", "dov", "elin", "fen", "gil", "hana", "ivo", "jun"]
LAST = ["stone", "rivers", "marsh", "hale", "brook", "fairley"]
CITY = ["oslo", "lima", "kyoto", "perth", "quito"]


def make_corpus(seed, n_chars=30_000):
    """Lines like 'ada hale, 34, lima.' from a template grammar."""
    gen = torch.Generator().manual_seed(seed)
    out, total = [], 0
    while total < n_chars:
        f, l, c = (int(torch.randint(0, len(v), (1,), generator=gen)) for v in (FIRST, LAST, CITY))
        age = 18 + int(torch.randint(0, 60, (1,), generator=gen))
        line = f"{FIRST[f]} {LAST[l]}, {age}, {CITY[c]}.\n"
        out.append(line)
        total += len(line)
    return "".join(out)[:n_chars]


class CharGRU(nn.Module):
    def __init__(self, vocab, d_emb=32, hidden=64, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_emb)
        self.gru = nn.GRU(d_emb, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, idx, h=None):            # idx (B, T) -> logits (B, T, V), h (layers, B, hidden)
        y, h = self.gru(self.emb(idx), h)
        return self.out(y), h


def streams(ids, batch):
    """Cut a 1-d id tensor into `batch` contiguous streams: (batch, L)."""
    length = ids.numel() // batch
    return ids[: batch * length].view(batch, length)


@torch.inference_mode()
def stream_loss(model, data, chunk):
    model.eval()
    h, total, count = None, 0.0, 0
    for t in range(0, data.shape[1] - 1, chunk):
        x = data[:, t : t + chunk]
        y = data[:, t + 1 : t + 1 + chunk]
        x = x[:, : y.shape[1]]
        logits, h = model(x, h)
        total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum"))
        count += y.numel()
    model.train()
    return total / count


@torch.inference_mode()
def sample(model, stoi, itos, prompt, n, temperature, gen):
    model.eval()
    idx = torch.tensor([[stoi[c] for c in prompt]])
    logits, h = model(idx)
    out = []
    for _ in range(n):
        probs = F.softmax(logits[:, -1] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=gen)
        out.append(itos[int(nxt)])
        logits, h = model(nxt, h)
    model.train()
    return prompt + "".join(out)


def run_gru(seed, steps=100, batch=64, chunk=16):
    text = make_corpus(seed)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    ids = torch.tensor([stoi[c] for c in text])
    n_tr = int(0.9 * ids.numel())
    train_ids, val_ids = ids[:n_tr], ids[n_tr:]
    data = streams(train_ids, batch)                # (64, 421)
    val = streams(val_ids, 8)

    # Unigram baseline: train character frequencies (add-one), cross-entropy on val.
    counts = torch.bincount(train_ids, minlength=len(chars)).float() + 1.0
    logp = (counts / counts.sum()).log()
    unigram = float(-logp[val_ids].mean())

    torch.manual_seed(seed)
    model = CharGRU(len(chars))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    sched = warmup_cosine(opt, steps)
    h, t = None, 0
    for _ in range(steps):
        if t + chunk + 1 > data.shape[1]:
            h, t = None, 0                          # wrap: start the streams again with a fresh state
        x, y = data[:, t : t + chunk], data[:, t + 1 : t + chunk + 1]
        if h is not None:
            h = h.detach()                          # truncated BPTT: keep the value, cut the graph
        logits, h = model(x, h)
        loss = F.cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        t += chunk
    gen = torch.Generator().manual_seed(seed)
    return {
        "val_loss": stream_loss(model, val, chunk),
        "unigram_loss": unigram,
        "train_last_loss": loss.item(),
        "vocab": len(chars),
        "corpus_chars": len(text),
        "params": count_params(model),
        "sample": sample(model, stoi, itos, "ada ", 100, 0.8, gen),
    }


# ------------------------------------------------------------------ part 4: transfer
SOURCE_CLASSES = (0, 1, 2)      # filled square, hollow square, disk
TARGET_CLASSES = (3, 2, 1, 0)   # plus is new; labels are permuted relative to the source
TARGET_NOISE = 0.25


def run_transfer(seed, **kw):
    with torch.backends.nnpack.flags(enabled=False):
        return _run_transfer(seed, **kw)


def _run_transfer(seed, source_steps=80, target_steps=40, n_target=100, batch=32,
                  target_noise=TARGET_NOISE, probe_lr=3e-2, base_lr=1e-2):
    s_tr, s_va, s_te = split_seeds(seed)
    # Source task: 3 classes, standard noise.
    xs, ys = make_shapes(2000, s_tr, classes=SOURCE_CLASSES)
    xs_va, ys_va = make_shapes(400, s_va, classes=SOURCE_CLASSES)
    torch.manual_seed(seed)
    source = SmallCNN(num_classes=len(SOURCE_CLASSES))
    train_cnn(source, xs, ys, source_steps, torch.Generator().manual_seed(seed + 11), batch=batch)
    source_acc = accuracy(source, xs_va, ys_va)
    backbone_state = {k: v.clone() for k, v in source.features.state_dict().items()}

    # Target task: 4 classes incl. the unseen plus, noisier, only n_target labels.
    xt, yt = make_shapes(n_target, s_tr + 50, classes=TARGET_CLASSES, noise=target_noise)
    xt_te, yt_te = make_shapes(400, s_te + 50, classes=TARGET_CLASSES, noise=target_noise)

    def pretrained():
        torch.manual_seed(seed + 1)
        m = SmallCNN(num_classes=4)       # fresh 4-way head = "replace the head"
        m.features.load_state_dict(backbone_state)
        return m

    # (a) linear probe: frozen backbone, BN running stats frozen too.
    probe = pretrained()
    probe.features.requires_grad_(False)
    trainable_probe = count_params(probe, trainable_only=True)
    train_cnn(probe, xt, yt, target_steps, torch.Generator().manual_seed(seed + 21),
              lr=probe_lr, batch=batch, bn_eval=True)
    probe_acc = accuracy(probe, xt_te, yt_te)
    frozen_ok = all(torch.equal(probe.features.state_dict()[k], v) for k, v in backbone_state.items())

    # (b) full fine-tune with discriminative learning rates.
    ft = pretrained()
    groups = [
        {"params": ft.features.parameters(), "lr": base_lr / 10},
        {"params": ft.head.parameters(), "lr": base_lr},
    ]
    trainable_ft = count_params(ft, trainable_only=True)
    train_cnn(ft, xt, yt, target_steps, torch.Generator().manual_seed(seed + 21),
              lr=base_lr, batch=batch, param_groups=groups)
    finetune_acc = accuracy(ft, xt_te, yt_te)

    # (c) from scratch on the same 100 images, same steps.
    torch.manual_seed(seed + 1)
    scratch = SmallCNN(num_classes=4)
    trainable_scratch = count_params(scratch, trainable_only=True)
    train_cnn(scratch, xt, yt, target_steps, torch.Generator().manual_seed(seed + 21),
              lr=base_lr, batch=batch)
    scratch_acc = accuracy(scratch, xt_te, yt_te)

    return {
        "source_acc": source_acc,
        "probe_acc": probe_acc,
        "finetune_acc": finetune_acc,
        "scratch_acc": scratch_acc,
        "backbone_unchanged_after_probe": frozen_ok,
        "trainable_params": {"probe": trainable_probe, "finetune": trainable_ft, "scratch": trainable_scratch},
        "head_params": count_params(probe.head),
    }


PARTS = {"mlp": run_mlp, "cnn": run_cnn, "gru": run_gru, "transfer": run_transfer}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--which", choices=[*PARTS, "all"], default="all")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    names = list(PARTS) if args.which == "all" else [args.which]
    result = {name: PARTS[name](args.seed) for name in names}
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
