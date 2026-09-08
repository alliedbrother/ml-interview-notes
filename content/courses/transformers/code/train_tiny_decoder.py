"""Overfit two tiny documents, restore training state, and test contextual recall.

Run from the repository root:
  python content/courses/transformers/code/train_tiny_decoder.py
No downloads or checkpoint files are created. This is not a generalization test.
"""

import copy

import torch
import torch.nn.functional as F

from modern_decoder import Config, ModernDecoder


def main():
    torch.manual_seed(161)
    torch.set_num_threads(1)
    cfg = Config(vocab_size=8, d_model=32, n_layers=2, n_heads=4,
                 n_kv_heads=2, d_head=8, d_ff=48, n_experts=0,
                 max_seq_len=32)
    model = ModernDecoder(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    # Common token 5 must predict 3 or 4 according to the preceding context.
    documents = torch.tensor([[1, 3, 5, 3, 5, 3, 5, 2],
                              [1, 4, 5, 4, 5, 4, 5, 2]])
    inputs, targets = documents[:, :-1], documents[:, 1:]

    def loss_for(net):
        logits, _ = net(inputs)
        return F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))

    initial = loss_for(model).item()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    final = loss_for(model).item()
    assert final < initial * 0.4

    snapshot = copy.deepcopy({"model": model.state_dict(),
                              "optimizer": optimizer.state_dict(),
                              "rng": torch.get_rng_state()})
    restored = ModernDecoder(cfg)
    restored.load_state_dict(snapshot["model"])
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_optimizer.load_state_dict(snapshot["optimizer"])
    torch.set_rng_state(snapshot["rng"])
    torch.testing.assert_close(model(inputs)[0], restored(inputs)[0])

    # Matching one subsequent update verifies optimizer state, not just logits.
    for net, opt in ((model, optimizer), (restored, restored_optimizer)):
        opt.zero_grad(set_to_none=True)
        loss_for(net).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    for left, right in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right)

    prefixes = documents[:, :3]
    generated = restored.generate(prefixes, max_new_tokens=3, temperature=0)
    assert generated[0, 3].item() == 3 and generated[1, 3].item() == 4
    print(f"Training loss: {initial:.4f} -> {final:.4f}")
    print("Context-dependent continuations:", generated.tolist())
    print("Restored model and optimizer reproduce the next update.")
    print("This demonstrates memorization and state contracts, not held-out quality.")


if __name__ == "__main__":
    main()
