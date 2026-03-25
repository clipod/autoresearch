"""
Compare model outputs before and after training.
Builds a model, shows random-weight gibberish, trains for 5 min,
then shows the same prompts again so you can see the improvement.

Usage: uv run sample.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb
from train import (
    GPT, build_model_config,
    DEPTH, DEVICE_BATCH_SIZE, TOTAL_BATCH_SIZE,
    EMBEDDING_LR, UNEMBEDDING_LR, MATRIX_LR, SCALAR_LR,
    ADAM_BETAS, WEIGHT_DECAY, WARMUP_RATIO, WARMDOWN_RATIO, FINAL_LR_FRAC,
)

# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt="The", max_new_tokens=100, temperature=0.8, top_k=50):
    """Generate text token-by-token from the model."""
    model.eval()
    device = next(model.parameters()).device

    token_ids = tokenizer.encode(prompt)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        input_tokens = tokens[:, -MAX_SEQ_LEN:]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_tokens)

        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < topk_vals[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0].tolist())


def show_samples(model, tokenizer, label):
    """Generate from a few prompts and print results."""
    prompts = ["The meaning of life is", "Once upon a time", "In the beginning"]
    print(f"\n{'=' * 60}")
    print(label)
    print('=' * 60)
    for prompt in prompts:
        print(f"\n  Prompt: '{prompt}'")
        text = generate(model, tokenizer, prompt=prompt, max_new_tokens=100)
        # Show first 300 chars, replace newlines for readability
        preview = text[:300].replace('\n', ' ')
        print(f"  Output: {preview}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()

    config = build_model_config(DEPTH, vocab_size)
    print(f"Model config: {asdict(config)}")

    # Build and init model with random weights
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params / 1e6:.1f}M")

    # ---- BEFORE TRAINING ----
    show_samples(model, tokenizer, "BEFORE TRAINING (random weights — expect gibberish)")

    model.eval()
    with autocast_ctx:
        before_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
    print(f"\n  val_bpb BEFORE: {before_bpb:.4f}")

    # ---- TRAINING ----
    print(f"\n{'=' * 60}")
    print(f"TRAINING for {TIME_BUDGET}s...")
    print('=' * 60)

    model.train()
    optimizer = model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR, embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR, adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR, weight_decay=WEIGHT_DECAY,
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    model_compiled = torch.compile(model, dynamic=False)
    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)

    tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

    def get_lr_multiplier(progress):
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        elif progress < 1.0 - WARMDOWN_RATIO:
            return 1.0
        else:
            cooldown = (1.0 - progress) / WARMDOWN_RATIO
            return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

    def get_muon_momentum(step):
        frac = min(step / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    def get_weight_decay(progress):
        return WEIGHT_DECAY * max(progress, 0.1)

    total_training_time = 0.0
    smooth_train_loss = 0.0
    step = 0

    while True:
        torch.cuda.synchronize()
        t0 = time.time()
        model_compiled.train()

        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = model_compiled(x, targets=y)
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = get_muon_momentum(step)
                group["weight_decay"] = get_weight_decay(progress)
        optimizer.step()
        model_compiled.zero_grad(set_to_none=True)

        train_loss_f = loss.item() * grad_accum_steps
        if math.isnan(train_loss_f) or train_loss_f > 100:
            print("\nFAIL - loss exploded")
            exit(1)

        torch.cuda.synchronize()
        dt = time.time() - t0
        if step > 10:
            total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased = smooth_train_loss / (1 - ema_beta ** (step + 1))
        remaining = max(0, TIME_BUDGET - total_training_time)
        pct = 100 * progress

        print(f"\r  step {step:05d} ({pct:.1f}%) | loss: {debiased:.4f} | remaining: {remaining:.0f}s    ", end="", flush=True)

        if step == 0:
            gc.collect(); gc.freeze(); gc.disable()

        step += 1
        if step > 10 and total_training_time >= TIME_BUDGET:
            break

    print()

    # ---- AFTER TRAINING ----
    show_samples(model, tokenizer, "AFTER TRAINING (same prompts — should be more coherent)")

    model.eval()
    with autocast_ctx:
        after_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
    print(f"\n  val_bpb AFTER: {after_bpb:.4f}")

    # ---- COMPARISON ----
    improvement = before_bpb - after_bpb
    pct_improvement = 100 * improvement / before_bpb
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print('=' * 60)
    print(f"  Before training:  {before_bpb:.4f} bpb")
    print(f"  After training:   {after_bpb:.4f} bpb")
    print(f"  Improvement:      {improvement:.4f} bpb ({pct_improvement:.1f}%)")
    print(f"  Steps trained:    {step}")
    print(f"  Peak VRAM:        {peak_vram:.0f} MB")
