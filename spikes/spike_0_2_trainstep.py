#!/usr/bin/env python3
"""Spike 0.2 — real training-step throughput and peak VRAM on the GTX 1650.

Matmul microbenchmarks said fp16 is ~8x slower than fp32 on TU117. This checks
whether that holds for an actual transformer train step, and measures the two
memory optimizations that matter most at 4GB:

  * gradient checkpointing
  * slicing the LM head to the loss positions only (`logits_to_keep`)

The second one is the big one for JEV specifically: decision SFT computes loss on
exactly ONE token, so materializing [batch, seq, vocab] logits is pure waste.

Output feeds train/configs/*.yaml. Run with the workspace venv.
"""

import argparse
import json
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoConfig, AutoModelForCausalLM

p = argparse.ArgumentParser()
p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
p.add_argument("--seq", type=int, default=192)
p.add_argument("--steps", type=int, default=6)
p.add_argument("--out", default="spikes/RESULTS_trainstep.json")
args = p.parse_args()

DEV = "cuda"


def make_model(dtype, grad_ckpt):
    m = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="sdpa"
    ).to(DEV)
    if grad_ckpt:
        m.gradient_checkpointing_enable()
        m.config.use_cache = False
    m.train()
    return m


def step_fn(model, opt, ids, labels, scaler, sliced):
    """One fwd+bwd+step. `sliced` computes the LM head on the last position only."""
    opt.zero_grad(set_to_none=True)
    if sliced:
        # hidden states for the whole sequence, logits for the final position only
        out = model.model(input_ids=ids)
        h = out.last_hidden_state[:, -1:, :]
        logits = model.lm_head(h)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), labels[:, -1]
        )
    else:
        loss = model(input_ids=ids, labels=labels).loss
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        opt.step()
    return loss


def bench(label, dtype, grad_ckpt, batch, sliced, amp=False):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = make_model(torch.float32 if amp else dtype, grad_ckpt)
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)
            optname = "AdamW8bit"
        except Exception:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=False)
            optname = "AdamW(fp32)"

        ids = torch.randint(0, model.config.vocab_size, (batch, args.seq), device=DEV)
        labels = ids.clone()
        scaler = torch.amp.GradScaler("cuda") if amp and dtype == torch.float16 else None

        ctx = (torch.amp.autocast("cuda", dtype=dtype) if amp
               else torch.autocast("cuda", enabled=False))

        for _ in range(2):  # warmup
            with ctx:
                step_fn(model, opt, ids, labels, scaler, sliced)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(args.steps):
            with ctx:
                step_fn(model, opt, ids, labels, scaler, sliced)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t

        toks = batch * args.seq * args.steps
        peak = torch.cuda.max_memory_allocated() / 2**30
        res = {
            "label": label, "ok": True, "opt": optname,
            "tok_per_s": round(toks / dt, 1),
            "step_ms": round(dt / args.steps * 1000, 1),
            "peak_gib": round(peak, 2),
            "batch": batch, "seq": args.seq,
            "grad_ckpt": grad_ckpt, "sliced_lm_head": sliced,
        }
        del model, opt, ids, labels
        torch.cuda.empty_cache()
        return res
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"label": label, "ok": False, "err": "OOM", "batch": batch,
                "grad_ckpt": grad_ckpt, "sliced_lm_head": sliced}
    except Exception as e:
        torch.cuda.empty_cache()
        return {"label": label, "ok": False, "err": f"{type(e).__name__}: {str(e)[:80]}",
                "batch": batch, "grad_ckpt": grad_ckpt, "sliced_lm_head": sliced}


cfg = AutoConfig.from_pretrained(args.model)
nparam = sum(p.numel() for p in
             AutoModelForCausalLM.from_config(cfg).parameters()) / 1e6
free, total = torch.cuda.mem_get_info()
print(f"model={args.model}  params={nparam:.0f}M  vocab={cfg.vocab_size}  "
      f"hidden={cfg.hidden_size}  layers={cfg.num_hidden_layers}")
print(f"VRAM free {free/2**30:.2f} GiB / {total/2**30:.2f} GiB   seq={args.seq}\n")

GRID = [
    # dtype story: is fp32 really the fastest on this card for a real step?
    ("fp32  ckpt sliced  b8", torch.float32, True, 8, True, False),
    ("bf16  ckpt sliced  b8", torch.bfloat16, True, 8, True, False),
    ("fp16amp ckpt sliced b8", torch.float16, True, 8, True, True),
    # does the sliced LM head matter?
    ("fp32  ckpt FULL    b8", torch.float32, True, 8, False, False),
    # does grad checkpointing cost throughput, and buy how much memory?
    ("fp32  NOckpt sliced b8", torch.float32, False, 8, True, False),
    # how far can batch go with the good config?
    ("fp32  ckpt sliced  b16", torch.float32, True, 16, True, False),
    ("fp32  ckpt sliced  b32", torch.float32, True, 32, True, False),
]

rows = []
print(f"{'config':<24} {'tok/s':>8} {'step ms':>8} {'peak GiB':>9}  opt")
print("-" * 68)
for label, dtype, gc, bs, sliced, amp in GRID:
    r = bench(label, dtype, gc, bs, sliced, amp)
    rows.append(r)
    if r["ok"]:
        print(f"{label:<24} {r['tok_per_s']:>8.0f} {r['step_ms']:>8.1f} "
              f"{r['peak_gib']:>9.2f}  {r['opt']}")
    else:
        print(f"{label:<24} {'-':>8} {'-':>8} {'-':>9}  {r['err']}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump({"model": args.model, "params_m": round(nparam, 1), "seq": args.seq,
           "vram_total_gib": round(total / 2**30, 2), "rows": rows},
          open(args.out, "w"), indent=2)

ok = [r for r in rows if r["ok"]]
if ok:
    best = max(ok, key=lambda r: r["tok_per_s"])
    print(f"\nfastest: {best['label']}  {best['tok_per_s']:.0f} tok/s  "
          f"peak {best['peak_gib']:.2f} GiB")
    tps = best["tok_per_s"]
    for n, d in [("SFT 100k x 128tok", 12.8e6), ("CPT 25M tok", 25e6),
                 ("CPT 100M tok", 100e6)]:
        print(f"  {n:<20} -> {d / tps / 3600:6.1f} h at this rate")
print(f"\nwrote {args.out}")
