#!/usr/bin/env python3
"""Build the continued-pretraining corpus: dedupe, mix in replay, pack, tokenize.

Scope warning up front: HARDWARE.md revision 6 cut this phase hard. PLAN.md 4a
asked for 0.5-2B tokens, which is 8-30 days on a GTX 1650. The budget here is
**50M tokens** (~8 h) with 100M as a stretch, and if that does not clear the
section-5 perplexity gate the right move is to **skip CPT entirely** and spend the
time on SFT data. For closed-set classification it is the most cuttable phase.

Implements:
  * exact dedupe (sha1) and near-dupe removal (MinHash LSH over 5-gram shingles)
  * a replay slice from the base model's original distribution, to bound
    catastrophic forgetting (PLAN.md 4a: 5-15%)
  * held-out domain and replay eval slices, so the section-5 gate
    ("domain ppl -15%, replay ppl +<5%") is actually measurable
  * fixed-length packing with document boundaries marked by EOS
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def iter_texts(paths: list[Path]):
    for p in paths:
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            if p.suffix == ".jsonl":
                d = json.loads(line)
                t = d.get("text") or d.get("input") or ""
            else:
                t = line
            if t:
                yield t


def dedupe(texts, threshold: float, num_perm: int = 64):
    """Exact dedupe always; MinHash near-dedupe when `datasketch` is available."""
    seen_exact: set[str] = set()
    staged = []
    for t in texts:
        h = hashlib.sha1(t.encode()).hexdigest()
        if h in seen_exact:
            continue
        seen_exact.add(h)
        staged.append(t)
    n_exact = len(staged)

    if threshold >= 1.0:
        return staged, {"exact_removed_to": n_exact, "near_removed": 0}

    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("  datasketch missing; exact dedupe only", file=sys.stderr)
        return staged, {"exact_removed_to": n_exact, "near_removed": 0,
                        "near_dedupe": "skipped"}

    def shingles(t: str, k: int = 5):
        w = t.lower().split()
        return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept = []
    for i, t in enumerate(staged):
        m = MinHash(num_perm=num_perm)
        for s in shingles(t):
            m.update(s.encode())
        if lsh.query(m):
            continue
        lsh.insert(str(i), m)
        kept.append(t)
    return kept, {"exact_removed_to": n_exact, "near_removed": n_exact - len(kept)}


def load_replay(n: int, name: str, seed: int) -> list[str]:
    """A slice of general text from the base model's original distribution.

    PLAN.md 4a: 5-15% replay bounds catastrophic forgetting. Without it, CPT on a
    narrow support-ticket corpus degrades basic language competence, which shows up
    as the replay perplexity climbing in the section-5 gate.
    """
    if n <= 0:
        return []
    try:
        from datasets import load_dataset
        ds = load_dataset(name, split="train", streaming=True)
        key = None
        out = []
        for row in ds:
            if key is None:
                key = next((k for k in ("text", "content", "raw_content")
                            if k in row), None)
                if key is None:
                    break
            t = (row.get(key) or "").strip()
            if len(t) > 200:
                out.append(t)
            if len(out) >= n:
                break
        if out:
            return out
        print(f"  replay dataset {name} had no usable text field", file=sys.stderr)
    except Exception as e:
        print(f"  replay load failed ({type(e).__name__}: {str(e)[:80]})",
              file=sys.stderr)
    print("  !! proceeding with NO replay slice: expect catastrophic forgetting, "
          "and the section-5 replay-ppl gate cannot be evaluated", file=sys.stderr)
    return []


def pack(tokenizer, texts: list[str], seq_len: int):
    """Concatenate with EOS boundaries and cut into fixed-length blocks."""
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no eos_token_id; cannot mark boundaries")
    buf, blocks = [], []
    for t in texts:
        buf.extend(tokenizer.encode(t, add_special_tokens=False))
        buf.append(eos)
        while len(buf) >= seq_len:
            blocks.append(buf[:seq_len])
            buf = buf[seq_len:]
    return blocks  # trailing partial block dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", nargs="+", default=["workspace/data/sft/seed.jsonl"])
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--outdir", default="workspace/data/cpt")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--target-tokens", type=int, default=50_000_000,
                    help="HARDWARE.md rev 6 budget: 50M ~ 8h on a GTX 1650")
    ap.add_argument("--replay-frac", type=float, default=0.10)
    ap.add_argument("--replay-dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--near-dupe-threshold", type=float, default=0.8)
    ap.add_argument("--eval-tokens", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=20261008)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    rng = random.Random(args.seed)

    paths = [Path(p) for p in args.inp]
    print(f"reading {[str(p) for p in paths]}")
    raw = list(iter_texts(paths))
    print(f"  {len(raw):,} documents")

    kept, stats = dedupe(raw, args.near_dupe_threshold)
    print(f"  dedupe: {len(raw):,} -> {len(kept):,} ({stats})")

    n_replay = int(len(kept) * args.replay_frac / max(1e-9, 1 - args.replay_frac))
    print(f"  loading {n_replay:,} replay docs from {args.replay_dataset}")
    replay = load_replay(n_replay, args.replay_dataset, args.seed)
    print(f"  got {len(replay):,} replay docs "
          f"({len(replay)/max(1,len(kept)+len(replay)):.1%} of mix)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Hold out eval slices BEFORE mixing, so domain and replay ppl are separable.
    rng.shuffle(kept)
    rng.shuffle(replay)
    eval_docs = max(1, args.eval_tokens // max(1, args.seq_len) // 2)
    dom_eval, kept = kept[:eval_docs], kept[eval_docs:]
    rep_eval, replay = replay[:eval_docs], replay[eval_docs:]

    manifest = {"model": args.model, "seq_len": args.seq_len, "dedupe": stats}
    for name, docs in [("domain_eval", dom_eval), ("replay_eval", rep_eval)]:
        if not docs:
            continue
        blocks = pack(tok, docs, args.seq_len)
        p = outdir / f"{name}.jsonl"
        with p.open("w") as fh:
            for b in blocks:
                fh.write(json.dumps({"input_ids": b}) + "\n")
        manifest[name] = {"blocks": len(blocks), "tokens": len(blocks) * args.seq_len}
        print(f"  {name:<12} {len(blocks):,} blocks "
              f"({len(blocks)*args.seq_len:,} tokens) -> {p}")

    mix = kept + replay
    rng.shuffle(mix)
    blocks = pack(tok, mix, args.seq_len)
    target_blocks = args.target_tokens // args.seq_len
    if len(blocks) > target_blocks:
        blocks = blocks[:target_blocks]
    elif len(blocks) < target_blocks:
        print(f"  NOTE: corpus yields {len(blocks)*args.seq_len:,} tokens, under the "
              f"{args.target_tokens:,} target. Add real domain text; do NOT raise "
              f"epochs to compensate (that memorizes rather than adapts).")

    p = outdir / "train.jsonl"
    with p.open("w") as fh:
        for b in blocks:
            fh.write(json.dumps({"input_ids": b}) + "\n")
    toks = len(blocks) * args.seq_len
    manifest["train"] = {"blocks": len(blocks), "tokens": toks,
                         "replay_frac_actual": len(replay) / max(1, len(mix))}
    print(f"  train        {len(blocks):,} blocks ({toks:,} tokens) -> {p}")
    print(f"  est. wall clock at 1767 tok/s: {toks/1767/3600:.1f} h")

    (outdir / "MIXTURE.md").write_text(
        "# CPT mixture\n\n"
        f"- base model / tokenizer: `{args.model}`\n"
        f"- sources: {[str(x) for x in paths]}\n"
        f"- documents after dedupe: {len(kept):,} domain + {len(replay):,} replay\n"
        f"- replay dataset: `{args.replay_dataset}` "
        f"({manifest['train']['replay_frac_actual']:.1%} of the mix)\n"
        f"- sequence length: {args.seq_len}, packed with EOS document boundaries\n"
        f"- train tokens: {toks:,} (budget {args.target_tokens:,}, "
        f"HARDWARE.md rev 6)\n"
        f"- near-dupe: MinHash LSH, 5-gram shingles, "
        f"threshold {args.near_dupe_threshold}\n"
        f"- held-out: `domain_eval.jsonl`, `replay_eval.jsonl` — the section-5 gate "
        "is domain ppl -15% with replay ppl +<5%\n")
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {outdir}/MIXTURE.md and manifest.json")


if __name__ == "__main__":
    main()
