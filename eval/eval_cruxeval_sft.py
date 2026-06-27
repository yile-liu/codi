"""Stage 1 baseline eval: CRUXEval-O output prediction via full-trace generation.

Feed the training prompt (seeds frame 0), let the SFT model generate the trace,
take main()'s last return value as the predicted output, score by execution.
Greedy => pass@1 is the exact-match fraction. Reuses cwm_andre eval logic.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.dataset import _prompt_str
from data.sources import load_cruxeval
from tokens import add_trace_tokens, token_ids

ARG_SEP, FRAME_SEP, RETURN_SEP = "<|arg_sep|>", "<|frame_sep|>", "<|return_sep|>"


def extract_answer_trace_full(gen: str) -> str | None:
    """Value of main()'s last RETURN frame: ...<|arg_sep|>"value"<|frame_sep|>."""
    r = gen.rfind(RETURN_SEP)
    if r == -1:
        return None
    a = gen.find(ARG_SEP, r)
    if a == -1:
        return None
    rest = gen[a + len(ARG_SEP):]
    end = rest.find(FRAME_SEP)
    val = (rest[:end] if end != -1 else rest).strip()
    if not val:
        return None
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        return val


def check_correct(code: str, expected: str, predicted: str, timeout: float = 3.0) -> bool:
    """Execute `code; assert expected == predicted` (CRUXEval semantics)."""
    test = f"{code}\nassert {expected} == {predicted}"
    try:
        return subprocess.run(
            [sys.executable, "-c", test], timeout=timeout, capture_output=True
        ).returncode == 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_samples", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--latent_steps", type=int, default=1)  # unused; for shared launcher
    args = ap.parse_args()

    # DDP-style data parallelism for inference: torchrun sets RANK/WORLD_SIZE/LOCAL_RANK.
    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        dist.init_process_group("nccl", timeout=timedelta(hours=1))  # ranks finish at different times under long gens
    torch.cuda.set_device(local_rank)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    add_trace_tokens(tok)  # idempotent; ensures trace tokens present
    tok.padding_side = "left"  # left-pad so all generated tokens start at the same offset
    eot_id = token_ids(tok)["<|end_of_text|>"]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(local_rank).eval()

    rows = load_cruxeval()
    if args.n_samples > 0:
        rows = rows[: args.n_samples]
    n = len(rows)
    shard = rows[rank::world]  # disjoint round-robin split across ranks

    n_correct = n_fmt = 0
    results = []
    for bi, batch_start in enumerate(range(0, len(shard), args.batch_size)):
        batch = shard[batch_start: batch_start + args.batch_size]
        enc = tok([_prompt_str(r["code"], r["input"]) for r in batch],
                  return_tensors="pt", padding=True, add_special_tokens=False).to(local_rank)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 eos_token_id=eot_id, pad_token_id=eot_id)
        for j, r in enumerate(batch):
            gen_ids = out[j, enc["input_ids"].shape[1]:].tolist()
            n_gen = gen_ids.index(eot_id) if eot_id in gen_ids else len(gen_ids)  # n_fwd==n_gen (no latent)
            gen = tok.decode(gen_ids, skip_special_tokens=False)
            pred = extract_answer_trace_full(gen)
            ok = pred is not None and check_correct(r["code"], r["output"], pred)
            n_fmt += pred is not None
            n_correct += ok
            results.append({"id": r["id"], "expected": r["output"], "predicted": pred, "correct": ok,
                            "n_gen": n_gen, "n_fwd": n_gen, "generation": gen})
        if rank == 0 and (bi + 1) % 5 == 0:
            done = batch_start + len(batch)
            print(f"  rank0 {done}/{len(shard)}  pass@1={n_correct/done:.4f}", flush=True)

    # Reduce metrics and gather per-row results across ranks.
    if ddp:
        t = torch.tensor([n_correct, n_fmt], device=local_rank)
        dist.all_reduce(t)
        n_correct, n_fmt = int(t[0]), int(t[1])
        gathered = [None] * world
        dist.gather_object(results, gathered if rank == 0 else None, dst=0)
        if rank == 0:
            results = [x for part in gathered for x in part]

    if rank == 0:
        mean_fwd = sum(r["n_fwd"] for r in results) / n
        mean_gen = sum(r["n_gen"] for r in results) / n
        print(f"\nCRUXEval-O pass@1={n_correct / n:.4f}  valid_format={n_fmt / n:.4f}  "
              f"(n={n}, greedy)  mean_fwd={mean_fwd:.1f} mean_gen={mean_gen:.1f}")
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"pass_at_1": n_correct / n, "valid_format": n_fmt / n, "n": n,
                           "mean_fwd": mean_fwd, "mean_gen": mean_gen, "results": results}, f, indent=2)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
