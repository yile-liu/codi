"""Diagnostic B: explicit-trace ("teacher") pass@1 of a CODI checkpoint.

Loads the CODI wrapper (eval_cruxeval_sft can't: state_dict has a model./prj.
prefix) and generates the explicit trace on the base LM, like the SFT eval.
Comparing to the latent eval (same ckpt) and standalone SFT (0.576) isolates
whether the gap is the latent path or a degraded co-trained teacher.
torchrun-compatible; run via eval.sbatch SCRIPT=eval.diag.teacher_eval.
"""

import argparse
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from data.dataset import _prompt_str
from data.sources import load_cruxeval
from eval.eval_cruxeval_codi import load_codi
from eval.eval_cruxeval_sft import check_correct, extract_answer_trace_full
from tokens import token_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_samples", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--latent_steps", type=int, default=1)  # for load_codi parity
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        dist.init_process_group("nccl", timeout=timedelta(hours=1))
    torch.cuda.set_device(local_rank)

    tok, ids, codi = load_codi(args.model, args.latent_steps, local_rank)
    model = codi.model  # base LM = co-trained teacher weights
    tok.padding_side = "left"
    eot_id = token_ids(tok)["<|end_of_text|>"]

    rows = load_cruxeval()
    if args.n_samples > 0:
        rows = rows[: args.n_samples]
    n = len(rows)
    shard = rows[rank::world]

    n_correct = n_fmt = 0
    results = []
    for bi, bs in enumerate(range(0, len(shard), args.batch_size)):
        batch = shard[bs: bs + args.batch_size]
        enc = tok([_prompt_str(r["code"], r["input"]) for r in batch],
                  return_tensors="pt", padding=True, add_special_tokens=False).to(local_rank)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 eos_token_id=eot_id, pad_token_id=eot_id)
        for j, r in enumerate(batch):
            gen = tok.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=False)
            pred = extract_answer_trace_full(gen)
            ok = pred is not None and check_correct(r["code"], r["output"], pred)
            n_fmt += pred is not None
            n_correct += ok
            results.append({"id": r["id"], "expected": r["output"], "predicted": pred, "correct": ok, "generation": gen})
        if rank == 0 and (bi + 1) % 5 == 0:
            print(f"  rank0 {bs+len(batch)}/{len(shard)}  pass@1={n_correct/(bs+len(batch)):.4f}", flush=True)

    if ddp:
        t = torch.tensor([n_correct, n_fmt], device=local_rank)
        dist.all_reduce(t)
        n_correct, n_fmt = int(t[0]), int(t[1])
        gathered = [None] * world
        dist.gather_object(results, gathered if rank == 0 else None, dst=0)
        if rank == 0:
            results = [x for part in gathered for x in part]

    if rank == 0:
        print(f"\nTEACHER (explicit-trace) pass@1={n_correct/n:.4f}  valid_format={n_fmt/n:.4f}  (n={n})")
        if args.out:
            json.dump({"pass_at_1": n_correct / n, "valid_format": n_fmt / n, "n": n, "results": results},
                      open(args.out, "w"), indent=2)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
