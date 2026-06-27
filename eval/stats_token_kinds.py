"""Token-kind breakdown of generated traces: control seps / locals (state) / action (source) / value.
Run: python -m eval.stats_token_kinds --results <json> --model <ckpt-for-tokenizer>"""
import argparse
import json

from transformers import AutoTokenizer

from tokens import add_trace_tokens, token_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--model", required=True)  # tokenizer source
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    add_trace_tokens(tok)
    ids = token_ids(tok)
    eot = ids["<|end_of_text|>"]
    EVENT = {ids["<|call_sep|>"], ids["<|line_sep|>"]}  # locals follow these until action_sep
    ACT, ARG = ids["<|action_sep|>"], ids["<|arg_sep|>"]
    CTRL = {ids[t] for t in ("<|trace_context_start|>", "<|call_sep|>", "<|line_sep|>", "<|return_sep|>",
            "<|exception_sep|>", "<|action_sep|>", "<|arg_sep|>", "<|frame_sep|>",
            "<|latent_start|>", "<|latent_end|>")}
    rows = json.load(open(args.results))["results"]
    agg = {"control": 0, "locals": 0, "action": 0, "value": 0}
    total = frames = 0
    for r in rows:
        seq = tok(r["generation"], add_special_tokens=False)["input_ids"]
        if eot in seq:
            seq = seq[:seq.index(eot)]
        cur = None
        for t in seq:
            if t in CTRL:
                agg["control"] += 1
                frames += t in EVENT
                cur = t
            elif cur in EVENT:
                agg["locals"] += 1
            elif cur == ARG:
                agg["value"] += 1
            else:
                agg["action"] += 1
        total += len(seq)
    n = len(rows)
    print(f"{args.results}  n={n}  mean_tokens={total/n:.1f}  frames/ex={frames/n:.1f}")
    for k, v in agg.items():
        print(f"  {k:8s} {v/n:7.1f}/ex  {100*v/total:5.1f}%")


if __name__ == "__main__":
    main()
