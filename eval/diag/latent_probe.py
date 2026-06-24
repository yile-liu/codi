"""Diagnostic C: does the latent block encode the per-frame $LOCALS state?

Teacher-forced on CRUXEval, per frame: KD residual (smooth_l1 + cosine of the
student post-latent <|action_sep|> hidden vs the teacher's, who saw explicit
locals); logit-lens recovery (fraction of the dropped-locals tokens in the top-k
of lm_head applied to the latent hiddens); post_is_asep sanity. Single GPU.
"""

import argparse
import json

import torch
import torch.nn.functional as F

from data.dataset import build_codi_example
from data.sources import load_cruxeval
from eval.eval_cruxeval_codi import load_codi
from tokens import token_ids


@torch.no_grad()
def probe_example(model, prompt_ids, trace_ids, spans, asep_id, topk):
    dev = model.model.get_input_embeddings().weight.device
    body, head, emb, prj = model.body, model.head, model._emb, model.prj
    L = model.latent_steps
    prompt = torch.tensor(prompt_ids, device=dev)
    trace = torch.tensor(trace_ids, device=dev)

    full = torch.cat([prompt, trace])
    t_hs = model.model(full[None], use_cache=False, output_hidden_states=True).hidden_states[-1][0]
    t_anchor = [t_hs[len(prompt) + j] for _, j in spans]  # teacher hidden at each action_sep

    out = model.model(inputs_embeds=emb(prompt[None]), use_cache=True)
    cache = out.past_key_values
    segs, prev, kd = [], 0, False
    for i, j in spans:
        segs.append(("text", trace[prev:i + 1], kd)); segs.append(("latent", None, False))
        prev, kd = j, True
    segs.append(("text", trace[prev:], kd))

    recs, frame, pend = [], 0, None
    for kind, seg_ids, kd in segs:
        if kind == "latent":
            o = body(inputs_embeds=emb(model._ls_tok), past_key_values=cache, use_cache=True)
            cache, h = o.past_key_values, o.last_hidden_state[:, -1:]
            step_h = []
            for _ in range(L):
                o = body(inputs_embeds=prj(h), past_key_values=cache, use_cache=True)
                cache, h = o.past_key_values, o.last_hidden_state[:, -1:]
                step_h.append(h[0, 0])
            o = body(inputs_embeds=emb(model._le_tok), past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            pend = (step_h, o.last_hidden_state[0, -1])  # post-latent hidden predicts action_sep
            continue
        out = model.model(inputs_embeds=emb(seg_ids[None]), past_key_values=cache,
                          use_cache=True, output_hidden_states=kd)
        cache = out.past_key_values
        if kd:
            s_anchor = out.hidden_states[-1][0, 0]  # action_sep hidden (last layer)
            t = t_anchor[frame]
            i, j = spans[frame]
            loc = set(trace_ids[i + 1:j])
            step_h, post_h = pend
            preds = set()
            for hh in step_h + [post_h]:
                preds.update(head(hh).topk(topk).indices.tolist())
            recs.append(dict(
                frame=frame, nloc=len(loc),
                sl1=F.smooth_l1_loss(s_anchor.float(), t.float()).item(),
                cos=F.cosine_similarity(s_anchor[None].float(), t[None].float()).item(),
                rec=(len(loc & preds) / len(loc)) if loc else None,
                post_is_asep=int(int(head(post_h).argmax()) == asep_id),
            ))
            frame += 1
    return recs


def agg(rows, key, sel=lambda r: True):
    v = [r[key] for r in rows if sel(r) and r[key] is not None]
    return sum(v) / len(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--latent_steps", type=int, default=1)
    ap.add_argument("--n", type=int, default=150, help="num CRUXEval examples to probe")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max_seq_len", type=int, default=3072)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tok, ids, model = load_codi(args.model, args.latent_steps, 0)
    asep_id = token_ids(tok)["<|action_sep|>"]

    all_recs, used = [], 0
    for r in load_cruxeval():
        if used >= args.n:
            break
        ex = build_codi_example(r["code"], r["input"], tok, max_seq_len=args.max_seq_len)
        if ex is None:
            continue
        all_recs += probe_example(model, ex["prompt_ids"], ex["trace_ids"], ex["spans"], asep_id, args.topk)
        used += 1
    print(f"probed {used} examples, {len(all_recs)} frames")
    print(f"sanity post_is_asep = {agg(all_recs,'post_is_asep'):.3f}")
    print(f"\noverall  KD smooth_l1={agg(all_recs,'sl1'):.4f}  cos={agg(all_recs,'cos'):.4f}  "
          f"locals_recovery@{args.topk}={agg(all_recs,'rec'):.4f}")

    print(f"\n{'frame_idx':>10} {'n':>5} {'sl1':>8} {'cos':>8} {'rec':>8}")
    bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 10**9)]
    for lo, hi in bins:
        sub = [r for r in all_recs if lo <= r["frame"] < hi]
        if not sub:
            continue
        lbl = f"{lo}" if hi - lo == 1 else f"{lo}-{hi-1 if hi<10**9 else ''}"
        print(f"{lbl:>10} {len(sub):>5} {agg(sub,'sl1'):>8.4f} {agg(sub,'cos'):>8.4f} {agg(sub,'rec'):>8.4f}")

    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"n_examples": used, "n_frames": len(all_recs), "topk": args.topk,
                   "post_is_asep": agg(all_recs, "post_is_asep"),
                   "overall": {"sl1": agg(all_recs, "sl1"), "cos": agg(all_recs, "cos"),
                               "rec": agg(all_recs, "rec")}},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
