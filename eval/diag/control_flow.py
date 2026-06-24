"""Diagnostic D: control-flow faithfulness of the latent student (offline).

The student emits each executed source line explicitly (only $LOCALS is latent),
so a generation reveals its control-flow path. Compare that path to ground truth:
wrong answer + matching path = value bottleneck; wrong answer + diverged path =
latent lost branch-driving state. Reads an eval_cruxeval_codi --out JSON, no GPU.
"""

import argparse
import json

from data.ground_truth import ground_truth_trace
from data.sources import load_cruxeval
from data.trace_format import parse_generated_trace


def _path(frames):  # (event, whitespace-collapsed source) per frame
    return [(f.event, " ".join(f.source.split())) for f in frames]


def first_divergence(gen, gt):
    for i, g in enumerate(gt):
        if i >= len(gen) or gen[i] != g:
            return i
    return len(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--max_frames", type=int, default=5000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    res = json.load(open(args.results))["results"]
    by_id = {str(r["id"]): r for r in load_cruxeval()}

    n = skipped = cm_c = cm_w = dv_c = dv_w = 0
    fracs = []
    for r in res:
        row = by_id.get(str(r["id"]))
        gt_frames = ground_truth_trace(row["code"], row["input"], max_frames=args.max_frames)[0] if row else None
        if not gt_frames:
            skipped += 1
            continue
        gt, gen = _path(gt_frames), _path(parse_generated_trace(r["generation"])[0])
        d = first_divergence(gen, gt)
        n += 1
        correct = bool(r["correct"])
        if d == len(gt) and len(gen) >= len(gt):
            cm_c += correct; cm_w += not correct
        else:
            dv_c += correct; dv_w += not correct
            fracs.append(d / max(1, len(gt)))

    cm, dv, wrong = cm_c + cm_w, dv_c + dv_w, cm_w + dv_w
    print(f"n={n}  (skipped {skipped})")
    print(f"control-flow MATCHES: {cm} ({cm/n:.3f})  correct={cm_c} wrong={cm_w}")
    print(f"control-flow DIVERGES: {dv} ({dv/n:.3f})  correct={dv_c} wrong={dv_w}"
          f"  [correct+diverged = source-text artifact floor]")
    if fracs:
        fracs.sort()
        print(f"first-divergence frame fraction: median={fracs[len(fracs)//2]:.2f}")
    if wrong:
        print(f"among {wrong} WRONG: {cm_w/wrong:.3f} correct-path/wrong-value, {dv_w/wrong:.3f} diverged-path")

    if args.out:
        json.dump({"n": n, "skipped": skipped, "cf_match": [cm_c, cm_w], "diverge": [dv_c, dv_w]},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
