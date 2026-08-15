import argparse
import os

import numpy as np
from tqdm import tqdm

from clip_features import ClipEncoder, pair_features
from vcr_data import load_split


def build(samples, encoder):
    # We produce two feature tables:
    #   answer stage  -> 4 rows per sample (one per candidate answer)
    #   rationale stage -> 4 rows per sample (one per candidate rationale)
    # "group" tells us which rows belong to the same sample so we can pick
    # an argmax within each group at evaluation time.
    ans = {"X": [], "y": [], "group": []}
    rat = {"X": [], "y": [], "group": []}

    for gi, s in enumerate(tqdm(samples, desc="encoding")):
        if not os.path.exists(s["image_path"]):
            continue
        img = encoder.encode_image(s["image_path"])

        ans_texts = [f'{s["question"]} {a}' for a in s["answers"]]
        ans_vecs = encoder.encode_texts(ans_texts)
        for i, tv in enumerate(ans_vecs):
            ans["X"].append(pair_features(img, tv))
            ans["y"].append(1 if i == s["answer_label"] else 0)
            ans["group"].append(gi)

        # condition the rationale on the (gold) answer, as the task defines it
        gold = s["answers"][s["answer_label"]] if s["answer_label"] is not None else ""
        rat_texts = [f'{s["question"]} {gold} {r}' for r in s["rationales"]]
        rat_vecs = encoder.encode_texts(rat_texts)
        for j, tv in enumerate(rat_vecs):
            rat["X"].append(pair_features(img, tv))
            rat["y"].append(1 if j == s["rationale_label"] else 0)
            rat["group"].append(gi)

    return ans, rat


def to_npz(d):
    return {
        "X": np.array(d["X"], dtype=np.float32),
        "y": np.array(d["y"], dtype=np.int64),
        "group": np.array(d["group"], dtype=np.int64),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--images", default="data/vcr1images")
    p.add_argument("--out", required=True, help="prefix, e.g. data/val")
    p.add_argument("--limit", type=int, default=0,
                   help="only use the first N samples (0 = all)")
    args = p.parse_args()

    samples = load_split(args.jsonl, args.images)
    if args.limit:
        samples = samples[: args.limit]
    print(f"{len(samples)} samples from {args.jsonl}")

    encoder = ClipEncoder()
    ans, rat = build(samples, encoder)

    np.savez(args.out + "_answer.npz", **to_npz(ans))
    np.savez(args.out + "_rationale.npz", **to_npz(rat))
    print("saved", args.out + "_answer.npz", "and", args.out + "_rationale.npz")


if __name__ == "__main__":
    main()
