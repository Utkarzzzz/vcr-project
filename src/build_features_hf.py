
import argparse

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from clip_features import ClipEncoder, pair_features


def build(split, limit, encoder):
    ds = load_dataset("pingzhili/vcr-qa", split=split, streaming=True)
    X, y, group = [], [], []
    for gi, row in enumerate(tqdm(ds, total=limit, desc=split)):
        if limit and gi >= limit:
            break
        img = encoder.encode_pil(row["image"])
        texts = [f'{row["question"]} {a}' for a in row["answer_choices"]]
        vecs = encoder.encode_texts(texts)
        for i, tv in enumerate(vecs):
            X.append(pair_features(img, tv))
            y.append(1 if i == row["answer_label"] else 0)
            group.append(gi)
    return {
        "X": np.array(X, dtype=np.float32),
        "y": np.array(y, dtype=np.int64),
        "group": np.array(group, dtype=np.int64),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["train", "validation"])
    p.add_argument("--limit", type=int, default=4000)
    p.add_argument("--out", required=True, help="prefix, e.g. data/hf_train")
    args = p.parse_args()

    encoder = ClipEncoder()
    d = build(args.split, args.limit, encoder)
    np.savez(args.out + "_answer.npz", **d)
    print(f"saved {args.out}_answer.npz  ({len(np.unique(d['group']))} samples)")


if __name__ == "__main__":
    main()
