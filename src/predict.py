import argparse
import json
import os

import numpy as np
from joblib import load

from clip_features import ClipEncoder, pair_features
from vcr_data import load_split


def score(clf, feats):
    return clf.predict_proba(np.array(feats, dtype=np.float32))[:, 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, help="test split jsonl")
    p.add_argument("--images", default="data/vcr1images")
    p.add_argument("--models", default="models")
    p.add_argument("--out", default="predictions.json")
    p.add_argument("--limit", type=int, default=0,
                   help="only predict on the first N samples (0 = all)")
    args = p.parse_args()

    ans_clf = load(f"{args.models}/answer_clf.joblib")
    rat_clf = load(f"{args.models}/rationale_clf.joblib")
    enc = ClipEncoder()

    samples = load_split(args.jsonl, args.images)
    if args.limit:
        samples = samples[: args.limit]
    results = []
    for s in samples:
        if not os.path.exists(s["image_path"]):
            continue
        img = enc.encode_image(s["image_path"])

        # Stage 1: pick the answer
        a_texts = [f'{s["question"]} {a}' for a in s["answers"]]
        a_vecs = enc.encode_texts(a_texts)
        a_feats = [pair_features(img, tv) for tv in a_vecs]
        a_pred = int(np.argmax(score(ans_clf, a_feats)))

        # Stage 2: condition the rationale on our predicted answer
        chosen_answer = s["answers"][a_pred]
        r_texts = [f'{s["question"]} {chosen_answer} {r}' for r in s["rationales"]]
        r_vecs = enc.encode_texts(r_texts)
        r_feats = [pair_features(img, tv) for tv in r_vecs]
        r_pred = int(np.argmax(score(rat_clf, r_feats)))

        results.append(
            {
                "annot_id": s["annot_id"],
                "answer": a_pred,
                "rationale": r_pred,
            }
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {len(results)} predictions to {args.out}")


if __name__ == "__main__":
    main()
