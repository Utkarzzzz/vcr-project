import argparse

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression


def load(prefix, stage):
    d = np.load(f"{prefix}_{stage}.npz")
    return d["X"], d["y"], d["group"]


def group_accuracy(clf, X, y, group):
    scores = clf.predict_proba(X)[:, 1]
    correct = 0
    total = 0
    picks = {}
    for gi in np.unique(group):
        mask = group == gi
        idx = np.where(mask)[0]
        best = idx[np.argmax(scores[idx])]
        picks[int(gi)] = int(np.where(idx == best)[0][0])
        if y[best] == 1:
            correct += 1
        total += 1
    return correct / total, picks


def train_stage(prefix, stage):
    X, y, group = load(prefix, stage)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X, y)
    acc, _ = group_accuracy(clf, X, y, group)
    print(f"[{stage}] train group-accuracy: {acc:.4f}")
    return clf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/train", help="feature prefix")
    p.add_argument("--out", default="models")
    p.add_argument("--stages", nargs="+", default=["answer", "rationale"],
                   choices=["answer", "rationale"])
    args = p.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)

    for stage in args.stages:
        clf = train_stage(args.train, stage)
        dump(clf, f"{args.out}/{stage}_clf.joblib")
        print(f"saved {args.out}/{stage}_clf.joblib")


if __name__ == "__main__":
    main()
