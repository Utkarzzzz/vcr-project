import argparse

import numpy as np
from joblib import load


def load_stage(prefix, stage):
    d = np.load(f"{prefix}_{stage}.npz")
    return d["X"], d["y"], d["group"]


def predict_choices(clf, X, group):
    scores = clf.predict_proba(X)[:, 1]
    chosen = {}   # group -> local index chosen
    correct = {}  # group -> local index of the correct candidate
    for gi in np.unique(group):
        idx = np.where(group == gi)[0]
        chosen[int(gi)] = int(np.argmax(scores[idx]))
    return chosen, scores


def correct_index(y, group):
    right = {}
    for gi in np.unique(group):
        idx = np.where(group == gi)[0]
        right[int(gi)] = int(np.argmax(y[idx]))
    return right


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/val", help="feature prefix")
    p.add_argument("--models", default="models")
    args = p.parse_args()

    ans_clf = load(f"{args.models}/answer_clf.joblib")
    rat_clf = load(f"{args.models}/rationale_clf.joblib")

    aX, ay, ag = load_stage(args.features, "answer")
    rX, ry, rg = load_stage(args.features, "rationale")

    a_pred, _ = predict_choices(ans_clf, aX, ag)
    r_pred, _ = predict_choices(rat_clf, rX, rg)
    a_true = correct_index(ay, ag)
    r_true = correct_index(ry, rg)

    groups = sorted(a_true.keys())
    q2a = np.mean([a_pred[g] == a_true[g] for g in groups])
    qa2r = np.mean([r_pred[g] == r_true[g] for g in groups])
    q2ar = np.mean(
        [(a_pred[g] == a_true[g]) and (r_pred[g] == r_true[g]) for g in groups]
    )

    print(f"Q -> A   accuracy: {q2a:.4f}")
    print(f"QA -> R  accuracy: {qa2r:.4f}")
    print(f"Q -> AR  accuracy: {q2ar:.4f}   <- primary metric")


if __name__ == "__main__":
    main()
