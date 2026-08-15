
import json
import os
import random

from PIL import Image

random.seed(0)

ROOT = "data"
IMG_DIR = os.path.join(ROOT, "vcr1images", "dummy")
os.makedirs(IMG_DIR, exist_ok=True)

OBJECTS = ["person", "person", "car", "dog"]
QUESTIONS = [
    [["Why", "is", [0], "smiling", "?"]],
    [["What", "is", [2], "doing", "?"]],
    [["Where", "is", [1], "looking", "?"]],
]
PHRASES = [
    "they are happy", "they are tired", "it is parked", "it is moving",
    "at the sky", "at the ground", "because it is sunny", "because it is late",
]


def make_record(i, split):
    q = random.choice(QUESTIONS)[0]
    fn = f"dummy/img_{i}.jpg"
    color = tuple(random.randint(0, 255) for _ in range(3))
    Image.new("RGB", (224, 224), color).save(os.path.join(ROOT, "vcr1images", fn))

    answers = [[random.choice(PHRASES)] for _ in range(4)]
    rationales = [[random.choice(PHRASES)] for _ in range(4)]
    rec = {
        "annot_id": f"{split}-{i}",
        "objects": OBJECTS,
        "img_fn": fn,
        "question": q,
        "answer_choices": answers,
        "rationale_choices": rationales,
    }
    if split != "test":
        rec["answer_label"] = random.randint(0, 3)
        rec["rationale_label"] = random.randint(0, 3)
    return rec


for split, n in [("train", 40), ("val", 15), ("test", 10)]:
    with open(os.path.join(ROOT, f"{split}.jsonl"), "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps(make_record(i, split)) + "\n")
    print(f"wrote data/{split}.jsonl ({n} samples)")
