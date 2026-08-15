import json
import os


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tag_objects(objects):
    # VCR object references are indices into the "objects" list.
    # We turn index -> a readable label like "person1", "car2" so the
    # question/answers read naturally after detokenizing.
    seen = {}
    tags = []
    for name in objects:
        seen[name] = seen.get(name, 0) + 1
        tags.append(f"{name}{seen[name]}")
    return tags


def detokenize(tokens, obj_tags):
    # A token is either a plain string ("Why") or a list of object
    # indices ([2] or [0, 1]). Flatten the whole thing into one string.
    out = []
    for tok in tokens:
        if isinstance(tok, list):
            out.append(" and ".join(obj_tags[i] for i in tok))
        else:
            out.append(tok)
    text = " ".join(out)
    # tidy up spaces before punctuation
    for p in [" ?", " .", " ,", " !", " '"]:
        text = text.replace(p, p.strip())
    return text


def read_sample(row):
    obj_tags = tag_objects(row["objects"])
    question = detokenize(row["question"], obj_tags)
    answers = [detokenize(a, obj_tags) for a in row["answer_choices"]]
    rationales = [detokenize(r, obj_tags) for r in row["rationale_choices"]]

    sample = {
        "annot_id": row.get("annot_id", ""),
        "img_fn": row["img_fn"],
        "question": question,
        "answers": answers,
        "rationales": rationales,
        # labels are missing on the test split
        "answer_label": row.get("answer_label"),
        "rationale_label": row.get("rationale_label"),
    }
    return sample


def load_split(jsonl_path, image_root):
    rows = load_jsonl(jsonl_path)
    samples = []
    for row in rows:
        s = read_sample(row)
        s["image_path"] = os.path.join(image_root, s["img_fn"])
        samples.append(s)
    return samples


if __name__ == "__main__":
    # quick sanity check on whatever data you point it at
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/val.jsonl"
    data = load_split(path, "data/vcr1images")
    print(f"loaded {len(data)} samples")
    s = data[0]
    print("Q:", s["question"])
    for i, a in enumerate(s["answers"]):
        mark = "*" if i == s["answer_label"] else " "
        print(f"  {mark} answer {i}: {a}")
