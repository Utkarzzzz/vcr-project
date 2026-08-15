# Visual Commonsense Reasoning — beginner build

A small, honest baseline for the VCR task (answer selection + rationale
selection). It leans on a pretrained CLIP model for the "seeing and reading"
part and a plain logistic-regression classifier for the "choosing" part — so
if you already know scikit-learn, most of this will feel familiar.

## The idea in one paragraph

Both stages are multiple choice: 4 candidates, pick one. CLIP turns the image
and each candidate sentence into vectors that live in the same space, so a good
image–text match has a high cosine similarity. We hand those vectors (plus their
similarity) to a logistic-regression model that learns to score each candidate,
and we pick the highest score. Stage 2 (rationale) works the same way, but the
text includes the chosen answer, matching the `P(R | I, Q, A*)` setup.

```
image ─► CLIP image encoder ─┐
                             ├─► feature row ─► LogisticRegression ─► score per candidate ─► argmax
"Q + candidate" ─► CLIP text ┘
```

## Files

| file | what it does |
|------|--------------|
| `src/vcr_data.py` | reads the VCR jsonl and turns token lists into plain sentences |
| `src/clip_features.py` | wraps CLIP; turns an image / texts into normalized vectors |
| `src/build_features.py` | encodes a whole split once and caches it to `.npz` |
| `src/train.py` | trains the two logistic-regression classifiers |
| `src/evaluate.py` | reports Q→A, QA→R and Q→AR accuracy on the val set |
| `src/predict.py` | full two-stage inference on the (unlabeled) test set |
| `src/make_dummy_data.py` | fake data so you can run everything before the real dataset arrives |

## Run it today (fake data, ~2 min on CPU)

```bash
pip install -r requirements.txt

python src/make_dummy_data.py
python src/build_features.py --jsonl data/train.jsonl --out data/train
python src/build_features.py --jsonl data/val.jsonl   --out data/val
python src/train.py    --train data/train
python src/evaluate.py --features data/val
python src/predict.py  --jsonl data/test.jsonl --out predictions.json
```

Accuracy on the fake data is random — this run only proves the plumbing works.

## Run it for real

1. Get the dataset from https://visualcommonsense.com/download/ (registration
   required). You need the annotation jsonl files (`train.jsonl`, `val.jsonl`,
   `test.jsonl`) and the `vcr1images` image folder.
2. Drop them into `data/` so you have `data/train.jsonl` and
   `data/vcr1images/...`.
3. Run the same commands above but point `--jsonl` at the real files. The
   feature-building step is the slow part (it runs CLIP on every image); a GPU
   helps but CPU works for a subset. To start small, keep the first few thousand
   lines of `train.jsonl` while you learn.

The real val set should land somewhere around 55–65% Q→A with this baseline —
well above the 25% random floor, and a solid, defensible starting point.

## Where to take it next (for extra marks)

- Swap `openai/clip-vit-base-patch32` for `openai/clip-vit-large-patch14`.
- Replace `LogisticRegression` with a small PyTorch MLP (2 layers).
- Use the object tags/bounding boxes instead of throwing them away.
- Try a stronger model (BLIP-2, LLaVA) for the encoder — same overall shape.
