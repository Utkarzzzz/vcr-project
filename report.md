# Visual Commonsense Reasoning — Joint Answer + Rationale Selection

**Author:** Utkarsh Anand
**Task:** 2Y Recruitment Round 2 — VCR (Q→A, QA→R, Q→AR)

> Fill every `[[ ... ]]` placeholder after you run the pipeline on the real VCR
> data. Delete this quote block before submitting.

---

## 1. Problem framing

Each VCR sample gives an image, a question, four candidate answers, and four
candidate rationales. The task is two multiple-choice decisions:

- **Stage 1 (Q→A):** pick the answer `A* = argmax_i P(A_i | I, Q)`.
- **Stage 2 (QA→R):** pick the rationale `R* = argmax_j P(R_j | I, Q, A*)`.

I treat both stages identically: score each of the four candidates and take the
argmax. The primary metric, Q→AR, requires *both* choices to be correct.

## 2. Model architecture

The system has two parts — a frozen encoder and a light trainable head.

**Encoder — CLIP (`openai/clip-vit-base-patch32`).** CLIP maps images and text
into a shared 512-dimensional space where a matching image and caption have high
cosine similarity. I use it purely as a feature extractor; its weights are never
updated. For each candidate I build a text string (`question + candidate`, plus
the chosen answer for the rationale stage) and encode it; the image is encoded
once and reused across all four candidates.

**Feature vector.** For an image vector `v` and a candidate-text vector `t`
(both L2-normalized) I form:

```
[ v , t , v * t , cos(v, t) ]     ->  512 + 512 + 512 + 1 = 1537 dims
```

The element-wise product and cosine similarity give a linear model an explicit
"do these match?" signal.

**Head — logistic regression (scikit-learn).** A binary classifier scores each
candidate as correct / incorrect. At inference I take the candidate with the
highest predicted probability within a sample. I train two separate heads, one
per stage. `class_weight="balanced"` compensates for the 1-correct-of-4 ratio.

```
image ─► CLIP image encoder ─┐
                             ├─► [v, t, v*t, cos] ─► LogisticRegression ─► argmax
"Q + candidate" ─► CLIP text ┘
```

## 3. Training strategy

- **Data prep.** VCR stores questions/answers as token lists with object
  references (e.g. `[2]`). I detokenize these into plain sentences, mapping each
  object index to a readable tag like `person1`.
- **Feature caching.** I run CLIP over every split once and cache the vectors to
  `.npz`, so training/evaluation is fast and repeatable (no repeated CLIP passes).
- **Stage 1.** Four rows per sample (one per answer), label 1 for the gold answer.
- **Stage 2.** Four rows per sample (one per rationale), text conditioned on the
  **gold** answer during training; at test time it is conditioned on the
  **predicted** answer, matching `P(R | I, Q, A*)`.
- **Optimizer.** scikit-learn `LogisticRegression`, `C=1.0`, `max_iter=2000`,
  L2 regularization. No GPU required; the whole thing trains on CPU.

## 4. Results (validation set)

> Run `python src/evaluate.py --features data/val` on the real data and paste
> the numbers here.

| Metric | Accuracy | Random baseline |
|--------|----------|-----------------|
| Q→A    | `[[ .. ]]` | 25% |
| QA→R   | `[[ .. ]]` | 25% |
| **Q→AR (primary)** | `[[ .. ]]` | ~6.25% |

Validation samples used: `[[ N ]]`. Encoder: CLIP ViT-B/32. Hardware: `[[ CPU/GPU ]]`.

## 5. Key design choices & trade-offs

- **Frozen CLIP + linear head over fine-tuning a VLM.** It runs on a laptop CPU,
  trains in seconds, and every part is inspectable — appropriate for the time
  budget and easy to defend. The cost is a lower accuracy ceiling than a
  fine-tuned LLaVA/Qwen model.
- **Cosine + element-wise product features.** These give the linear head the
  matching signal it needs without a deep network.
- **Two-stage conditioning.** Stage 2 uses the predicted answer at test time, so
  errors can propagate — this is inherent to the joint Q→AR metric.
- **Known limitations.** CLIP's text encoder truncates at 77 tokens, so very
  long rationales lose their tail; and CLIP has no explicit reasoning over the
  object boxes.

## 6. Possible improvements

- Larger encoder (`clip-vit-large-patch14`).
- A small 2-layer MLP head instead of logistic regression.
- Use the object bounding boxes / tags instead of discarding them.
- Ensemble the answer and rationale scores jointly rather than in a strict cascade.
