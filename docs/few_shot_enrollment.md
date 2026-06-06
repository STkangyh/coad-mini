# Few-shot Enrollment of New Action Classes

Register a brand-new action class from a handful of examples, growing the model
from **48 → 48+K** classes **without catastrophically forgetting** the original
48. This directly showcases the project's A-GEM continual-learning core.

## Components

`src/enroll/few_shot.py`

- **`expand_classifier(model, n_new)`** — grows the final `cls` Linear from
  `num_classes` to `num_classes + n_new`. Old weight/bias rows are copied
  **bit-exactly**; new rows are small-init (`N(0, 0.01)`, bias `0`). Updates
  `model.num_classes`. So immediately after expansion the model behaves almost
  identically to before (zero structural forgetting).
- **`FewShotEnroller`** — wraps a loaded `GRUDetector` and holds a replay buffer
  of old-class `(window_tensor, label)` exemplars (reservoir sampling).
  `enroll(new_examples, new_labels, epochs, lr, replay_per_step)` trains the
  expanded model on the new class while interleaving/replaying old exemplars.
  With `use_agem=True` (default) it additionally applies **A-GEM gradient
  projection** (the exact formula from `src/utils/gem.py`): if the new-class
  gradient conflicts with the replay gradient (`dot(g_cur, g_ref) < 0`), it is
  projected to be non-conflicting:

  ```
  g_new = g_cur - (g_cur·g_ref / g_ref·g_ref) * g_ref
  ```

  Everything operates on **in-memory** feature windows of shape `(16, 512)` — no
  files required.
- **`retention_accuracy(model, eval_set, device)`** — top-1 accuracy helper over
  `[(window, label), ...]`.

## Demo

```bash
python3 experiments/demo_few_shot.py
```

Loads `checkpoints/agem_48cls.pt`, expands to 49 classes, enrolls a synthetic
new class, and prints old-class retention + new-class accuracy. Runs with no
external data.

## FastAPI endpoint

`POST /enroll` (multipart):

- `label` — new class name (form field)
- `files` — one or more example videos

It extracts CLIP features via the existing `video_to_feature` helper, expands the
A-GEM model to 49 classes, enrolls with replay, and writes an updated checkpoint
`checkpoints/agem_enrolled.pt` plus labels `checkpoints/class_labels_enrolled.json`.
Full end-to-end use needs `ffmpeg` + real videos; the endpoint logic itself is
covered by synthetic-input tests.

## Tests

```bash
python3 -m pytest tests/test_few_shot_enroll.py -q
```

Covers: structural growth + bit-exact weight preservation; a self-contained
4-class synthetic continual scenario showing small forgetting + new-class
learning, and that replay forgets no more than naive fine-tuning; and an
integration test on the real `agem_48cls.pt`.
