# Codebase Explanation — MobileCLIP Phishing Classifier

This document explains every folder, file, and key design decision in plain language. It is written as a teaching guide — you do not need prior experience with computer vision or transformers to follow it.

---

## Part 1 — The Big Picture

### What problem are we solving?

When someone visits a suspicious URL, we take a screenshot of the website and ask: **does this page look like a phishing page?**

We are not reading the URL, the page source, or any text. We are looking at the screenshot like a human would — does the login form look fake? Is the logo slightly off? Does the layout feel wrong?

### Why screenshots instead of HTML?

- Phishing pages constantly change their HTML to evade text-based detection
- Attackers register new domains hourly
- But the visual design of a phishing page — the fake PayPal login, the urgency banner, the off-brand colours — stays consistent because they are copying templates
- A model trained on visual patterns generalises better than one memorising URLs or HTML structure

### What is MobileCLIP2?

CLIP stands for **Contrastive Language-Image Pretraining**. It is a model trained by OpenAI (and later improved by many labs) on hundreds of millions of image-text pairs from the internet. During training, CLIP learns to match images with their text descriptions. The side effect is that the image encoder learns extremely rich visual features — it understands objects, scenes, layouts, colours, and styles.

MobileCLIP is Apple's efficient version of CLIP. Instead of a giant ViT-L/14 backbone, it uses a compact hybrid architecture (CNN + lightweight transformer) that runs fast on mobile devices.

We use **MobileCLIP2-S2** — the second generation, pretrained on **DFN-2B** (a high-quality filtered dataset of 2 billion image-text pairs, vs the 1 billion used for v1). Same ~36M parameter count as v1 but meaningfully stronger visual features because of the better training data. The pretrained tag in OpenCLIP is `dfndr2b`.

We take MobileCLIP2's image encoder (the "visual backbone") and add a small classification head on top. This is called **transfer learning** — we start with a model that already understands the visual world, and we fine-tune it to recognise the specific patterns of phishing pages.

### Why False Negatives are worse than False Positives

In security, the cost of errors is asymmetric:
- **False Positive** (FP): We flag a legitimate page as phishing. The user is mildly inconvenienced. They can appeal, or try again.
- **False Negative** (FN): We let a phishing page through as legitimate. The user's credentials are stolen. Potentially irreversible.

This asymmetry shapes every design decision in this codebase: the loss function, the sampler, the threshold choice, the evaluation metric.

---

## Part 2 — Frameworks Used

### PyTorch

PyTorch is the deep learning framework. It handles:
- **Tensors**: multi-dimensional arrays that live on CPU or GPU
- **Autograd**: automatic computation of gradients (needed for training)
- **nn.Module**: the base class for all neural network layers
- **DataLoader**: feeds data to the model in batches during training

Every model, loss function, and transform in this codebase is a PyTorch object.

### OpenCLIP

OpenCLIP (`open_clip_torch`) is an open-source implementation of CLIP models maintained by LAION. It provides pretrained MobileCLIP checkpoints. We use it to load the pretrained image encoder:

```python
import open_clip
model, _, _ = open_clip.create_model_and_transforms('MobileCLIP2-S2', pretrained='dfndr2b')
encoder = model.visual   # we only want the image encoder, not the text encoder
```

The pretrained tag `dfndr2b` means the model was trained on DFN-2B, a dataset of 2 billion filtered image-text pairs (higher quality than the original DataComp-1B used for MobileCLIP v1).

### torchvision

torchvision provides image transforms (resize, crop, colour jitter, etc.) and DataLoader utilities. We use it for all preprocessing in `datasets/transforms.py`.

### scikit-learn

Used only for computing validation metrics (F1, F2, recall, ROC-AUC) during training and evaluation. It provides optimised implementations of these metrics that handle edge cases (e.g. division by zero when a class is missing from a batch).

### PyYAML

Reads `configs/config.yaml` into a Python dictionary. Every hyperparameter is in the YAML file — nothing is hardcoded in the training scripts.

---

## Part 3 — Folder Structure, File by File

---

### `configs/config.yaml`

**What it is:** The single source of truth for all hyperparameters.

**Why one file:** If hyperparameters are scattered across multiple scripts, experiments become unreproducible. By centralising everything in one YAML file, you can version-control your experiment configs, share them with teammates, and reproduce any experiment exactly.

**Key sections:**

```yaml
model:
  backbone: "MobileCLIP2-S2"  # which OpenCLIP model to use
  pretrained: "dfndr2b"       # which pretrained checkpoint (DFN-2B dataset)
  freeze_backbone_epochs: 5   # how long to keep backbone frozen (Phase 1)
  embedding_dim: 512          # size of the feature vector output
```

```yaml
training:
  optimizer:
    lr_head: 1.0e-3           # learning rate for Phase 1 (head only)
    lr_head_phase2: 1.0e-4    # learning rate for the head in Phase 2
    lr_backbone: 1.0e-5       # learning rate for backbone in Phase 2
```

The learning rate difference between backbone (1e-5) and head (1e-4) is intentional. We want to gently nudge the pretrained backbone features, not overwrite them. This is called **differential learning rates** — a standard technique in transfer learning.

---

### `datasets/`

This package is responsible for loading images from disk and preparing them for the model.

---

#### `datasets/phishing_dataset.py`

**Core class: `PhishingDataset`**

Inherits from `torch.utils.data.Dataset`. PyTorch's `DataLoader` expects a `Dataset` object with two methods:
- `__len__()`: how many samples are in the dataset
- `__getitem__(idx)`: return the sample at position `idx`

**How image discovery works:**

```python
for path in sorted(class_dir.rglob("*")):
    if path.suffix.lower() in cls.VALID_EXTENSIONS:
        brand = path.relative_to(class_dir).parts[0]
        records.append(ImageRecord(...))
```

`rglob("*")` recursively finds every file under the directory. We check the file extension, extract the brand name (the immediate subfolder), and create an `ImageRecord` — a simple data container holding the path, label (0 or 1), class name, and brand.

**Why `ImageRecord` has a `brand` field:**

The brand field is critical for the split strategy. If `amazon/homepage.png` goes into training, `amazon/signup.png` must also go into training — not validation. If they split, the model could learn "this is Amazon's colour scheme" from training and get easy points in validation just by recognising the brand, not the phishing patterns. This is called **data leakage**.

**How `__getitem__` works:**

```python
def __getitem__(self, idx):
    record = self.records[idx]
    image = Image.open(record.path).convert("RGB")  # load as RGB
    image = self.transform(image)                    # resize, crop, augment
    return image, torch.tensor(record.label)
```

`.convert("RGB")` is important. Screenshots can be:
- RGBA (with transparency channel for PNG) → convert removes alpha
- Grayscale → convert adds 3 identical channels
- Palette mode (indexed colour) → convert renders to true colour

The model expects exactly 3 channels.

**Class: `DatasetSplitter`**

Splits the dataset into train (70%), validation (15%), test (15%) with brand-awareness for legitimate images.

For legitimate images: brands are shuffled with a fixed seed, then greedily assigned to train until 70% is reached, then val until 15%, rest goes to test. Whole brand groups stay together.

For phishing images (all in one `generic/` folder, no brand structure): files are shuffled and split at the file level.

**`compute_class_weights()`:**

```python
weight[c] = total / (n_classes * count[c])
```

For our dataset:
- `weight[legitimate] = 3485 / (2 × 2935) = 0.594`
- `weight[phishing]   = 3485 / (2 × 550)  = 3.168`

These weights are passed to the Focal Loss as `alpha` values. Every phishing error is penalised 5.3× harder than a legitimate error.

---

#### `datasets/transforms.py`

**What are transforms?**

A transform is a function that takes a PIL image and returns a modified version. The model expects a normalised tensor, so every image must pass through the same preprocessing pipeline.

**Training transforms (augmentation):**

```
RandomResizedCrop(256, scale=(0.85, 1.0))
    ^-- Randomly zoom in slightly and crop to 256×256.
        scale=(0.85, 1.0) means we keep 85-100% of the image.
        This simulates different viewport sizes and screenshot crops.
        We keep the scale tight to preserve layout structure.

ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.0)
    ^-- Randomly adjust brightness and contrast.
        Simulates different monitor calibrations and OS themes.
        hue=0.0 intentionally — shifting brand colours (PayPal blue,
        Google red) would confuse the model.

RandomGrayscale(p=0.03)
    ^-- 3% chance of converting to grayscale.
        Handles screenshots captured without colour (accessibility tools,
        headless browsers in certain configs).

RandomApply([GaussianBlur(...)], p=0.15)
    ^-- 15% chance of mild blur.
        Simulates JPEG compression, low-DPI captures.

ToTensor()
    ^-- Converts PIL image (H×W×3, uint8 0-255) to
        PyTorch tensor (3×H×W, float32 0.0-1.0).

Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ^-- Shifts pixel values to match the distribution the
        MobileCLIP weights were trained on. This is critical —
        if you use the wrong normalization, pretrained features
        won't transfer correctly.

RandomErasing(p=0.10, scale=(0.01, 0.08))
    ^-- 10% chance of erasing a small random rectangle.
        Simulates partial page loads, cookie banners, watermarks.
        Applied AFTER normalization.
```

**Why no horizontal flip?**

Website layouts are directional. Navigation bars, logos, and form labels are not symmetric. Flipping a PayPal login page horizontally would make it look wrong to a human and should look wrong to the model. Including flip would teach the model that flipped pages are valid, hurting its ability to detect visual anomalies.

**Validation transforms:**

```
Resize(292)           # resize shortest side to 292
CenterCrop(256)       # crop centre 256×256
ToTensor()
Normalize(...)
```

No augmentation. The validation and test transforms must be deterministic so that metrics are comparable across runs.

---

### `models/`

---

#### `models/backbone.py`

**What the backbone does:**

Takes a batch of images `[B, 3, 256, 256]` and produces embeddings `[B, 512]`. These embeddings are dense numerical representations that capture the visual semantics of each image. Similar-looking pages will have similar embeddings (close in 512-dimensional space).

**L2 normalisation:**

```python
features = F.normalize(features, dim=-1, p=2)
```

This scales every embedding to unit length (norm = 1.0). It sits on the surface of a 512-dimensional sphere. This is done for two reasons:
1. The head and future ensemble can use cosine similarity directly
2. It stabilises training — no embedding can grow unboundedly large

**The freeze/unfreeze mechanism:**

```python
def freeze(self):
    for param in self.encoder.parameters():
        param.requires_grad = False

def unfreeze(self):
    for param in self.encoder.parameters():
        param.requires_grad = True
```

`requires_grad=False` tells PyTorch not to compute gradients for those parameters during backpropagation. This means:
- The backbone weights do not change in Phase 1
- Backpropagation is faster (no need to store the autograd graph through the backbone)
- The head can learn stable features without the backbone shifting underneath it

**Weight decay split:**

```python
NO_DECAY_PATTERNS = ("bias", "norm", "ln", "bn", "layernorm", ...)
```

In AdamW, weight decay is a regularisation term that pushes all weights toward zero. But bias terms and normalisation layer parameters should NOT be pushed toward zero — they serve specific purposes (shifting distributions) and pushing them to zero breaks that. This is a well-established practice from the original BERT and GPT papers.

**`_load_encoder()` — why we discard the text encoder:**

```python
full_model, _, _ = open_clip.create_model_and_transforms(...)
encoder = full_model.visual   # only the image encoder
# full_model (including text encoder) is garbage-collected
```

CLIP has two encoders: image and text. We only need the image encoder. Loading and immediately discarding the text encoder wastes RAM during the first load but saves ~50% memory during training.

---

#### `models/classifier.py`

**The classification head:**

```
LayerNorm(512)         <- normalise the embedding per-sample
Linear(512 → 256)      <- compress to a smaller space
GELU()                 <- smooth activation (better than ReLU for transformers)
Dropout(0.3)           <- randomly zero 30% of neurons during training (regularisation)
Linear(256 → 2)        <- output: one score per class (raw logits)
```

**Why two linear layers instead of one?**

A single `Linear(512 → 2)` is a linear probe. It can only draw straight lines through the embedding space. Two layers with a non-linearity (GELU) in between can learn curved boundaries — important when the embedding space has non-linear structure. However, we keep it shallow (just two layers) to avoid overfitting on our small dataset.

**Why `Dropout(0.3)`?**

During training, at each forward pass, 30% of the 256 hidden units are randomly set to zero. This forces the network to not rely on any single neuron — every neuron must learn redundant, robust representations. At inference time, dropout is disabled (`model.eval()`).

**The forward output contract:**

```python
return {
    "logits":     logits,      # [B, 2] — raw scores, used for loss
    "probs":      probs,       # [B, 2] — after softmax, used for decisions
    "embeddings": embeddings,  # [B, 512] — backbone features, for ensemble
}
```

Always returning `embeddings` is the contract with the future ensemble model. The ensemble repository can load this model and call `model.backbone.extract_features(images)` to get the visual feature vector without running the head.

---

#### `models/focal_loss.py`

**What is loss?**

During training, the model makes a prediction and we compare it to the true label. The difference is the "loss" — a single number. We use the loss to compute gradients (via backpropagation) and update the model weights to reduce the loss.

**Standard Cross-Entropy:**

```
CE(p_t) = -log(p_t)
```

where `p_t` is the model's probability for the true class. If the model predicts 90% correctly, the loss is `-log(0.9) = 0.105`. If it predicts 10% correctly, the loss is `-log(0.1) = 2.3`.

Problem: on an imbalanced dataset, the model can minimise CE by simply predicting "legitimate" for everything (correct 84% of the time). The loss on rare phishing examples is overwhelmed by the volume of easy legitimate examples.

**Focal Loss:**

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

Two additions:
- `alpha_t`: a class weight. For phishing, alpha = 3.168 — every phishing error is penalised 3.168× harder.
- `(1 - p_t)^gamma`: the focusing term. When the model is very confident (`p_t` near 1), `(1 - p_t)^gamma` approaches zero — the loss contribution is tiny. When the model is wrong and uncertain (`p_t` near 0), `(1 - p_t)^gamma` approaches 1 — the loss contribution is large.

With `gamma=2`:
- Easy correct prediction (p_t = 0.9): weight = (1-0.9)^2 = 0.01 — this example barely contributes to training
- Hard wrong prediction (p_t = 0.1): weight = (1-0.1)^2 = 0.81 — this example drives most of the learning

This is exactly what we want: force the model to focus on the hard phishing examples that look legitimate.

**`gamma=0` reduces to weighted cross-entropy.** This is verified in the smoke tests.

---

### `training/`

---

#### `training/sampler.py`

**What is `WeightedRandomSampler`?**

Normally, a DataLoader picks random samples from the dataset. With 5.34:1 imbalance, each mini-batch would have ~5 legitimate images for every 1 phishing image. The model sees so few phishing examples per batch that it never learns to distinguish them.

`WeightedRandomSampler` assigns each sample a sampling probability:
```
weight[i] = 1.0 / count[class_of_sample_i]
```

For our dataset:
- Each legitimate image: weight = 1/2935 = 0.000341
- Each phishing image:   weight = 1/550  = 0.001818

Phishing images are sampled 5.34× more frequently. Each training batch now has roughly equal legitimate and phishing samples — even though the original dataset is heavily imbalanced.

**Why both sampler AND focal loss?**

The sampler balances at the batch level (equal numbers of each class per batch). The focal loss still penalises phishing errors harder at the gradient level. These are complementary protections: even if a batch happens to have slightly more legitimate images, the loss ensures phishing errors still dominate the gradient signal.

---

#### `training/callbacks.py`

**What is a callback?**

A callback is an object that receives information at the end of each epoch and decides whether something should happen (save a checkpoint, stop training, log the LR). Callbacks are separated from the Trainer so they can be tested independently and swapped without changing training logic.

**`EarlyStopping`:**

Tracks whether `val_f2` has improved. If it hasn't improved by at least `min_delta=1e-4` for `patience=10` consecutive epochs, it tells the Trainer to stop.

Why `val_f2` instead of `val_loss`?
- Loss can decrease while recall drops (the model becomes more confident about legitimate pages, reducing loss, but misses more phishing)
- F2 score directly measures what matters: catching phishing

Why F2 instead of F1?
- F1 = (2 × Precision × Recall) / (Precision + Recall) — equally weights both
- F2 = (5 × Precision × Recall) / (4 × Precision + Recall) — weights recall 4× more than precision
- F2 declines faster when recall drops, so early stopping protects recall more aggressively

**`ModelCheckpoint`:**

Decides when to save `best_model.pt`. It compares the current epoch's `val_f2` to the historical best. Only saves when there is a genuine improvement. This means `best_model.pt` always contains the weights from the epoch that performed best on the validation set.

---

#### `training/trainer.py`

**The two-phase strategy:**

```
Phase 1 (epochs 1-5):
  backbone frozen → no gradients flow through the 35.8M backbone params
  only the 0.1M head params are updated
  why: if we update everything from the start, the randomly-initialised
       head generates chaotic gradients that corrupt the pretrained features

Phase 2 (epochs 6-50):
  backbone unfrozen → gradients flow everywhere
  backbone LR = 1e-5 (100x smaller than Phase 1 head LR)
  head LR = 1e-4 (10x smaller than Phase 1, already adapted)
  why: small backbone LR gently adjusts pretrained features without
       destroying them; head LR can be a bit larger since it needs
       to adapt to the new gradient signal from the unfrozen backbone
```

**Mixed Precision (AMP):**

On CUDA GPUs, calculations can be done in float16 instead of float32. float16 uses half the memory and runs ~2-3× faster on modern GPUs (which have dedicated tensor cores for float16). The `autocast` context manager automatically converts eligible operations to float16. The `GradScaler` prevents float16 gradients from underflowing to zero.

On CPU, AMP is disabled because CPU doesn't have float16 hardware — it would actually be slower.

**Gradient clipping:**

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Before the optimizer step, all gradients are rescaled so that their combined norm is at most 1.0. This prevents "gradient explosions" — rare situations where a batch causes extremely large gradients that would push the model weights to infinity. Particularly important in the first few batches of Phase 2 when the backbone first starts receiving gradients.

**The warmup + cosine scheduler:**

```
Phase 2 schedule:
  Epochs 6-7 (warmup): LR rises linearly from 0 to full LR
  Epochs 8-50 (cosine): LR decays as a half-cosine curve from full LR to 1e-7
```

Warmup prevents large updates immediately after the backbone unfreezes. Cosine decay provides a smooth, gradual reduction in LR — the model makes smaller and smaller adjustments as it converges.

**EMA (Exponential Moving Average) — `_EMATracker`:**

During training, model weights bounce around as the optimizer takes steps — each batch pushes the weights in a slightly different direction based on whatever samples happened to be in that batch. The final weights (at the last step of training) can be at a local "noisy" point rather than a smooth one.

EMA keeps a running average of all weight values over the entire training history:

```
shadow_weight = decay × shadow_weight + (1 - decay) × current_weight
```

With `decay=0.999`, the shadow weight moves very slowly — it effectively represents the average of the last ~1000 optimizer steps (the last 6-7 epochs). This smooth average lands in a flatter, more stable region of the loss landscape and generalises better to unseen data.

**How it is used in this codebase:**

1. After every `optimizer.step()`, `self._ema.update(self.model)` updates all shadow weights.
2. When validation runs, `with self._ema.apply(self.model):` temporarily swaps EMA weights into the model. The validation loop runs with EMA weights. At the end of the `with` block, the original training weights are automatically restored.
3. When `best_model.pt` is saved, `self._ema.ema_state_dict(self.model)` builds a state dict with EMA weights substituted for all trainable parameters. This is what inference uses.
4. `last_checkpoint.pt` always saves the live training weights plus the EMA shadow dict, so training can be resumed correctly.

**Why validation uses EMA weights:**

The val_f2 that early stopping watches should reflect the EMA model (the one that goes to production), not the instantaneous training model. Without EMA, early stopping might save a checkpoint that looks good in training but is actually at a local "noisy" peak. With EMA, what you see in val is what you get in production.

**The `apply()` context manager:**

```python
@contextlib.contextmanager
def apply(self, model):
    # 1. Save current training weights
    backup = {name: param.data.clone() for name, param in model.named_parameters()}
    # 2. Load EMA weights
    for name, param in model.named_parameters():
        param.data.copy_(self.shadow[name])
    try:
        yield   # validation runs here
    finally:
        # 3. Always restore training weights, even if validation crashes
        for name, param in model.named_parameters():
            param.data.copy_(backup[name])
```

The `try/finally` ensures training weights are ALWAYS restored, even if an exception occurs inside the validation loop. This is important — if restoration failed, the optimizer would be updating EMA weights during the next training epoch, which would corrupt everything.

**To disable EMA:**

Set `training.ema_decay: 0` in `configs/config.yaml`. The Trainer checks `if self._ema is not None` at every point, so EMA cleanly drops out with no other changes.

---

### `evaluation/`

---

#### `evaluation/metrics.py`

**The confusion matrix (with phishing as positive):**

```
              Predicted
              Legit  Phishing
Actual Legit    TN     FP
Actual Phishing FN     TP
```

- **TP** (True Positive): Phishing page correctly detected
- **TN** (True Negative): Legitimate page correctly passed
- **FP** (False Positive): Legitimate page incorrectly flagged (annoying but harmless)
- **FN** (False Negative): Phishing page that slips through (dangerous)

**Key derived metrics:**

```
Precision    = TP / (TP + FP)    -- of all pages flagged, how many are actually phishing?
Recall       = TP / (TP + FN)    -- of all actual phishing pages, how many did we catch?
FNR          = FN / (FN + TP)    -- what fraction of phishing pages did we miss?
Specificity  = TN / (TN + FP)    -- of all legitimate pages, how many did we correctly pass?

F1  = 2 * P * R / (P + R)        -- harmonic mean of precision and recall
F2  = 5 * P * R / (4P + R)       -- weights recall 4x over precision

MCC (Matthews Correlation Coefficient):
     = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
     -- single metric ranging from -1 (worst) to +1 (perfect)
     -- more informative than accuracy on imbalanced datasets
```

**ROC-AUC vs PR-AUC:**

- ROC-AUC: area under the curve of TPR vs FPR at different thresholds. Good for balanced datasets. A random classifier scores 0.5.
- PR-AUC: area under the curve of Precision vs Recall. More informative for imbalanced datasets because it does not credit the model for correctly identifying the majority class.

For our 5.34:1 imbalanced dataset, PR-AUC is the more meaningful of the two.

---

#### `evaluation/threshold.py`

**Why 0.5 is wrong:**

The softmax output is not a calibrated probability. The model was not trained to output exactly 0.5 for cases it is uncertain about. A model trained with weighted focal loss tends to produce outputs shifted away from 0.5 for the minority class.

The threshold sweep finds the threshold `t` such that classifying as phishing when `P(phishing) >= t` gives the best performance for the chosen strategy.

**The three operating points:**

```
Max F1 threshold:
  Finds t that maximises F1.
  Use this for general-purpose deployment where precision and recall
  are equally important.

Max F2 threshold (recommended):
  Finds t that maximises F2.
  Since F2 weights recall 4x over precision, this threshold is
  lower (more aggressive) than the F1 threshold.
  Catches more phishing at the cost of more false alarms.

Min FNR threshold:
  Finds the HIGHEST threshold where FNR <= 5%.
  "Give me the most selective classifier that still catches 95% of phishing."
  This maximises precision while meeting a recall guarantee.
  Use in maximum-security scenarios.
```

**The vectorised sweep:**

```python
preds = (phishing_probs[np.newaxis, :] >= thresholds[:, np.newaxis]).astype(int)
```

Instead of looping over 181 thresholds one by one, this single line creates a `[181, N]` boolean matrix: `preds[i, j]` is 1 if sample j would be classified as phishing at threshold i. Then we compute all confusion matrix values with numpy vectorised operations in microseconds.

---

#### `evaluation/evaluator.py`

**The 9-step pipeline:**

1. Run inference over the full test set (no augmentation, no shuffling)
2. Compute all metrics at threshold=0.5 (baseline)
3. Run threshold sweep — find F1-opt, F2-opt, MinFNR operating points
4. Compute all metrics at the recommended threshold
5. Write `classification_report.txt` (sklearn format, per-class P/R/F1)
6. Write `predictions.json` (per sample: path, true label, predicted label, probability)
7. Extract misclassifications — false negatives sorted by highest P(phishing) first
8. Generate diagnostic plots (confusion matrix, ROC, PR, threshold sweep)
9. Write `evaluation_report.json` (the recommended threshold is stored here)

**Why false negatives are sorted by P(phishing) descending:**

The most instructive errors are the ones where the model is most confident in the wrong direction. A phishing page with P(phishing) = 0.48 (just below the threshold) is a borderline case — expected to be hard. A phishing page with P(phishing) = 0.12 is catastrophic — the model completely failed on it. These highest-confidence errors reveal the model's blind spots and should be prioritised for analysis.

---

### `utils/`

---

#### `utils/seed.py`

**Why seeding matters:**

Neural network training involves randomness at many points:
- Weight initialisation
- Data shuffling and sampling
- Dropout masks
- Data augmentation

Without a fixed seed, two identical training runs can produce different models. By calling `seed_everything(42)` at the start, we make experiments reproducible: same config, same data, same result every time.

```python
random.seed(seed)              # Python's random module
np.random.seed(seed)           # NumPy
torch.manual_seed(seed)        # PyTorch CPU
torch.cuda.manual_seed_all(seed)  # PyTorch GPU
torch.backends.cudnn.deterministic = True   # disable non-deterministic CUDA ops
torch.backends.cudnn.benchmark = False      # disable auto-tuner (also non-deterministic)
```

#### `utils/device.py`

**Auto device resolution:**

```python
if torch.cuda.is_available():
    resolved = torch.device("cuda")
elif torch.backends.mps.is_available():
    resolved = torch.device("mps")   # Apple Silicon
else:
    resolved = torch.device("cpu")
```

`device: "auto"` in config means "use the best available hardware without changing any code."

#### `utils/logging_utils.py`

**Two-handler logging:**

```
Console handler (INFO+):  shown in terminal during training
File handler   (DEBUG+):  everything written to outputs/logs/training.log
```

All loggers in the project are children of `"phishing_clip"`:
```
phishing_clip                      <- root
phishing_clip.datasets.phishing_dataset
phishing_clip.models.backbone
phishing_clip.training.trainer
...
```

Changing the level on `phishing_clip` instantly controls all child loggers.

#### `utils/io_utils.py`

**Checkpoint format design:**

The checkpoint is a dictionary saved with `torch.save()`:
```python
{
    "epoch":            17,         <- so we know where to resume from
    "model_state":      {...},      <- all model weights
    "optimizer_state":  {...},      <- optimizer momentum buffers
    "scheduler_state":  {...},      <- LR schedule position
    "scaler_state":     {...},      <- AMP scaler state
    "metrics":          {...},      <- val metrics at this epoch
    "phase":            2,          <- were we in phase 1 or 2?
    "early_stopping":   {...},      <- counter and best metric
    "model_checkpoint": {...},      <- best metric seen so far
}
```

Saving optimizer state is critical for proper resumption. The Adam/AdamW optimizer stores momentum statistics (running averages of gradients) for every parameter. Without these, resuming training is like restarting from scratch for the optimizer — the first few epochs after resumption will have wrong gradient estimates.

---

### `visualization/`

#### `visualization/gradcam.py`

**What is GradCAM?**

Gradient-weighted Class Activation Mapping is a technique for visualising which regions of an image the model focused on when making its prediction.

**How it works:**
1. Register a hook on the last convolutional layer to capture its activation maps
2. Do a forward pass to get the prediction
3. Do a backward pass to compute gradients of the phishing score with respect to the activation maps
4. Global-average-pool the gradients to get a weight per feature map: `alpha_k = mean(grad_k)`
5. Compute the weighted sum: `L = ReLU(sum_k alpha_k * A_k)`
6. Resize `L` to the original image size
7. Overlay as a heatmap on the original image

Regions that light up bright red caused the model to predict "phishing". This tells you whether the model is attending to the right things (a suspicious login form) or spurious things (background colour).

#### `visualization/plot_utils.py`

Generates:
- **Confusion matrix**: 2×2 grid with TP/TN/FP/FN counts and percentages
- **ROC curve**: TPR vs FPR at all thresholds, with AUC
- **PR curve**: Precision vs Recall at all thresholds, with AUC
- **Threshold sweep**: All three operating points visualised against the full sweep
- **Training history**: Loss, accuracy, F2, and LR curves across epochs

---

### `scripts/`

---

#### `scripts/train.py`

The main entry point. Does five things in order:
1. Parse arguments and load `config.yaml`
2. Apply any `--override` patches
3. Set up logging and seed
4. Build all components (dataset, model, trainer config)
5. Call `trainer.train()`

**`build_dataloaders()`:**

```python
sampler = build_weighted_sampler(train_rec)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, ...)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,   ...)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False,   ...)
```

Note: `sampler` and `shuffle=True` are mutually exclusive in PyTorch. When you provide a sampler, PyTorch uses the sampler's ordering instead of random shuffle.

`num_workers=0` on Windows: DataLoader uses multiple processes to load data in parallel (one worker per CPU core). On Windows, creating these processes uses the "spawn" method which is slower and requires more careful setup. For simplicity, `num_workers=0` means the main process loads all data. On Linux/Mac with a GPU, set `num_workers=4` for a significant speedup.

#### `scripts/evaluate.py`

Runs evaluation on the held-out test set. The test set is never seen during training — not used for early stopping, not used for model selection. It is a clean estimate of real-world performance.

Critically, `_build_test_loader()` reproduces the **exact same split** as training by using the same seed and fractions. Without this, the test set would contain data the model was trained on, giving falsely optimistic results.

#### `scripts/infer.py`

**Threshold resolution priority chain:**

```
--threshold argument
       ↓ (if not provided)
outputs/predictions/evaluation_report.json → "recommended_threshold"
       ↓ (if file not found)
configs/config.yaml → evaluation.threshold (default 0.5)
```

After you run `evaluate.py`, the optimal threshold is written to `evaluation_report.json`. Subsequent `infer.py` calls automatically use this threshold without you having to type it every time.

**`--embed-only` flag:**

```python
embedding = model.backbone.extract_features(tensor)  # [1, 512]
```

This is the hook for the future ensemble. Instead of a binary prediction, it returns the raw 512-dimensional feature vector. The ensemble model will take this vector alongside URL features (domain age, SSL certificate, WHOIS data) and make the final decision.

**`--tta` flag — Test-Time Augmentation:**

Test-Time Augmentation (TTA) runs the model multiple times on slightly different versions of the same image and averages the predictions. No retraining needed — it is a pure inference-time trick.

Why it works: a single center-crop might accidentally hide a suspicious element near the edge of the screenshot — for example, a fake bank logo that's slightly cut off, or a suspicious login form that's partially outside the center crop. Different random crops expose different regions. Averaging the phishing probability across all views reduces this "luck of the crop" variance.

```
Views used:
  View 0: deterministic center crop (val transform)  ← the reliable anchor
  Views 1-5: random crops via train transform (mild augmentation)
  Total: 6 forward passes, probabilities averaged
```

The train transform used for TTA is the same mild augmentation applied during training — only slight zooms and colour shifts. No aggressive augmentation that would distort meaning.

```python
def run_inference_tta(model, image_path, threshold, device, n_augments=5):
    all_probs = []

    # View 0: deterministic center crop
    tensor = val_transform(image).unsqueeze(0).to(device)
    probs = model(tensor)["probs"][0, 1]
    all_probs.append(probs)

    # Views 1..N: random crops
    for _ in range(n_augments):
        tensor = train_transform(image).unsqueeze(0).to(device)
        probs = model(tensor)["probs"][0, 1]
        all_probs.append(probs)

    # Average and threshold
    avg_prob = sum(all_probs) / len(all_probs)
    predicted_class = "phishing" if avg_prob >= threshold else "legitimate"
```

The output includes `tta_individual_probs` — the raw probability from each view. If all views agree (e.g. [0.91, 0.89, 0.93, 0.88, 0.92, 0.90]), the model is very confident. If they spread wide (e.g. [0.61, 0.32, 0.78, 0.41, 0.55, 0.69]), the image is genuinely ambiguous — the phishing content might only be visible from certain crops.

TTA is recommended for production deployments where the cost of a False Negative (missed phishing) justifies the extra inference time (~6× slower but still milliseconds on GPU).

---

## Part 4 — How to Read the Training Output

When training runs, you see output like this:
```
Epoch   1/50 [P1] *  loss=0.8234/0.7891  acc=0.612/0.641  rec=0.521  f2=0.572  auc=0.701
Epoch   2/50 [P1]    loss=0.7234/0.7121  acc=0.671/0.681  rec=0.601  f2=0.631  auc=0.741
```

Breaking it down:
- `Epoch 1/50` — current epoch out of total
- `[P1]` — Phase 1 (backbone frozen). Changes to `[P2]` after `freeze_backbone_epochs`
- `*` — star means this epoch's `val_f2` was the best seen so far (EMA weights saved to `best_model.pt`)
- `loss=0.82/0.79` — train loss / val loss (val loss computed with EMA weights)
- `acc=0.61/0.64` — train accuracy / val accuracy (val computed with EMA weights)
- `rec=0.521` — validation recall with EMA weights (fraction of phishing pages caught)
- `f2=0.572` — validation F2 score with EMA weights (the primary metric for early stopping)
- `auc=0.701` — validation ROC-AUC with EMA weights

All validation metrics use EMA weights. This means they represent what the production model will actually achieve, not a snapshot of the noisy training weights at that exact step.

**What to look for:**

| Sign | What it means |
|---|---|
| `rec` near 0.0 | Model isn't detecting phishing at all — check loss weights |
| `rec` near 1.0, `val_loss` high | Model flags everything as phishing — threshold too low |
| `train_loss` drops but `val_loss` rises | Overfitting — add dropout, reduce LR, use more augmentation |
| `val_f2` improves then plateaus | Normal convergence — early stopping will trigger after patience epochs |
| `val_f2` never improves from epoch 1 | Something may be wrong — check dataset paths and class balance |

---

## Part 5 — End-to-End Flow Summary

```
scripts/train.py
    |
    ├── build_dataloaders()
    |       PhishingDataset.discover("dataset")
    |           -> 3,485 ImageRecord objects
    |       DatasetSplitter.split()
    |           -> 2,439 train / 522 val / 524 test
    |       WeightedRandomSampler(train_records)
    |           -> phishing sampled 5.34x more often
    |
    ├── build_model()
    |       MobileCLIPBackbone("MobileCLIP2-S2", pretrained="dfndr2b")
    |           -> loads 36M pretrained weights from HuggingFace (DFN-2B dataset)
    |       ClassificationHead (LayerNorm -> Linear -> GELU -> Dropout -> Linear)
    |       PhishingClassifier(backbone, head)
    |
    ├── Trainer.train()
    |       _EMATracker initialised (shadow weights = starting model weights)
    |
    |       Phase 1 (epochs 1-5):
    |           backbone.freeze()
    |           optimizer = AdamW([head_params], lr=1e-3)
    |           for each batch:
    |               images -> backbone -> 512-dim embedding -> head -> [B, 2] logits
    |               loss = FocalLoss(logits, labels, alpha=[0.594, 3.168], gamma=2)
    |               loss.backward() -> gradients on head only
    |               optimizer.step()
    |               ema.update(model)  <- shadow = 0.999*shadow + 0.001*weights
    |           with ema.apply(model):
    |               val loop -> val_f2 computed using EMA weights
    |           callbacks -> if val_f2 improved: save best_model.pt (EMA weights)
    |                        always:              save last_checkpoint.pt (training weights)
    |
    |       Phase 2 (epochs 6-50):
    |           backbone.unfreeze()
    |           optimizer = AdamW([backbone_params (lr=1e-5), head_params (lr=1e-4)])
    |           scheduler = warmup (2 epochs) -> cosine decay to 1e-7
    |           same loop as Phase 1, gradient now flows through backbone too
    |
    └── outputs/checkpoints/best_model.pt  <- EMA weights from best val_f2 epoch

scripts/evaluate.py
    |
    ├── load best_model.pt (contains EMA weights automatically)
    ├── run inference on 524 test images (never seen during training)
    ├── ThresholdOptimizer.optimize()
    |       sweeps 181 thresholds -> finds F2-optimal threshold (e.g. 0.35)
    ├── compute all metrics at recommended threshold
    └── save predictions.json, false_negatives.json, evaluation_report.json
           evaluation_report.json stores recommended_threshold = 0.35

scripts/infer.py --image screenshot.png --tta
    |
    ├── load best_model.pt (EMA weights, loaded transparently)
    ├── read threshold 0.35 from evaluation_report.json
    ├── load image, apply val_transform -> [1, 3, 256, 256] tensor (view 0)
    ├── apply train_transform 5 more times -> views 1-5
    ├── run model forward pass 6 times, collect phishing_prob each time
    ├── avg_prob = mean([p0, p1, p2, p3, p4, p5])
    ├── predicted_class = "phishing" if avg_prob >= 0.35 else "legitimate"
    └── print JSON with avg_prob, individual probs, inference_time_ms
```
