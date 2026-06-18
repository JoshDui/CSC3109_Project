# FocalNet Plan

## Owner

- **William**

## Project Context

This project is a **4-class aerial image classification** task for Group 12.

Classes:

- bridge
- freeway
- overpass
- railway

Dataset summary:

- 2,800 training images total
- 400 held-out validation images total
- RGB images, already 256x256

The task is to train a model that predicts the correct class label for each aerial image.

---

## What is FocalNet?

**FocalNet** is a vision architecture proposed by Microsoft.

Instead of using standard CNN convolutions as the main modeling idea, and instead of using plain global self-attention like a standard Vision Transformer (ViT), FocalNet uses **focal modulation** to combine information from different spatial ranges.

In simple terms:

- it looks at **local visual details**
- it also gathers **wider scene context**
- it combines both to make a classification decision

This makes it a good fit for aerial imagery, where both small details and overall layout matter.

Reference repository:

- Microsoft FocalNet: <https://github.com/microsoft/FocalNet>

---

## Why FocalNet for This Project?

The four classes are visually similar transport/infrastructure scenes.

Important cues include:

- **bridge**: crossing structure, often spanning water or land gaps
- **freeway**: broad road corridors and lane structure
- **overpass**: road-over-road crossing patterns
- **railway**: long parallel track-like lines

This means the model likely needs both:

1. **Local detail understanding**
   - edges
   - track patterns
   - road markings
   - bridge boundaries

2. **Global scene/layout understanding**
   - crossing geometry
   - long corridor patterns
   - whether a structure spans another region

FocalNet is a strong candidate because it is designed to combine **multi-scale context** while still being practical to fine-tune with pretrained weights.

---

## Main Goals with FocalNet

William's FocalNet work should aim to:

1. **Implement a distinct non-CNN model contribution**
   - avoid duplicating another standard CNN baseline
   - contribute a meaningfully different architecture to the model comparison

2. **Fine-tune a pretrained FocalNet model on the Group 12 dataset**
   - use transfer learning, not training from scratch
   - adapt the model to 4 output classes

3. **Evaluate FocalNet using the project rubric metrics**
   - accuracy
   - precision
   - recall
   - F1-score
   - confusion matrix

4. **Compare FocalNet against the other planned model families**
   - CNN / ResNet-style baseline
   - ViT
   - CLIP
   - other team-owned architectures

5. **Support deployment/efficiency analysis later**
   - model size
   - inference latency
   - deployment compatibility
   - possible ONNX / runtime benchmarking if needed

---

## Planned Implementation Approach

### Recommended model variant

Start with:

- **`focalnet_tiny_srf`**

Reason:

- small enough to be practical
- available through `timm`
- suitable as a first fine-tuning target

### Training strategy

- use **pretrained weights**
- replace the classification head with **4 output classes**
- fine-tune on the project dataset
- use an **internal train/tune split** from the training set
- reserve the official validation set for final evaluation only

### Initial training settings

- image size: **224x224**
- optimizer: **AdamW**
- loss: **cross entropy**
- epochs: **15-20** as a starting point
- early stopping based on **macro F1** on the tuning split

---

## Repo Integration Plan

The existing project already has reusable `timm`, data loading, and evaluation
helpers. FocalNet should reuse those helpers, but the primary workflow is now a
notebook-first run so the model-specific training/evaluation control flow stays
visible for review.

Primary workflow file:

- `notebooks/06_focalnet_training_and_evaluation.ipynb`

Supporting integration:

- FocalNet aliases are exposed through `src/models/timm_classifier.py`.
- The notebook imports shared helpers from `src.config`, `src.data`,
  `src.evaluation`, and `src.models`.
- No standalone `src/training/train_focalnet.py` or
  `src/evaluation/evaluate_focalnet.py` is required for the current workflow.

### Practical approach

1. Use the `focalnet-tiny-srf` alias, which resolves to `focalnet_tiny_srf`.
2. Reuse the current data loaders and internal split logic from `data/raw/train`.
3. Reuse the existing evaluation metric pipeline.
4. Keep the notebook cells for optimizer, scheduler, checkpoint selection, and
   early stopping visible.
5. Save:
    - best checkpoint
    - tuning metrics JSON
    - confusion matrix image
    - training history CSV
    - run configuration JSON
6. Run final evaluation on `data/raw/val` only in the final notebook section
   after a clean top-to-bottom pretrained run.

---

## Suggested Report Positioning

FocalNet can be presented as:

> an alternative multi-scale vision architecture for aerial scene classification, used to test whether focal modulation improves discrimination between visually similar transport-infrastructure classes compared with standard CNN baselines and plain transformer models.

This gives a clear comparison story:

- **CNNs**: convolution-based local feature learning
- **ViT**: global patch-based self-attention
- **CLIP**: vision-language pretrained representation
- **FocalNet**: focal modulation / multi-scale context aggregation

---

## Immediate Next Steps

1. Open `notebooks/06_focalnet_training_and_evaluation.ipynb`.
2. Restart the kernel and run top-to-bottom for the final pretrained
   FocalNet-Tiny SRF fine-tuning run.
3. Run the final held-out evaluation section once on `data/raw/val`.
4. Record metrics and compare with the rest of the team models.

---

## Notes

- Start with the smallest practical FocalNet variant first.
- Do **not** train from scratch on this dataset.
- Keep the official validation set for final evaluation only.
- Reportable final results require ImageNet pretrained FocalNet weights;
  `pretrained=False` is diagnostic only.
- Deployment/MLOps benchmarking can be added after the first working model checkpoint is available.
