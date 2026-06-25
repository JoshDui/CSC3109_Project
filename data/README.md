# Data Folder

Place the assigned dataset here after the team receives it.

Canonical local structure:

```text
data/
  raw/
    train/
      bridge/
      freeway/
      overpass/
      railway/
    val/
      bridge/
      freeway/
      overpass/
      railway/
```

The canonical raw split has 700 training images per class under `data/raw/train`
and 100 held-out validation images per class under `data/raw/val`.

Validation images must be used only for evaluation.

Derived pseudo-label data, such as semantic masks, must stay outside the raw
image split. Use separate locations such as `data/semantic_masks/`,
`reports/tables/`, and `reports/figures/` for derived artifacts.

Do not commit the raw dataset unless the course instructions explicitly allow it.
