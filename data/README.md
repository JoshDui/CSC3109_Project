# Data Folder

Place the assigned dataset here after the team receives it.

Current local structure:

```text
data/
  set 12/
    bridge/
    freeway/
    overpass/
    railway/
```

The current extracted set has 700 images per class. This appears to be the training portion of the assignment dataset. If a separate validation set is provided later, place it under a separate folder and do not mix it with training data.

Expected final structure after validation data is available:

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

Validation images must be used only for evaluation.

Do not commit the raw dataset unless the course instructions explicitly allow it.
