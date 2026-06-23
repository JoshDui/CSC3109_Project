# Curated artifacts

This folder is for curated artifacts needed to make final notebooks and reports
pull-and-read or pull-and-run reproducible.

Guidelines:

- Commit only curated final evidence, not scratch outputs.
- Use Git LFS for `.pt`, `.pth`, `.onnx`, and other large binary model files.
- Include an `ARTIFACTS.json` or README in each bundle with provenance,
  checksums, and the pipeline command that created it.
- Keep raw datasets and full intermediate mask trees out of git unless the team
  explicitly decides otherwise.

Suggested semantic-guided bundle path:

```text
artifacts/semantic_guided_cgaf/final_20260616/
```
