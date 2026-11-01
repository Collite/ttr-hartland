# generated — what the cluster loads

Built by `just regenerate` (scripts/regenerate-local.sh; regenerate.yml runs the same) with the toolchain
`toolchain.lock` pins, and committed: the model-gate's `check-model` fails when it is stale.

- `resolved-packages.json` — the packages, entities and areas of `model/` (the ttr-core Modeler CLI).
- `lexicon.tar.zst` — the Czech lexicon archive: lands at M3 S3.1 (`just build-lexicon`), with `check-lexicon` in the
  model-gate. Until then there is none.
