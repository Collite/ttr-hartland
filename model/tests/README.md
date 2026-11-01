# model/tests — the model suites

`*.test.mjs` under `model/`, run by `just verify-model` (node's test runner, no dependencies of its own).
`project-harness.mjs` loads `model/` as one project through a built ttr-core at toolchain.lock's ref
(`TATRMAN_PACKAGES`, default `../tatrman/packages`) and lists the diagnostics the gate accepts, each with its reason.
