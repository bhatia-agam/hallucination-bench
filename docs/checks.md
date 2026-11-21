# `hbench check`: gates and sanity checks in one command

```bash
uv run hbench check                  # list every check and what it guards
uv run hbench check bank_manifests   # run one: prints PASS or FAIL and one line of why
uv run hbench check all              # run every check that needs no option; the others are listed as SKIP
```

Exit code: 0 = PASS, 1 = FAIL or a usage error (unknown check, missing or unknown option).

| check | what it guards | options |
|---|---|---|
| `bank_manifests` | Every frozen data set under `data/` is byte-identical to its `MANIFEST.sha256`. | |
| `index_manifest` | `data/index/v8_manifest.json` names the shipped x3 bank and judge versions that are still registered. | |
| `calibration_sets` | The calibration case files under `data/calibration/` are present and parse. | |
| `judge_prompts` | The frozen judge prompts and schemas still hash to their pinned values (`tests/test_registry_pins.py`, `tests/test_v5c_frozen.py`). One test compares against a workspace-bench clone (`$HBENCH_WSBENCH_PUB`) and skips without it; the PASS line then says what was not checked. | |
| `readouts_contract` | A readouts file parses under the contract: no malformed, duplicate or mixed-kind rows. | `readouts=FILE` |
| `dry_run` | The pipeline builds its judge prompts end to end on `examples/readouts`, with no API call. Listed only when `hbench try` is installed (it wraps it). | `[family=F]` |

All of them run offline.

## Adding a check

A check is a function that takes its options as keyword arguments (strings) and returns
`(ok, why)`: a bool and one line. Register it by dotted path, so nothing is imported until it runs.

In this repo, add the function to `src/hallucination_bench/checks.py` and one
`register_check(name, guards, "hallucination_bench.checks:<function>")` line below the others. A
check whose assertions already live in a test file is one line:
`return pytest_outcome("tests/test_x.py")`.

From outside the repo (for example a reader's HF-vs-SGLang parity check), name it in the
environment; its one-line description is the function's docstring:

```bash
export HBENCH_CHECKS="reader_parity=reader.checks:parity"
uv run hbench check reader_parity option=lens=my-lens option=engine=sglang
```
