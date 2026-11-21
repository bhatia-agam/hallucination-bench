# HallucinationBench

HallucinationBench measures how much of what an activation-reading lens says is hallucinated.
A lens (O-lens, NLA, J-lens, logit lens, ...) reads a model's activations and describes them;
a judge maps every claim in every readout to grounded, hallucinated or neither, across several
families of prompts (chat, stories, agentic scenarios, user modelling, refusals, role-play).
The headline number is the **hallucination index**: the weighted share of a lens's claims that
are hallucinated, lower is better.

## Install

```bash
git clone https://github.com/camila-blank/hallucination-bench && cd hallucination-bench
uv sync                        # Python >= 3.12; dev tools come with the dev group
uv run hbench check dry_run    # 1 s, offline, no key: the pipeline builds its judge prompts
```

Judging calls Claude (`claude-opus-5-5` by default). Any one of these works:

- `export ANTHROPIC_API_KEY=...`
- `ant auth login` once (the keyless `ant` CLI profile; nothing to export)
- `ANTHROPIC_AUTH_TOKEN` set to an OAuth bearer token

With none of them a judging command stops at its first call with `JUDGE CONFIG ERROR: no
Anthropic credentials` (exit 3). Paths outside the repo (readouts roots, sibling checkouts) are
`HBENCH_*` environment variables with defaults, listed in `src/hallucination_bench/paths.py`.
The full test suite is `uv run pytest -q` (offline, about 10 minutes).

## Try it in a minute

Two items of one family, end to end, with no key and no readouts of your own:

```bash
uv run hbench try --dry_run                  # the main chat family
uv run hbench try family=story --dry_run
```

```text
== hbench try: wsbench_chat, 2 item(s), readouts examples/readouts/wsbench_chat.jsonl
[1] chat-dailydialog-0000
    question: Where's your better half?
    model's reply: That’s a charming question! Since I’m an AI, I don’t have a spouse ...
    expected: every claim a readout makes is supported by the conversation up to the token it was read at
    readouts in the file: 3 cells at layer(s) [44]
...
== judge (dry run: prompts are built, no call is made)
6 cell(s) would be judged with claimmap-v2.9: prompts built, 0 calls made, nothing spent
== the first judge prompt (first 12 lines)
== report (empty in a dry run: nothing was judged)
== outputs
  judge cache (one row per call; resumable): outputs/try/wsbench_chat/judged/wsbench_chat/claude-opus-5-5/try/cache.jsonl
  per-cell verdicts: outputs/try/wsbench_chat/judged/wsbench_chat/claude-opus-5-5/try/cells.jsonl
  report: outputs/try/wsbench_chat/judged/wsbench_chat/claude-opus-5-5/report.md
```

It shows each item, the prompt the judge would get, and where every output file is.

- **For real:** drop `--dry_run`. The two example items are 6 cells and about 20 judge calls,
  about $0.20 at list prices, under a minute. It needs the credentials from Install. A judge
  answer that fails validation is asked again, up to three passes; if `failed calls remain`
  after that, run the same command again: only those calls are re-asked.
- **Your own lens:** `readouts=<file>`, one JSONL row per cell (`{"id", "layer", "pos",
  "samples": [...]}` or `"tokens"` + `"scores"`); `n=` for more items.
- **What it covers:** `family=wsbench_chat` is the chat family, which the index calls `main`;
  `family=story` is `story`. The other five index families have their own drivers and no
  try-out ([Families and weights](#families-and-weights)). The try-out judges with
  `claimmap-v2.9`, the claim-map prompt of the current index.

## The commands you will use

Every command takes its options as `key=value` (a boolean as `--key`); `--show` prints the
resolved config without running, `--help` lists every key with its default. The old
`--some-flag value` spelling still works.

| command | what it does |
|---|---|
| `hbench try [family=F] [--dry_run]` | The two-question try-out above. |
| `hbench check [name\|all]` | Gates and sanity checks: list them, or run one and get PASS / FAIL ([Checks](#checks)). |
| `hbench run out=DIR readouts_root=R arms=a,b` | Judge arms of the main family (`eval=story` for stories, `eval=agentic_misalignment` for the agentic family). Resumable; `limit=N` for a smoke run. |
| `hbench report out=DIR` | Cross-arm report and audit samples. |
| `hbench index out=DIR` | The hallucination index (hindex-v8, the only index version): one row per method and layer, from the families' outputs. Nothing is re-judged. |
| `hbench final manifest=data/index/v8_manifest.json` | Check the frozen manifest, build the index from exactly its roots, draw the final plots. |
| `hbench calibrate split=dev` | Score the classifier against hand-labelled cases. |

The families with their own drivers have their own subcommands (`hbench overrefusal ...`,
`hbench benign-roleplay ...`). Every pipeline's full command list is in
[docs/commands.md](docs/commands.md).

## Where outputs go

Everything is written under `out=` (default `outputs/`, git-ignored):

| path | what it holds |
|---|---|
| `<out>/<eval>/<judge model>/<arm>/cache.jsonl` | One row per judge call, so a re-run only pays for what is missing. |
| `<out>/<eval>/<judge model>/<arm>/cells.jsonl` | Per-cell verdicts. |
| `<out>/<eval>/<judge model>/<arm>/summary.json` | Rates, health and spend of one arm. |
| `<out>/<eval>/<judge model>/report.md` | The cross-arm report (`report.json`, `audit_samples.md` beside it). |
| `<out>/index/<judge model>/index.{json,md}` | The hallucination index. |
| `outputs/try/<family>/` | Everything `hbench try` writes. |

Every command prints the paths it wrote.

## Checks

```bash
uv run hbench check          # list every check, what it guards and the options it takes
uv run hbench check all      # run the 5 that need no option: a few seconds, offline, no key
uv run hbench check readouts_contract readouts=examples/readouts/wsbench_chat.jsonl
```

Each check prints PASS or FAIL and one line of why. `check all` covers the frozen data
manifests, the index manifest, the calibration sets, the pinned judge prompts and a dry run of
the pipeline; it lists `readouts_contract` as SKIP because it needs a readouts file. Details
and how to add a check: [docs/checks.md](docs/checks.md).

## Results viewer

The results viewer lives in WorkspaceBench and reads HallucinationBench outputs too. From a
[workspace-bench](https://github.com/camilablank/workspace-bench) checkout, for the try-out
you just judged:

```bash
uv run wsbench viewer ingest <this repo>/outputs/try/wsbench_chat/judged adapter=hb hb_repo=<this repo> lens=try store=/tmp/hb_viewer
uv run wsbench viewer serve store=/tmp/hb_viewer      # http://127.0.0.1:8600
```

For a full run the directory is `<out>` of `hbench run`, after `hbench index out=<out>` has
written `index/` there (the viewer reads its scores from the index; a live `hbench try` builds
it for you). Install and the other options: `src/wsbench/viewer/README.md` in that checkout.

## Score your own lens

```bash
# readouts: <root>/<arm>/hallucination.jsonl, one row per cell (the contract above)
uv run hbench run out=outputs/mylens readouts_root=<root> arms=mylens bank=x3 map_prompt=claimmap-v2.9 limit=5   # smoke: 5 cells per layer
uv run hbench run out=outputs/mylens readouts_root=<root> arms=mylens bank=x3 map_prompt=claimmap-v2.9           # the whole arm; resumable
uv run hbench report out=outputs/mylens bank=x3    # the cross-arm report
```

That is the `main` family. The other families and the full index run are in
[docs/commands.md](docs/commands.md).

## Families and weights

The index (hindex-v8) is a weighted mean over seven families, one row per (method, layer):

| family | weight | what it measures | how it runs |
|---|---|---|---|
| `main` | 9 | Claims about a chat conversation, on the 3x chat bank (`data/wsbench_chat_x3`). | `hbench run bank=x3` |
| `am_claims` | 3 | Claims a lens makes at mentions of misalignment in agentic scenarios. | `hbench run eval=agentic_misalignment` |
| `am_situation` | 3 | Readouts that misrepresent the agentic scenario as a whole. | `hbench run eval=agentic_misalignment stages=situation` |
| `sibling` | 1 | User modelling: a same-category but wrong attribute of the user (Korean user, readout says China). | `scripts/sibling_probe/` |
| `story` | 1 | Claims about a story the model is writing. | `hbench run eval=story` |
| `overrefusal` | 1 | Readouts that claim a refusal on prompts the model never refuses. | `hbench overrefusal ...` |
| `benign_roleplay` | 2 | Readouts that see a jailbreak in harmless role-play prompts. | `hbench benign-roleplay ...` |

Weights sum to 20. `full` is every family; `matched` leaves out `sibling` (sum 19) for methods
without a sibling score. hindex-v8 is the only index version, and `hbench index` and `hbench
final` are the only commands that build it. Details and the excuse rules:
[docs/hallucination_index.md](docs/hallucination_index.md); every eval with its plan:
[docs/evals.md](docs/evals.md).

## More documentation

| topic | where |
|---|---|
| Every eval and its plan | [docs/evals.md](docs/evals.md) |
| The hallucination index: versions, weights, joint rows, manifests | [docs/hallucination_index.md](docs/hallucination_index.md) |
| The claim map and the taxonomy | [docs/claim_map.md](docs/claim_map.md) |
| Every pipeline's commands | [docs/commands.md](docs/commands.md) |
| Judge settings and known caveats | [docs/judge_and_caveats.md](docs/judge_and_caveats.md) |
| Judge backends: live calls or Message Batches, sizing concurrency | [docs/judge_backends.md](docs/judge_backends.md) |
| Gates and sanity checks | [docs/checks.md](docs/checks.md) |
| Results of record (frozen hindex-v8) | [docs/results/0029_index_v8.md](docs/results/0029_index_v8.md), every row: [0029_index_v8_full.md](docs/results/0029_index_v8_full.md); figures in [`plots/final/`](plots/final/) |
| Design history | `plans/` (this repo supersedes the archived `camilablank/hallucination-bench-v1`; WorkspaceBench's hallucination eval is the main family here) |
| Conventions for contributors and agents | [CLAUDE.md](CLAUDE.md) |

## Credits

HallucinationBench ships text from public datasets; their licences apply to those files:

- **WildChat** (Zhao et al. 2024, `allenai/WildChat`, ODC-BY): the `benign_roleplay` prompts
  (`data/benign_roleplay/`), via WorkspaceBench's `jailbreak_recognition` bank.
- **LMSYS-Chat-1M** (Zheng et al. 2023, LMSYS-Chat-1M licence) and **DailyDialog** (Li et al.
  2017, CC BY-NC-SA 4.0): the `wsbench_chat` turns (`data/wsbench_chat/`, via WorkspaceBench, and
  `data/wsbench_chat_x3/`).
- **OR-Bench** (Cui et al. 2024): the `overrefusal` prompts (`data/overrefusal/`).
- **WorkspaceBench** (github.com/camilablank/workspace-bench): the banks, judges and the
  `jailbreak_recognition` family that `benign_roleplay` runs.
