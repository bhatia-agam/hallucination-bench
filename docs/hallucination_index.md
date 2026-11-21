# Hallucination index

`uv run hbench index` scores every model (one row per method x layer) from the families' outputs
(nothing is re-judged) and writes `outputs/index/<judge model>/index.{json,md}`. `hbench index`
and `hbench final` are the only index builders.

**`hindex-v8`** ([plan 0029](../plans/0029-index-v8-overhaul.md), `version=v8`) re-judges every claim-map family with
**`claimmap-v2.9`** (`map_prompt=claimmap-v2.9`, plan 0029 Amendment 1, which replaced
`claimmap-v2.8` after the pilot2 tag gate failed; recorded in `arm.json`; a run on an arm dir
judged with another prompt is refused, exit 3) into separate roots (`out=outputs/v8`).

**The continuation excuse is decided by the verifier `cverify-v1`** (plan 0029 Amendment 2;
the judge's tags failed the blind gate twice, 0.52 and 0.733).
- Every incorrect segment whose quoted contradiction is only after the read point is a
  candidate, whatever the judge's tags say.
- One dedicated call per candidate (Opus 5.5, effort medium, Sonnet on refusal) answers c1, c2
  and c3:
  - c1: it is a literal continuation;
  - c2: it is possible given the text before the marker, i.e. it fills the **same slot** (same
    list number and level, same speaker and addressee, every entity fixed before the marker
    kept; invented specifics inside the slot are fine);
  - c3: it is on topic.
- The candidate scores `continuation_miss` (weight 0, still a claim) iff all three hold. A
  failed verifier call never excuses and makes `hbench run` exit 4 until it is re-asked.
- The setting is on by default for claimmap-v2.9 roots (off under `anchor=v5c`, which has no
  continuation excuse; an explicit verifier with `anchor=` exits 3). `excuse_verifier=off` is
  for diagnostics only. It is recorded in `arm.json`, and a run that changes it exits 3.
- A cell with a failed or pending verdict is not scored by the index, and `hbench final`
  refuses a manifest with any such cell (no 1% allowance) unless it is a listed persistent
  failure (plan 0029 Amendment 4, below).
- Known limitation: for readouts over 2,500 characters the verifier's readout window anchors on
  the first `<<<`, so a readout that itself contains `<<<` can lose the marked segment (counted
  as `n_verifier_segment_unmarked`).
- The judge's `continuation_text` / `contradicted_before_marker` tags are kept and reported
  (agreement counts) but no longer gate the excuse.

A different possible continuation is not a hallucination (Camila, 2026-11-04). A wrong list number ("2." as a first
heading) is never possible (Camila, 2026-11-05): such a heading stays `detail` (another heading
of the same list; the number does not upgrade the leaf) and is not excused; claims or predictions about later text,
commentary, the agent's own first-person plan read inside the prompt (AM), and continuations
the prefix rules out still count. A
wrong role or title is a substitution; every date or time the conversation contradicts
(real-world events too) and quantities of the conversation's own events are `detail`. The main family runs on plan 0028's 3x bank (`bank=x3`). Families and
weights: main 9, `am_claims` 3 (the AM claim map on candidate claims, `claimmap-v2.9+am3` on
Sonnet with a scenario card), `am_situation` 3 (`amsit-v2`: readouts that misrepresent the
agentic scenario as a whole (a reading of the local syntax at the marker does not), `stages=situation`), sibling 1, story 1 (only `detail`
down-weighted), overrefusal 1, benign_roleplay 2 (Σ 20; `matched` = all but sibling, Σ 19). The
cheating eval is not on the index (archived 2026-11-04 as a diagnostic, plan 0029 Amendment 3;
every version refuses a cheating root). Every method also gets a **joint**
row (layer `all`: sample 0 of all 11 layers of a read site in one judge call; `hbench run --joint
[joint_from=a,b]`, AM `stages=joint`, `scripts/overrefusal/sj_run.py --joint` + `hbench
overrefusal offtopic-export --joint`, `hbench benign-roleplay index-export --joint`) with
`full_joint`; per-layer rows carry `full`, `matched`, `full_j6` (the six joint families) and
`full_v7` / `matched_v7`; `over_layers` holds the 11-layer means (shared item bootstrap).
`hbench final manifest=data/index/v8_manifest.json out=outputs/final/ plots=plots/final/`
checks the manifest (sha256 per file, <= 1% failed per family x method x unit, every claim map
judged with `claimmap-v2.9` (a v2.8 or mixed manifest is refused) and the continuation
verifier `cverify-v1` (manifest + every arm dir + a chat / story joint `joint.json`; no failed
/ pending verdict at all); else exit 2), builds hindex-v8 from exactly those roots and draws the final plots
(`index_per_layer.png`, `families_per_method.png`, `macro_all11.png`). Judge noise is not measured
for v8. **Persistent judge failures** (plan 0029 Amendment 4): a cell still failed after >= 3
attempts at its blocking stage (both models refused, or an invalid answer after the in-run
retries; never a transport error) is listed in `data/index/v8_persistent_failures.json`
(`scripts/v8/persistent_failures.py`; hashed in the manifest); `final` excludes listed cells from
the 1% / zero-verdict gates (each must be failed in the exact hashed file; <= 15% per entry),
never scores them, and reports them in `index.json`, `index.md` and `excluded.md`.
**hindex-v8 is the only index version** (plan [0030](../plans/0030-v8-only-index.md) removed
hindex-v3 to hindex-v7 from the code; `version=` accepts only `v8`). The frozen v8 results:
[`docs/results/0029_index_v8.md`](results/0029_index_v8.md). The family definitions v8
inherits are in plans
[0013](../plans/0013-final-sibling-eval-and-hallucination-index.md),
[0019](../plans/0019-safety-families-on-the-index.md),
[0021](../plans/0021-am-candidate-claims-index-v5.md),
[0022](../plans/0022-overrefusal-wrong-topic-refusals.md) and
[0027](../plans/0027-benign-roleplay-on-index-v7.md).
Arms are matched by name across families; legacy names and exclusions live in
`data/index/methods.json`. Each family root is overridable (`main_root=`, `am_root=`,
`story_root=`, `um_root=`, repeatable); the overrefusal root is **opt-in**
(`overrefusal_root=`, a dir holding `index_counts.jsonl` + `index_meta.json`, or
`HBENCH_SAFETY_OR_OUT` when set; never a dev-box default).

External data is configured by environment variables, each defaulting to the dev-box path
(`src/hallucination_bench/paths.py`): `HBENCH_READOUTS_ROOT` (WorkspaceBench paper readouts),
`HBENCH_WSBENCH_PUB` (a workspace-bench checkout), `HBENCH_GLOBAL_WORKSPACE`,
`HBENCH_SIBLING_SRC_ROOT`, `HBENCH_ALI_PROBE_ROOT`, `HBENCH_COMBOS_READOUTS`, `HBENCH_X3_READOUTS`
(plan 0028's 3x-bank readouts, the `bank=x3` default; the bank dir is `HBENCH_X3_BANK`, default
`data/wsbench_chat_x3`); the shell drivers
also read `HBENCH_WSBENCH_PY`, `HBENCH_AM_GEN`, `HBENCH_AM_READOUTS`, `HBENCH_MLNLA_GEN`,
`HBENCH_EXTRA_READOUTS`, `HBENCH_STORY_GEN`, `HBENCH_STORY_READOUTS`, `HBENCH_GW_CAPTURE`
(a global-workspace checkout for the capture jobs). Tests that need one of them skip and name the variable.

Per cell (arm × item × read site × layer) `wsbench_chat` runs up to five calls:

1. **summarise** (token lenses only): the `interp-v1` summariser turns the decoded top-10 token
   bag into one interpretation, which is judged as the readout (k = 1);
2. **span** (`v5c-chat`, verbatim): wrong / off-topic spans per readout and a class per readout
   (`hallucinated` / `off_topic` / `consistent` / `generic`);
3. **verify** (`v5c-chat-verify-v1`, verbatim): every other specific claim as `true` /
   `unverifiable` / `disputed` / `off_topic`;
4. **classify** (`taxonomy-v1`, new): labels every hallucinated span found by 2–3. It never
   decides whether a span is a hallucination;
5. **map** (`claimmap-v2.3`, plan 0002 rev 13): code splits every readout into units; one
   judge call puts every unit, or verbatim sub-spans of it, into one taxonomy-v2 leaf; code
   rolls the leaves up. Single-call by default (needs only stage 1; no established spans, no
   verify); `anchor=v5c` restores rev 12 (the v5c verified wrong spans become their own
   "established" units; needs stage 2).

`hbench run stages=summary,span,verify,classify,map` picks the stages that may call the API
(default: all; dependencies run first if missing: `stages=map` runs summary, plus span on a
seeded 10% item subsample for the v5c comparison; under `anchor=v5c` span on every cell).
