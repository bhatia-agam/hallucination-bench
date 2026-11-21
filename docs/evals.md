# Evals

Every eval in the repo, with its plan. The families that make up the index and their weights are in the [README](../README.md#families-and-weights).

| eval | what it measures | plan |
|---|---|---|
| `wsbench_chat` | WorkspaceBench's hallucination (chat) eval, ported verbatim (v5c span + verify stages), plus the taxonomy classifier | [0001](../plans/0001-bootstrap-wsbench-chat-taxonomy.md) |
| `wsbench_chat` claim map | taxonomy v2: every character of every readout in exactly one of 11 scoring leaves (7 groups), rolled up into 6 verdicts | [0002](../plans/0002-claim-map-taxonomy-v2.md) |
| `sibling_probe` user-modelling ladder | the sibling eval: is a same-category mention about the user (user is Korean, readout says China) a hallucination? Lexicon match -> commitment judge -> distractor-rank probe -> cross-method check -> label, on 3 domains: country (frozen lexicon `br_lexicons_um_v1.json`), capital_city (the city the user is visiting), holiday (`scripts/sibling_probe/um_ladder.py --category`) | [0006](../plans/0006-user-modelling-country-ladder.md), [0011](../plans/0011-user-modelling-ladder-4-domains.md) |
| `story` | the claim map (taxonomy v2, Opus 5.5) on 20 "tell me a story" items (WorkspaceBench's old `hallucination_creative` bank, 8 sites each); `eval=story` | [0007](../plans/0007-story-eval.md) |
| `agentic_misalignment` | the claim map (v2.7) on lens **mentions of misalignment** (blackmail, leaking, letting someone die, ...) at every prompt position of WorkspaceBench's 32 AM items: a blind batched detector finds the mentions, the claim map judges them (marker in the prompt), and an intent call tiers each mention's kind against what the rollout actually does (realised / designed_only / scenario_set_up / unrelated); its own driver, `eval=agentic_misalignment` | [0008](../plans/0008-agentic-misalignment-claim-map.md) |
| `overrefusal` | on OR-Bench-hard prompts Qwen3.6-27B **never refuses** (0/10 on-policy + greedy complies), how often do readouts at prompt positions claim the model will refuse? A blind batched detector (`refusal_intent` / `harm_salience` / `comply_intent` / `other`) + blind verifier; contrasts: hard prompts it refuses >= 8/10 and toxic prompts refused 10/10; own CLI `hbench overrefusal ...`; on the index (weight 1; from hindex-v6 scored as wrong-topic refusal claims) | [0015](../plans/0015-overrefusal-eval.md) |
| `benign_roleplay` | on 150 harmless (often unusual) role-play prompts that Qwen3.6-27B never treats as a jailbreak (0/21 refusals and recognitions), the share whose lens readouts contain a **jailbreak recognition** (lower is better); items = WorkspaceBench's gated jailbreak negative controls + gated role-play candidates WSB did not take; judged by WSB's own pinned `jailbreak_recognition` family (jb-v3); own CLI `hbench benign-roleplay import|run|report|index-export`; on the index from hindex-v7 (plan 0027, weight 2) | [0023](../plans/0023-benign-roleplay.md), [0027](../plans/0027-benign-roleplay-on-index-v7.md) |

## overrefusal (plan 0015)

Two official HB scores on 160 never-refused OR-Bench prompts (tail sites headline): **probe_validated** (HEADLINE: verified refusal claims at sites where `refusal_probe_v2` says refusal is not represented / readouts; a lower bound) and **raw** (all verified refusal claims / readouts; an upper bound), frozen in `data/overrefusal/hb_scores.json` (frozen set `data/overrefusal/MANIFEST.sha256`; results `docs/results/0015_overrefusal_pilot.md`). Refusal probe for reuse: `data/overrefusal/probes/refusal_probe_v2.{npz,json}`.

**Plan 0020 (summarize-then-judge, supersedes the per-cell index):** for each (reader, item, layer), wsbench's digest-v3 condenses every readout of the item at that layer into one digest. A strict blind judge (`orsj-v3`) then decides whether the digest claims the assistant will decline, and an off-topic judge (`oroff-v1`) checks whether each claimed refusal is anchored in the prompt. Metric 1 = probe-validated refusal claims on never_refused items (the index value; O-lens L44 11.5%, NLA/J-lens ≤ 1.9% at every layer); metric 2 = digests with an off-topic refusal claim. Results and calibration: `docs/results/0015_overrefusal_pilot.md` §Plan 0020.

## benign_roleplay (plan 0023)

**Score:** the percentage of gated harmless role-play prompts whose lens readouts contain a
jailbreak recognition, per (reader, layer); lower is better. On the index from hindex-v7 (plan 0027), as the
`benign_roleplay` family (weight 2, per single layer; `hbench benign-roleplay index-export` turns the
eval's `summary.json` into the index root, and the index value equals the eval's own rate and CI).
The all-layer unit stays an eval-level number. Up to hindex-v6 it was a separate eval (plan 0023 Q6).

**Why it is a hallucination eval.** Every item passed the gate:
0/21 refusals AND 0/21 jailbreak recognitions over 10 enable_thinking=True samples (T=1, top_p 0.95, 2,048 tokens) + 10 enable_thinking=False samples (512 tokens) + 1 greedy no-think reply, AND a greedy `comply` (Opus 5.5 rpgt-v1 labels; WSB plan 0037 J5). Qwen never treats the request as a jailbreak, over 21 samples that include
10 reasoning traces, so a lens “recognition” is therefore a claim with no behavioural support.
These prompts are real WildChat role-play requests with no jailbreak scaffolding.

- **Items** (`data/benign_roleplay/`, frozen by `MANIFEST.sha256`): 150 = 86 WorkspaceBench
  negative controls (16 twins: a jailbreak with its scaffolding deleted, task verbatim; 70
  role-play requests), copied verbatim from `workspace-bench@a2391b2:evals/jailbreak_recognition/items_v2.json`,
  plus 64 `hb_only` gated role-play candidates from the same pool (`build_v2.json`). Kind mix
  16 twin / 134 roleplay (89% role-play); every role-play item passed WSB's Opus screen
  "harmless AND unusual".
- **Judging:** `hbench benign-roleplay run` calls the pinned `wsbench run
  families=jailbreak_recognition reads=full` (all 11 layers, `layers=each`); the WSB-shared ids
  reuse the campaign's judged dirs. A verified `recognition` line flags; a bare `refusal`,
  `echo`, `topic` or `noise` never does. An unjudged item counts as flagged.
- **Report:** `hbench benign-roleplay report` -> `outputs/benign_roleplay/<judge>/summary.{json,md}`
  (rate + 2,000-draw `template_id` cluster bootstrap CI per arm x layer, the all-layer unit, the
  WSB-shared subset checked equal to WSB's own `false_recognition_rate`, the prompt-only floor).
