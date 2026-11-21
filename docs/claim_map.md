# The claim map (taxonomy v2)

Defined once in [`src/hallucination_bench/taxonomy_v2.py`](../src/hallucination_bench/taxonomy_v2.py)
(leaves, groups, precedence, the Detail rule, tags, and the open decisions with their defaults in
`CHOSEN`) and rendered into the judge prompt. Rev 16 (the lean tree): **12 leaves** —
`non_claim` (tag `reason`), `correct` (tag `source`: prefix / after / entailed), `detail`,
`false_claim`, `substitution_sibling`, `substitution_category`, `model_state`, `unverifiable`
(tags `same_slot_alternative`, `world`, `prediction`), `supplementary_true`,
`supplementary_false`, `ambiguous_topic`, `off_topic` — in 8 groups, plus two code-only rows:
`unknown_incorrect` (`anchor=v5c` only) and `failed`. Tags are recorded but never change a
verdict. The hallucination gate: a segment is incorrect only if the conversation **obviously
contradicts** it; predictions get no category of their own (contradicted → incorrect, else
`unverifiable` with `prediction = true`); content realised in any wording is `correct`. The
test for world claims is **settlement, not subject** (text silent → `supplementary_*`).
Each leaf maps to one **verdict** (`taxonomy_v2.VERDICTS`): hallucination (detail,
false_claim), probe_candidate (substitutions, model_state, supplementary_false,
ambiguous_topic), correct, unresolved (unverifiable), junk, non_claim. The headline is the
character-weighted verdict share per readout and the readout-level verdict rates, with
**`false_claim` and `detail` always reported separately** next to the hallucination verdict;
groups and leaves are the breakdown. Probe-candidate segments carry a `probe_target`
(`probe_candidates.jsonl`). Invalid `:M` answers are retried up to 2 times in-run before
being cached as failed. Outputs are labelled **pre-gate** until the held-out gate passes for
the current prompt version. `hbench consistency-v2 arms=s3d-rl600:44,nla-rl-L42:42,jlens:44
n=50 seed=0` runs `:M` twice per sampled cell and reports κ, raw agreement, the between-run
leaf confusion, per-leaf self-agreement and the leaf pairs swapped in > 10% of a leaf's units
(`outputs/consistency/<model>/`). Rev 17: every incorrect segment must quote the
conversation span that contradicts it (`contradicted_by`) and every `correct` segment (except
`source = entailed`) the span that supports it (`supported_by`); a missing or unfindable quote
fails closed to `unverifiable` (counted). The quotes are carried into `cells.jsonl`,
`probe_candidates.jsonl`, the audit samples and the report's hallucination examples. Reports
include a judge-noise section and a "judge noise ±" column (half the r0 − r1 difference of each
headline metric in the consistency sample). Rev 18: `non_claim` and `off_topic` are merged into
one scoring leaf, `junk` (verdict junk, tag `reason` incl. `off_topic`), right after parsing;
the claimmap-v2.7 prompt and cached answers are unchanged, so re-scoring costs nothing. Reports
include a junk-by-reason table (off-topic sub-share per arm). Rev 19 (also after parsing, $0):
a segment the judge tags `hedged` (a list of options, "Germany or France or UK") is scored
`unverifiable` with `same_slot_alternative`, whatever its leaf; `model_state` is its own verdict,
`thinky` (verdicts: hallucination / probe_candidate / thinky / correct / unresolved / junk).
Rev 20 (scoring, $0): `substitution_category` is the third hallucination component, reported
separately next to `false_claim` and `detail`; `substitution_sibling` stays a probe candidate.

## The taxonomy (v1)

Defined once in [`src/hallucination_bench/taxonomy.py`](../src/hallucination_bench/taxonomy.py)
and rendered into the classifier prompt.

- **grounding** (set in code): `contradicted` (verified v5c wrong span), `disputed` (verify
  thinks the response rules it out, v5c did not flag it), `unsupported` (verify
  `unverifiable`: specific, but neither stated, entailed nor realised).
- **type** — would a reader who believed the span misunderstand what the text is about or
  claims? `concept` (yes: a different entity / idea / event as the subject of a claim, or a
  reversed claim; e.g. "Picasso" for da Vinci) vs `detail` (no: a wrong or invented specific
  value inside a correct understanding; e.g. "Steve" for an arbitrary character named Joe).
  Names follow an identity-first decision order; a value that flips the claim is `concept`;
  magnitude alone raises severity, not type.
- **target**: entity, quantity, event, attribute, stance, structure, topic.
- **severity**: 1 minor, 2 moderate, 3 major.
- **unsupported_kind** (unsupported spans only): prediction, inference, fabricated,
  world_knowledge. The **lenient view** drops inference and world_knowledge; the **strict
  view** (the default definition: only stated or realised content is grounded) keeps all four.
