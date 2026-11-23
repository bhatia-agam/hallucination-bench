# Producing readouts

HallucinationBench judges readouts; it does not make them. This repository needs no GPU, no
model weights and no serving stack. To score a lens you read it at the cells the plan names,
with any producer, and hand the bench one readouts file per family. This page is the
interface between the two halves: the plan, the file, the check, and the commands that take
a checked file to the index.

```
hbench plan  ->  your producer (GPU)  ->  hbench readouts  ->  the family's judge  ->  hbench index
```

## 1. Get the read plan

```bash
uv run hbench plan out=outputs/plan                      # every family, the index's 11 layers
uv run hbench plan out=outputs/plan families=story layers=44    # a try-out
```

One JSONL file per family, one row per item, in the row format of WorkspaceBench's
`wsbench plan`, so a producer that reads WorkspaceBench plans reads these unchanged:

```json
{"family": "story", "id": "hal-cre-000", "render": "captured",
 "positions": {"kind": "positions", "positions": [29, 85, 153, 233, 305, 359, 432, 495]},
 "layers": [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60],
 "extra": {"input_ids": [248045, 846, 198, "..."]}, "lite_positions": {"kind": "same"}}
```

| key | meaning |
|---|---|
| `render` | how the row becomes a token sequence: `captured` = the exact token ids in `extra.input_ids`; `plain` = `text` verbatim, no chat template and no special tokens added (the text already holds the chat markup the model read); `chat_context` = the subject's chat template over `messages`, thinking off, with the generation prompt |
| `positions` | the token positions to read: absolute, 0-based indices into that sequence |
| `layers` | the decoder layers to read (the residual stream after that layer) |
| `extra.n_tokens` | on `plain` rows, the length of the sequence in tokens: if your tokenizer gives another length, you are reading other positions |
| `lite_positions` | WorkspaceBench's reduced read; not used here |

What each family's plan holds (Qwen3.6-27B, the subject of the shipped banks):

| family | items | render | positions per layer | cells at 11 layers | read site |
|---|---|---|---|---|---|
| `main` | 446 | `captured` | 5,924 | 65,164 | up to 16 punctuation sites in the model's reply |
| `story` | 20 | `captured` | 160 | 1,760 | 8 sites in the story |
| `agentic_misalignment` | 32 | `plain` | 50,452 | 554,972 | every token of the prompt, through `<\|im_start\|>assistant` |
| `overrefusal` | 427 | `plain` | 11,325 | 124,575 | every token of the prompt |
| `benign_roleplay` | 64 | `chat_context` | 47,179 | 518,969 | every token of the last user turn |

`agentic_misalignment` feeds two index families (the claims and the situation judge) from
one file. `benign_roleplay` has 150 items: the 64 planned here are the bench's own; the other
86 belong to WorkspaceBench's `jailbreak_recognition` family and are judged on a full read of
that family (172 items), whose plan is WorkspaceBench's: `wsbench plan
families=jailbreak_recognition` in the checkout `hbench benign-roleplay` pins. The 64 rows are
exactly what that checkout plans for them, so the plan hash is the one its judge expects.

`hbench plan` prints each file's `plan_sha`: WorkspaceBench's hash of the prompts and
positions (layers are not hashed). A producer writes it beside the readouts and
`hbench readouts` compares it.

## 2. Read the lens

For every row, every position and every layer: run the lens on the residual activation
there and keep what it produces. Any producer works.

WorkspaceBench's producer takes these files as they are once it has the `plan=` key
(`wsbench capture plan=<file>` to store the activations, `wsbench produce plan=<file>` to
read them); until that key is in your WorkspaceBench checkout, use your own producer. It
captures the activations of a plan once, serves the lens, and writes the readouts file and
its sidecar in the form below.

## 3. The readouts file

One JSONL file per family, one row per (item, layer, position):

```json
{"id": "hal-cre-000", "layer": 44, "pos": 29, "token": ".", "samples": ["..."]}
{"id": "hal-cre-000", "layer": 44, "pos": 29, "token": ".", "tokens": ["Ġthe", "Ġa"], "scores": [9.1, 7.7]}
```

- `id`, `layer`, `pos`: the plan row's id, one of its layers, one of its positions.
- `samples`: the lens's text, one string per sample. A token lens writes `tokens` (vocabulary
  strings, best first) and optionally `scores` instead; the bench summarises them into prose
  before judging. A file is all text or all tokens.
- `token` (optional): the token at the position, as `tokenizer.decode([id])`. Write it: the
  check compares it with the bank's token, which catches a wrong render or tokenizer.
- Write one JSON object per line and escape U+0085, U+2028 and U+2029 inside strings (as
  ` `; `json.dumps` with `ensure_ascii=True` does). The loader splits lines on them, so
  a raw one turns a row into two malformed halves and the cell goes missing.

Beside the file, a producer may declare how it read, in `<file>.meta.json`:

```json
{"reads": "full", "sampling": "prose1", "reader": "<method>", "arm": "<arm>",
 "layers": [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60], "plan_sha": "<from hbench plan>"}
```

`sampling` is `prose1` (one sample per cell), `bullet4` (four) or `tokens10` (a token
lens). The bench's own judges do not read the sidecar. WorkspaceBench's judge, which scores
`benign_roleplay`, enforces it when it is there and only reports its checks when it is not,
so write it for that family.

## 4. Check the file before judging

```bash
uv run hbench readouts outputs/readouts/mylens/story.jsonl family=story
uv run hbench readouts outputs/readouts/mylens/story.jsonl family=story layers=44
```

It loads the file with the loader the judges use and exits 2, naming the problem, when rows
are dropped (malformed, duplicate, text and tokens mixed), a cell of the plan is missing at
a layer asked for (default: the 11), a row's `token` is not the bank's token at that
position, or the sidecar names another plan, lacks a layer or declares a sampling the rows
do not have. Notes (no sidecar, rows without `token`, extra layers, empty cells) do not fail
it. It makes no judge call and takes seconds: about 4 s and under 1 GB of memory for the
largest file (554,972 rows).

Without it, the families differ in what they do with an incomplete file: `main` and `story`
stop on a missing cell, `agentic_misalignment` stops on an incomplete layer or a stray row,
and `overrefusal` judges what is there.

## 5. From a checked file to the index

`<arm>` is your reader's name and `$R` the folder that holds its files; every family below
writes under one `out=`. The judge needs Anthropic access (README). A command that exits 4
has failed calls cached: run it again and only those are asked (`sj_run.py` does not signal
it: run it twice).

**main and story**

```bash
uv run hbench run out=outputs/mylens bank=main map_prompt=current stages=map,cverify \
    arm=<arm>=$R/main.jsonl
uv run hbench run out=outputs/mylens bank=main map_prompt=current --joint arm=<arm>=$R/main.jsonl
uv run hbench run out=outputs/mylens eval=story map_prompt=current stages=map,cverify \
    arm=<arm>=$R/story.jsonl
uv run hbench run out=outputs/mylens eval=story map_prompt=current --joint arm=<arm>=$R/story.jsonl
```

The `--joint` run judges the all-layer unit (the index's `all` row) and needs all 11 layers.

**agentic_misalignment** (file at `$R/<arm>/agentic_misalignment.jsonl`)

```bash
A="eval=agentic_misalignment out=outputs/mylens readouts_root=$R arms=<arm> map_prompt=current \
   judge_model=claude-opus-5-5 map_model=claude-sonnet-5-5"
uv run hbench run $A stages=intent                                # once per out=, shared by every arm
uv run hbench run $A stages=detect,map,cverify,situation,joint    # a token lens: detect,summ,map,...
```

**overrefusal** (file at `$R/<arm>/overrefusal.jsonl`)

```bash
export HBENCH_OR_OUT=outputs/mylens HBENCH_OR_READOUTS_V3=$R
uv run python scripts/overrefusal/sj_run.py --tag full --readers <arm>
uv run python scripts/overrefusal/sj_run.py --tag joint --joint --readers <arm>
uv run hbench overrefusal offtopic-export out=outputs/mylens
uv run hbench overrefusal offtopic-export out=outputs/mylens --joint
```

**benign_roleplay** (the bench's items at `$R/<arm>/hb_only/jailbreak_recognition.jsonl`, the
full WorkspaceBench family at `$R/<arm>/jailbreak_recognition.jsonl`, each with its sidecar)

```bash
B="arms=<arm> out=outputs/mylens layout=nested campaign=outputs/mylens/campaign"
uv run hbench benign-roleplay run $B readouts_root=$R --dry_run
uv run hbench benign-roleplay report $B
uv run hbench benign-roleplay index-export dest=outputs/mylens/benign_roleplay/index_root --joint \
    summary=outputs/mylens/benign_roleplay/claude-sonnet-5/summary.json
```

`campaign=` is the folder with the judged results of the full WorkspaceBench family, which
cover the shared items: one output folder of `wsbench run families=jailbreak_recognition
reads=full` per unit, `<campaign>/<arm>/all` and `<campaign>/<arm>/L<k>`, judged in the pinned
checkout. `run` names the first one that is missing, then judges the bench's own items
through the same checkout (`wsb=`, default `$HBENCH_WSBENCH_PUB`); `--dry_run` prints the
`wsbench run` calls it would issue, drop the flag to issue them.

**The index**

```bash
uv run hbench index out=outputs/mylens \
    overrefusal_root=outputs/mylens/overrefusal/claude-opus-5-5/offtopic_index \
    benign_roleplay_root=outputs/mylens/benign_roleplay/index_root
```

`main`, `story` and `agentic_misalignment` are found under `out=`; the two safety roots are
always named. A family that was not judged is absent from the row, and the macro that needs
it is not computed. `hbench index --help` lists every root key.

Reader names: the over-refusal export, the role-play commands and the index's safety
families keep lists of reader names and skip a reader that is on none of them. Put yours on
them for every command of this section with `export HBENCH_EXTRA_READERS=<arm>`; without it
the reader is scored on `main`, `story` and `agentic_misalignment` only.

## 6. Hardware and time

Nothing in this repository uses a GPU: `hbench plan` takes under a second, `hbench readouts`
seconds, and the judges are API calls from a CPU machine. An 11-layer
`agentic_misalignment` file is about 555,000 rows; loading it takes about 1 GB of memory.

Producing is where the GPU time goes. The bench reads 1,265,440 cells over its five plans at
11 layers, plus WorkspaceBench's `jailbreak_recognition` family (838,827 cells) for the
shared role-play items: 2,104,267 cells, 21.5 GB of stored activations (float32).

Measured on NVIDIA B200 GPUs (183 GiB) with Qwen3.6-27B and an oracle lens of the same size,
one text sample per cell:

| producer | rate per GPU | `main` + `story` (66,924 cells) | everything above (2,104,267 cells) |
|---|---|---|---|
| lens served by an inference server from stored activations | about 78 cells/s | 0.24 GPU-hours (measured) | about 7 GPU-hours (projected from the rate) |
| lens decoded with `transformers`, one process | 12 to 15 cells/s | 1.5 GPU-hours (projected) | about 39 GPU-hours (projected) |

Wall time is GPU-hours divided by the GPUs used, plus a few minutes of start-up per
container; the served producer also merges the lens's adapter once (about 2 minutes on one
GPU) and fills the activation store once per plan, which every lens then shares.
