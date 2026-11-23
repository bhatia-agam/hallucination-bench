# Judge backends: live and Message Batches

Claude judge calls (`claude-*` models) can travel two ways. Both send the same requests and fill
the same caches, so a run can start on one and finish on the other.

| backend | how | price | speed |
|---|---|---|---|
| `live` (default) | one Messages request per call, `concurrency=` at a time | list price | bounded by `concurrency=`, `rpm=` and the account's rate limits |
| `batches` | waiting calls are gathered into [Message Batches](https://platform.claude.com/docs/en/build-with-claude/batch-processing) | half price on every token | outside the live rate limits; each batch ends when the API gets to it: from 1 to over 20 minutes in our checks, at most 24 h, and every stage of a run waits for its own batch |

## Turn it on

```bash
uv run hbench run arms=mylens judge_backend=batches concurrency=2000
# or, for every command that builds a judge client:
export HBENCH_JUDGE_BACKEND=batches
```

Leave it out (or `judge_backend=live`) for today's behaviour. From Python:
`LLM(backend="batches")`.

`hbench run` keeps `2 x concurrency=` cells in flight, and a cell waits while its call is in a
batch, so under `batches` give `concurrency=` a large value (1,000 to 5,000): that many cells
share each round. `rpm=` and `concurrency=` still bound the calls that run live.

## What changes and what does not

- **Same requests.** A batch entry is the live call's request (`llm.anthropic_params`: model,
  `max_tokens`, the cached system block, messages, `output_config`), minus the HTTP `timeout`.
  Long joint calls are not streamed in a batch (they do not need to be).
- **Same cache fingerprints and files.** `cache.jsonl` rows written by either backend are hits
  for the other; rerun into the same `out=` to resume.
- **Stages still run in order.** A cell's map call, then its verifier calls, then any re-ask
  each wait for a batch: one round of batches per stage.
- **How calls are gathered.** A call waits until no new call has arrived for 2 s, then
  everything waiting goes out together. A round with fewer than 20 calls runs live instead.
- **Anything a batch does not settle is re-run live**, with the live path's retries: errored,
  expired, canceled and missing entries and unparsable bodies. A refusal is the call's answer,
  as live (the caller's fallback-model rule then applies).
- **Live only:** OpenRouter models and preflights.
- **Spend.** Batch calls are tallied at half the list price in `usage.usd`.

## Resume without paying twice

Set a journal directory and keep it with the run's outputs:

```bash
export HBENCH_JUDGE_BATCH_DIR=outputs/.judge_batches
```

Every submitted batch is recorded there. If the process stops while a batch is running (a
revoked token, a killed container), the next run with the same directory collects that batch
instead of sending its requests again. Without the variable an interrupted run re-sends them.

## Settings

| variable | default | meaning |
|---|---|---|
| `HBENCH_JUDGE_BACKEND` | `live` | `live` or `batches` |
| `HBENCH_JUDGE_BATCH_DIR` | unset | journal directory (see above) |
| `HBENCH_JUDGE_BATCH_MAX_REQUESTS` | 5000 | requests per batch (the API allows 100,000) |
| `HBENCH_JUDGE_BATCH_MAX_BYTES` | 64000000 | request bytes per batch (the API allows 256 MB) |
| `HBENCH_JUDGE_BATCH_MIN_REQUESTS` | 20 | a smaller round runs live |
| `HBENCH_JUDGE_BATCH_LINGER_SECONDS` | 2 | quiet time before the waiting calls are sent |
| `HBENCH_JUDGE_BATCH_POLL_SECONDS` | 20 | longest wait between status checks |

## Sizing the live backend

`concurrency=` (default 64) and `rpm=` (default 1500) set the live throughput; neither is
hardcoded anywhere in the package. Size them to the account, not to a habit:

1. Read the account's limits from any response's `anthropic-ratelimit-*` headers (requests,
   input tokens and output tokens a minute; cached input does not count as input).
2. Output tokens a minute (OTPM) is the limit a judge run meets first:
   `calls in flight x output tokens per call x 60 / seconds per call`, summed over every process
   on the account, must stay under it.
3. Keep `rpm=` above `concurrency x 60 / seconds per call`, or the pacer becomes the limit.

Measured on the x3 main bank with `claude-opus-5-5` (50 map + 50 verifier calls,
0 retries and 0 rate-limit errors at either setting):

| call | uncached input | cached input | output | seconds |
|---|---|---|---|---|
| map (`effort=low`) | 1,080 tok | 13,240 tok | 710 tok | 13 |
| verifier (`effort=medium`) | 860 tok | 1,430 tok | 330 tok | 7 |

So one in-flight map call uses about 3,300 OTPM. On an account with 15M OTPM, 150,000 requests a
minute and 100M input tokens a minute, `concurrency=16 rpm=600` uses under 1% of the output
limit. Recommended there: `concurrency=256 rpm=6000` per process (about 6% of OTPM), and
keep the sum over all processes on the account under about 2,000 calls in flight (half the
limit). 64 was the largest value tested; go up in steps and watch `retries=` in the stage log.
The first `concurrency=` calls of a cold run each write the cached system prompt (they start
before any of them has filled the cache); that is paid once per run.

## Credentials the environment cannot carry

`llm.use_anthropic_client(factory)` makes every Anthropic client with `factory()` (an
`AsyncAnthropic`), e.g. one built with workload identity federation and a token provider.
