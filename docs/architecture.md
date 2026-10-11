# Architecture — Plant Operations Loop Agent

## The design in one paragraph

A shift's exceptions live on a board. Two nested loops work that board, and both loops are **controlled by code, not by the model**: the model is called for three bounded judgements per pass (diagnose, plan, critique) and everything else — which case is next, whether a plan is acceptable, when to stop, what happens to the record — is deterministic code with tests. The outer loop decides *what to work on*; the inner loop decides *when a case is done being thought about*; a human decides *what actually happens*.

## The board loop (outer)

```
        ┌──────────────────────────────────────────────────┐
        │                                                  │
        ▼                                                  │
  order (priority, code) ──► select next ──► run case loop ──► record at gate
                                │
                                ├─ queue empty ────────────► BOARD_CLEAR
                                └─ budget spent ───────────► SHIFT_BUDGET_EXCEEDED
                                                             + written handoff
```

- **Ordering** (`priority.py`): base weight by exception type — `safety_observation` 1000, `machine_down` 600, `quality_nonconformance` 400, `material_shortage` 250, `maintenance_alert` 150 — scaled by `1 + 0.25 × (line_criticality − 1)`, plus for `machine_down` half the minutes already down (capped). Ties break by `reported_at`, then case id. The formula and its reason string are stored on every record: anyone can recompute why a case went first.
- **Budget** (`BoardPolicy`): max cases and a token budget for the run, checked in code before each case. A case mid-loop is never cut off; the budget binds *between* cases.
- **Handoff**: on a budget exit, the unworked remainder is listed with each case's own reason ("shift case budget reached (3 cases)"), the worked cases are grouped by exit, and the summary states plainly that everything worked waits at the gate and nothing was actioned. The handoff is data on the run record, not a log line.
- Implementation: a LangGraph graph (`order → select → process → select … → handoff`) whose `process` node invokes the case graph per case with gate mode *record* (see below), persisting each record via callback as it completes.

## The case loop (inner)

```
 intake ─► screen ─► prepare ─┬─ safety_observation ──────────────► draft (escalation)
                              │
                              ▼
              ┌──► diagnose ─► plan ─► verify ──┐
              │                                 │ gaps, rounds left,
              │        focus = open gaps only   │ budget left
              └─────────────────────────────────┘
                                                │ exit decided in code
                                                ▼
                              draft (work order | escalation) ─► SUPERVISOR GATE ─► record
```

### Rounds

Each pass is three calls behind the `Reasoner` port (`ports.py`), each returning a strict schema (`schemas.py`), each validated in code with one bounded retry:

1. **Diagnose** → `Diagnosis`: ranked `Hypothesis` list. Contract: every hypothesis carries `evidence` (SOP clause, history match, or reading) and a `confidence`. Retrieval over the KB (SOPs + history, behind the `Retriever` port) is scoped by machine before the call.
2. **Plan** → `ActionPlan`: ordered `PlanStep`s built on the leading hypothesis's SOP, `required_parts` resolved against inventory (stock, or an approved substitute), estimated downtime. The SOP's mandatory safety steps are inserted by the planner as steps with `kind="safety"`, `mandatory=True` — and the verify round treats their absence as a blocking safety gap, so "the model forgot the lockout" cannot converge.
3. **Verify** → `Critique` (the model's own second look) **merged with code checks** (`guardrails.verify_plan`): `missing_safety_step`, `part_unavailable`, `unevidenced_hypothesis`, `sop_contradiction`, `banned_action`. Merge rule: code gaps win — where both flag the same `(kind, ref)`, the code version is kept; the model's critique can add gaps, never remove a code gap.

### Loop control (`loops.CaseLoopController`)

- After each verify, the controller computes the round's **gap delta** (opened/closed vs the previous round, keyed `kind:ref`) and records a `RoundRecord` — the loop timeline the console renders.
- On a loop-back, the next diagnose/plan receive **only the open gaps as focus**. Reasoners are expected to work the gaps (the offline rules engine applies discriminating checks for focused rival hypotheses; the model prompts carry the gap list verbatim). Focus is also where the Responses adapter's reasoning continuity pays off: round N+1 resumes round N's thread.
- Per-round token use is sliced from cumulative usage, so telemetry per round is exact, and the case budget is enforced on real numbers.

### Exit conditions (all in code, `decide_exit`)

Evaluated in this order after every verify round (and at prepare for the first):

| Exit | Condition | Draft |
|---|---|---|
| `safety_stop` | Case type is `safety_observation` (at prepare, round 0 — the case is never diagnosed), **or** any open gap is a safety-flagged `banned_action` | escalation |
| `blocked_external` | A `part_unavailable` gap names a part that is not stocked at all, or is stocked out with no *available* approved substitute — more rounds cannot fix a shelf | escalation |
| `converged` | No open gaps after verify | work order |
| `budget_exceeded` | Cumulative case tokens exceed `LOOP_CASE_TOKEN_BUDGET` with gaps still open | escalation |
| `exhausted` | Round count reaches `LOOP_MAX_ROUNDS` (default 3) with gaps still open | escalation, gaps named |

The ordering matters: a safety stop outranks a blocked shelf; a clean verify converges even if the budget is nearly spent; a hopeless part exits in round 1 instead of burning three rounds to reach the same escalation.

### The supervisor gate

Every case — converged or escalated, including safety stops — ends `awaiting_supervisor` with its draft. In the case graph this is a LangGraph **interrupt** before the decision node: the run pauses, the record persists, and `record_decision` either **resumes** the graph (checkpoint live) or **applies the decision directly** to the stored record (checkpoint gone — API restart, checkpoints off). Both paths produce the same record; the record is the source of truth. Board runs use *record* mode (no pause) because a board run must finish; the gate state is identical, only the pause differs.

The gate is a hard boundary, stated on every record (`GATE_NOTICE`): the agent never issues commands to machines, never marks work performed, never closes a case. A guardrail also screens the draft text itself for performed-work phrasing ("we replaced…", "back in production") — a draft that claims the work happened fails its own record's checks.

## After the gate: outcomes, live accuracy, the watch

The gate decision (approve/reject) is not the end of the record's life — it is the start of its evidence. Round 2 closes the loop:

- **Outcomes** (`POST /api/cases/{id}/outcome`, `SupervisorOutcome` on the record): the supervisor records the plan's disposition (`accepted` / `accepted_with_changes` / `rejected` — a separate axis from the gate decision: whether the *drafted plan* was the plan that got executed), the confirmed fault, work performed, actual downtime, parts used, resolved. Validation happens at the transport (decision enum; confirmed fault must be in the KB vocabulary, `kb.fault_classes()`; non-negative downtime), because the outcome is ground truth for everything below. An outcome is recorded once; a second is a 409, not an edit. It rides inside the case record's JSONB payload — no new table, and the record on file remains exactly the record on file.
- **Live diagnosis accuracy** (`stats.py`, `GET /api/stats/diagnosis`): a pure aggregation over records — the same code path serves both stores. A record enters fault scoring only when a fault was confirmed *and* the loop produced hypotheses (a safety stop confirms a fault with no diagnosis to score; it is counted separately as `confirmed_without_diagnosis`, never silently dropped). Metrics: top-1/top-2 accuracy, per-class precision/recall with counts, mean rounds over confirmed convergences, plan-acceptance counts/shares, downtime error (signed mean = bias, MAE = accuracy). With zero outcomes the payload is `available: false` with an explanatory note — honest empty states over flattering zeroes.
- **Shift handoff** (`handoff.py`, `GET /api/board/handoff`, `?format=text` for plain text): composed entirely in code from the current records — awaiting cases in board order, escalations with open gaps, blocked-external cases with their unsatisfied parts, safety stops, open reliability items, and the latest board run with its budget-cut remainder. The JSON and the text are rendered from the same dict, so the shift-log paste cannot drift from the API. No model writes any part of it: a handoff is a safety-adjacent document, and an invented gap (or a dropped safety stop) is worse than a terse one.
- **Recurring-fault watch** (`reliability.py`, `ReliabilityPolicy` in `loops.py`): invoked from the service wherever a case reaches the gate (`_persist_case`, used by both the single-case path and the board graph's result callback) and again on outcome recording. The fault a case counts as is its *effective* class — confirmed when an outcome carries one, else the top diagnosis — so a confirmation can complete a pattern or dissolve one. Counting is a rolling window over record timestamps anchored at the check moment; at the threshold (default 3 in 14 days) the watch persists one `reliability_review` `BoardItem` (its own table, migration `0003` — an item is not a case and has no loop record) with the contributing case ids as evidence, deduplicated on (machine, fault class) while an item is open. The item is the entire action: it appears on the board and in the handoff, and triggers nothing automatic.

## Guardrails as code

| Guardrail | Where | Effect |
|---|---|---|
| Evidence required | critique contract + code gap `unevidenced_hypothesis` | a hypothesis with empty evidence can never converge |
| Mandatory safety steps | `guardrails.verify_plan` vs the SOP's code-side list (matched by step id or ≥0.55 token overlap, so rewording can't smuggle one out) | missing step = blocking safety gap |
| Banned actions | `screening.screen_plan_step` over every plan step (interlock/guard bypass, defeating a safety device, hot work on live equipment) | safety gap → `safety_stop` the same round |
| Injection screen | `screening.screen_text` on title/description/notes before anything else: instruction-shaped spans (incl. "skip the lockout…") are struck, flagged on the record, and the surviving text is fenced as untrusted data in prompts | planted instructions never reach a plan |
| Parts reality | planner resolution + `part_unavailable` gap + blocked-exit rule | a plan can only promise what the shelf holds |
| Draft language | `guardrails.draft_language_ok` | no performed-work claims in drafts |

## Backends behind the port

`backends.get_reasoner` selects by `REASONING_BACKEND`. All model-backed reasoners share `StructuredReasoner` (prompts, schema validation, one bounded retry, usage/latency accounting, translated provider errors — a dead provider fails the run loudly and is written to the error log; nothing is ever silently re-routed to another backend):

- **`MuseSparkReasoner`** (`muse`): Muse Spark 1.3 via the OpenAI SDK against `https://api.meta.ai/v1` (Chat Completions). Refuses at construction without `MODEL_API_KEY`.
- **`MuseResponsesReasoner`** (`muse` + `MODEL_SURFACE=responses`): the Responses surface with `include=["reasoning.encrypted_content"]`. The per-case `session` dict on `CaseContext` accumulates reasoning items across rounds; each call replays them ahead of the new prompt. Without this, a multi-round loop restarts the model's reasoning every round — the continuity is the reason the surface exists in this blueprint.
- **`LLMReasoner`** (`llm` / provider aliases): Anthropic Messages, or any OpenAI-compatible endpoint (which includes NVIDIA NIM, hosted or self-hosted, via `OPENAI_BASE_URL`), or Ollama.
- **`OfflineReasoner`** (`offline`, default): deterministic SOP-threshold rules with a small signal function per fault class, discriminating checks for focused rivals, keyword fallback, and an honest `unknown_fault` when nothing matches — which the loop then escalates as `exhausted` rather than dressing up. It exists so tests/evals are hermetic; it is labelled a plumbing backend everywhere it appears.

Cost estimation uses a small price table (`backends._PRICE_PER_MTOK`): Muse Spark standard tier $1.25/M in · $4.25/M out (contributor tier priced separately); unknown models report no estimate rather than a guessed one.

## Persistence & topology

- **Store port** → `PostgresStore` (records, board runs, board items, error log as JSONB rows) or `InMemoryStore` (test double). Board/metrics/stats/handoff read models are computed from records in both.
- **Checkpointer**: Postgres saver when `DATABASE_URL` is set, in-memory saver otherwise, `CHECKPOINTS=off` to disable (gate decisions then always take the direct-apply path).
- **Vector index** for KB retrieval: pgvector table (`migrations/0002`), dimension-free column so embedder swaps don't migrate; in-memory cosine index as the double. Embedder: hashing (offline) or a provider `/embeddings` endpoint.
- **Object store**: raw case payloads archived to S3 (compact stdlib SigV4 client — the blueprint's only outbound storage call) or memory.
- **Migrations**: numbered SQL in `migrations/`, applied in order on first Postgres connection by store and index alike, recorded in `schema_migrations`; `MIGRATIONS_DIR` overrides the location (the Docker image ships them at `/app/migrations`).
- **API**: FastAPI, JSON logs with request IDs, `/health` + `/readiness` (503 when Postgres is configured but unreachable; explicit test-double labelling otherwise), optional `API_KEY` gate on non-probe routes. The console is one static HTML file with no dependencies.

## Testing the loops themselves

The eval pack and suite are designed around the loop semantics, not just outputs: scripted-critique controller tests isolate exit logic from any reasoner; defect planners (drop-safety / wrong-part / banned-step) prove the verify round catches realistic component failures; board evals pin the priority order and both budget cuts; the Muse adapter tests pin the wire shape and the reasoning-continuity round-trip against stubbed transports, which is the strongest claim available before a live key exists.
