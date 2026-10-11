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
                    verify's parts pass runs in code inside the
                    verify stage: shelf truth → KB substitute search
                    → replan around a passing substitute, or the
                    search is documented and the case blocks
                                                │ exit decided in code
                                                ▼
                              draft (work order | escalation) ─► SUPERVISOR GATE ─► record
```

### Rounds

Each pass is three calls behind the `Reasoner` port (`ports.py`), each returning a strict schema (`schemas.py`), each validated in code with one bounded retry:

1. **Diagnose** → `Diagnosis`: ranked `Hypothesis` list. Contract: every hypothesis carries `evidence` (SOP clause, history match, or reading) and a `confidence`. Retrieval over the KB (SOPs + history, behind the `Retriever` port) is scoped by machine before the call. The context also carries the machine's **merged history** (`CaseContext.history`, assembled by `history.py`): the KB's recorded entries plus the supervisor-confirmed write-backs for that machine, confirmed first, every entry marked with its `source` — and the same entries ride the prompt context as chunks, marked in text, for the model backends. The machine's own history is context by right, not by retrieval luck.
2. **Plan** → `ActionPlan`: ordered `PlanStep`s built on the leading hypothesis's SOP, `required_parts` named from the SOP, estimated downtime. The SOP's mandatory safety steps are inserted by the planner as steps with `kind="safety"`, `mandatory=True` — and the verify round treats their absence as a blocking safety gap, so "the model forgot the lockout" cannot converge. Parts are deliberately *not* resolved here: a planner's stock claims (including any substitute it names) are treated as unverified claims, and the verify stage re-measures them.
3. **Verify** → `Critique` (the model's own second look) **merged with code checks** (`guardrails.verify_plan`): `missing_safety_step`, `part_unavailable`, `unevidenced_hypothesis`, `sop_contradiction`, `banned_action`. Merge rule: code gaps win — where both flag the same `(kind, ref)`, the code version is kept; the model's critique can add gaps, never remove a code gap. First, though, the verify stage runs its **parts pass in code** (`substitution.py`):

   - **Shelf truth.** Every requirement is re-measured against inventory — `available_qty` and `satisfied` come from the shelf, never from the planner.
   - **The search.** For a requirement the shelf cannot satisfy, the KB's compatibility relation for that slot is walked in order. Each entry carries the pairing's basis and its spec deltas; SOPs carry `part_constraints` for slots where physics demands them (the COMP-7 fan slot requires airflow ≥ 420 m³/h). A candidate that is not stocked, is short of the required quantity, violates a constraint, or lacks the spec to even be checked is **rejected**, with the reason named.
   - **Replanning.** The first candidate that clears shelf and constraints is applied: plan steps are revised onto it, the requirement is satisfied *by substitute*, and a `Substitution` record (original, substitute, basis, spec deltas, outcome) rides on the plan. The code checks then measure the *revised* plan, and a critic's part gap for a part the pass resolved is retired as stale. When nothing passes, the surviving `part_unavailable` gap's detail is rewritten to document the search.

### Loop control (`loops.CaseLoopController`)

- After each verify, the controller computes the round's **gap delta** (opened/closed vs the previous round, keyed `kind:ref`) and records a `RoundRecord` — the loop timeline the console renders.
- On a loop-back, the next diagnose/plan receive **only the open gaps as focus**. Reasoners are expected to work the gaps (the offline rules engine applies discriminating checks for focused rival hypotheses — readings first, then the plant's confirmed memory: a supervisor-confirmed precedent for the leading fault separates a rival the readings could not, re-scored to the ruled-out level with the precedent named in its evidence; the model prompts carry the gap list verbatim). Focus is also where the Responses adapter's reasoning continuity pays off: round N+1 resumes round N's thread.
- Per-round token use is sliced from cumulative usage, so telemetry per round is exact, and the case budget is enforced on real numbers.

### Exit conditions (all in code, `decide_exit`)

Evaluated in this order after every verify round (and at prepare for the first):

| Exit | Condition | Draft |
|---|---|---|
| `safety_stop` | Case type is `safety_observation` (at prepare, round 0 — the case is never diagnosed), **or** any open gap is a safety-flagged `banned_action` | escalation |
| `blocked_external` | A `part_unavailable` gap survives whose part the verify stage's own search could not cover — no substitute listed, none in stock, or every candidate rejected by the SOP's constraints — more rounds cannot fix a shelf; the escalation documents the search | escalation |
| `converged` | No open gaps after verify | work order |
| `budget_exceeded` | Cumulative case tokens exceed `LOOP_CASE_TOKEN_BUDGET` with gaps still open | escalation |
| `exhausted` | Round count reaches `LOOP_MAX_ROUNDS` (default 3) with gaps still open | escalation, gaps named |

The ordering matters: a safety stop outranks a blocked shelf; a clean verify converges even if the budget is nearly spent; a hopeless part exits in round 1 instead of burning three rounds to reach the same escalation.

### The supervisor gate

Every case — converged or escalated, including safety stops — ends `awaiting_supervisor` with its draft. In the case graph this is a LangGraph **interrupt** before the decision node: the run pauses, the record persists, and `record_decision` either **resumes** the graph (checkpoint live) or **applies the decision directly** to the stored record (checkpoint gone — API restart, checkpoints off). Both paths produce the same record; the record is the source of truth. Board runs use *record* mode (no pause) because a board run must finish; the gate state is identical, only the pause differs.

The gate is a hard boundary, stated on every record (`GATE_NOTICE`): the agent never issues commands to machines, never marks work performed, never closes a case. A guardrail also screens the draft text itself for performed-work phrasing ("we replaced…", "back in production") — a draft that claims the work happened fails its own record's checks.

**The work-order packet** (`packet.py`, `GET /api/cases/{id}/packet`, JSON or `?format=text`): the floor copy of a gated case, composed in code from the record only — case summary, diagnosis with evidence, plan steps in order, the mandatory safety steps again as a check-off list, the parts with any substitution flagged (basis and spec deltas) and each part marked when it stands claimed for this case on its run, estimated downtime, and blank sign-off lines (supervisor / technician / completion with actual downtime and parts used). Only a converged case packets — a non-converged exit or a rejected draft raises a 409, because an unverified plan must not reach the floor dressed as a work order. An awaiting case packets under its draft banner; an approved one carries the approval on its face.

## The shared shelf: allocation (`allocation.py`)

Until round 5, the verify stage's parts pass measured the shelf per case, independently — correct for a lone case, wrong for a board. Two cases could both plan around the same last unit of a part, both converge, and the conflict surfaced at the workbench. Round 5 makes the board plan against one shelf:

- **Claims.** When a record is persisted at the gate (`service._persist_case` — the one funnel every gated record passes through, board graph and single runs alike), a converged record claims one entry per satisfied requirement, on the part actually drawn (the substitute, when the plan runs on one; the requirement it satisfies is kept on the claim). Cases reach the gate in board order, so claims follow the board's priority by construction — a case can only ever lose a part to an earlier, higher-priority case. The claim id derives from case + drawn part, so re-persisting a record claims once. Only converged records claim: `safety_stop` records in particular never hold a part, and the claim machinery sits entirely downstream of the loops, so it cannot delay or alter a safety path.
- **Effective stock.** The case controller reads shelf truth through two providers the service binds to the ledger (the same binding idiom as the merged-history provider): a stock provider (`effective_qty` = shelf − outstanding claims) and a claim-note provider. Both the parts pass and the substitute search — and the blocked-exit recheck, so the exit decision can never disagree with the pass — measure through them. Unbound (plain-code controller use, unit tests), the providers default to the bare shelf and behaviour is exactly the pre-round-5 behaviour.
- **Naming the claim.** When a shortfall is a claim's doing — the bare shelf could satisfy the requirement, the effective shelf cannot — the blocked gap and any substitute rejection carry the note `claimed by <case>, higher priority on this board` (for a claim left standing by an earlier run, the note names that run instead). The claimer is computed by walking the part's claims in recording order to the one whose quantity first makes the remainder insufficient. A shortfall the bare shelf also cannot satisfy keeps the plain stockout wording: the ledger never blames a claim for a shortage it did not cause. The shift handoff reads the claimer back out of the blocked record's gap text, so it reports the truth of the run that blocked even after the claim has been released.
- **Release.** Gate decisions land in `service.record_decision`, on both the resume and direct-apply paths: a rejection releases the case's standing claims, each stamped `released_by_gate_rejection` with the time and the deciding supervisor; an approval leaves them standing, because the approved work order will draw the parts. Released quantities return to the effective pool for *future* runs and re-plans only — a finished board run is a record and is never recomputed, so the neighbour that blocked on the claim keeps its blocked record until a fresh run re-plans it. That asymmetry is deliberate: the alternative (silently unblocking a case nobody re-verified) would put an unverified plan back in play.
- **Storage.** Claims are their own store collection (migration `0005_part_claims.sql`), following the board-items/history-entries precedent: the ledger is plant-wide runtime state that later runs must read and a later gate request must mutate, so scattering it across case payloads would make "what is outstanding?" a scan of other records' contents. The full claim rides as a JSONB payload with promoted `case_id` / `part_id` / `status` columns; the in-memory double keeps the same shape.
- **Boundary.** A claim is a planning reservation on the record. It is not a warehouse transaction, it draws no stock, and it changes nothing about the gate: the agent still never marks work performed, and the packet's claim marking exists so the technician sees the unit is spoken for — by this work order, against the rest of the board.

Surfaces: the board run record and its API response carry the run's claims; `GET /api/allocations` returns the whole ledger; the console's board view renders both; the packet marks claimed parts; the handoff names allocation-blocked cases. Diagnosis stats do not read claims — allocation is a planning fact, not an accuracy fact.

## After the gate: outcomes, live accuracy, the watch

The gate decision (approve/reject) is not the end of the record's life — it is the start of its evidence. Round 2 closes the loop:

- **Outcomes** (`POST /api/cases/{id}/outcome`, `SupervisorOutcome` on the record): the supervisor records the plan's disposition (`accepted` / `accepted_with_changes` / `rejected` — a separate axis from the gate decision: whether the *drafted plan* was the plan that got executed), the confirmed fault, work performed, actual downtime, parts used, resolved. Validation happens at the transport (decision enum; confirmed fault must be in the KB vocabulary, `kb.fault_classes()`; non-negative downtime), because the outcome is ground truth for everything below. An outcome is recorded once; a second is a 409, not an edit. It rides inside the case record's JSONB payload — no new table, and the record on file remains exactly the record on file.
- **The learning loop** (`history.py`): a confirmed fault does not stay trapped in the record that earned it — it is written back to the machine's history as a `HistoryEntry`, where future diagnoses read it. The seam follows the repo's storage idiom: the packaged KB is authored, versioned ground truth and is never appended to at runtime, so confirmations — runtime state — get their own store table (migration `0004`, JSONB payload like board items), keyed `hist-<case_id>` so the once-per-case outcome makes the write idempotent by construction. Only a confirmed fault writes, under any disposition (a rejected plan with a confirmed fault still teaches the fault; an outcome confirming nothing writes nothing). A safety stop refuses a confirmed fault in code, computed *before* the outcome persists, so the refusal is total — no outcome on the record, no entry in memory — because a safety case has no diagnosis to confirm. The diagnoser reads one merged view (confirmed first, sources marked): the offline signals cite the first matching entry through the same evidence path as always — now a confirmation is cited as the floor's own finding — and, on a focused loop-back, a supervisor-confirmed precedent for the leading fault separates a rival the readings could not (the same re-score a readings discriminator performs; recorded history never separates). The stats payload gains a `learning_loop` section — write-back counts by machine, and repeat-fault top-1 (scored cases whose machine + fault an earlier case's outcome had already confirmed) beside the overall top-1, both plain, with an explicit note when there are no repeats yet — and the handoff names the confirmed faults the next shift inherits.
- **Live diagnosis accuracy** (`stats.py`, `GET /api/stats/diagnosis`): a pure aggregation over records — the same code path serves both stores. A record enters fault scoring only when a fault was confirmed *and* the loop produced hypotheses (a record carrying a confirmed fault but no hypotheses is counted separately as `confirmed_without_diagnosis`, never silently dropped). Metrics: top-1/top-2 accuracy, per-class precision/recall with counts, mean rounds over confirmed convergences, plan-acceptance counts/shares, downtime error (signed mean = bias, MAE = accuracy), and the learning-loop section above. With zero outcomes the payload is `available: false` with an explanatory note — honest empty states over flattering zeroes.
- **Shift handoff** (`handoff.py`, `GET /api/board/handoff`, `?format=text` for plain text): composed entirely in code from the current records — awaiting cases in board order, escalations with open gaps, blocked-external cases with their unsatisfied parts, safety stops, open reliability items, the supervisor-confirmed faults written back to machine history, and the latest board run with its budget-cut remainder. The JSON and the text are rendered from the same dict, so the shift-log paste cannot drift from the API. No model writes any part of it: a handoff is a safety-adjacent document, and an invented gap (or a dropped safety stop) is worse than a terse one.
- **Recurring-fault watch** (`reliability.py`, `ReliabilityPolicy` in `loops.py`): invoked from the service wherever a case reaches the gate (`_persist_case`, used by both the single-case path and the board graph's result callback) and again on outcome recording. The fault a case counts as is its *effective* class — confirmed when an outcome carries one, else the top diagnosis — so a confirmation can complete a pattern or dissolve one. Counting is a rolling window over record timestamps anchored at the check moment; at the threshold (default 3 in 14 days) the watch persists one `reliability_review` `BoardItem` (its own table, migration `0003` — an item is not a case and has no loop record) with the contributing case ids as evidence, deduplicated on (machine, fault class) while an item is open. The item is the entire action: it appears on the board and in the handoff, and triggers nothing automatic.

## Guardrails as code

| Guardrail | Where | Effect |
|---|---|---|
| Evidence required | critique contract + code gap `unevidenced_hypothesis` | a hypothesis with empty evidence can never converge |
| Mandatory safety steps | `guardrails.verify_plan` vs the SOP's code-side list (matched by step id or ≥0.55 token overlap, so rewording can't smuggle one out) | missing step = blocking safety gap |
| Banned actions | `screening.screen_plan_step` over every plan step (interlock/guard bypass, defeating a safety device, hot work on live equipment) | safety gap → `safety_stop` the same round |
| Injection screen | `screening.screen_text` on title/description/notes before anything else: instruction-shaped spans (incl. "skip the lockout…") are struck, flagged on the record, and the surviving text is fenced as untrusted data in prompts | planted instructions never reach a plan |
| Parts reality | verify-stage parts pass: shelf truth for every requirement + the KB substitute search with SOP constraints (`substitution.py`), then the `part_unavailable` gap + blocked-exit rule | a plan can only promise what the shelf holds — directly, or via a substitute that was searched, checked and recorded |
| One shelf per board | allocation ledger (`allocation.py`): converged plans claim their drawn parts; every measurement reads the shelf net of standing claims | two cases on one board can no longer both plan around the same last unit — the loser blocks with the claiming case named, and a gate rejection releases the claim for future runs |
| Draft language | `guardrails.draft_language_ok` | no performed-work claims in drafts |

## Backends behind the port

`backends.get_reasoner` selects by `REASONING_BACKEND`. All model-backed reasoners share `StructuredReasoner` (prompts, schema validation, one bounded retry, usage/latency accounting, translated provider errors — a dead provider fails the run loudly and is written to the error log; nothing is ever silently re-routed to another backend):

- **`MuseSparkReasoner`** (`muse`): Muse Spark 1.3 via the OpenAI SDK against `https://api.meta.ai/v1` (Chat Completions). Refuses at construction without `MODEL_API_KEY`.
- **`MuseResponsesReasoner`** (`muse` + `MODEL_SURFACE=responses`): the Responses surface with `include=["reasoning.encrypted_content"]`. The per-case `session` dict on `CaseContext` accumulates reasoning items across rounds; each call replays them ahead of the new prompt. Without this, a multi-round loop restarts the model's reasoning every round — the continuity is the reason the surface exists in this blueprint.
- **`LLMReasoner`** (`llm` / provider aliases): Anthropic Messages, or any OpenAI-compatible endpoint (which includes NVIDIA NIM, hosted or self-hosted, via `OPENAI_BASE_URL`), or Ollama.
- **`OfflineReasoner`** (`offline`, default): deterministic SOP-threshold rules with a small signal function per fault class, discriminating checks for focused rivals, keyword fallback, and an honest `unknown_fault` when nothing matches — which the loop then escalates as `exhausted` rather than dressing up. It exists so tests/evals are hermetic; it is labelled a plumbing backend everywhere it appears.

Cost estimation uses a small price table (`backends._PRICE_PER_MTOK`): Muse Spark standard tier $1.25/M in · $4.25/M out (contributor tier priced separately); unknown models report no estimate rather than a guessed one.

## Persistence & topology

- **Store port** → `PostgresStore` (records, board runs, board items, history entries, part claims, error log as JSONB rows) or `InMemoryStore` (test double). Board/metrics/stats/handoff read models are computed from records in both.
- **Checkpointer**: Postgres saver when `DATABASE_URL` is set, in-memory saver otherwise, `CHECKPOINTS=off` to disable (gate decisions then always take the direct-apply path).
- **Vector index** for KB retrieval: pgvector table (`migrations/0002`), dimension-free column so embedder swaps don't migrate; in-memory cosine index as the double. Embedder: hashing (offline) or a provider `/embeddings` endpoint.
- **Object store**: raw case payloads archived to S3 (compact stdlib SigV4 client — the blueprint's only outbound storage call) or memory.
- **Migrations**: numbered SQL in `migrations/`, applied in order on first Postgres connection by store and index alike, recorded in `schema_migrations`; `MIGRATIONS_DIR` overrides the location (the Docker image ships them at `/app/migrations`).
- **API**: FastAPI, JSON logs with request IDs, `/health` + `/readiness` (503 when Postgres is configured but unreachable; explicit test-double labelling otherwise), optional `API_KEY` gate on non-probe routes. The console is one static HTML file with no dependencies.

## Testing the loops themselves

The eval pack and suite are designed around the loop semantics, not just outputs: scripted-critique controller tests isolate exit logic from any reasoner; defect planners (drop-safety / wrong-part / banned-step) prove the verify round catches realistic component failures; board evals pin the priority order and both budget cuts; the learning entries run a case, record the outcome, and re-run a similar case against the same store, so the write-back is the only variable; the allocation entries run two cases at the shelf's only belt — contention, the named claim, the gate release, and a fresh run converging — against the same store for the same reason; the Muse adapter tests pin the wire shape and the reasoning-continuity round-trip against stubbed transports, which is the strongest claim available before a live key exists.
