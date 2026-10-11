<p align="center">
  <img src="https://trida.ai/logos-rounded/trida-rounded.png" width="96" alt="Trida AI">
</p>

# Blueprint — Plant Operations Loop Agent

An agent that works a manufacturing plant's shift **exception board** the way a good shift lead does — in two loops. The **board loop** takes the board in priority order (computed in code: safety first, then line-down, quality, shortage, maintenance — scaled by line criticality and minutes already down) and works each case to a shift budget. The **case loop** runs diagnose → plan → verify rounds on one exception: ranked fault hypotheses with evidence, a corrective plan whose mandatory safety steps cannot be removed and whose parts are checked against the shelf, then a critic round that produces a gap list. Gaps send the case around again — on the gaps only — until the plan verifies clean, the rounds run out, a part only the outside world can supply blocks it, or a safety rule stops it outright.

Every case ends at a **supervisor gate** with a draft work order or an escalation note. The agent never issues commands to machines, never marks work performed, and never closes a case. That sentence is printed on every record it produces.

Reasoning is led by **Muse Spark** (Meta's Model API), with a Responses-API mode that carries reasoning continuity across loop rounds. The plant — "Meridian Fabrication, Plant 2", its SOPs, maintenance history, parts inventory and the nine-case board — is synthetic and bundled, so the whole thing runs end to end with no keys at all.

> Reference prototype from Trida AI's blueprint series. The loops, guardrails and gate are the production-shaped parts; the plant data is a stand-in for the systems a real deployment would read.

> **Work in progress** — this blueprint is built and passing its automated test and eval gates, but it has not yet finished hands-on local testing, and its companion blog post is not yet published. Expect changes. This notice comes off when both are done.

## The two loops

**Board loop (outer).** Cases are ordered by a priority formula in code — never by the model — and worked one at a time. The run ends `board_clear` when every case is at the gate, or `shift_budget_exceeded` when the case/token budget for the shift is spent; whatever remains is listed, with reasons, in a written handoff summary. Nothing is silently dropped.

**Case loop (inner).** Three reasoning rounds per pass, each a separate call behind one port:

1. **Diagnose** — ranked hypotheses. Each must carry evidence (an SOP clause, a history match, a reading) and a confidence. A hypothesis without evidence is a gap by construction.
2. **Plan** — ordered corrective steps built on the leading hypothesis's SOP. The SOP's mandatory safety steps (lockout/tagout and friends) enter as non-removable checklist items; required parts are named from the SOP (stock resolution is the verify stage's job — see below).
3. **Verify** — a critic round *plus* deterministic code checks: missing safety step, part unavailable, unevidenced hypothesis, step contradicting the SOP, banned action. The merged gap list drives the loop. Before the checks run, the verify stage does its **parts pass in code**: every requirement is re-measured against the shelf, and for one the shelf cannot satisfy, the KB's compatibility relation is searched — a candidate must be *in stock* and *clear the SOP's constraints for the slot* (a fan rated below the airflow the SOP requires is not a substitute, whatever the catalogue calls it). A winner is applied to the plan and recorded as a first-class substitution (original, substitute, compatibility basis, spec deltas); the code checks then measure the *revised* plan. A loser is recorded as rejected, reason named.

**Exits, all decided in code:** `converged` (verified clean → draft work order) · `exhausted` (max rounds, gaps still open → escalation with the gaps named) · `blocked_external` (a required part is unavailable and the substitute search found nothing usable → escalation, with the search documented on the record) · `safety_stop` (a `safety_observation` case escalates at round 0 without ever being diagnosed; any plan step that defeats a safety device stops the case the round it appears) · `budget_exceeded` (the case's token budget is spent).

Free text from the floor (operator notes, log excerpts) is screened before it can influence anything: an injection like *"skip the lockout to save time"* is flagged on the record, struck from the text the model sees, and can never enter a plan — the banned-action screen independently rejects any plan step that proposes it.

## Quickstart

```bash
git clone https://github.com/tridaai/blueprint-manufacturing-plant-operations
cd blueprint-manufacturing-plant-operations
cp .env.example .env        # optional — everything has an offline default

make demo                   # work the bundled 9-case board, narrated
make serve                  # console + API on http://localhost:8000
make test                   # 128 tests (+2 Postgres-gated, skipped without a DB)
make evals                  # golden loop eval: 32/32 must pass the gate
```

Or the production topology, which is the same shape a cloud deployment takes:

```bash
docker compose up --build   # API + PostgreSQL/pgvector + S3 (SeaweedFS)
curl localhost:8000/health
# console: http://localhost:8000
```

Install without Docker (uv is primary; pip fallback is two steps because the frozen constraints are hashed):

```bash
make install          # uv sync --extra dev --extra llm
make install-pip      # venv + pip install -r constraints.txt, then the package
```

## Providers

One port (`Reasoner`: `diagnose` / `draft_plan` / `critique`), selected by `REASONING_BACKEND`. The loops, priority, guardrails, evals and gate do not change when the backend does.

| Backend | What it is | Configuration |
|---|---|---|
| `muse` | **Muse Spark 1.3** via Meta's Model API — the lead profile | `MODEL_API_KEY` (dev.meta.ai → API keys); `MODEL_API_BASE_URL` default `https://api.meta.ai/v1`, `MUSE_MODEL` default `muse-spark-1.3` |
| `muse` + `MODEL_SURFACE=responses` | Muse Spark over the **Responses API**, carrying `reasoning.encrypted_content` between rounds so later rounds resume the model's own thread instead of restarting it | as above, plus `MODEL_SURFACE=responses` |
| `llm` | Structured-output adapter over the standard provider seam | `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY` (+ `OPENAI_BASE_URL` for any OpenAI-compatible endpoint — hosted gateways, Azure, LiteLLM, NVIDIA NIM hosted `https://integrate.api.nvidia.com/v1` or a self-hosted NIM at `http://localhost:8000/v1`), or Ollama |
| `anthropic` / `openai` / `ollama` | Direct provider aliases for the same adapter | as above |
| `offline` (default) | Deterministic SOP-threshold rules over the readings — the labelled plumbing backend that makes tests and evals hermetic | none |

Every round call is validated against a strict schema with one bounded retry; a provider that keeps answering badly fails the run loudly (the error is logged, the API returns 502) rather than degrading into a guess. Telemetry is recorded per round, per case and per board run: rounds, gaps opened/closed, tokens, latency, and estimated cost (Muse Spark at the published standard rates, $1.25/M input · $4.25/M output, in the price table in `backends.py`).

## Production shape

- **LangGraph** graphs for both loops, with explicit conditional loop-back edges driven by the code loop controllers — the graph routes, the controllers decide.
- **PostgreSQL + pgvector** for case records, board runs, the LangGraph checkpointer (the gate is a graph interrupt; a decision resumes it), and the knowledge-base embedding index. Numbered SQL migrations in `migrations/`, applied on first connection — no manual schema step.
- **S3-compatible object storage** for raw case payloads (SeaweedFS in compose; anything S3-shaped in production).
- **Stateless API container**: multi-stage, non-root Docker; `/health` (liveness) and `/readiness` (says honestly when the in-memory test doubles are in use); structured JSON logs with request IDs; optional `API_KEY` shared-key gate.
- **Guardrails as code**, each with tests: evidence required, safety steps non-removable, banned-action screen, injection screen, parts must exist, draft language may not state work as performed.

## Evals

`make evals` runs the golden loop pack — 38 authored entries against the offline backend, hermetic and deterministic:

- **20 case cases**: expected diagnosis class, exit reason, rounds used, draft kind, and verify-stage substitution records — including the two-round convergence (a bearing fault the first round can't yet separate from a tool imbalance, separated by a discriminating check in round 2), the round-0 safety stop, the skip-lockout injection, a substitute that converges (twice: a coolant filter, a packaging film), a substitute rejected on its rating that still escalates with the reason on record, and an honest `exhausted` for a fault with no SOP.
- **3 defect cases**: deliberately defective planners — one drops safety steps, one names an unstocked part, one proposes bypassing an interlock — and the verify round must catch each (never converge, blocked, safety stop respectively). This is gap-recall, tested against the failure modes the loop exists for.
- **3 board cases**: the seeded board drains in the exact priority order; a case budget of 3 and a token budget both cut the run at the right place and hand over the right remainder.
- **2 outcome cases**: cases run end to end, supervisor outcomes are recorded against them, and the harness asserts the live-accuracy stats and the recurring-fault watch — including a partial outcome (disposition only) and a confirmation that reclassifies a case out of its diagnosed pattern.
- **2 learning cases**: a case runs, a supervisor outcome is recorded, and a similar case runs against the *same* store — the confirmation written back to the machine's history is the only difference between the two halves. In one, an ambiguous CNC-3 vibration case exhausts at 3 rounds un-learned, then converges in 2 once the bearing fault stands confirmed on that machine; in the other, an outcome that confirms nothing is recorded and the follow-up behaves exactly as the baseline — the guard case proving the write-back, not the outcome ritual, is what teaches.
- **2 allocation cases**: the board sharing one shelf. Two conveyor cases want the shelf's only belt section — the higher-priority case converges and claims it, the lower blocks with the claim named in its gap; the supervisor rejects the winner at the gate, the claim is released, and a *fresh* run lets the lower case converge on the freed part (the finished run's records do not change). The second runs a lone case with no contention: it converges exactly as it always did, claims the substitute it draws, and the claim is the only new thing on the record.
- **4 economics cases**: the board priced at the plant model's rates, with expected dollars hand-computed from the KB — a two-case rated board (exposure per case and in total, largest named), a lone case on the unrated Utilities line (minutes present, dollars null), a $7,200 maintenance alert entered ahead of a $633 line-down case that the board must still work *first* (exposure never reorders), and cost accuracy against recorded actuals, per line and plant-wide, with the unrated case counted in the minutes and never in the dollars.
- **2 carryover cases**: two board runs and the turnover between them. Run 1 ends on the case budget with one case converged-then-decided-and-done, one converged and still waiting, one blocked on the winner's claim, one never worked; run 2 accepts the carry-over — the blocked case resumes with its rounds intact and takes the belt the finished case's released claim freed, the waiting case's claim is renewed with its provenance stamped, and the plant still counts four cases, not five. The second case drops a carried case at acceptance: the drop is on the record, its claim is released, and the case it was blocking converges on the freed part.

Gate: ≥95% overall. Current: **38/38**. Plus 162 pytest tests — loop-controller units with scripted critiques (converge r1/r2, exhaust, budget, safety), the verify-stage substitution branches (applied, rejected on rating, none listed, a planner's unverified substitute claim stripped), the board-level allocation (contention, gate release, the substitute search over effective stock, safety cases never claiming), the shift carry-over (the computed manifest, resume with rounds intact, claim renewal and the run-ended release, drops at acceptance, safety cases carried as records and never re-worked, inheritance chaining), the work-order packet contents and its refusals, the confirmed-fault write-back (per-decision behaviour, idempotency, the safety-stop refusal, the merged history's markings), the Muse adapters against stubbed transports (request shaping + reasoning continuity), guardrails, priority, the service and API, and hand-computed fixtures for the outcome stats, the learning-loop stats, the downtime economics, the handoff and the watch — with 2 Postgres-gated tests that skip without a database.

## Closing the loop: outcomes

A gate decision says a draft may proceed. It does not say the draft was *right*. So the record stays open for what happened next, and the loop is measured against it:

- **Outcome capture** — `POST /api/cases/{id}/outcome` (and the form on the case detail in the console): the supervisor records the plan's disposition (`accepted` / `accepted_with_changes` / `rejected`), the **confirmed fault** from the same vocabulary the diagnoses speak, the work performed, the actual downtime, the parts used, and whether the case resolved. Validation is strict — an unknown fault class is a 422, because ground truth with a typo in it poisons every number downstream. Partial outcomes count for what they recorded.
- **The plant remembers** — a confirmed fault is written back to the machine's history, where the next diagnosis reads it. The write-back is a store record of its own (migration `0004`), never an edit to the packaged KB: the KB's seeded history stays immutable *recorded* history, and every written-back entry is marked `supervisor_confirmed` with the case that confirmed it — the two kinds are never confused. Only a confirmed fault writes (any disposition counts — the fault the floor found is ground truth even when the plan was refused; an outcome confirming nothing writes nothing), one entry per case, ever. A safety stop refuses a confirmed fault outright, before anything persists: a safety case has no diagnosis to confirm. The diagnose context carries the machine's merged history — confirmed entries first, sources marked — so the offline signals cite a confirmation as the floor's own finding, the model backends see it in their context marked the same way, and when the readings cannot separate two fault classes, a supervisor-confirmed precedent for the leader is the discriminating fact that separates them. Recorded history alone never separates anything; it is cited as evidence, nothing more.
- **Live diagnosis accuracy** — `GET /api/stats/diagnosis`: top-1/top-2 accuracy of the diagnosis against confirmed faults, per-fault-class precision/recall with their counts, mean rounds-to-converge for confirmed cases, plan-acceptance rates, estimated-vs-actual downtime error, and a downtime-economics section that prices the same comparison per line and plant-wide (see "The board in dollars" below). Pure aggregation over the records, identical over both stores; with no outcomes yet it says so instead of printing zeroes. A **learning-loop section** reports the write-backs (confirmed faults, by machine) and repeat-fault performance — the top-1 rate on cases whose machine + fault a supervisor had already confirmed, set beside the overall top-1 rate, both as plain counts; when there are no repeat cases yet it says so.
- **Shift handoff** — `GET /api/board/handoff` (add `?format=text` for the shift-log text): cases at the gate, escalations with their open gaps, blocked-external cases with the part they're waiting on (and, when the wait is the board's own allocation, the case holding the claim), safety stops, open reliability items, the faults supervisors have confirmed into the machines' history, the last board run — including what a budget cut left unworked — and the carry-over manifest the next shift inherits (see "The next shift inherits the board" below). Every fact is composed in code from the records; no model writes any of it.
- **Recurring-fault watch** — when a case reaches the gate, or an outcome confirms a fault, a code-side watch counts the same machine + fault class inside a rolling window (default: 3 in 14 days, `RELIABILITY_THRESHOLD` / `RELIABILITY_WINDOW_DAYS`). At the threshold it raises **one** `reliability_review` item on the board, carrying the contributing cases as evidence. Fixing the same bearing three times in a fortnight is a reliability problem, not three maintenance jobs — the item says so, and then does nothing else: it triggers no automatic action and waits at the supervisor gate like everything else.

## Parts reality: substitutions and the floor packet

A stocked-out part is where a demo ends and a plant tool begins. Two mechanisms handle it:

- **Substitute-part replanning** — substitution lives in the verify stage, not in any planner, so it runs for every backend and leaves paperwork. The KB carries an explicit **compatibility relation** per part slot (basis + spec deltas: rating, fit, grade), and SOPs carry **constraints** for a slot where the physics demands them. When the shelf cannot satisfy a requirement, the search walks the relation in order: not stocked or short → rejected; a spec that violates an SOP constraint — or can't be verified at all → rejected, reason named. The first candidate that clears both is applied: plan steps are revised onto it, the substitution is recorded as a plan element, and a converged plan that uses one is flagged **SUBSTITUTION** on the record and in the work-order draft — the supervisor approves it knowingly, or not at all. When nothing passes, the case still exits `blocked_external`, but the escalation documents the search: what was considered, and why each candidate failed.
- **Board-level allocation** — the board shares one shelf, and the loop plans like it. When a case converges, the parts its plan will draw are **claimed** for it in an allocation ledger (its own store table, migration `0005`), in board-run order — which is priority order by construction. Every later case measures the shelf *net of the claims standing*, in the parts pass and in the substitute search alike: a substitute claimed by an earlier case is exactly as unavailable as one the shelf never held. A case that loses a part to a claim takes the ordinary paths with the reason named precisely — its blocked gap reads "claimed by EXC-…, higher priority on this board", never a bare stockout that hides the board's own decision — and the shift handoff names the allocation and the claiming case. A gate **approval** leaves the claim standing (the work order will draw the part); a gate **rejection** releases it, stamped with the reason, the time and the supervisor, back into the pool for *future* runs. A release never rewrites the finished run it came from — runs are records; the case that blocked stays blocked on its record until a fresh run re-plans it. A claim is a planning reservation on the record, not a warehouse transaction: it draws no stock and marks no work performed, and safety cases never claim — they escalate at round 0, before any of this machinery exists. The ledger is on the board run record, at `GET /api/allocations`, and in the console's board view.
- **The work-order packet** — `GET /api/cases/{id}/packet` (JSON, or `?format=text`) renders a gated case for the technician who carries it to the machine: case summary, diagnosis with its evidence, the plan steps in order, the mandatory safety steps again as a **check-off list**, the parts list with substitutions flagged (basis and deltas included) and each part marked when it stands claimed for this case on its run, the estimated downtime, and blank sign-off lines — supervisor, technician, completion with actual downtime and parts used. Composed in code from the record only. Only a converged case packets: a non-converged exit produced an escalation draft instead, and a rejected draft must not reach the floor dressed as a work order — both are a 409. An awaiting case packets with its draft banner; an approved one carries the approval on its face.

## The board in dollars: downtime economics

The priority formula speaks in severity points. A plant runs on money per hour of downtime — and the supervisor, not the formula, is where a dollar judgement belongs. So every case is also priced, in code, from facts the loop already carries:

- **Rates are plant-model data** — each line in the bundled KB carries a `downtime_rate_usd_per_hour`, authored alongside its criticality like the SOPs and the shelf. Utilities (line-e) deliberately carries none: its downtime is not charged to a product line in this plant model, so its cases are **minutes only** — an absent rate is a fact, never a licence to invent one. Every dollar figure names the rate it used and where the rate lives.
- **Exposure** — a case's estimated exposure is its plan's own estimated downtime priced at its line's rate, stamped on the record when it reaches the gate, shown beside the case on the board and in the console, and totalled on the board run ("this board is carrying ~$19,053.33 of estimated downtime exposure" on the seeded board) with the largest single exposure named. The shift handoff carries the same section: open exposure, the largest open case, and the closed cases' recorded actuals.
- **Economics inform; they do not reorder** — exposure is printed beside the priority order, never fed into it. A $7,200 maintenance alert still works *after* a $633 line-down case, because the plant's triage doctrine is a safety-and-severity judgement, not a financial one. The supervisor sees both numbers and applies the overlay — that stays a human's job, and the golden pack pins the order to prove it.
- **Safety is never priced** — a safety stop carries no economics at all: not a zero, an absence. Its record, its board row and its handoff section stay exactly as bare of money as they were before this feature existed.
- **Cost accuracy** — when a supervisor records the outcome, the actual minutes are priced at the same rate the estimate used, and the stats report estimated-vs-actual minutes *and* dollars, plant-wide and per line, over the same population as the downtime error — counts beside every figure, and minutes-only cases counted in the minutes and never in the dollars.

## The next shift inherits the board: shift carry-over

A board run ends; the plant does not. Until this round, a run's unfinished business survived only as records plus handoff prose, and the next shift's board was assembled fresh, by hand, from the prose. Now the handover is a mechanic with the same standing as the loops:

- **The manifest** — when a board run ends, its unfinished business is computed in code and stored on the run record: cases the shift budget never reached (*not worked*), cases that exited `blocked_external` and still wait (*blocked*), and cases still awaiting the supervisor at the gate (*at gate*) — each with its state and what it is waiting on. The run record also carries every queued case's intake payload, so the offer is self-contained. The shift handoff gains the same manifest as a section: the contract the next shift is handed, not a summary of it.
- **Acceptance is explicit** — a new board run names its predecessor (`carry_over_from` on `POST /api/board/run`, `--carry-over-from` on the CLI) or it starts clean; there is no implicit inheritance. Acceptance settles the offer against the store *as it stands then* — decisions land between runs, and the settlement, not the manifest, is the truth. A case decided in the meantime is closed, not carried.
- **Re-work is a resume, not a re-intake** — a blocked case re-enters the loop at its loop-back point with its loop state intact: diagnosis, plan, open gaps, every round record, and its cumulative token spend. The next round is round N+1 against the same `max_rounds` bound — the budget governs the case's total work across shifts, not per shift — and the first resumed round re-diagnoses focused on the carried gaps, exactly as a loop-back round does, while the parts pass re-measures the shelf as it stands *now*. That re-measurement is the point: a stockout is a fact about a moment, and the moment has passed. Gate cases are inherited standing — a human is already their next step, so they consume no budget and are never re-worked — and unworked cases join as the fresh intakes they always were. Inheritance chains: a gate case one shift inherited and left standing is re-offered on that shift's own manifest, so an undecided case is never dropped from the chain by being inherited once.
- **Claims turn over with the board** — claims are run-scoped planning reservations, and the turnover adjudicates every standing claim in the ledger. An accepted case's claims are **renewed** into the accepting run, stamped `carried_from_run` with the hop's provenance, and stay releasable by a gate rejection exactly as before. A claim whose case is finished — decided and done, dropped, superseded — is **released** with the reason `released_run_ended`, and the settlement lands on the run record and in the handoff ("2 claim(s) renewed for carry-over, 1 claim(s) released — run ended"). One exemption, carried from the allocation round: a claim whose case is **approved and still awaiting its outcome** keeps standing — the approved work order will draw the part — and the first turnover after the outcome lands releases it. A part a finished board was holding is therefore provably free on the next board: the golden pack runs the contention board across two shifts and watches the blocked case take the freed belt.
- **Drops** — at acceptance the supervisor may drop a carried case (actor + reason, on the record): the case leaves the board with the drop stamped on it, and its claims release with the turnover. Dropping is the only way a case leaves the board without a gate decision, and it is never silent. A fresh submission of the same case at acceptance supersedes the carried entry instead — the new intake wins, the old claim falls to the sweep.
- **Safety carries as a record, never as work** — a safety stop that is still waiting carries as an *at gate* escalation for the human; it is never resumed into the diagnosis path, and the settlement refuses to build a resume seed for a safety case in code, not by convention. An unworked safety case carries as a fresh intake and stops at round 0 when the new board reaches it, exactly as it would have on the old one.
- **Records, not recounts** — a re-worked case's new record replaces its old one (one record per case, as always), so the stats and the economics read each case once, at its latest record: a case worked on two boards is one case in the plant's numbers, with its exposure recomputed from the same pricing functions — never summed across runs. No new storage was needed: run records, case records and claims already persist as payloads, and every field this round adds is additive with a default.

The manifest of any stored run is at `GET /api/board/runs/{run_id}/carry-over`, including how that run's own inheritance settled.

## Honest status

- **Muse Spark adapters are built to Meta's documented API and unit-tested against stubbed transports** (Chat Completions request shape; Responses shape; the round-1 encrypted reasoning item verifiably rides along on round 2). **Live verification is pending a Model API key** — no live call to `api.meta.ai` has been made from this repo yet, and nothing here claims otherwise. The offline backend runs the identical loops today.
- **Docker/compose not run in the build sandbox** (no daemon); the topology mirrors the sibling blueprints' verified compose stacks, and the Dockerfile builds from the same frozen constraints. First `docker compose up --build` on a real machine is the check.
- **PostgreSQL paths are covered by gated tests** that skip without `TEST_DATABASE_URL`; they ran against the doubles everywhere else.
- The offline backend is a rules engine, not a model: its diagnoses are only as good as the SOP thresholds in the synthetic KB. It exists so the loops can be evaluated deterministically — swap in `muse` or `llm` for real reasoning.
- The KB, histories, inventory and board are synthetic. A real deployment points the ports at the plant's CMMS/EAM, document store and historian; the loop code does not change.

## Layout

```
src/plant_loop/
  loops.py        the two loop controllers: rounds, gap bookkeeping, exits, budgets
  graph.py        LangGraph wiring: case graph (loop-back edges, gate interrupt), board graph
  priority.py     board priority: doctrine order x line criticality + downtime
  backends.py     the Reasoner port's engines: muse (chat + responses), llm, offline rules
  guardrails.py   verify-round code checks + draft language screen
  screening.py    injection screen + banned-action screen for floor text and plan steps
  kb.py           the synthetic plant KB (SOPs, history, inventory) + retriever
  history.py      the plant's memory: confirmed-fault write-backs + the merged history view
  substitution.py the verify stage's substitute search: shelf truth, KB relation, SOP constraints
  allocation.py   the allocation ledger: claims on converged plans, effective stock, gate release
  carryover.py    shift carry-over: the computed manifest, acceptance settlement, claim turnover
  economics.py    downtime economics: cases and boards priced at the plant model's line rates
  drafts.py       work-order / escalation draft builders (deterministic, gate-noticed)
  packet.py       the work-order packet for the floor, composed in code (JSON + text)
  stats.py        live diagnosis accuracy over recorded outcomes (pure aggregation)
  handoff.py      the shift handoff report, composed in code (JSON + text)
  reliability.py  the recurring-fault watch: reliability_review board items
  store.py        record stores: PostgreSQL / in-memory double
  service.py      runs the graphs, owns persistence and the gate
  api.py          FastAPI transport + the single-file console
  data/           kb.json, sample_cases.json (the seeded board), eval_results.json
migrations/       numbered SQL (records, board runs, errors, pgvector KB index, board items, history entries, part claims)
evals/            golden.jsonl (38 entries) + run_evals.py (the gate)
tests/            pytest suite
```

## Extension seams

- **New reasoning backend**: implement the three-method `Reasoner` protocol and register it in `backends.get_reasoner` — the loops never see the difference.
- **Real plant systems**: replace `KBRetriever` (SOPs/history/inventory reads) and the `Store`/`ObjectStore` implementations behind their ports; keep the controllers.
- **Loop policy**: rounds, budgets and the priority weights live in `loops.py`/`priority.py` config, not scattered through the graph.
- **Actuation**: deliberately absent. If a deployment ever connects the gate's approval to a CMMS work-order API, that integration belongs behind the supervisor's decision — the agent's boundary does not move.

## License

Apache 2.0. From Trida AI's open blueprint series — working prototypes of production-shaped AI agents, published so the engineering can be read, run and argued with.
