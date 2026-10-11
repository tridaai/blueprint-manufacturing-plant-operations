<p align="center">
  <img src="https://trida.ai/logos-rounded/trida-rounded.png" width="96" alt="Trida AI">
</p>

# Blueprint — Plant Operations Loop Agent

An agent that works a manufacturing plant's shift **exception board** the way a good shift lead does — in two loops. The **board loop** takes the board in priority order (computed in code: safety first, then line-down, quality, shortage, maintenance — scaled by line criticality and minutes already down) and works each case to a shift budget. The **case loop** runs diagnose → plan → verify rounds on one exception: ranked fault hypotheses with evidence, a corrective plan whose mandatory safety steps cannot be removed and whose parts are checked against the shelf, then a critic round that produces a gap list. Gaps send the case around again — on the gaps only — until the plan verifies clean, the rounds run out, a part only the outside world can supply blocks it, or a safety rule stops it outright.

Every case ends at a **supervisor gate** with a draft work order or an escalation note. The agent never issues commands to machines, never marks work performed, and never closes a case. That sentence is printed on every record it produces.

Reasoning is led by **Muse Spark** (Meta's Model API), with a Responses-API mode that carries reasoning continuity across loop rounds. The plant — "Meridian Fabrication, Plant 2", its SOPs, maintenance history, parts inventory and the nine-case board — is synthetic and bundled, so the whole thing runs end to end with no keys at all.

> Reference prototype from Trida AI's blueprint series. The loops, guardrails and gate are the production-shaped parts; the plant data is a stand-in for the systems a real deployment would read.

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
make test                   # 98 tests (+2 Postgres-gated, skipped without a DB)
make evals                  # golden loop eval: 28/28 must pass the gate
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

`make evals` runs the golden loop pack — 28 authored entries against the offline backend, hermetic and deterministic:

- **20 case cases**: expected diagnosis class, exit reason, rounds used, draft kind, and verify-stage substitution records — including the two-round convergence (a bearing fault the first round can't yet separate from a tool imbalance, separated by a discriminating check in round 2), the round-0 safety stop, the skip-lockout injection, a substitute that converges (twice: a coolant filter, a packaging film), a substitute rejected on its rating that still escalates with the reason on record, and an honest `exhausted` for a fault with no SOP.
- **3 defect cases**: deliberately defective planners — one drops safety steps, one names an unstocked part, one proposes bypassing an interlock — and the verify round must catch each (never converge, blocked, safety stop respectively). This is gap-recall, tested against the failure modes the loop exists for.
- **3 board cases**: the seeded board drains in the exact priority order; a case budget of 3 and a token budget both cut the run at the right place and hand over the right remainder.
- **2 outcome cases**: cases run end to end, supervisor outcomes are recorded against them, and the harness asserts the live-accuracy stats and the recurring-fault watch — including a partial outcome (disposition only) and a confirmation that reclassifies a case out of its diagnosed pattern.

Gate: ≥95% overall. Current: **28/28**. Plus 98 pytest tests — loop-controller units with scripted critiques (converge r1/r2, exhaust, budget, safety), the verify-stage substitution branches (applied, rejected on rating, none listed, a planner's unverified substitute claim stripped), the work-order packet contents and its refusals, the Muse adapters against stubbed transports (request shaping + reasoning continuity), guardrails, priority, the service and API, and hand-computed fixtures for the outcome stats, the handoff and the watch — with 2 Postgres-gated tests that skip without a database.

## Closing the loop: outcomes

A gate decision says a draft may proceed. It does not say the draft was *right*. So the record stays open for what happened next, and the loop is measured against it:

- **Outcome capture** — `POST /api/cases/{id}/outcome` (and the form on the case detail in the console): the supervisor records the plan's disposition (`accepted` / `accepted_with_changes` / `rejected`), the **confirmed fault** from the same vocabulary the diagnoses speak, the work performed, the actual downtime, the parts used, and whether the case resolved. Validation is strict — an unknown fault class is a 422, because ground truth with a typo in it poisons every number downstream. Partial outcomes count for what they recorded.
- **Live diagnosis accuracy** — `GET /api/stats/diagnosis`: top-1/top-2 accuracy of the diagnosis against confirmed faults, per-fault-class precision/recall with their counts, mean rounds-to-converge for confirmed cases, plan-acceptance rates, and estimated-vs-actual downtime error. Pure aggregation over the records, identical over both stores; with no outcomes yet it says so instead of printing zeroes.
- **Shift handoff** — `GET /api/board/handoff` (add `?format=text` for the shift-log text): cases at the gate, escalations with their open gaps, blocked-external cases with the part they're waiting on, safety stops, open reliability items, and the last board run — including what a budget cut left unworked. Every fact is composed in code from the records; no model writes any of it.
- **Recurring-fault watch** — when a case reaches the gate, or an outcome confirms a fault, a code-side watch counts the same machine + fault class inside a rolling window (default: 3 in 14 days, `RELIABILITY_THRESHOLD` / `RELIABILITY_WINDOW_DAYS`). At the threshold it raises **one** `reliability_review` item on the board, carrying the contributing cases as evidence. Fixing the same bearing three times in a fortnight is a reliability problem, not three maintenance jobs — the item says so, and then does nothing else: it triggers no automatic action and waits at the supervisor gate like everything else.

## Parts reality: substitutions and the floor packet

A stocked-out part is where a demo ends and a plant tool begins. Two mechanisms handle it:

- **Substitute-part replanning** — substitution lives in the verify stage, not in any planner, so it runs for every backend and leaves paperwork. The KB carries an explicit **compatibility relation** per part slot (basis + spec deltas: rating, fit, grade), and SOPs carry **constraints** for a slot where the physics demands them. When the shelf cannot satisfy a requirement, the search walks the relation in order: not stocked or short → rejected; a spec that violates an SOP constraint — or can't be verified at all → rejected, reason named. The first candidate that clears both is applied: plan steps are revised onto it, the substitution is recorded as a plan element, and a converged plan that uses one is flagged **SUBSTITUTION** on the record and in the work-order draft — the supervisor approves it knowingly, or not at all. When nothing passes, the case still exits `blocked_external`, but the escalation documents the search: what was considered, and why each candidate failed.
- **The work-order packet** — `GET /api/cases/{id}/packet` (JSON, or `?format=text`) renders a gated case for the technician who carries it to the machine: case summary, diagnosis with its evidence, the plan steps in order, the mandatory safety steps again as a **check-off list**, the parts list with substitutions flagged (basis and deltas included), the estimated downtime, and blank sign-off lines — supervisor, technician, completion with actual downtime and parts used. Composed in code from the record only. Only a converged case packets: a non-converged exit produced an escalation draft instead, and a rejected draft must not reach the floor dressed as a work order — both are a 409. An awaiting case packets with its draft banner; an approved one carries the approval on its face.

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
  substitution.py the verify stage's substitute search: shelf truth, KB relation, SOP constraints
  drafts.py       work-order / escalation draft builders (deterministic, gate-noticed)
  packet.py       the work-order packet for the floor, composed in code (JSON + text)
  stats.py        live diagnosis accuracy over recorded outcomes (pure aggregation)
  handoff.py      the shift handoff report, composed in code (JSON + text)
  reliability.py  the recurring-fault watch: reliability_review board items
  store.py        record stores: PostgreSQL / in-memory double
  service.py      runs the graphs, owns persistence and the gate
  api.py          FastAPI transport + the single-file console
  data/           kb.json, sample_cases.json (the seeded board), eval_results.json
migrations/       numbered SQL (records, board runs, errors, pgvector KB index, board items)
evals/            golden.jsonl (28 entries) + run_evals.py (the gate)
tests/            pytest suite
```

## Extension seams

- **New reasoning backend**: implement the three-method `Reasoner` protocol and register it in `backends.get_reasoner` — the loops never see the difference.
- **Real plant systems**: replace `KBRetriever` (SOPs/history/inventory reads) and the `Store`/`ObjectStore` implementations behind their ports; keep the controllers.
- **Loop policy**: rounds, budgets and the priority weights live in `loops.py`/`priority.py` config, not scattered through the graph.
- **Actuation**: deliberately absent. If a deployment ever connects the gate's approval to a CMMS work-order API, that integration belongs behind the supervisor's decision — the agent's boundary does not move.

## License

Apache 2.0. From Trida AI's open blueprint series — working prototypes of production-shaped AI agents, published so the engineering can be read, run and argued with.
