# PRD-276: Durable background work — one job queue, and work that outlives the request

**Status:** Proposed, for decision. Written 11 Nov 2026 by Dave. No code is changed by this PRD.
**Maintainer approval:** needed, from Gerard. §10 lists the decisions, each with a recommendation.
**The PRD number is provisional** until the maintainer confirms it.
**Grounded on:** `main @ 2e3c99286`. Labels: **[T]** read in code at that commit; **[I]** inferred; **[?]** not known from the repository.

## 1. Why

Automatos started as one API process doing everything a request needed. That was right then. The platform has since grown past that design:
- hosted runs four uvicorn workers in one container;
- runs take minutes, not seconds;
- other products now depend on runs finishing and callbacks arriving (PRD-274).

Today the API process is also the worker, the scheduler and the retry engine. Most background work runs in that process's memory, so a deploy or crash loses it. Each loss has been patched in its own place:
- a boot reaper that fails what a restart orphaned;
- a resend job for callbacks;
- poll loops;
- per-feature retry loops.

PRD-161 already built one durable queue: board-task dispatch, with claims, leases and requeue. The rest of the platform doesn't use it.

The question this PRD asks: should every piece of work that must not be lost go through one durable job queue, with workers and a scheduler that are separate roles from the API?

### 1.1 This week's examples

| What happened | Why | Where |
|---|---|---|
| A run's signed callback was lost on a restart | Its three attempts (0, 5 and 15 s) happen in the API process's memory | `services/run_callbacks.py` (`BACKOFF_SECONDS`, `_send_with_retries`) [T] |
| A job was added to resend pending callbacks every 10 minutes | To recover the callbacks a restart cut off | #1481 |
| Two API replicas could send the same callback twice, with different delivery ids | The scheduler's single-owner lock is a file in `/tmp`, so it holds within one container only | #1482; `main.py:429` [T] |

None of these is a bug in the PR that met it. They are what this design costs.

## 2. Goals

- **One way to say "do this, later or now, and don't lose it":** a job, queued in the same database transaction as the write that caused it.
- **Work survives a deploy or a crash.** A job whose worker died is claimed again after its lease ends. A job is never failed just because its process died.
- **Runs exactly once in effect.** Delivery is at least once. Handlers are written to be idempotent, and each job carries an idempotency key.
- **Retries, backoff and a dead-letter state in one place,** replacing the per-feature retry loops (§4.4).
- **The API answers requests and queues work. Workers run the work. One scheduler queues timed work.** These are three roles of the same image.
- **The local edition still works with `docker compose up`.** One process takes all three roles, as today.

## 3. Non-goals

- **No new broker.** No RabbitMQ, Kafka or NATS. The design uses the Postgres the platform already runs (§6.2, §10-1).
- **Chat turns stay in the request.** A streamed chat answer isn't background work.
- **Missions stay as they are.** The coordinator's reconcile loop already resumes and cleans up running Missions (`core/boot/reaper.py` docstring) [T].
- **Board-task dispatch keeps its own claim in the first waves.** PRD-161's `board_tasks` lease columns already work. §8 Wave 4 asks whether board tasks become one kind of job.
- **Best-effort work stays fire-and-forget:** cache warming, embeddings refreshed at boot, telemetry. §8 Wave 5 sorts these out case by case.

## 4. What exists today

### 4.1 Processes

| Process | What it runs | How it gets work |
|---|---|---|
| `automatos-ai-api`, 4 uvicorn workers in one container (`WEB_CONCURRENCY=4`, `orchestrator/Dockerfile:408`) [T] | HTTP; the board dispatch loop in **every** worker (`main.py:632`); the unified scheduler in **one** worker per container (an `fcntl` lock on `/tmp/automatos_scheduler.lock`, `main.py:425-553`) [T] | Requests; `pg_notify` and a 5-second poll for board tasks (`BOARD_DISPATCH_POLL_SECONDS`) [T] |
| `agent-workspace-worker` | Agent tasks in isolated workspace directories | Redis lists, polled in priority order (`services/workspace-worker/main.py`); the API pushes with `lpush` (`api/workspace_github.py:294`) [T] |
| A paired CLI host (`services/cli-host`) | Tickets for `runtime: cli` agents | The **same** `FOR UPDATE SKIP LOCKED` claim and lease as the dispatcher, with a runtime filter (`services/cli_host_service.py`, PRD-234) [T] |
| `agent-opt-worker`, `media-render`, `voice-tts` | Their own work | HTTP calls from the API with an internal token [T for `agent-opt-worker`; I for the rest] |

Railway's replica count for the API isn't set in the manifest (`infrastructure/railway-manifest.json`, generated 9 Apr 2026, `"replicas": null`). Whether hosted runs more than one API container is **[?]**.

### 4.2 Background work in the API process (counts from a grep at the commit above, excluding tests) [T]

| Mechanism | Count | Survives a restart? |
|---|---|---|
| `asyncio.create_task` / `ensure_future` | ~67 sites; most in `modules/tools`, `api/webhooks.py` (8, replies to inbound chat-platform webhooks), `consumers/chatbot`, `modules/memory` | No |
| FastAPI `BackgroundTasks` | 9 modules (documents, workflows, skills, workspace move, workspace purge, admin, GitHub webhooks) | No |
| `threading.Thread` | 11 (deny-list refresh, watchdog, logging, thumbnails, voice) | No; most are meant to be best-effort |
| APScheduler jobs on the unified scheduler | 36 `add_job` sites (heartbeats, Playbook schedules, memory jobs, credits metering, socials, digests, reconcilers) | Timed: the schedule is rebuilt at boot from the database; a run in progress isn't resumed |
| Playbook runs | `PlaybookEngine.launch` → `launch_recipe_task` → `asyncio.create_task` (`api/recipe_executor.py:1042`) | No; the boot reaper marks a stale run failed (`core/boot/reaper.py`) |
| Claimed board tasks | `_launch_one` → `_launch_task_execution`, in the worker that claimed it (`services/board_dispatcher.py:695`) | Yes: the lease (600 s by default) runs out and the sweeper requeues it, up to `BOARD_DISPATCH_MAX_ATTEMPTS` |

### 4.3 The durable pieces that already exist [T]

- **PRD-161's dispatch spine** (`services/board_dispatcher.py`) is a complete queue:
  - claims with `FOR UPDATE SKIP LOCKED`;
  - a lease (`lease_until`) and an attempt count (`attempts`);
  - a sweeper that requeues expired leases;
  - lease renewal for long runs;
  - wake-ups through `pg_notify` on `board_task_available`, with polling as the fallback;
  - per-agent slots.
- **LISTEN/NOTIFY fan-out** for the live UI (`services/board_events.py`, the `board_events` channel). Its own docstring says it "buys latency, not correctness".
- **The board as a queue for more than one consumer:** the CLI host claims through the dispatcher's own claim (PRD-234). PRD-161's primitives already generalise.
- **Advisory locks** in 14 modules, including the boot leader (`core/database/boot_lock.py`). That lock covers seeding at boot only and is released afterwards.
- **A pluggable task runner** (`core/task_runner/`, PRD-56): `TASK_RUNNER_BACKEND` = `local`, `queued` (Redis, for the workspace worker) or `kubernetes`, with `local` as the default (`config.py:716`). The hosted API has the setting; its value is [?].
- **A persistent scheduler job store when Redis is configured:** `UnifiedScheduler.start` uses APScheduler's `RedisJobStore` when `REDIS_URL` is set, and memory otherwise (`services/scheduler.py:40-50`). The hosted API sets `REDIS_URL` (the manifest's env keys) [I].

### 4.4 Two copies of the same retry loop [T]

- `services/run_callbacks.py`: three attempts, backoff `(0, 5, 15)` seconds, in memory.
- `services/destinations/dispatcher.py` (widget callbacks to Telegram, Slack and webhooks): three attempts, backoff `(0, 5, 15)` seconds, in memory. Its docstring says: "For v1 the queue is in-process via `asyncio.create_task` … the callsite swaps from `create_task` to `enqueue_redis`" once a worker is wired.

## 5. What goes wrong

| Event | Today | With this PRD |
|---|---|---|
| Deploy or crash during a Playbook run | The run dies. At the next boot, a run in flight for longer than `BOOT_REAPER_STALE_MINUTES` (30 by default) is marked failed (`core/boot/reaper.py`) [T] | The job's lease runs out, and another worker runs it again, or resumes from its last finished step if the handler supports that (§9) |
| Deploy during a callback's retries | The callback stays pending; #1481's job resends it 5 to 15 minutes later | The delivery job is still queued; a worker sends it at the next backoff step |
| Two API containers | Each container's scheduler owns its own `/tmp` lock, so timed jobs run once per container. Both read and write one shared `RedisJobStore`, which APScheduler 3.x doesn't support across processes [I] | One scheduler across containers (§6.4); duplicate ticks are harmless (idempotency key) |
| A restart while the API sends its reply to an inbound chat-platform webhook (`api/webhooks.py`, `asyncio.create_task`) [T] | Lost: the webhook was already answered, so nothing sends it again [I] | Queued in the webhook's own transaction; sent once a worker is free |
| A slow LLM job in the API process | Holds a uvicorn worker's event loop and memory while requests wait | Runs in a worker; the API stays responsive (§6.1) |

## 6. Design

### 6.1 Three roles, one image

`PROCESS_ROLES` in `config.py` is a comma-separated set of `api`, `worker` and `scheduler`. The default is all three. A process starts each role it lists, and no other.

| Role | Runs | Hosted | Local edition |
|---|---|---|---|
| `api` | HTTP and SSE. Queues jobs; runs none | The existing API service: `PROCESS_ROLES=api`, `WEB_CONCURRENCY` unchanged | Listed by default |
| `worker` | The job loop (§6.3) and, from Wave 4, the board dispatch loop | A new Railway service from the same image: `PROCESS_ROLES=worker,scheduler`, scaled independently | Listed by default |
| `scheduler` | The timed jobs. Each only **queues** a job (§6.4). It runs only while it holds a Postgres advisory lock, so a process can list it safely on every replica | Listed on the worker service: every replica tries for the lock, one holds it, and another takes over if that one dies | Listed by default |

The local edition keeps the default, so `docker compose up` runs everything in one process, as today. **So a deployment can't run without timed work unnoticed,** the API's readiness reports a problem when no process has held the scheduler lock for longer than the shortest timed-job interval.

### 6.2 The job table and how a job is queued

One new table, `jobs`:

| Column | Purpose |
|---|---|
| `id` (uuid), `kind` (text), `workspace_id` (uuid, nullable for platform jobs) | What the job is, and its tenant |
| `payload` (jsonb) | The handler's input: ids, never large blobs |
| `status` | `queued` (including a failed job waiting for its next attempt), `running`, `done`, `dead` (its attempts used up) |
| `run_at`, `attempts`, `max_attempts` | When it may run next, and the retry budget |
| `lease_until`, `claimed_by`, `claim_id` (uuid, new on every claim) | PRD-161's lease, generalised, and the token that fences a stale worker (§6.3) |
| `idempotency_key` (unique per `kind`) | A second queue with the same key is a no-op |
| `last_error`, `created_at`, `updated_at`, `finished_at` | What happened |

```python
from core.jobs import enqueue

enqueue(db, "run_callback.deliver", {"task_id": task.id},
        workspace_id=task.workspace_id, key=f"run_callback:{task.id}")
db.commit()  # the job exists only if the ticket's write committed
```

- **In the caller's transaction:** a job and the write that caused it commit together, or neither does. Postgres delivers a `NOTIFY` sent in a transaction only when it commits, so the wake-up is never early.
- **Handlers are registered per kind,** with that kind's backoff, attempt limit, lease and concurrency:

```python
@job_handler("run_callback.deliver", backoff=(5, 15, 60, 300, 900), max_attempts=6, lease_seconds=60)
async def deliver(job: Job) -> None: ...
```

- **How attempts map to delays:** `attempts` counts claims, starting at 1 on the first. After the *n*th attempt fails, the job waits `backoff[n - 1]` seconds. If the tuple is shorter than `max_attempts - 1`, its last delay repeats. So every retry has a delay, and a handler that lists one delay waits the same time between all its attempts.

### 6.3 The worker loop

This is PRD-161's dispatch loop, generalised to any `kind` [T: the primitives it reuses]:
- **Claiming:** `FOR UPDATE SKIP LOCKED` over `jobs` where `status = 'queued' AND run_at <= now()`, a batch at a time, within each kind's concurrency limit.
- **Waking up:** `LISTEN jobs_available`, with a poll as the fallback.
- **Leases:** the lease is renewed while a long handler runs.
  - **An expired lease counts as a failed attempt,** because `attempts` was raised when the job was claimed. With attempts left, the job goes back in the queue at its next delay. With none left, it becomes `dead`, and is reported as a failure would be. So a job whose worker keeps dying can't loop forever. This is PRD-161's rule: `requeue_expired_leases` fails a ticket once its attempts reach the limit (`services/board_dispatcher.py:355`) [T].
- **Fencing a stale worker:** a slow worker may still be running after its lease has expired and another worker has claimed the job.
  - Every claim writes a new `claim_id`, and every renewal, success or failure is `UPDATE … WHERE id = :id AND claim_id = :claim_id`.
  - A worker whose update matches no row has lost the job. It stops, logs, and writes nothing more, so it can never requeue or complete a newer claim.
  - PRD-161 already fences renewals by run (F209: `runtime_ref->>'run_id'` in `renew_lease`) [T]. Here every write is fenced, not just renewals.
- **Failure:** a handler that raises is logged with `logger.exception` and recorded through `record_error` under `subsystem="jobs"`. The job then goes back in the queue at its next delay (§6.2). Once its attempts are used up it becomes `dead`, and the owner is told where a person should know (callbacks, Playbook runs).
- **Shutdown:** on SIGTERM the worker stops claiming and finishes or abandons what it holds. An abandoned job's lease runs out and another worker picks it up.
- **Retention:** a timed job deletes `done` rows older than `JOBS_RETENTION_DAYS`.

### 6.4 The scheduler queues; it never runs

- **Every timed job becomes `enqueue(kind, key=f"{schedule_id}:{slot}")`,** where the key is unique per `kind`.
  - **`schedule_id`** is the schedule's own stable identity: `playbook:<id>` for a Playbook schedule, `heartbeat:<agent id>`, `scheduled_task:<id>`, or `platform` for a platform job such as metering. Those rows belong to a workspace, so two schedules, in one workspace or two, never share a key.
  - **`slot`** is the scheduled time the tick is for, not the time it ran.
  - Two schedulers firing the same schedule's same slot queue one job, so a duplicate tick is harmless.
- **One scheduler across containers:** a Postgres advisory lock, held on a session for as long as the process runs. This replaces the `/tmp` file lock, which holds within one container only. Another process waits for the lock and takes over if the holder dies.
- **APScheduler's job store goes back to memory.** The schedule is rebuilt from the database at start-up, as `load_active_tasks_to_scheduler` and the reconcile tick already do (`main.py:509-527`). The schedule never needs sharing across processes.
- **Slots missed while no scheduler ran:** a rebuilt schedule doesn't by itself queue the slots that passed during a deploy. Today, APScheduler skips a slot that fires too late (its misfire grace time) [I]. Instead:
  - **A durable cursor per schedule:** a `job_schedules` row (`kind`, `schedule_id`, `last_slot`). Each due slot is queued in the same transaction that moves `last_slot` forward, so a slot is either queued and recorded, or neither.
  - **At start-up, and on each tick,** the scheduler queues every slot between `last_slot` and now, by the kind's **catch-up policy:**
    - `all`: every missed slot, for work that is counted per slot;
    - `latest`: the most recent missed slot only, for heartbeats, digests and reconcilers;
    - `none`: skip, for best-effort work.
  - **Metering needs `latest`, not `all`.** `meter_tick` debits every call still waiting and marks it charged (`services/credits/jobs.py:37`, `meter_pending`) [T], so one late tick covers the missed ones.
  - With the cursor in place, APScheduler is only a timer that wakes the scheduler. §10-3 asks whether a plain tick loop over the cursors should replace it.

### 6.5 Outbound deliveries are one kind of job

`run_callback.deliver` and `widget_destination.deliver` replace the two in-memory retry loops (§4.4):
- **A stable delivery id per callback** (the job's id), the same on every attempt;
- **backoff long enough** to ride out a receiver's restart;
- **a `dead` state** shown where `run_callbacks` and `widget_event_log` already report delivery.

**The guarantee is at least once.** A worker can die after the receiver accepted a send, but before the job is marked done, and the retry sends it again. What that means for each receiver:
- **Run callbacks:** the receiver dedupes. The contract (#1462) already says `X-Automatos-Delivery` names the delivery and retries keep it (`services/run_callbacks.py:9`) [T]. With a stable id per callback, a receiver that drops an id it has seen gets each callback once in effect. The docs for callback receivers say so, and Wave 1 tests that the id is the same on every attempt.
- **Widget destinations** (Telegram, Slack, a workspace webhook, sent through `send_workspace_notification`): Telegram and Slack messages have no delivery id to dedupe on [I]. A rare duplicate message is possible and is documented as such. Wave 1 gives a workspace webhook the same `X-Automatos-Delivery` header, so it can dedupe as a run callback's receiver does.

#1481's resend job is deleted in Wave 1, and #1482's race goes with it.

### 6.6 Jobs and events are different things

- **A job:** "do this, exactly once in effect." It goes in `jobs`.
- **An event:** "this happened; anyone may listen." It stays on LISTEN/NOTIFY (`board_events`) or Redis pub/sub for the live UI, where losing one costs only latency.
- **An event that must not be lost** (an audit trail, a billing record) is written in the business transaction first. A job then publishes it, which makes it an outbox. No event needs this yet.

### 6.7 Editions

- **Local edition:** the default `PROCESS_ROLES` (all three). Same compose file, same single container, no new service.
- **Hosted:**
  - the API service sets `PROCESS_ROLES=api`;
  - a new `automatos-ai-worker` service, from the same image, sets `PROCESS_ROLES=worker,scheduler`;
  - every worker replica runs jobs, and one at a time holds the scheduler lock.
- Both editions run the same code. No hosted-only path.

## 7. What is reused, and what is deleted

| Reused | From |
|---|---|
| Claims, leases, the sweeper and NOTIFY wake-ups | `services/board_dispatcher.py` (PRD-161), lifted into `core/jobs/` |
| Error recording | `record_error` (the boot reaper and the dispatcher already use it) |
| The schedule rebuilt from the database | `load_active_tasks_to_scheduler`, `schedule_reconcile` |
| The single seam for Playbook launches | `PlaybookEngine.launch` (PRD-142 W3-S12): the one place that changes |

**Deleted, each in the wave that replaces it** ("Replace, don't shim"):
- the in-memory retries in `run_callbacks` and `destinations/dispatcher`;
- #1481's resend job;
- the `/tmp/automatos_scheduler.lock` file lock;
- `RedisJobStore` for the unified scheduler;
- the boot reaper's surfaces as each moves to a lease: `RecipeExecution` first.

## 8. Rollout

| Wave | Delivers | Proven by |
|---|---|---|
| **0** | This PRD decided (§10) | — |
| **1** | `core/jobs` (table, `enqueue`, handler registry, fenced claims, the worker loop under the default `PROCESS_ROLES`); run callbacks and widget destinations moved; the resend job and both retry loops deleted | Tests:<br>• a worker is killed mid-delivery, and the retry carries the same delivery id;<br>• a stale worker's completion matches no row;<br>• a job whose lease keeps expiring becomes `dead` at its limit |
| **2** | The scheduler queues only, behind the advisory lock; keys of `schedule_id` plus slot; `job_schedules` cursors with each kind's catch-up policy; memory job store | Tests:<br>• two schedulers: each schedule's slot is queued once;<br>• two schedules with the same slot each queue their own job;<br>• a scheduler stopped across three slots queues them by each kind's policy on start-up |
| **3** | Playbook runs as `playbook.run` jobs through `PlaybookEngine.launch`; inbound webhook replies as jobs; the reaper's `RecipeExecution` surface replaced by the lease | A restart during a run: the run is retried, not failed |
| **4** | `PROCESS_ROLES` split on hosted (§6.7): a worker service; the board dispatch loop moves to the worker role; the decision on board tasks as jobs (§10-5) | The API's event loop stays free while runs execute (`core/loop_watchdog.py` reports no stalls under load) |
| **5** | The remaining `create_task` and `BackgroundTasks` sites sorted: move what a person would notice losing; keep what is best-effort | A short list in the PR, site by site |

Waves 1 and 2 stand on their own, and each is a few hundred lines. Every wave keeps the default `PROCESS_ROLES`, all three in one process, working.

## 9. Limits, stated plainly

- **At least once, not exactly once.** A handler can run twice if its worker dies after the work but before it marks the job done. Every handler must be idempotent. The idempotency key stops a second queue, not a second run.
- **A retried Playbook run starts again** unless its executor can resume from its finished steps. Retrying a run that is half done may repeat a side effect, such as a sent email. Wave 3 needs a rule for each kind: retry, resume, or fail and tell the owner, as today.
- **Postgres takes the queue's load.** At today's volume that's well within what PRD-161 already does [I]. Polling stays modest because NOTIFY does the waking. §12 covers growth to thousands of hosted customers, and the path off Postgres if it is ever needed.
- **A long LLM step holds a lease.** Renewal is per kind, and the lease must outlast the longest step between renewals.

## 10. Decisions for the maintainer

1. **The backing store.** *Recommendation:* a `jobs` table in Postgres, generalised from PRD-161: no new dependency, and jobs are queued in the same transaction.
   - **Alternatives:** a Postgres queue library (`procrastinate`, `pgqueuer`), which is a new dependency and needs an issue under AGENTS.md; or Redis (ARQ), which can't queue in the same transaction.
2. **Hosted topology.** Does hosted run, or plan to run, more than one API container? Should hosted have a worker service? *Recommendation:* one worker service, with replicas allowed, and the API unchanged.
3. **One scheduler across containers,** using a Postgres advisory lock, `job_schedules` cursors and APScheduler's memory job store. Should a plain tick loop over the cursors replace APScheduler, now that it is only a timer? *Recommendation:* the lock and cursors in Wave 2; keep APScheduler until the cursors have run a while, then decide.
4. **The workspace worker's Redis queue** (`TASK_RUNNER_BACKEND=queued`). Keep it as it is, or move its intake onto `jobs`? *Recommendation:* keep it in this PRD; it is a separate isolated consumer and works. What does hosted set `TASK_RUNNER_BACKEND` to?
5. **Board tasks as a kind of job** (Wave 4), or keep PRD-161's own columns? *Recommendation:* decide in Wave 4, once Waves 1 to 3 have shown `jobs` working.
6. **#1481 and #1482 in the meantime.** *Recommendation:* merge #1481 as it is. Close #1482 in favour of this PRD's Waves 1 and 2, which delete the resend job.
7. **The backend behind an interface, with switching triggers written down** (§12). *Recommendation:*
   - `enqueue` and `@job_handler` are the only way in, and Postgres is the one backend for now;
   - jobs get their own schema, with an optional `JOBS_DATABASE_URL`;
   - queue metrics from Wave 1;
   - the outbox relay to a broker is the planned next step, built only when a trigger is met.
8. **The PRD number.**

## 11. Open questions

- **Playbook runs and side effects:** should a step that has a side effect (a send, a post, a write to another system) record that it finished, so a retried run skips it? That is the difference between "retry" and "resume" in §9.
- **Who sees a dead job:** the workspace owner (as today for a failed run), an operator view, or both?
- **Credit metering** (`services/credits/jobs.py`) spends money. A missed tick is caught up by the next (§6.4).
  - **Can two overlapping runs double-debit?** `_pending` selects waiting calls without locking them (`services/credits/meter.py`) [T], so it depends on how `meter_call` marks a call charged [?].
  - **Today** one scheduler per container and `max_instances=1` stop overlap within a container. Wave 2's per-slot key stops two schedulers queueing the same slot, but not a slow run overlapping the next slot's.
  - Settle this before Wave 2 moves metering: a conditional "mark charged", or a concurrency of 1 for the `credits.meter` kind.

## 12. Notes: scale, and a path off Postgres

Written after a review question: if hosted served hundreds or thousands of customers, would a Postgres queue be the wrong choice, and should the queue sit behind an interface so its provider can change?

### 12.1 Customers aren't queue throughput

- **A rough estimate [I]:** a few hundred jobs an hour per workspace (runs, callbacks, heartbeats, schedules).

  | Workspaces | Jobs per second |
  |---|---|
  | 1,000 | ~30–80 |
  | 10,000 | a few hundred |

  A `SKIP LOCKED` queue on a well-sized Postgres handles that. Outside this repository, Rails' default job queue (Solid Queue) and Elixir's Oban both run on the database in large production systems.
- **The queue isn't what limits throughput.** Most jobs are LLM calls lasting seconds to minutes. Worker count, model providers' rate limits and cost run out first. A broker improves none of them.

### 12.2 Where Postgres starts to hurt, and what to do first

| Pressure | What it looks like | First response |
|---|---|---|
| Write churn on one hot table | Dead rows, autovacuum lag, index bloat on `jobs` | Short retention for `done` rows; tune autovacuum for the table |
| Sharing the main database | Queue writes slow the app's own reads and writes | Its own schema from Wave 1, and an optional `JOBS_DATABASE_URL` that moves it to its own Postgres. Queueing in the same transaction then needs the outbox (§12.4) |
| `NOTIFY` at high rates | `NOTIFY` takes a lock as each transaction commits, so commits contend at thousands of notifies a second [I] | Coalesce notifies, or poll on a short interval |
| Connections | Every worker replica holds a pool | A pooler, and a fixed pool size per worker |

**What Postgres isn't built for** is high-volume fan-out of events to many consumers, replaying history, and streaming across regions. Those are the signs that a broker or a log is needed. None is needed today.

### 12.3 Fairness matters more than the broker

With thousands of tenants, one workspace can flood the queue and starve the rest. The claim needs per-workspace concurrency limits, and fair ordering (round-robin by workspace, not just oldest first).
- Postgres does this in the claim query.
- PRD-161 already limits slots per agent (`BOARD_DISPATCH_AGENT_SLOTS`) [T].
- A broker doesn't solve it: a queue per tenant gets hard to run at thousands of tenants.

*For Wave 1:* a per-workspace concurrency limit on each kind, from the start.

### 12.4 One interface, and switching triggers

- **The interface is the seam.** `enqueue()` and `@job_handler` are the only way in, and handler code never touches the backend. A backend meets one contract:
  - queue a job, delayed by `run_at`, and inside the caller's transaction;
  - claim, then acknowledge or fail;
  - renew a lease;
  - per-kind retries, backoff and dead-letter;
  - per-kind and per-workspace concurrency;
  - idempotency keys.
- **The contract is what Automatos needs, not what every provider has in common:**

  | Need | Postgres | Brokers |
  |---|---|---|
  | Queue in the caller's transaction | Native | Only through an outbox table and a relay that forwards it |
  | Delayed jobs | Native | RabbitMQ needs a plugin or a TTL workaround; SQS caps delays at 15 minutes; Kafka has none [I] |
  | Long-running jobs | Lease renewal | A visibility timeout, with different limits per provider |
- **The Postgres design is the first step, not a dead end.** Moving to a broker keeps Postgres as the outbox: jobs still commit with the write that caused them, and a relay forwards them. The `jobs` table becomes that outbox.
- **A broker, not a log.** If the queue ever outgrows Postgres, the move is to a task broker (RabbitMQ, SQS, NATS JetStream or Redis Streams), with acknowledgements, retries and dead-letter built in. Kafka is a durable event log. It is awkward for per-message retries, delays and leases, and fits later as an event backbone, such as usage, audit and analytics events for many consumers, not as this job queue.
- **A library interface, not a queue-manager service.** A service between the app and the queue would:
  - add a network hop and a single point of failure to every queue call;
  - break queueing inside the caller's transaction;
  - add a container to the local edition.

  It would earn its place only if services in other languages needed to queue or consume work. Even then the broker's own protocol is usually better.
- **Few backends, all tested the same way.**
  - Postgres now, and at most one broker backend when a trigger is met.
  - One contract test suite runs against every backend in CI.
  - A backend no one runs decays unnoticed, as can happen with `TASK_RUNNER_BACKEND`'s three backends (`local` by default) [I].
- **Switching triggers, measured from Wave 1:**
  - the time from a job's `run_at` to its claim (p95, per kind);
  - queue depth per kind;
  - jobs queued a second, sustained;
  - write volume and autovacuum lag on `jobs`;
  - the queue's share of the main database's load.

  A sustained breach that its own schema or database doesn't fix is the signal to build the relay.
