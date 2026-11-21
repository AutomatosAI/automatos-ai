# PRD-282: Operating a self-hosted install — named private addresses an operator may allow, a readiness check that means ready, and logs a machine can read

**Status:** Draft for Dave's review. Written 20 Nov 2026. No code has been written.
**Owner:** Gerard · **Reviewer:** Dave
**Grounded on:** `automatos-ai` `main @ 97c727136`, read on 20 Nov 2026.
**Labels:** **[T]** read in code at that commit; **[I]** inferred; **[G]** Gerard's decision; **[?]** not known.
**Relates to:** PRD-275 (partner tool packs; its §9 and §10-2 leave private addresses as an open decision). PRD-277 (sign-in through an organisation's own identity provider, which may sit on a private address). PRD-240 (web access). PRD-256 (OpenTelemetry). PRD-279 (the version an install reports).

**Decisions already taken [G] (Gerard, 19 Nov 2026):**

| | Decision |
|---|---|
| G1 | A self-hosted install runs inside an organisation's own network, and the systems it has to reach there are often on private addresses. |
| G2 | Nothing leaves the install unless the operator switches it on. The same holds in reverse: nothing inside the network is reachable from the install unless the operator names it. |

---

## 1. Introduction

Three things an operator of a self-hosted install runs into, each small, each in the way.

**Private addresses are always refused.** Core refuses every outbound call to a private address, and says so: "Always refused, not configurable". That is the right default, and it protects a hosted service from being used to reach its own network. But an install inside an organisation's network exists to reach systems there. A partner's API or a model server on a private address cannot be used at all. PRD-277 already makes one exception of this kind, for the identity provider an install is configured with.

**"Ready" does not mean ready.** The route the load balancer asks answers "ready" once start-up has finished, and checks nothing. An API that has lost its database still says it is ready.

**The logs are for people.** Each line is plain text. The request id exists, and is not in the line. An organisation's monitoring system wants one structured record per line, with the request in it.

## 2. What exists today

### 2.1 Outbound addresses

| Area | Today | Where |
|---|---|---|
| The guard | One function resolves a host, checks every address it resolves to, and returns the checked address so the connection is made to that address and no other. | `orchestrator/core/security/web_access.py:91-135`, `:168-186` **[T]** |
| What is refused | Private ranges, loopback, link-local (which covers the cloud metadata address), carrier-grade NAT, and the IPv6 equivalents. An IPv6 address that carries an IPv4 one is checked as that IPv4 address. | `core/security/url_validator.py:18-37`, `:49-77` **[T]** |
| Whether it can be changed | No. The module says "Always refused, not configurable", and the settings file says private ranges "are ALWAYS refused". `WEB_ACCESS_DENY` can only add refusals. | `web_access.py` docstring; `config.py:528`, `:545` **[T]** |
| Order of checks | The web-access switch, the scheme, the host name, the deny list, the port, then the lookup and the range check. | `web_access.py:104-135` **[T]** |
| Redirects | Never followed automatically. Each hop is checked again, three at most. Partner calls refuse any redirect. | `modules/documents/pinned_fetch.py:32`, `:109-115`; `modules/tools/partner/client.py:79` **[T]** |
| Who uses the guard | Partner tool calls, the agent's web fetch, heartbeat webhooks, run callbacks, document and image fetches, cloud file downloads, socials files, and a workspace's own model endpoint in the hosted edition. | `modules/tools/partner/client.py:111`; `modules/tools/discovery/handlers_web.py:113`; `services/heartbeat_service.py:1397`; `services/run_callbacks.py:152`; `modules/documents/pinned_fetch.py:66`; `core/llm/byok_endpoint.py:88` **[T]** |
| A second helper | `validate_webhook_url` resolves and checks without fixing the address. No production code calls it. | `core/security/url_validator.py:80` **[T]** |
| What PRD-275 says | "A partner API on a private address is refused. … A self-hosted install whose partner API is only reachable privately needs a decision." | `docs/PRDS/PRD-275-PARTNER-TOOL-PACKS.md:164` **[T]** |

### 2.2 Health

| Route | Today | Where |
|---|---|---|
| `/health/ready` | 503 while starting. 200 `{"status":"ready"}` once start-up has finished, whether or not its stages succeeded. Checks nothing live. | `orchestrator/main.py:1291`, `:533`, `:562-564` **[T]** |
| `/health` | Reports the API, the database and configuration. The database check knows about a full connection pool and never waits on one. 503 only when the database is "stuck". Redis is not checked. | `main.py:1304-1354`; `core/health_probe.py` **[T]** |
| Its configuration line | "healthy" only when `DATABASE_URL` is set, so an install configured with the separate `POSTGRES_*` settings reports "degraded". | `main.py:1304-1354` **[T]** |
| The chart's probes | Start-up and liveness ask `/health`. Readiness asks `/health/ready`. | `charts/automatos/templates/api-deployment.yaml:66-84` **[T]** |
| Metrics | `/metrics`, with no sign-in, from the API itself. | `core/monitoring/automatos_metrics.py:156-160` **[T]** |

### 2.3 Logs

| Area | Today | Where |
|---|---|---|
| Format | `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`, text, to the console. | `core/monitoring/automatos_logging.py:389-391` **[T]** |
| Level | `INFO`, fixed in the code. The chart, the compose file and the worker's image each pass a `LOG_LEVEL` that nothing reads. | `automatos_logging.py:351`; `charts/automatos/values.yaml:39` **[T]** |
| Request id | A middleware takes `X-Request-ID` or makes one, and keeps it for the request. It is not in the console line. | `main.py:1044-1065` **[T]** |
| Structured output | Only through an optional relay that ships batches to a separate service. The console stays text. | `config.py:944-959` **[T]** |
| The web server's access log | On, in the server's own default format. | `charts/automatos/templates/api-deployment.yaml:41-49` **[T]** |
| The worker | Its own logging module and its own text format. | `services/workspace-worker/main.py:39-40` **[T]** |

## 3. Goals

- The operator of a self-hosted install can allow a named private address for a named purpose, and nothing wider.
- An address an agent or a user supplies can never use that allowance.
- The hosted edition cannot be given the allowance at all.
- The readiness route says "ready" only when the API can reach its database.
- The existing health route reports each dependency's state, without ever hanging.
- Every log line can be one structured record that carries the request id, in the API and the worker alike.
- With no new setting, everything behaves as today.

## 4. Non-goals

- **Loopback, link-local and the cloud metadata address.** They stay refused for everyone, with no setting.
- **Ranges.** An allowance names a host or one address, and a port. Not a network.
- **An allowance a workspace can set.** It is the operator's, in the install's own settings. It never appears in a screen.
- **Putting sign-in on `/metrics`.** It is kept off the public address by routing, as now.
- **Shipping logs anywhere.** Core writes them. Collecting them is the install's.
- **A health check that restarts the API when a dependency is down.** Restarting does not bring a database back.

## 5. Design

### 5.1 Every outbound call has a purpose

The guard takes one more argument: why the call is being made.

| Purpose | Who sets the address | Callers today |
|---|---|---|
| `partner` | The operator or a workspace manager, when installing a pack | Partner tool calls |
| `identity` | The operator, in the install's settings | PRD-277's sign-in client, when it is built |
| `model` | The operator or a workspace, as a model endpoint | A workspace's own model endpoint |
| `callback` | A workspace's own settings | Heartbeat webhooks, run callbacks |
| `fetch` | An agent, or whatever a user typed or uploaded | Web fetch, document and image fetches, cloud files, socials |

US-101 lists every caller and its purpose before anything changes.

### 5.2 The allowance

One setting, `OUTBOUND_PRIVATE_ALLOW`, empty by default. Each entry is a purpose, a host and a port:

```
partner=orders.internal.example:443,model=models.internal.example:443
```

- **A host is an exact name or one address.** No wildcard, no range.
- **Only two purposes may appear:** `partner` and `model`. An entry for any other stops the install at start-up.
- **The identity provider needs no entry.** PRD-277 already makes the configured issuer's host reachable on a private address, and nothing else (its US-401 and FR-9). That rule stays where it is. Its client passes the `identity` purpose, so the same guard applies it.
- **An entry may only allow a private address.** One that names loopback, a link-local address or the metadata address stops the install at start-up.
- **The check is otherwise unchanged.** The name is still resolved, every address it resolves to is still checked, and the connection is still made to the checked address. An allowed name passes only if every address it resolves to is a private address that may be allowed. A name that also resolves to loopback or link-local is refused.
- **Redirects are checked hop by hop with the same purpose,** as now.
- **The list is logged at start-up,** and each use of an allowance is logged with its purpose and host, never a path or a query.

### 5.3 Where the allowance is not offered

- **The hosted edition refuses the setting.** A hosted service must never be able to reach its own network on a tenant's behalf. With `AUTH_EDITION=saas` and the setting non-empty, the install does not start.
- **`fetch` and `callback` can never use it,** whatever is configured. A test calls the guard with each and an allowed host, and expects a refusal.

### 5.4 One guard

Every place that takes an address from an operator, a workspace or an agent goes through the guard, with its purpose. No module keeps its own list of refused ranges. `validate_webhook_url`, which nothing calls, is deleted. A test fails when a module outside `core/security` defines its own.

### 5.5 Ready means the database answers

`/health/ready` returns 200 only when start-up has finished and the database check passes.

- **It uses the existing check,** which already knows a busy pool from a broken one and never waits on a full pool **[T]**. "Busy" is still ready. "Unreachable" and "stuck" are not.
- **It never blocks.** The answer comes from a result refreshed in the background on a short timer, so the route does no work of its own.
- **Redis is not part of it.** Core runs with Redis features switched off when it has none **[T]**, so its absence must not take the API out of service.
- **Liveness is unchanged.** It stays on `/health`.

### 5.6 What each dependency is doing

`/health` already reports a list of components **[T]**. It gains one entry for each dependency: Redis, object storage, the search index, the memory store and the model provider, beside the database it has today. Each is up, down or not configured, with when it was last checked.

- **No new route is added.** The route that exists is extended.
- **Each dependency is checked in the background,** on its own timer, with its own short time limit. The route only reads the last results, so it stays as fast as it is.
- **A check is a ping or a metadata call.** The model provider is never called to find out. Its entry says whether one is configured and how the last real call ended.
- **The route's status code does not change.** It is 503 only when the database is stuck, as today. A dependency being down never fails the liveness check.
- **The entries hold no address, credential or error text.**
- The same results appear in `/metrics` as one gauge per dependency.

The configuration line in `/health` stops calling an install "degraded" for using the `POSTGRES_*` settings.

### 5.7 Logs a machine can read

Two settings.

| Setting | Default | Meaning |
|---|---|---|
| `LOG_FORMAT` | `text` | `json` writes one JSON record per line |
| `LOG_LEVEL` | `INFO` | The level. It is passed today and ignored. |

A JSON record carries: the time, the level, the logger, the message, the service, the version (PRD-279), the request id, the workspace id when there is one, the method and path of the request, and the trace and span ids when a trace is being recorded. An exception is a field, not extra lines.

- **The web server's access lines use the same format,** so every line a pod writes is the same shape.
- **The request id is added to the text format too.** It costs nothing and makes a text log searchable by request.
- **What is already kept out of logs stays out.** The filters that hide file hashes in paths and stop request addresses from other libraries being logged apply to both formats **[T]**.
- **No new dependency.** The formatter is written with the standard library.
- **The worker does the same,** in its own logging module.

## 6. Waves and user stories

Each wave is one pull request or a short series. CI is the only gate.

### Wave 1: Named private addresses

#### US-101: A table of every outbound caller · M · analysis only
**Description:** As a maintainer, I want every place core calls an address someone else chose listed with its purpose, so that the allowance reaches only the callers it is meant for.

**Acceptance Criteria:**
- [x] A table lists each caller: file, line, who chooses the address, and which of the five purposes it is. (Callers that go through the guard are in the table; fixed and infrastructure destinations are in a short list. [PRD-282-US-101-OUTBOUND-CALLERS.md](PRD-282-US-101-OUTBOUND-CALLERS.md), read at `2d90dab0a`.)
- [x] Each row says whether the call goes through the guard today. (Each table row names its guard call.)
- [x] A caller that does not is listed for Dave, with what it does instead. (Handed to a maintainer privately.)
- [x] A row whose purpose is not obvious is listed for Gerard, not guessed. (Section 4 of the page.)

#### US-102: The guard takes a purpose · M
**Description:** As a developer, I want every outbound check to say why it is being made, so that a rule can depend on it.

**Acceptance Criteria:**
- [ ] `resolve_outbound` and its async form require a purpose, and every caller in US-101's table passes one.
- [ ] With `OUTBOUND_PRIVATE_ALLOW` empty, every existing guard test passes unchanged.
- [ ] A test fails if the guard is called without a purpose.

#### US-103: The allowance · M
**Description:** As the operator of a self-hosted install, I want to name the private systems my install may reach, so that it can use them and nothing else in my network.

**Acceptance Criteria:**
- [ ] `OUTBOUND_PRIVATE_ALLOW` exists in `config.py` and `config-surface.json`.
- [ ] An entry for `partner` or `model`, with a host and a port, lets that purpose reach that host on that port when it resolves to a private address.
- [ ] The same host on another port, or for another purpose, is refused.
- [ ] A name that resolves to an allowed private address and also to a loopback or link-local address is refused.
- [ ] The connection is made to the checked address, as today. A test shows a second lookup is not made.
- [ ] Each use is logged with purpose and host, and no path or query.

#### US-104: What the setting refuses · M
**Description:** As a security reviewer, I want the setting itself to refuse anything dangerous, so that a mistake in configuration cannot open the install's own network.

**Acceptance Criteria:**
- [ ] Start-up fails for an entry with a wildcard, a range, no port, or a purpose other than the two allowed.
- [ ] Start-up fails for an entry naming a loopback, link-local or metadata address.
- [ ] Start-up fails when the edition is `saas` and the setting is not empty.
- [ ] Each failure names the entry and the reason.
- [ ] The valid entries are logged once at start-up.

#### US-105: An agent can never use it · S
**Description:** As a security reviewer, I want proof that an address an agent or a user supplies cannot use the allowance, so that allowing a partner's host does not let a prompt reach it.

**Acceptance Criteria:**
- [ ] With a host allowed for `partner`, a `fetch` to that host is refused. So is a `callback`.
- [ ] The test covers the agent's web fetch, a document fetch and a webhook, each through its real caller.
- [ ] A redirect from a public address to an allowed private one is refused for `fetch`.

#### US-106: One guard · M
**Description:** As a maintainer, I want one place that decides which addresses are refused, so that a rule changed there is changed everywhere.

**Acceptance Criteria:**
- [ ] Every caller in US-101's table goes through the guard.
- [ ] `validate_webhook_url` is deleted, with its export.
- [ ] A test fails if a module outside `core/security` defines its own list of refused ranges.
- [ ] Partner calls pass `partner`, and PRD-275's §9 and §10-2 are updated to say how a private partner API is allowed.

### Wave 2: Health

#### US-201: Ready means the database answers · M
**Description:** As an operator, I want an API that has lost its database to stop receiving traffic, so that people get a clear failure and not a broken page.

**Acceptance Criteria:**
- [ ] `/health/ready` returns 200 only when start-up has finished and the last database check passed.
- [ ] "Busy" is ready. "Unreachable" and "stuck" are not.
- [ ] The route does no database work itself. A test with a full pool shows it answers at once.
- [ ] `test_f229_health_off_the_loop.py`, `test_f458_health_never_waits_on_a_full_pool.py` and `test_prd262_health_gates.py` pass, changed only where they pin the old meaning of ready.
- [ ] The chart's probes are unchanged.

#### US-202: The state of each dependency, in `/health` · M
**Description:** As an operator, I want the health route to tell me which dependency is down, so that I do not work it out from error logs.

**Acceptance Criteria:**
- [ ] `/health`'s list of components has an entry for Redis, object storage, the search index, the memory store and the model provider, each up, down or not configured, with the time of its last check.
- [ ] Each check runs in the background with its own time limit. The route only reads results, and a test with every dependency hanging shows it still answers at once.
- [ ] The model provider is never called by a check. Its entry comes from whether one is configured and how the last real call ended.
- [ ] With Redis down, `/health` still returns 200. Its status code changes only for the database, as today.
- [ ] No entry holds an address, a credential or error text from a dependency.
- [ ] No new route is added, and the three existing health tests pass.

#### US-203: The same in metrics, and the configuration line · S
**Description:** As an operator, I want an alarm to be able to watch each dependency, so that I hear about it before a person does.

**Acceptance Criteria:**
- [ ] `/metrics` has one gauge per dependency, 1 for up and 0 for down, absent when not configured.
- [ ] `/health`'s configuration line is "healthy" for an install configured with `POSTGRES_*` and no `DATABASE_URL`.

### Wave 3: Logs

#### US-301: One JSON record per line · M
**Description:** As an operator, I want logs my monitoring system can parse, so that I can search by request, workspace and level.

**Acceptance Criteria:**
- [ ] `LOG_FORMAT` and `LOG_LEVEL` exist in `config.py` and `config-surface.json`.
- [ ] With `LOG_FORMAT=json`, every line the API writes is one JSON object with the fields in section 5.7.
- [ ] An exception is one record with the trace in a field.
- [ ] The web server's access lines are in the same format.
- [ ] With `LOG_FORMAT=text`, a line is as today with the request id added.
- [ ] `test_1178_request_log_guard.py` and `test_prd256_o5_logs_metrics.py` pass, and the filters they cover apply to both formats.
- [ ] No new dependency is added.

#### US-302: The worker · S
**Description:** As an operator, I want the worker's logs in the same shape, so that one parser reads everything.

**Acceptance Criteria:**
- [ ] The worker reads the same two settings in its own config module.
- [ ] With `json`, its lines carry the same fields, with its own service name.
- [ ] `test_prd262_worker_starlette.py` passes.

#### US-303: The chart · S
**Description:** As an operator, I want to choose the format when I install, so that I need no patch.

**Acceptance Criteria:**
- [ ] The chart passes `LOG_FORMAT` to the API, the migration job and the worker, beside the `LOG_LEVEL` it already passes.
- [ ] With `LOG_FORMAT` unset, the rendered chart is unchanged. A test pins it.

## 7. Functional requirements

- FR-1: Every outbound call to an address chosen by an operator, a workspace or an agent must go through the guard with a purpose.
- FR-2: With `OUTBOUND_PRIVATE_ALLOW` empty, the guard must behave exactly as today.
- FR-3: An allowance must name a purpose, an exact host or one address, and a port.
- FR-4: Only `partner` and `model` may be allowed by the setting. `identity` reaches only the configured issuer's host, by PRD-277's rule. `fetch` and `callback` must be refused a private address whatever is configured.
- FR-5: Loopback, link-local and metadata addresses must be refused for every purpose, with no setting.
- FR-6: The hosted edition must refuse to start with a non-empty allowance.
- FR-7: The connection must be made to the address that was checked.
- FR-8: `/health/ready` must return 200 only when the database check passes, and must never block.
- FR-9: `/health` must report each dependency from background results, must not change its status code because a dependency other than the database is down, and must expose no address, credential or error text.
- FR-10: With `LOG_FORMAT=json`, every line the API and the worker write must be one JSON record that carries the request id when there is one.
- FR-11: What is kept out of logs today must be kept out in both formats.

## 8. Decisions for Gerard

Each has a recommendation, which stands unless he changes it.

| # | Decision | Recommendation | Why |
|---|---|---|---|
| D-1 | What an allowance names | A purpose, an exact host or address, and a port | A range allows whatever is later put in it. A name says what was meant. |
| D-2 | Which purposes may be allowed | `partner` and `model` | Each is an address an operator or a manager chose on purpose. An agent's fetch is not. The identity provider is already covered by PRD-277. |
| D-3 | The hosted edition | Refuses the setting outright | It must never reach its own network for a tenant |
| D-4 | Who sets it | The operator, in the install's settings. Never a workspace. | It widens what the whole install can reach |
| D-5 | What readiness depends on | The database only | Without it nothing works. Without Redis, core still serves. |
| D-6 | The default log format | Text, as today, with the request id added | A local user reads the console. An organisation sets `json`. |

## 9. Technical considerations

- **House rules that apply:** settings only through `config.py` and `config-surface.json`; replace, don't shim; no blocking work on the event loop; no new dependency; signed-off commits; pull requests under about 400 lines.
- **Ask first.** The allowance changes a guardrail. AGENTS.md asks for an issue before that, so Wave 1 opens with one, and US-101's table is its first comment.
- **PRD-277.** Its sign-in client does not exist yet. When it is built it passes `identity`, and nothing here waits for it. Its US-401 and FR-9 say no other private address becomes reachable. This PRD adds two named kinds, so those two sentences are changed in this pull request to say "through sign-in", with a pointer here.
- **PRD-275.** The partner client already skips the web-access switch and keeps every other check **[T]** (`modules/tools/partner/client.py:111`). It gains only the purpose.
- **A workspace's own model endpoint** is checked only in the hosted edition today **[T]** (`core/llm/byok_endpoint.py:84-85`). What the `enterprise` edition does there is decided by PRD-277's capability table, and this PRD gives it the `model` purpose to use.
- **Tests that pin today's shape and must be changed on purpose:** `test_ssrf_blocked_ranges.py`, `test_prd240_web_access.py`, `test_f375_a_document_fetch_is_pinned_and_checks_every_hop.py`, `test_heartbeat_webhook_pinned.py`, and the three health tests named in US-201.

## 10. Success metrics

- A self-hosted install calls a partner API on a private address after one line of configuration, and an agent's web fetch to the same address is still refused.
- An API whose database is unreachable stops being sent traffic within its probe's period.
- An operator finds every line of one request by its id, in the API's and the worker's logs.
- The hosted edition's behaviour is unchanged, shown by its existing tests.

## 11. Risks

| Risk | What limits it |
|---|---|
| The allowance becomes a way to reach an install's network from a prompt | Only two purposes can use it, neither of them an agent's, and a test proves a fetch is refused |
| A name on the list is pointed at a dangerous address later | Every address it resolves to is checked on every call, and loopback and link-local are never allowed |
| An operator allows too much | An entry is one host and one port. There is no range to write. |
| Readiness that depends on the database takes the only API out of service during a short blip | That is the correct signal. The check tolerates a busy pool, and liveness does not restart the pod. |
| A background check adds load or cost | Each is a ping or a metadata call on a slow timer, and none is a model call |
| JSON logs carry something text logs did not | The same filters apply to both, and the record's fields are a fixed list |

## 12. Open questions

1. **Does every caller reach the guard today?** US-101 answers it, and anything it finds goes to Dave before it is written up.
2. **Should `callback` ever be allowed a private address,** for an organisation whose own systems receive webhooks? Not in this PRD. It would need its own decision.
3. **How often should each dependency be checked?** Often enough for an alarm, rarely enough to cost nothing.
