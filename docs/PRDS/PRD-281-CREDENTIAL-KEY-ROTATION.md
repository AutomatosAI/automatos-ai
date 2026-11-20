# PRD-281: Rotating the credential key — more than one key at a time, a job that re-encrypts, and no key made by accident

**Status:** Draft for Dave's review. Written 20 Nov 2026. No code has been written.
**Owner:** Gerard · **Reviewer:** Dave
**Grounded on:** `automatos-ai` `main @ 97c727136`, read on 20 Nov 2026.
**Labels:** **[T]** read in code at that commit; **[I]** inferred; **[G]** Gerard's decision; **[?]** not known.
**Relates to:** PRD-276 (durable background work, which runs the re-encryption). PRD-277 (the `enterprise` edition and its install admin). PRD-261 (step-up on sensitive routes). PRD-273 (moving a workspace, which leaves these secrets behind on purpose).

**Decisions already taken [G] (Gerard, 19 Nov 2026):**

| | Decision |
|---|---|
| G1 | An organisation's install must be able to rotate every secret it holds. The credential key is the one it cannot rotate today. |

---

## 1. Introduction

People save credentials in Automatos: the key to a business system, a model provider's API key, a signing secret. Core encrypts each one before it reaches the database, with one key the install holds, the credential key.

That key can never be changed. The code has a `rotate_key` method, and its body is one line that raises "Key rotation not implemented in MVP". An organisation that rotates every secret on a schedule, or that has to react to a suspected leak, has no way to do either for this one.

There is a second problem beside it. When the key is missing, core does not stop. It makes a new one, writes it to a file and carries on. Every credential already in the database is then unreadable, and the only sign is a warning in the log.

This PRD lets an install hold more than one key at a time, adds a job that re-encrypts everything under the newest, and stops core inventing a key when data already exists.

## 2. What exists today

| Area | Today | Where | What it means |
|---|---|---|---|
| The cipher | One Fernet key, used as given. No key derivation. | `orchestrator/core/credentials/encryption.py:12`, `:43` **[T]** | One key encrypts and decrypts everything. |
| Where the key comes from | In order: the `CREDENTIAL_ENCRYPTION_KEY` setting, a key file, and failing both, a newly generated key written to that file. | `encryption.py:46-134`; `config.py:589`, `:595` **[T]** | An install with no key gets a fresh one without being asked. |
| A missing key | A warning: "This will NOT be able to decrypt existing credentials!" No edition is treated differently. | `encryption.py:94-97` **[T]** | Existing credentials are orphaned, and the install keeps running. |
| A malformed key | Refused, with `EncryptionKeyError`. | `encryption.py:76`, `:91` **[T]** | A bad key stops the install. A missing one does not. |
| What is stored | A bare Fernet token. Nothing in it says which key made it. | `encryption.py:136-213` **[T]** | Core cannot tell an old key's values from a new key's without trying to decrypt them. |
| Rotation | `rotate_key()` raises `NotImplementedError("Key rotation not implemented in MVP")`. | `encryption.py:215-225` **[T]** | Not possible. |
| The service object | One shared instance holds the cipher. Three places in one module build their own. | `encryption.py:246-259`; `modules/nl2sql/service.py:138`, `:424`, `:914` **[T]** | A change to how the key is loaded has to reach four constructions. |
| `get_key_info()` | Reports the default key file's path and ignores the configured one. | `encryption.py:227` **[T]** | It can describe a file the install is not using. |
| Where encrypted values live | Three columns, two JSON values and one job payload. | below **[T]** | Six stores to re-encrypt. |
| Tests | The key-file path, and one integration's token. None for rotation. | `tests/test_prd209_credential_key_file.py`, `tests/test_prd173_shopify_token_encryption.py` **[T]** | Rotation starts with no coverage. |

The six stores:

| Store | Holds | Where |
|---|---|---|
| `credentials.encrypted_data` | Saved integration credentials | `core/models/credentials.py:65`, `:74` **[T]** |
| `user_api_keys.encrypted_key` | A workspace's own model provider keys | `core/models/core.py:157`, `:164` **[T]** |
| `workspace_signing_secrets.secret_encrypted` | A workspace's callback signing secret | `core/models/run_callbacks.py:29`, `:32` **[T]** |
| `workspaces.settings`, one key | One integration's access token, inside a JSON setting | `api/shopify.py:212` **[T]** |
| `sites.secrets` | A site's server-side credentials, "encrypted at rest by app layer". How the value is laid out inside the column is not read yet **[?]**. | `core/models/sites.py:88`; `services/sites.py:234` **[T]** |
| A pending job's payload, one key | Contact details sealed for a delivery job, removed when the job ends | `services/destinations/delivery_jobs.py:77-93` **[T]** |

## 3. Goals

- An install can replace its credential key without losing a saved credential and without stopping.
- During a rotation both keys work, so no request fails because of it.
- One job re-encrypts every stored value under the new key, can be stopped and run again, and says exactly what it did.
- An operator can find out whether anything still depends on an old key before removing it.
- Core never invents a key for an install that already holds encrypted data.
- A new place that encrypts cannot be added without rotation knowing about it.

## 4. Non-goals

- **Wrapping the key with a cloud's key service.** Keeping the key itself inside a key management service is a different design, and one that ties core to a provider. Not here.
- **A key for each workspace.** One key for the install, as now.
- **Rotating on a timer by itself.** An operator starts a rotation. Core makes it safe.
- **Other secrets.** Signing secrets for webhooks, the platform key and session secrets are rotated by replacing a setting, and need nothing from this PRD.
- **A screen for it.** The work is an operator's, done rarely. Section 8 asks whether that should change.

## 5. Design

### 5.1 More than one key

Two settings instead of one:

| Setting | Holds | Used for |
|---|---|---|
| `CREDENTIAL_ENCRYPTION_KEY` | The current key | Encrypting, and decrypting |
| `CREDENTIAL_ENCRYPTION_KEYS_PREVIOUS` | Older keys, newest first. Empty unless a rotation is under way. | Decrypting only |

The cipher becomes the cryptography library's own multi-key form: it encrypts with the first key and tries each in turn to decrypt. The stored format does not change, so there is no migration, and a value written before this change reads exactly as before.

### 5.2 A rotation, start to finish

1. **Add the new key.** The operator makes a new key, sets it as the current one, and moves the old one to the previous list. The install restarts. From here new values are written under the new key, and old values still read.
2. **Re-encrypt.** The operator starts the job in section 5.4. It rewrites every stored value under the new key.
3. **Check.** The report in section 5.5 says nothing depends on the old key.
4. **Remove the old key.** The operator empties the previous list. The install restarts.

Nothing is unreadable at any step. If the job is never run, the install keeps working for as long as the old key stays in the list.

### 5.3 A list of every store

One module lists each place an encrypted value lives, and for each one how to read the values, and how to write one back. The six stores of section 2 are its first entries.

A test fails when a caller of the encryption service appears that the list does not account for. Someone who adds a sixth store finds out from CI, not from a rotation that silently skipped it.

### 5.4 The job

A durable job (PRD-276), started by an operator.

- **It walks each store in batches,** and for each value re-encrypts it under the current key. A value already under the current key is left alone.
- **It can be stopped and started again.** It keeps its place, and running it twice does no harm.
- **A value no key can read is counted, not fatal.** The job records which store and which row, never the value, and carries on.
- **It does not hold a long lock.** Each batch is its own short transaction, so the install keeps serving.
- **It ends with a report:** for each store, how many values it rewrote, how many were already current, and how many no key could read.
- **Its start and its end are written to the audit log,** with who started it.

### 5.5 Is anything still on an old key

A report an operator can ask for at any time. For each store: how many values the current key reads by itself, and how many need a key from the previous list.

It reads values to answer, so it is a job with a result, not something the API works out on each request. Step 4 of a rotation is safe when every store reports zero.

### 5.6 No key made by accident

- **A key is generated only on a true first run:** in the `local` edition, and only when no store holds an encrypted value.
- **In every other case a missing key stops the install,** with a message that says the key is missing and that data encrypted under it exists.
- **At start-up core tries the keys on one stored value.** If none can read it, the install stops and says the key does not match the data. That is a wrong key or the wrong database, and either is better found at start-up than at the first failed connection.

### 5.7 One service

The three places that build their own encryption service use the shared one. `get_key_info()` reports the file the install actually uses, and how many previous keys are configured. It never reports a key.

### 5.8 Who can start it

Starting a rotation job and reading the report are for a platform admin, behind the existing step-up check. In the `enterprise` edition that is the install admin (PRD-277). In the `local` edition, which has one operator and no step-up, they are the operator's, as its other admin routes are.

The key itself is never sent to, or returned from, any route. It arrives only as a setting, from wherever the install keeps its secrets.

## 6. Waves and user stories

Each wave is one pull request or a short series. CI is the only gate.

### Wave 1: More than one key, and no accidents

#### US-101: A current key and previous keys · M
**Description:** As an operator, I want core to accept an old key beside a new one, so that changing the key does not break what is already saved.

**Acceptance Criteria:**
- [ ] `CREDENTIAL_ENCRYPTION_KEYS_PREVIOUS` exists in `config.py` and `config-surface.json`.
- [ ] A value encrypted under a previous key decrypts. A new value is encrypted under the current key.
- [ ] With the previous list empty, behaviour is exactly as today, and a test pins it with a value written before this change.
- [ ] A malformed key in either setting stops the install with a message that names the setting and quotes no key.
- [ ] `ruff check` and the backend test lanes pass in CI.

#### US-102: No key made by accident · M
**Description:** As an operator, I want a missing or wrong key to stop the install, so that I never learn about it from a credential that will not open.

**Acceptance Criteria:**
- [ ] A key is generated only in the `local` edition and only when no store holds a value.
- [ ] With a value stored and no key configured, start-up fails with a message that says so.
- [ ] With a value stored and no configured key able to read it, start-up fails with a message that says the key does not match the data.
- [ ] The check reads one value from one store, runs off the event loop, and logs nothing from the value.
- [ ] A fresh `local` install still starts with no key and makes one, as today. A test pins it.

#### US-103: One encryption service · S
**Description:** As a developer, I want every caller to use the same service object, so that how the key is loaded is decided once.

**Acceptance Criteria:**
- [ ] The three direct constructions in `modules/nl2sql/service.py` use the shared service.
- [ ] A test fails if `EncryptionService(` is called anywhere outside its own module and its tests.
- [ ] `get_key_info()` reports the configured key file and the number of previous keys, and never a key.

### Wave 2: Re-encrypting

#### US-201: The list of stores · M
**Description:** As a maintainer, I want one list of everywhere an encrypted value lives, so that a rotation cannot miss one.

**Acceptance Criteria:**
- [ ] One module lists the six stores of section 2, each with how to read its values in batches and how to write one back.
- [ ] The ones that are a value, or a key, inside a JSON column are handled as such: only that part changes. The pull request says how `sites.secrets` is laid out.
- [ ] A test lists every caller of the encryption service and fails when one is not accounted for by an entry.
- [ ] The pull request includes the table of callers, with file and line, that the test is built from.

#### US-202: The re-encryption job · L
**Description:** As an operator, I want one job to move every stored value to the current key, so that the old key can be retired.

**Acceptance Criteria:**
- [ ] A durable job walks every store in the list, in batches, each batch in its own transaction.
- [ ] A value already under the current key is not rewritten.
- [ ] Stopped part-way and started again, it finishes without rewriting what it already did.
- [ ] A value no key can read is counted with its store and row, and the job continues. The value is never logged.
- [ ] A value changed by a user while the job runs is not overwritten with its old content. A test shows it.
- [ ] The job's report gives, for each store, the three counts in section 5.4.
- [ ] Every query is scoped as the store's own code scopes it. The job crosses workspaces only because it is the install's own maintenance, and the pull request says so.

#### US-203: The report · M
**Description:** As an operator, I want to know whether anything still needs an old key, so that I remove it only when it is safe.

**Acceptance Criteria:**
- [ ] A job reports, for each store, how many values the current key reads alone and how many need a previous key.
- [ ] It changes nothing.
- [ ] After US-202's job has finished on a test database, it reports zero for every store.
- [ ] With a pending job whose payload was sealed under the old key, it reports that one.

### Wave 3: Running it for real

#### US-301: Starting it, and who may · M
**Description:** As an install admin, I want to start a rotation and read its result through the API, so that I need no shell in the install.

**Acceptance Criteria:**
- [ ] One route starts the re-encryption job, one starts the report, and one returns the latest result of each.
- [ ] Each requires a platform admin, and a recent step-up in the editions that have one. A lower role is refused, and a test shows it.
- [ ] No route accepts or returns a key.
- [ ] Starting and finishing each job writes an audit entry with who started it and the counts.
- [ ] The routes are in `route-manifest.json`.

#### US-302: The whole rotation, in a test · M
**Description:** As a maintainer, I want the four steps of section 5.2 run end to end in CI, so that the procedure is proven and stays proven.

**Acceptance Criteria:**
- [ ] One test stores a value in each of the six stores under a first key.
- [ ] It adds a second key as current, and reads every value.
- [ ] It runs the job, and the report shows zero on the old key.
- [ ] It removes the first key, and reads every value.
- [ ] It then shows that with the first key alone, the rewritten values cannot be read.

#### US-303: The chart and the page · S
**Description:** As an operator, I want the chart to carry the previous keys and a page to tell me the steps, so that a rotation is a procedure and not an investigation.

**Acceptance Criteria:**
- [ ] `CREDENTIAL_ENCRYPTION_KEYS_PREVIOUS` is an optional key of the chart's secret, given to the same components that receive the current key.
- [ ] A page in `docs/` gives the four steps, how to make a key, how to read the report, and what to do when the job reports values no key can read.
- [ ] The page says what a rotation does not cover: values exported or copied elsewhere before it.

## 7. Functional requirements

- FR-1: Core must encrypt with the current key and decrypt with the current key or any previous key.
- FR-2: The stored format of an encrypted value must not change.
- FR-3: With no previous key configured, behaviour must be unchanged.
- FR-4: Core must not generate a key unless the edition is `local` and no store holds an encrypted value.
- FR-5: Core must refuse to start when stored data exists and no configured key can read it.
- FR-6: Every place an encrypted value is stored must be in one list, and CI must fail when a caller of the encryption service is not accounted for.
- FR-7: The re-encryption job must be safe to stop, start again and run twice.
- FR-8: The job must never log, return or store a decrypted value or a key.
- FR-9: The job must not overwrite a value that changed while it ran.
- FR-10: Starting a rotation and reading its report must require a platform admin and, where the edition has step-up, a recent one. Both must be audited.
- FR-11: No route may accept or return a key.

## 8. Decisions for Gerard

Each has a recommendation, which stands unless he changes it.

| # | Decision | Recommendation | Why |
|---|---|---|---|
| D-1 | How core holds more than one key | The cryptography library's multi-key cipher, with a current key and a previous list | No change to what is stored, no migration, and nothing new to maintain |
| D-2 | What happens when the key is missing or wrong | Stop, in every edition, unless it is a first run of `local` with nothing stored | The alternative has already been written into a warning: credentials that cannot be opened |
| D-3 | Who starts a rotation | A platform admin, through the API, with step-up | An organisation's install may give nobody a shell |
| D-4 | A screen for it | None for now | It is rare, and an operator's. A screen can follow the API. |
| D-5 | The pending job's payload | Re-encrypted by the same job | Otherwise removing the old key breaks a delivery that was waiting |

## 9. Technical considerations

- **House rules that apply:** settings only through `config.py` and `config-surface.json`; every route in `route-manifest.json`; replace, don't shim; no blocking work on the event loop; signed-off commits; pull requests under about 400 lines.
- **Ask first.** The job rewrites existing data and reads across workspaces, and the new routes are admin routes. AGENTS.md asks for an issue before a change to existing data, to tenancy or to permissions, so Wave 2 opens with one.
- **No new dependency.** The multi-key cipher is in the `cryptography` package core already uses.
- **PRD-276.** The job and the report are registered with `@job_handler` and enqueued like any other durable job.
- **PRD-273.** A moved workspace leaves these three columns behind on purpose (`services/workspace_move/table_map.py:111-118` **[T]**). Rotation changes nothing there.
- **The workspace worker** is not given the credential key **[T]**, so a rotation restarts the API and the migration job's next run, and nothing else.
- **Tests that pin today's shape and must be changed on purpose:** `test_prd209_credential_key_file.py`.

## 10. Success metrics

- A rotation is completed on a test database with a value in every store, and every value reads afterwards.
- No request fails during it.
- The report reads zero before the old key is removed.
- An install started against a database with the wrong key stops within seconds, with a message that says why.

## 11. Risks

| Risk | What limits it |
|---|---|
| A store is missed, and its values become unreadable when the old key is removed | The list is checked by a test that fails on any caller it does not cover, and the report must read zero first |
| The job overwrites a credential a user just changed | It writes back only if the stored value is still the one it read, and a test shows it |
| The start-up check stops an install that used to start | It stops only when stored data cannot be read, which was already broken |
| An operator removes the old key before running the job | Nothing is lost while the key still exists. The page says to keep it until the report reads zero, and start-up refuses to run against data it cannot read. |
| The job is slow on a large install | Batches, short transactions, and it can be stopped and resumed |

## 12. Open questions

1. **Does anything else hold a value made by this service** that the six stores do not cover, such as an export or a cache? US-201's table answers it.
2. **Should the report also run by itself after a rotation job ends?** It would save a step.
3. **How many previous keys should core accept?** One is enough for a rotation. More than a few suggests a rotation was never finished.
