# Ralph Review Prompt — PRD-270 W2 Cloud bundles

You are a fresh-context **adversarial reviewer**. The build claims PRD-270 Wave 2 is complete (US-270-11a, 11b, 11c, 12a, 12b, 12c, 12d, 13a, 15a, 13b, 13c, 13d, 14a, 14g, 14b, 14c, 14d, 14e, 16, 14f). This wave serves approved community content to anyone on SaaS, writes marketplace rows and plugin files into every local install, and puts a gate in front of the web's install calls. Review it as someone trying to:
- read a credential, connection, email, workspace or row id, configuration, schedule or webhook from a bundle or a public route;
- make a local install fetch a URL of the attacker's choosing, or write a row from a bundle the validator should have refused;
- get a key past the strict spec, or make a bundle item look Official;
- make a Sync overwrite an operator's own agent, Playbook, package or plugin, or a workspace's installed copy;
- store a plugin's bytes before its checks, or make a Sync call a model;
- install or enable a Cloud item from the web without its disclosure's Add, or from Auto's chat without its approval;
- break an install on SaaS, or a seeded item's install from chat.

## What to look for

**The migration and model (US-270-11a):**
- the revision doesn't chain onto the base's head, or alters a table, or changes a row;
- a missing guard (`IF NOT EXISTS`), CHECK or partial index;
- the model's CHECK text not equal to the DDL's;
- the downgrade deleting anything but the table;
- a pin loosened or a test lost;
- a setting read outside `config.py` or missing from `config-surface.json`.

**The format (US-270-11b, 11c):**
- a spec level that accepts unknown keys;
- a field that can carry configuration, schedule_config, an id, an email, a path, a token or matched file text;
- a cap missing against its column (name 255, category 100, version 50 after `local_version`);
- a ref accepted where its kind is wrong;
- canonical bytes that depend on key order;
- a message that echoes input;
- the corpus not covering every code.

**The export (US-270-12a, 12b):**
- a seeded, unapproved, inactive, system or role-restricted agent exported;
- a seeded or unapproved Playbook exported;
- two rows of one lineage both exported;
- the submitter's workspace agent ids in a step;
- schedule_config, configuration or an email in a bundle;
- a plugin exported that is not approved, active and safe in both the row and its scan;
- `matched_text`, an S3 bucket or path, or `approved_by` in a plugin bundle;
- the publisher an email or `created_by`;
- anything claiming Official;
- a bundle the strict validator would refuse being listed.

**The hosted routes (US-270-12c, 12d):**
- W1's catalogue themes changed, or the items added after W1's size cap, ETag or cache header;
- a bundle served for an unlisted revision;
- a plugin zip served for an unexported plugin or another version;
- binary files corrupted (decoded as text);
- a missing SERVING_HEADERS on a 404;
- the routes answering on a local edition, carrying `get_request_context_hybrid`, or unthrottled;
- a non-GET route added.

**The local reads of the catalogue (US-270-13a):**
- a URL taken from an entry or a bundle fetched;
- W1's `fetch.py` edited or bypassed, or another HTTP client used;
- the size cap, sha256, validation or the entry match missing;
- an entry claiming Official accepted;
- an unknown type failing the Sync or counted;
- W1's theme checks changed;
- credentials sent.

**Plugins (US-270-15a):**
- any byte stored before the zip, manifest and static checks;
- the upload's zip rules weaker;
- a critical static finding stored;
- `PluginScanService.scan_plugin` or `llm_security_scan` called;
- a failure leaving objects or rows;
- a locally-held slug overwritten;
- the plugin cap ignored;
- a plugin row sharing the item pass's transaction;
- enablement changed (it stays the existing route and permission).

**The writers (US-270-13b, 13c):**
- a row without a mapping updated;
- a workspace copy read or written;
- an agent name collision not refused;
- a package written with an unresolvable member;
- a Playbook step pointing at an agent outside this install's marketplace;
- schedule_config other than manual;
- the model's validators skipped;
- the item pass in more than one transaction, or a failure leaving some rows;
- `slug` set on a Cloud agent;
- the counts wrong.

**The Sync (US-270-13d):**
- the item passes running when W1's pass failed, on SaaS, with the URL empty, or concurrently with another Sync;
- the catalogue fetched twice;
- who can Sync changed;
- W1's answer keys or status broken;
- a route missing from the manifest or shadowed.

**Auto's chat installs (US-270-14g):**
- a Cloud agent, package or plugin installed from chat with no grant, or on a grant for another call;
- the grant not single-use, or the dial or the owner's own turn skipping it;
- a seeded or operator item now asking;
- the card's text missing the prompt (or the no-prompt line), the apps, the model or the trust label;
- bundle text that renders as a link, image or HTML in the card;
- an error that installs instead of asking;
- `platform_executor.py`, `tool_grants.py`, `api/approval_grants.py` or the action definitions edited;
- a web install meeting this gate (asked twice).

**The local reads and the web (US-270-14a–f):**
- a local read answering on SaaS, reading another workspace's copies, or calling the network;
- an install route reachable from the web for a Cloud item without the disclosure's Add (try each of the four calls, from each card, list, modal and notice);
- the gate awaiting anything on SaaS or for a non-install request;
- the gate importing api-client (a cycle);
- `api-client.ts` grown;
- an untouchable component touched;
- server text rendered as HTML or markdown;
- 'Official' shown on a bundle item;
- a second ink primary;
- the update notice wrong (a copy at the current version still noticed, or another workspace's copy counted);
- the spec untagged, reaching the real Automatos Cloud, or adding a dependency.

**Docs (US-270-16):**
- a local path, partner name or secret;
- `self-hosting.md` grown;
- the settings or the plugin update-by-reference behaviour missing.

**The wave as a whole:**
- a story's evidence doesn't exist;
- an AC is marked DONE that the code doesn't meet;
- a W1 seam changed beyond the one change K4 names;
- another kit's file changed;
- a file over 800 lines grew (the K3 exceptions aside);
- a pin file lost a test;
- a new dependency, HTTP client, model call or private key;
- the PR undrafted before this review passed.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd270w2.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD -- . ':!scripts/ralph'
```

The base is W1's tip while W1 is unmerged: W1's own diff is not this wave's. Never file a finding against a line this diff doesn't touch, except where an AC says the base's behaviour had to change and it didn't.

Read beside the diff:
- `scripts/ralph/prd-270w2.json` (binding: its `decisions` and each story's ACs and `notes`);
- `docs/PRDS/PRD-270-SYNC-WITH-AUTOMATOS-CLOUD.md` §5.1–5.3, §5.6, §5.8, §6 (Phase 2) and §8;
- W1's code the notes mark RE-VERIFY (its catalogue route, `feed.py`, `pull.py`, `fetch.py`, the Sync route);
- `AGENTS.md` and `frontend/DESIGN_SYSTEM.md` §22;
- the `ownerTest` list.

## Verification

- **code-review** skill, or the code-reviewer agent, on the diff, **in the foreground**.
- **security-reviewer** agent, in the foreground, on:
  - `orchestrator/services/marketplace_bundles/` (all of it);
  - `orchestrator/api/marketplace_catalog_items.py` and `marketplace_cloud_items.py`;
  - the changed lines of W1's `api/marketplace_catalog.py` (or `services/marketplace_cloud/catalog.py`), `services/marketplace_cloud/feed.py`, `pull.py` and `api/marketplace_cloud_sync.py`;
  - `modules/tools/discovery/cloud_install_asks.py` and `confirmation_gate.py`'s changed lines;
  - `frontend/lib/marketplace/cloud-install-gate.ts`, `frontend/lib/api-client.ts`'s changed lines, `frontend/components/marketplace/cloud/` (W2's files).

  Ask it explicitly about:
  - what an unauthenticated caller can read from the bundle and plugin routes;
  - whether any key can pass the strict spec;
  - SSRF and redirects in the bundle and plugin fetches;
  - bytes stored before verification;
  - a model call anywhere in a Sync;
  - an operator's own row or a workspace copy written;
  - every path to an install or enable of a Cloud item from the web without Add, or from Auto's chat without its approval;
  - the gate on SaaS.
- **database-reviewer** agent, in the foreground, on `orchestrator/alembic/versions/prd270_w2_cloud_bundles.py`, the model, the item pass's one transaction (`import_items.py`, `import_agents.py`, `import_playbooks.py`, `import_packages.py`) and the plugin pass's commits (`plugin_import.py`).
- **Integration seams.** W1's review fixes met W2 code at the merge 3a41ff8e0, in files that merged with no conflict. Check both, and add a test where one is missing:
  1. **The catalogue build.** W1's ce215a1c0 makes `build_catalog` do every database read first, end its read transaction, then read each theme.json from storage. W2 wraps it as `with_items(build_catalog(db), db)`, so the agent, Playbook, package and plugin exports read the database afterwards, in a new transaction. Confirm that no pool connection is held over a storage read. Also check whether W2's Postgres catalogue test exercises the path that releases: its seeded rows are flushed, so its transaction is kept.
  2. **The Sync's item passes.** W1's RVW-3 rule: no database transaction stays open while Automatos Cloud answers (`ensure_stored` takes the Cloud row's columns; `_run_sync` rolls back before the fetch). Check `services/marketplace_bundles/sync_items.py`, and the bundle and plugin fetches it makes, against that rule.
- Run `bash scripts/ralph/acceptance-prd270w2.sh` yourself. It waits for the wave's one CI run and the browser lane on the tested commit, and reads the PR body and its base.
- Spot-check five marked acceptance criteria at random against the code and the tests. Always include one from US-270-11b, one from US-270-15a, one from US-270-14g and one from US-270-14d.
- **Nothing runs on the owner's machine.** Never end your turn with anything running in the background.

## Verdict

**No CRITICAL, HIGH or MEDIUM:** write a 7-line summary:
- (a) the format: strict, canonical, what it can never carry;
- (b) the export: which rows, the lineage, nothing private, nothing Official;
- (c) the hosted routes: what is public, SaaS only, the limit;
- (d) the import: derived URLs, revision and validation before any row, plugins checked before storage, rows without a mapping and copies untouched, one transaction;
- (e) the Sync: one click, every pass, who can press it unchanged;
- (f) the web and the chat: the badges, the disclosure, the gate's coverage of the four calls, Auto's chat installs on approval only, update notices;
- (g) the owner's next step: the `ownerTest` list (images.yml first; W1 merged first, this PR retargeted to main before it; SaaS deployed before a local Sync), then undraft the PR (`gh pr ready`).

Then `REVIEW_PASS` as the last line. REVIEW_PASS and REVIEW_FINDINGS appear on your last line only, never in prose. You never undraft the PR.

**Findings:**
1. Append `P270W2-RVW-n` stories to `scripts/ralph/prd-270w2.json`, n counting from 1. Same shape: id, title, description, acceptanceCriteria (with the cause, the change and the test), priority after the last story, passes false, notes.
2. Commit with `git commit -s -m 'chore(prd-270): W2 review findings → fix stories [skip ci]' -- scripts/ralph/prd-270w2.json`.
3. `REVIEW_FINDINGS` as the last line.
