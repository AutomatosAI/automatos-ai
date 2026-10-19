# Detecting content made by Automatos AI

Article 50(2) of the EU AI Act says AI-generated audio, images, video and text must be marked in a machine-readable format and be detectable as AI-generated. The Commission's Guidelines (para. 75) add that the people who receive the content must have a way to detect it. This page explains how anyone can check a file, what the answer means, and where the check stops working. PRD-264 story EUAI-10 built it.

## How to check a file

1. Open **`/verify`** on the Automatos web app, for example `https://<your Automatos host>/verify`. You don't need an account. The page works in the hosted edition and in the local edition.
2. Choose the file.
3. The browser computes the file's SHA-256 fingerprint on your own device, using Web Crypto. **The file is never uploaded.** Only the 64-character fingerprint is sent.
4. The page shows the answer.

You can also call the route yourself:

```
GET /api/public/provenance/{sha256}
```

`{sha256}` is the file's SHA-256 written as 64 lower-case hex characters (for example, the output of `sha256sum file.mp4` or `shasum -a 256 file.mp4`). Any other value is refused with `422`.

## What the answer means

| Answer | Meaning |
|---|---|
| `{"made_by_automatos_ai": true, "kind": "video", "date": "2026-10-19"}` | Automatos produced and marked this exact file. `kind` is `video`, `image`, `audio` or `document`. `date` is the UTC day the file was first recorded. |
| `{"made_by_automatos_ai": false}` | Automatos has no record of this exact file. |

The answer never says which workspace, customer or person made the file. It also never includes a prompt or a file name. Those details aren't sent back, and most of them aren't stored at all (see below).

## Limits

- **A changed file has a different fingerprint and reads "no".** Any edit gives the file a new SHA-256, even one you can't see or hear. That includes re-encoding, compressing, cropping, resizing, trimming, taking a screenshot or converting the format. Social networks and messaging apps usually re-encode uploads, so a copy downloaded from one of them reads "no" even when Automatos made the original. The check proves a match. A "no" does not prove that AI wasn't involved.
- **Only marked files are recorded.** A row is written when Automatos produces a marked file. The writers are the stories that mark files: generated documents and images (PRD-264 EUAI-09) and media-render outputs (EUAI-06b). **Until those stories merge, every file reads "no".** Chat answers are marked inside the message (EUAI-09) and are not recorded, so you can't look up a chat answer by its hash.
- **Content Credentials (C2PA).** Once EUAI-06b ships, signed media files also carry a C2PA manifest. You can read a manifest with any Content Credentials tool, for example [contentcredentials.org/verify](https://contentcredentials.org/verify). That tool works on files from any C2PA tool, but it can only read a manifest the file still carries. The `/verify` page links to it.
- **No watermark yet.** An imperceptible watermark that survives re-encoding is not built (PRD-264 decision D11). Until it is, the metadata marks and this hash record are how a file is detected. `docs/compliance/ARTICLE-50-GAP-ANALYSIS.md` (EUAI-14) explains the gap.
- **Rate limit.** Each IP address gets `PROVENANCE_RATE_PER_MINUTE` checks a minute (default 30). An IPv6 caller is counted per /64. Over that, the route answers `429` with a `Retry-After` header. If Redis is unreachable, the limiter lets checks through and logs an error.
- **The local edition answers for its own files only.** A self-hosted instance knows only the files it produced itself, so ask the instance that made the file.

## What is stored

The table is `content_provenance`. It has one row per marked file per workspace:

| Column | Holds |
|---|---|
| `sha256` | The file's SHA-256, 64 lower-case hex characters (indexed) |
| `kind` | `video`, `image`, `audio` or `document` |
| `manifest_id` | The C2PA manifest id when the file is signed, otherwise empty |
| `generator` | The toolkit or model name, for example `media-render/<version>` |
| `created_at` | When the file was recorded |
| `workspace_id` | The workspace, for tenancy and erasure only. The public answer never reads it |

No prompt, user, file name or path is stored (Guidelines para. 94). Code writes rows through `services/content_provenance.py` `record_marked_file(db, workspace_id, sha256, kind, generator, manifest_id=None)`. Writing the same file twice in one workspace keeps the first row.

**Erasure.** Workspace erasure (`gdpr_service.erase_workspace`) deletes the workspace's rows. It uses the same self-maintaining SQL purge as every other workspace-scoped table, and the foreign key cascades when the workspace row goes. After erasure, the workspace's files read "no".

## The one public route (decision D16)

AGENTS.md requires every endpoint to be authenticated. This route is the named exception (PRD-264 D16), because the people who receive generated content have no Automatos account. It is kept narrow:

- It takes a hash, never a file.
- It reads one indexed column.
- It returns only yes or no, the kind and the date.
- It is rate-limited per IP and listed in the route manifest.
- A bearer token or workspace header sent along with the request is ignored.

### Security review of the route (19 Oct 2026)

The route, `services/content_provenance.py` and the rate-limit wiring were reviewed against the security-reviewer checklist (secrets, injection, authentication and tenancy, user-controlled input, leaking errors, unlimited expensive endpoints). What it found:

- **The client address.** The limiter keys on the first `X-Forwarded-For` hop, as `main.py`'s global limiter does. That is only safe behind a proxy that replaces a client-sent `X-Forwarded-For`. Railway's staff state that its edge strips the header a client sends and that the first value is the connecting address; a further hop may be appended inside Railway's network, so the last value is not the client. On Railway the first hop is therefore right. **Behind any other proxy (for example the Helm chart's ingress) or with the API port reachable directly, a caller can choose that value and get a fresh bucket per request.** Whoever runs such a deployment must make its proxy overwrite `X-Forwarded-For`. The value is parsed as an IP address before it becomes a bucket, so a forged header can't write an arbitrary Redis key or log line. IPv6 is counted per /64, because one subscriber can otherwise rotate addresses inside it.
- **Redis down.** The limiter fails open, as the widget limiter does by decision, and logs an error for each request it lets through. While Redis is down the route is unlimited. Each check is one read on an indexed column with a fixed-size answer. Failing closed instead (answering `503`) would need the shared `RateLimitStore` to report that Redis was unreachable; that is not built, and is a decision for the maintainer.
- **The date.** The answer gives the UTC day of the earliest record of that exact file, across all workspaces. It doesn't say which workspace, or how many workspaces hold the file. Only someone who already holds the file, or its hash, can ask, and knowing that Automatos made it is the point of the route.
- **Nothing else found.** The hash is matched against a fixed pattern before it is read, and the query is parameterised. The 422 and 429 messages are fixed text. A token or workspace header sent along is ignored. No workspace, user or file data is read for the answer. The route itself logs nothing. The server's access log records the request path, which holds the hash, and the caller's address, as it does for every route.

Code: `orchestrator/api/provenance.py`, `frontend/app/verify/page.tsx`, `frontend/lib/provenance.ts`.
