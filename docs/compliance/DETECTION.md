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
- **Only marked files are recorded.** A row is written when Automatos produces a marked file. The writers are the stories that mark files: generated documents and images (PRD-264 EUAI-09) and media-render outputs (EUAI-06b). **Until EUAI-06b merges, media-render files read "no".** Chat answers are marked inside the message (EUAI-09) and are not recorded, so you can't look up a chat answer by its hash. A document a person fills in by hand (the Studio's Generate, a template preview) is not AI-generated, so it carries no mark and reads "no".
- **Content Credentials (C2PA).** Once EUAI-06b ships, signed media files also carry a C2PA manifest. You can read a manifest with any Content Credentials tool, for example [contentcredentials.org/verify](https://contentcredentials.org/verify). That tool works on files from any C2PA tool, but it can only read a manifest the file still carries. The `/verify` page links to it.
- **No watermark yet.** An imperceptible watermark that survives re-encoding is not built (PRD-264 decision D11). Until it is, the metadata marks and this hash record are how a file is detected. [ARTICLE-50-GAP-ANALYSIS.md](./ARTICLE-50-GAP-ANALYSIS.md) (EUAI-14) explains the gap.
- **Rate limit.** Each IP address gets `PROVENANCE_RATE_PER_MINUTE` checks a minute (default 30). An IPv6 caller is counted per /64. Over that, the route answers `429` with a `Retry-After` header. If Redis is unreachable, the limiter lets checks through and logs an `ERROR` line for each one, naming the provenance rate limiter and the caller's bucket (an IP address or an IPv6 /64), never the hash.
- **Nothing keeps the answer.** Every answer, refusals included, carries `Cache-Control: no-store`, so no browser, proxy or CDN stores a record of the check.
- **The local edition answers for its own files only.** A self-hosted instance knows only the files it produced itself, so ask the instance that made the file.

## The marks inside a file (EUAI-09)

Each file Automatos generates also says in its own metadata that AI made it, so tools can read it without asking Automatos. The term is the IPTC Digital Source Type: `compositeWithTrainedAlgorithmicMedia` for a document (an agent's text in the workspace's own template and letterhead), `trainedAlgorithmicMedia` for an image an image model made.

| File | Where the mark is |
|---|---|
| PDF | The XMP packet in the document catalogue's `/Metadata` stream: `Iptc4xmpExt:DigitalSourceType` and `xmpMM:InstanceID` |
| DOCX, XLSX | Custom document properties (File > Properties > Custom): `AIGenerated` holds the IPTC term, `AIInstanceID` the instance id |
| PNG | An XMP packet in an `iTXt` chunk (`XML:com.adobe.xmp`) |
| JPEG | An XMP packet in an `APP1` segment |
| Chat answers | `"ai_generated": true` on the frames that open and close each assistant message in the chat stream, and `data-ai-generated="true"` on the message in the page |
| Blog posts | `ai_generated: true` in the blog API for a post an agent wrote |

The mark holds the term and a random instance id (`xmp.iid:<uuid>`) only: no workspace, person, prompt or file name. The instance id is new for every file, so two files made from identical input never share a fingerprint, and a predictable document (a templated letter) can't be confirmed by hashing guesses. An image that already carries a C2PA manifest from the model that made it is kept byte for byte, with its manifest. WebP and GIF images carry no metadata mark (no library the platform ships writes one without re-encoding the picture) and get no provenance row.

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

No prompt, user, file name or path is stored (Guidelines para. 94). Code writes rows through `services/content_provenance.py` `record_marked_file(db, workspace_id, sha256, kind, generator, manifest_id=None)`, or `record_produced_file(...)` with a session of its own (documents and images). Writing the same file twice in one workspace keeps the first row. A failed write is logged by its error class, never with the hash.

**Erasure.** Workspace erasure (`gdpr_service.erase_workspace`) deletes the workspace's rows. It uses the same self-maintaining SQL purge as every other workspace-scoped table, and the foreign key cascades when the workspace row goes. After erasure, the workspace's files read "no". The admin hard delete removes them the same way. An onboarding reset (`wipe_built`) does not: the marked files are already out in the world, so the reset keeps their rows and published content still reads "yes".

## The one public route (decision D16)

AGENTS.md requires every endpoint to be authenticated. This route is the named exception (PRD-264 D16), because the people who receive generated content have no Automatos account. It is kept narrow:

- It takes a hash, never a file.
- It reads one indexed column.
- It returns only yes or no, the kind and the date.
- It is rate-limited per IP and listed in the route manifest.
- A bearer token or workspace header sent along with the request is ignored.

### Security review of the route (19 Oct 2026)

The route, `services/content_provenance.py` and the rate-limit wiring were reviewed against the security-reviewer checklist (secrets, injection, authentication and tenancy, user-controlled input, leaking errors, unlimited expensive endpoints). What it found:

- **The client address.** The limiter keys on the first `X-Forwarded-For` hop, as `main.py`'s global limiter does. That is only safe behind a proxy that replaces a client-sent `X-Forwarded-For`. Railway's staff state that its edge strips the header a client sends and that the first value is the connecting address; a further hop may be appended inside Railway's network, so the last value is not the client. On Railway the first hop would therefore be right, but **that premise is unverified for this deployment until a probe is run**: send a request with a made-up `X-Forwarded-For` through the public edge and confirm the route counts it against the real address. Running the probe, and acting on its result, is an owner decision. **Behind any other proxy (for example the Helm chart's ingress) or with the API port reachable directly, a caller can choose that value and get a fresh bucket per request.** Whoever runs such a deployment must make its proxy overwrite `X-Forwarded-For`. The value is parsed as an IP address before it becomes a bucket, so a forged header can't write an arbitrary Redis key or log line. IPv6 is counted per /64, because one subscriber can otherwise rotate addresses inside it.
- **Redis down.** The limiter fails open, as the widget limiter does by decision, and logs an error for each request it lets through. While Redis is down the route is unlimited. Each check is one read on an indexed column with a fixed-size answer. Each of those error lines names the provenance rate limiter (the shared `RateLimitStore` takes a label from its caller; the widget limiter's line is unchanged) and the caller's bucket, never the hash. Failing closed instead (answering `503`) would need the shared `RateLimitStore` to report that Redis was unreachable; that is not built, and is an owner decision.
- **The date.** The answer gives the UTC day of the earliest record of that exact file, across all workspaces. It doesn't say which workspace, or how many workspaces hold the file. Only someone who already holds the file, or its hash, can ask, and knowing that Automatos made it is the point of the route.
- **The hash in metrics, logs and traces (PRD-264 FR-17).** The hash is in the request path, and several places copy a request path out of the process. Each one now collapses a 64-character hex segment to `{sha256}` (`core/monitoring/path_digests.py`):
  - the Prometheus request metrics (`/metrics` is unauthenticated, and every checked hash used to become its own label);
  - the `path` field on every log line shipped to the log relay, and the `http_path` field on console lines;
  - uvicorn's access log line, which keeps the method, status and caller address;
  - the "Unhandled error" line of the JSON-500 converter.
  A database error on the lookup answers `503` and is logged by its error class only, because its text carries the bound hash. With `OTEL_ENABLED`, the server span's URL attributes (`http.target`, `http.url`, `url.full`, `url.path`) carried the full path including the hash; the span's request hook now normalises them the same way, and the span name was already the route template.
- **Logging.** The route logs two lines of its own: the fail-open `ERROR` line, which holds the caller's bucket (an IP address), and the database-error line, which holds an error class. Neither holds the hash.
- **Other checks.** The hash is matched against a fixed pattern before it is read, and the query is parameterised. The 422, 429 and 503 messages are fixed text. A token or workspace header sent along is ignored. No workspace, user or file data is read for the answer.

### Review record

security-reviewer agent, 19 Oct 2026, commit 7c530cdb1: no CRITICAL/HIGH; M1 metrics labels (fixed in this commit), L1 X-Forwarded-For trust (owner decision, documented), L2 Redis fail-open (owner decision, documented), L3 confirmation oracle on low-entropy files (FR-16: random per-file instance id in every mark), L4 onboarding reset (fixed), L5 public pattern (fixed), I1 no-store (fixed), I2 doc corrections (this edit).

"This commit" is the PRD-264 `fix(prd-264): EUAI-10 security review fixes` commit. FR-16 is built for documents and images (EUAI-09, above); media-render files get theirs with EUAI-06a and EUAI-06b. The `/verify` page is public at `/verify` and `/verify/...` only, so `/verify-email` or `/verification` stay behind sign-in.

Code: `orchestrator/api/provenance.py`, `orchestrator/core/monitoring/path_digests.py`, `frontend/app/verify/page.tsx`, `frontend/lib/provenance.ts`.
