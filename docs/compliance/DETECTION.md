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
- **Rate limit.** Each IP address gets `PROVENANCE_RATE_PER_MINUTE` checks a minute (default 30). Over that, the route answers `429` with a `Retry-After` header. If Redis is unreachable, the limiter lets checks through and logs an error.
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

Code: `orchestrator/api/provenance.py`, `frontend/app/verify/page.tsx`, `frontend/lib/provenance.ts`.
