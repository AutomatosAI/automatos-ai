# hello-c05 — the package a revocation takes down (PRD-261 C-05l)

The fixture behind `specs/theme-revocation.spec.ts`. Nothing is intercepted: the package is
uploaded privately to the local backend (`POST /api/themes/import`), opted in to and chosen
(`PUT /api/themes/choice`), then revoked by its digest through the admin route
(`POST /api/admin/themes/revocations`; the local operator is `super_admin`, and the local edition
has no step-up), and lifted again (`POST /api/admin/themes/revocations/{revocationId}/lift`).

| File | What it is |
|---|---|
| `build.py` | Builds A-06a's `hello` package under the id `automatos-e2e/hello-c05`, so its digest is its own: revoking it touches no other spec's theme. The spec runs it with `python3 build.py --out <file>`. |

The build is byte-for-byte the same every run, so the second browser's upload answers 200 with the
version already held. Every revocation the spec writes has a reason that starts with
`C-05l browser proof:`, and the spec lifts every live one of those before and after each test.
