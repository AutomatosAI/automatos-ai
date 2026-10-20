# hello-c04b — the package behind the Themes tab (PRD-261 C-04b)

The fixture behind `specs/marketplace-themes-tab.spec.ts`. Nothing is intercepted: the package is
submitted to the local backend's Marketplace (`POST /api/marketplace/themes/submit`), approved
through the admin route (`POST /api/admin/themes/{packageId}/approve`; the local operator is
`super_admin`, and the local edition has no isolated review stage, so the person's approval is the
review), then installed from the Marketplace's Themes tab and chosen there.

| File | What it is |
|---|---|
| `build.py` | Builds A-06a's `hello` package under the id `automatos-e2e/hello-c04b`, so its listing and install never meet another spec's (A-07d's spec uninstalls every `automatos-e2e/hello` install). The spec runs it with `python3 build.py --out <file>`. |

The build is byte-for-byte the same every run, so a second submission answers 200 with the version
already held, approved or not.
