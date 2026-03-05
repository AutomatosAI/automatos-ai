# Dependency Audit — PRD-70 FIX-09

**Date:** 2026-03-03
**Audited by:** Claude (automated)

## Python (pip-audit)

**39 known vulnerabilities in 13 packages.**

### Critical / High Priority (upgrade ASAP)

| Package | Current | Fix | CVEs | Notes |
|---------|---------|-----|------|-------|
| **fastapi** | 0.104.1 | 0.109.1+ | PYSEC-2024-38 | DoS via multipart |
| **starlette** | 0.27.0 | 0.40.0+ | CVE-2024-47874, CVE-2025-54121 | Request smuggling |
| **python-multipart** | 0.0.6 | 0.0.22+ | CVE-2024-53981, CVE-2026-24486 | DoS via multipart parsing |
| **gitpython** | 3.1.40 | 3.1.41+ | PYSEC-2024-4 | RCE via crafted repo |
| **langchain-core** | 1.0.2 | 1.2.11+ | CVE-2025-65106, CVE-2026-26013 | Multiple issues |
| **pypdf** | 4.3.1 | 6.7.4+ | 12 CVEs | PDF parsing DoS, major version bump required |

### Medium Priority

| Package | Current | Fix | CVEs | Notes |
|---------|---------|-----|------|-------|
| python-jose | 3.3.0 | 3.4.0+ | PYSEC-2024-232/233 | JWT token forgery |
| sqlparse | 0.5.0 | 0.5.4+ | GHSA-27jp-wm6q-gp25 | ReDoS |
| pdfminer-six | 20250506 | 20251230+ | CVE-2025-64512, CVE-2025-70559 | PDF DoS |
| nltk | 3.8.1 | 3.9.3+ | CVE-2025-14009 | ReDoS |
| scikit-learn | 1.3.2 | 1.5.0+ | PYSEC-2024-110 | Arbitrary code execution |
| black | 23.11.0 | 24.3.0+ | PYSEC-2024-48 | Dev dependency only |

### Low Priority

| Package | Current | Fix | Notes |
|---------|---------|-----|-------|
| ecdsa | 0.19.1 | - | CVE-2024-23342, timing side-channel (low practical risk) |

## Node.js (npm audit)

**17 vulnerabilities (0 critical, 2 high, 14 moderate, 1 low).**

All are in dev/build dependencies (esbuild, vite, vitest chain). No production runtime CVEs.

| Package | Severity | Notes |
|---------|----------|-------|
| esbuild <=0.24.2 | Moderate | Dev server request smuggling (dev-only) |
| vite 0.11.0-6.1.6 | Moderate | Depends on vulnerable esbuild |
| ajv <6.14.0 | Moderate | ReDoS with $data option |
| diff 4.0.0-4.0.3 | Moderate | DoS in parsePatch |
| glob 10.2-10.4.5 | Low | ReDoS |

## Recommended Actions

### Immediate (this sprint)
1. `pip install --upgrade fastapi starlette python-multipart gitpython python-jose sqlparse`
2. Verify no breaking API changes (fastapi 0.104→0.109 is minor)
3. Pin `gitpython>=3.1.41` — critical for this PRD (git clone security)

### Next sprint
1. Upgrade `langchain-core` to 1.2.11+ (check LangChain compatibility)
2. Upgrade `pypdf` to 6.x (major version — test PDF processing pipeline)
3. Run `npm audit fix` for frontend dev deps

### Deferred
1. `ecdsa` — timing side-channel, very low practical risk
2. `black` — dev-only formatter, no production exposure
