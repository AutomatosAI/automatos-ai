# qrcodegen (vendored)

Origin: Project Nayuki's **QR Code generator library**, TypeScript edition, **v1.8.0**: `typescript-javascript/qrcodegen.ts` from https://github.com/nayuki/QR-Code-generator at tag `v1.8.0` (commit `720f62bddb7226106071d4728c292cb1df519ceb`, released 17 Apr 2022; upstream file SHA-256 `c4749095a91bf9696e3a303998b9905e467094f53041e64393e65e6d887737fd`). Project page: https://www.nayuki.io/page/qr-code-generator-library.

Licence: MIT. Copyright (c) Project Nayuki. The upstream licence header is kept, unchanged, at the top of every file here.

Approved by Gerard on 24 Oct 2026 (PRD-266 decision K17) for the QR image in Settings → Phones & devices → Pair a phone (US-019c, `components/settings/phones-devices/pair-phone-qr.tsx`). It is vendored, not installed, so `frontend/package.json` and `package-lock.json` do not list it.

## What changed from upstream

Upstream is one 991-line file holding a global `namespace qrcodegen`. This repository's code-shape gate (AGENTS.md: a new file at most 800 lines, a function at most 50 code lines) checks every line of a new file, so:

1. **Three ES modules instead of one namespace.** `bits.ts` holds the type aliases (`bit`, `byte`, `int`) and the helpers `appendBits`, `getBit` and `assert`, now exported. `qrsegment.ts` holds `QrSegment` and `QrSegment.Mode`. `qrcodegen.ts` holds `QrCode` and `QrCode.Ecc` and is the entry point. The outer `namespace qrcodegen { … }` is gone, so its contents sit one indent level out. `namespace qrcodegen.QrCode` and `namespace qrcodegen.QrSegment` are now `export namespace QrCode` and `export namespace QrSegment`, merged with their classes, so `QrCode.Ecc.MEDIUM` and `QrSegment.Mode.BYTE` read as they do upstream.
2. `encodeText` and `encodeBinary` call the imported `QrSegment`, not `qrcodegen.QrSegment`.
3. `QrCode.getPenaltyScore` was 62 code lines. Its two scans (rows, then columns) had identical bodies. That body is now one method, `getLinePenalty`, which `getPenaltyScore` calls for each row and each column. The penalty is the same sum, so the same mask is chosen.
4. Trailing whitespace is dropped.

Every other line is upstream's.

## Updating

Take `typescript-javascript/qrcodegen.ts` from a newer tag, apply the four changes above, and update the version, commit and SHA-256 here. The checks in `components/settings/phones-devices/__tests__/pair-phone-qr.test.tsx` cover the encoder's output.
