# Provisional Patent Application — Filing Checklist

**Inventor:** Gerard Kavanagh
**Filing Date Target:** Monday, March 23, 2026
**Filing Fee:** $320 (micro entity)
**Filing Method:** USPTO EFS-Web (https://patentcenter.uspto.gov)

---

## Before You Start

- [ ] Create a USPTO.gov account (if you don't have one): https://patentcenter.uspto.gov
- [ ] Have a credit/debit card ready for the $320 filing fee

## Documents to Upload (all in this folder)

| # | Document | File | USPTO Category |
|---|----------|------|----------------|
| 1 | Cover Sheet | `01-COVER-SHEET.md` | Provisional Cover Sheet (SB/16) |
| 2 | Specification | `02-SPECIFICATION.md` | Specification |
| 3 | Claims | `03-CLAIMS.md` | Claims |
| 4 | Abstract | `04-ABSTRACT.md` | Abstract |
| 5 | Drawings | `05-DRAWINGS.md` | Drawings |
| 6 | Inventor Declaration | `06-INVENTOR-DECLARATION.md` | Application Data Sheet |

### Supporting Evidence (optional but strengthens filing)

| # | Document | File | Purpose |
|---|----------|------|---------|
| 7 | Technical Disclosure | `../PRD-108-TECHNICAL-DISCLOSURE.md` (repo root docs/) | Full technical detail |
| 8 | Algorithm Specification | `../PRD-108-ALGORITHMS.md` (repo root docs/) | All 15 algorithms |
| 9 | Production Evidence | `runs/2026-03-21/09-PRODUCTION-EVIDENCE.md` | First production run |
| 10 | Test Results | `runs/2026-03-21/01-proof-suite.txt` through `07-multi-scenario-pytest.txt` | 119 assertions |

## Filing Steps

### Step 1: Go to USPTO Patent Center
https://patentcenter.uspto.gov

### Step 2: Start New Submission
- Click "New Submission" > "Provisional Application"
- Select "Utility" as application type

### Step 3: Entity Status
- Select **Micro Entity** ($320 fee)
- You qualify if:
  - You haven't been named as inventor on more than 4 previously filed US patent applications
  - Your gross income last year was less than 3x the US median household income (~$232,000 for 2025)
  - You haven't assigned/licensed/conveyed rights to an entity that exceeds the income limit

### Step 4: Upload Documents
Upload the files from this folder as PDFs:
1. Convert each .md file to PDF (use any markdown-to-PDF tool, or print from browser)
2. Upload `02-SPECIFICATION.md` as the "Specification"
3. Upload `03-CLAIMS.md` as the "Claims"
4. Upload `04-ABSTRACT.md` as the "Abstract"
5. Upload `05-DRAWINGS.md` as the "Drawings"
6. Fill in the cover sheet info from `01-COVER-SHEET.md`

### Step 5: Pay the Fee
- $320 for micro entity provisional application

### Step 6: Submit and Save
- Download and save the filing receipt
- Note the **Application Number** — this is your priority date reference
- Save the confirmation email

## After Filing

- [ ] Save filing receipt PDF
- [ ] Record application number
- [ ] Calendar reminder: 12 months from filing date to convert to non-provisional
- [ ] Consult a patent attorney within 3 months to review claims and advise on non-provisional strategy
- [ ] Continue building production evidence (more missions, more scenarios)

## Important Notes

- A provisional patent application is NOT examined — it just establishes your priority date
- You have exactly 12 months to file a non-provisional application claiming benefit of this provisional
- If you don't file a non-provisional within 12 months, the provisional expires (but your technical disclosure still serves as prior art)
- The provisional does NOT need to be perfect — it needs to adequately describe the invention
- All the documents in this folder are designed to meet that bar

## Cost Summary

| Item | Cost | When |
|------|------|------|
| Provisional filing (micro entity) | $320 | Monday |
| Patent attorney consultation | $500-1,500 | Within 3 months |
| Non-provisional filing (micro entity) | $800 | Within 12 months |
| Patent attorney for non-provisional | $8,000-15,000 | Within 12 months |

The $320 provisional locks your priority date. Everything else can wait.

---

*Prepared March 21, 2026*
