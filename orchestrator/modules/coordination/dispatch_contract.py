"""PRD-226 US-003 — the shared 4-part dispatch contract.

Single source for the dispatch-contract shape. Every handoff — a mission task
decomposed by the planner, or a board ticket filed by the PRD-224 ASSIGN lane —
is written in this same four-part shape so the owner needs nothing else to do
the work. Defined ONCE here and imported by both consumers (planner prompt
builder + ASSIGN directive); it is never copy-pasted into either.

House rule (PRD-226 §7): prompt fragments shared between the ASSIGN lane and the
planner live in one place, not duplicated in code.
"""

# The four parts, as prompt text. References, not pasted content — the contract
# points at artifacts rather than inlining them so it stays short.
DISPATCH_CONTRACT_FRAGMENT = """\
A dispatch contract has four parts, written so the owner needs nothing else to do the work:
- **OBJECTIVE** — the outcome, in one line.
- **OUTPUT** — the concrete Deliverable and its shape.
- **TOOLS** — which tools to use, which to avoid, and references (docs, prior Deliverables, ids) to READ instead of re-deriving.
- **BOUNDARIES** — scope limits and the definition of done (the checklist that says the work is finished).
Reference artifacts by name or id; never paste their content in."""
