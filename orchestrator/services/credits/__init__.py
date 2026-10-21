"""PRD-265 — credits: the ledger, the trial grant, the gate in front of the
platform's AI keys, and ask accounting.

- ``ledger`` — the one writer of ``credit_ledger`` / ``credit_balances``.
- ``trial`` — the one-time trial grant, once per person.
- ``routing`` — who pays for a call: the customer's own key, or credits.
- ``asks`` — one ask per user turn or run: groups its calls; the gate stops it at the balance.
- ``meter`` — every platform-funded call costs its real cost × ``CREDITS_MARKUP``.
- ``alerts`` — running low, run out.

The local edition never meters: every entry point returns early there.
"""
