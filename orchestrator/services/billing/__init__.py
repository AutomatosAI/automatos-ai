"""Billing for hosted Studio (PRD-265 W2): in-page Stripe checkout, in-app management, webhooks.

The credit ledger is ``services/credits``; this package moves plans and credits
when a payment provider says money moved. See ``provider.py`` for the seam.
"""
