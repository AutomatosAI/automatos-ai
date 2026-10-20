"""PRD-266 — Automatos Go on the server: pairing, devices, discovery and remote access.

One domain, one package (§4.10), as ``services/themes/``. The routers in
``api/mobile_devices.py`` and ``api/well_known.py`` validate, call these and
map errors; the table is ``core.models.mobile_devices``; the secrets come from the one
pairing kernel, ``core.auth.pairing``.
"""
