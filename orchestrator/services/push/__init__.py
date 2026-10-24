"""PRD-266 W2 — push to Automatos Go phones: the sender, the payload, the fan-out and the receipts.

One domain, one package (§4.10), as ``services/mobile/``:

* ``sender.py`` — the ``PushSender`` protocol and the frozen messages, tickets and receipts it
  carries. No HTTP.
* ``expo.py`` — ``ExpoPushSender``, the one module in the package (and the wave) that speaks HTTP
  to anyone: Expo's push service (§4.10, AGENTS.md exception 2).
* ``payload.py`` — which events push (``PUSH_TYPE_EVENTS``) and each push type's signed message.
* ``consent.py`` — the operator's answer, on a local install, to sending through Expo.
* ``recipients.py`` and ``fanout.py`` — the right phones for an event, with membership and role
  read at send time, handed over off the event loop.
* ``receipts.py`` — the job that reads Expo's receipts and turns off phones that left.
* ``preferences.py`` — a person's push switches, read at fan-out.

The device table is ``core.models.mobile_devices``; the notification dispatcher hands its push
leg to ``fanout.py``, as it hands Telegram rows to ``channels.sender``.
"""
