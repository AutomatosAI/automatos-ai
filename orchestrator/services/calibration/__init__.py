"""PRD-268 §6.11.5 — calibration on day one: ``Calibration@1``, what Auto thinks it knows.

Ships off (K15) until §12-Q22 is answered: US-C22b's routes (``api/auto_calibration.py``) are
gated by its switch (``switch``).

* ``model``: the Pydantic shapes (``Calibration``, ``CalibrationLine``, ``CalibrationSource``);
* ``sources``: the titles and kinds the person may see, and never a body, a chunk or a file;
* ``prompt``: the one model call's messages, every title quoted as data under a ref;
* ``answer``: what of the model's answer is kept (cited sources only, no percentage or
  confidence number, §12-Q26) and the three offers;
* ``strings``: the consent line, the source-count words, the starters and the text part, in
  English and Portuguese (Brazil) (K18);
* ``build``: ``build_calibration``, at most one model call;
* ``claim``: the person's record, offered once (the unique claim), and its answers' state;
* ``offer``: consent first, then one calibration posted as one Auto message, or declined;
* ``answers`` and ``memory``: Reject and Sounds right, remembered through the memory service (K16);
* ``audit``: an audit row after each commit.
"""
