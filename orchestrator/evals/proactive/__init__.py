"""PRD-268 §6.7 (US-C13, K14) — the proactive evals: each feature is measured before it goes live.

``bars`` holds the pass bars, ``scoring`` one pure scorer per eval, ``gold`` the gold-set reader.
The gold sets stay local (public repo): they are read from ``config.PROACTIVE_EVAL_GOLD_DIR``, a
folder outside the repository, and none is ever committed. The runners are run by hand, by the
owner, never by CI; CI tests the scorers on inline synthetic cases only.
"""
