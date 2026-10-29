"""PRD-268 §6.4, §6.11.2 — the brief: ``Brief@1``, the structured brief every surface draws.

* ``model``: the Pydantic shapes (``Brief``, ``BriefRow`` and their parts);
* ``strings``: the greeting, the section labels, the empty sentence, "Show N more", the row
  action labels and the rows' reasons, in English and Portuguese (Brazil) (K7, K16);
* ``needs``, ``suggested``, ``today``, ``since`` and ``unchecked``: one small reader per section;
* ``build``: ``build_brief``, which assembles them with Auto's Read, folds each section and
  writes the spoken words;
* ``claim``: one brief per person per local date, shared by the scheduled digest and the brief
  on open (``claim_brief``, ``todays_brief_message``, ``post_brief``);
* ``on_open``: ``open_brief``, the brief on open for one person and local date (the claim won,
  taken or pending), behind ``POST /api/proactive/brief/open``.
"""
