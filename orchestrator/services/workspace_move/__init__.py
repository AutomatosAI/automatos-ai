"""PRD-273 — Move a workspace: export it from one install to one file (wave 1).

* ``rules`` / ``table_map``  every workspace table classified MOVED or NOT_MOVED, with a reason,
  its secret and derived columns (§5.2, §5.3); ``schema`` reads this database and is the guard;
* ``auto_part``            ``workspace.json`` (allow-listed settings) and ``auto.json`` (Auto's soul);
* ``rows`` / ``refs``      the ``sql/`` parts, streamed in batches, and the install-level rows they name;
* ``bundles``              the agent and Playbook bundles (PRD-270's format);
* ``memory`` / ``files``   durable and field memory; document and Deliverable files by content hash;
* ``crypto`` / ``writer``  the optional passphrase (AES-256-GCM, scrypt) and the zip itself;
* ``manifest`` / ``export`` the manifest, and one function that writes the whole file;
* ``jobs`` / ``storage``   the background job, its file in object storage, the single download, expiry.

Nothing is imported here, so the GDPR export can read the table map without the rest.
"""
