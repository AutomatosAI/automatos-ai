"""SSH host-key validation — CodeQL py/paramiko-missing-host-key-validation.

``paramiko.AutoAddPolicy`` and ``paramiko.WarningPolicy`` both accept ANY
unknown host key and never persist it to the ``SSHClient`` instance's caller
(a fresh ``SSHClient()`` is built per call here), so every single connection
is treated as first contact: there is no protection against a
man-in-the-middle, ever, not even on the second call to the same host.

This module gives trust-on-first-use (TOFU) instead, the same model OpenSSH's
own client uses: the first connection to a host records its key on disk;
every later connection loads that file first, and paramiko's own
``SSHClient.connect`` raises ``BadHostKeyException`` itself — before any
policy runs — when a known host presents a DIFFERENT key. Our policy class is
only ever consulted for a host paramiko has no key for yet.
"""

import logging
import os
from pathlib import Path

import paramiko

logger = logging.getLogger(__name__)

# Mirrors the pattern already used for SkillLoaderConfig.SKILLS_BASE_DIR: a
# fixed, non-secret path under the service's own home directory, not a secret
# or a per-request value, so it does not need to go through config.py.
KNOWN_HOSTS_PATH = os.path.expanduser("~/.automatos/ssh_known_hosts")


class TrustOnFirstUsePolicy(paramiko.MissingHostKeyPolicy):
    """Accept and persist a host key paramiko has never seen for this host.

    paramiko calls ``missing_host_key`` only when the host has no key on file
    at all. A host that already has a *different* key on file is rejected by
    paramiko itself, before this policy is ever consulted.
    """

    def missing_host_key(self, client: paramiko.SSHClient, hostname: str, key: paramiko.PKey) -> None:
        logger.warning(
            "[ssh] trust-on-first-use: recording new host key for %s (%s %s)",
            hostname, key.get_name(), key.get_fingerprint().hex(),
        )
        client.get_host_keys().add(hostname, key.get_name(), key)
        _persist_host_keys(client)


def _persist_host_keys(client: paramiko.SSHClient) -> None:
    """Best-effort save; a write failure must not fail the connection."""
    try:
        Path(KNOWN_HOSTS_PATH).parent.mkdir(parents=True, exist_ok=True)
        client.get_host_keys().save(KNOWN_HOSTS_PATH)
    except OSError:
        logger.exception("[ssh] failed to persist known host keys to %s", KNOWN_HOSTS_PATH)


def apply_trust_on_first_use_policy(client: paramiko.SSHClient) -> None:
    """Load any previously-trusted host keys, then apply TOFU for new hosts.

    Call this instead of ``client.set_missing_host_key_policy(AutoAddPolicy())``
    or ``WarningPolicy()`` on every ``paramiko.SSHClient()`` this service
    creates.
    """
    if os.path.exists(KNOWN_HOSTS_PATH):
        try:
            client.load_host_keys(KNOWN_HOSTS_PATH)
        except OSError:
            logger.exception("[ssh] failed to load known host keys from %s", KNOWN_HOSTS_PATH)
    client.set_missing_host_key_policy(TrustOnFirstUsePolicy())
