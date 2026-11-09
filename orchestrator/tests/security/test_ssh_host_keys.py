"""Tests for core.security.ssh_host_keys — CodeQL py/paramiko-missing-host-key-validation.

Covers the trust-on-first-use policy: a never-seen host key is recorded and
persisted, and a second client loading the same known_hosts file sees it.
"""

import paramiko
import pytest

import core.security.ssh_host_keys as ssh_host_keys


@pytest.fixture
def known_hosts_path(tmp_path, monkeypatch):
    path = tmp_path / "ssh_known_hosts"
    monkeypatch.setattr(ssh_host_keys, "KNOWN_HOSTS_PATH", str(path))
    return str(path)


def _fake_key():
    return paramiko.RSAKey.generate(1024)


class TestTrustOnFirstUsePolicy:
    def test_accepts_and_records_unknown_host(self, known_hosts_path):
        client = paramiko.SSHClient()
        policy = ssh_host_keys.TrustOnFirstUsePolicy()
        key = _fake_key()

        policy.missing_host_key(client, "example.com", key)

        assert client.get_host_keys().lookup("example.com") is not None

    def test_persists_the_new_key_to_disk(self, known_hosts_path):
        client = paramiko.SSHClient()
        policy = ssh_host_keys.TrustOnFirstUsePolicy()
        key = _fake_key()

        policy.missing_host_key(client, "example.com", key)

        reloaded = paramiko.HostKeys(known_hosts_path)
        assert reloaded.lookup("example.com") is not None

    def test_a_write_failure_does_not_raise(self, known_hosts_path, monkeypatch):
        monkeypatch.setattr(ssh_host_keys, "KNOWN_HOSTS_PATH", "/nonexistent-root-only/ssh_known_hosts")
        client = paramiko.SSHClient()
        policy = ssh_host_keys.TrustOnFirstUsePolicy()

        # Must not raise even though the directory cannot be created.
        policy.missing_host_key(client, "example.com", _fake_key())


class TestApplyTrustOnFirstUsePolicy:
    def test_sets_the_tofu_policy(self, known_hosts_path):
        client = paramiko.SSHClient()

        ssh_host_keys.apply_trust_on_first_use_policy(client)

        assert isinstance(client._policy, ssh_host_keys.TrustOnFirstUsePolicy)

    def test_loads_a_previously_trusted_host_key(self, known_hosts_path):
        first_client = paramiko.SSHClient()
        ssh_host_keys.apply_trust_on_first_use_policy(first_client)
        key = _fake_key()
        first_client._policy.missing_host_key(first_client, "example.com", key)

        second_client = paramiko.SSHClient()
        ssh_host_keys.apply_trust_on_first_use_policy(second_client)

        stored = second_client.get_host_keys().lookup("example.com")
        assert stored is not None
        assert stored.get(key.get_name()).get_base64() == key.get_base64()

    def test_a_missing_known_hosts_file_is_not_an_error(self, known_hosts_path):
        client = paramiko.SSHClient()

        ssh_host_keys.apply_trust_on_first_use_policy(client)  # file does not exist yet

        assert isinstance(client._policy, ssh_host_keys.TrustOnFirstUsePolicy)
