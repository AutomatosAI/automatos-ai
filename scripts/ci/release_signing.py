#!/usr/bin/env python3
"""Check a release's signatures against the identity that made them (PRD-279 US-103, US-202).

images.yml signs each published image, and on a ``v*`` tag the chart, with cosign and no
key: Fulcio issues a short-lived certificate for the job's GitHub OIDC token, and the
certificate names the workflow run. On a ``v*`` tag the workflow then runs ``verify``
on what it just signed, and a failed check fails the run.

The check pins every part of the identity exactly, never by pattern:

* ``--certificate-identity``: ``https://github.com/<owner>/<repo>/.github/workflows/images.yml@<ref>``,
  the certificate's subject (the run's ``job_workflow_ref``);
* ``--certificate-oidc-issuer``: GitHub Actions' token issuer;
* the GitHub claims Fulcio copies into the certificate: the repository, the ref, the
  commit and the workflow's name.

So a signature made by a pull request build (ref ``refs/pull/<n>/merge``), by a branch
(``refs/heads/…``), by another tag, by another workflow or in a fork does not pass the
check a release uses. ``accepts`` models that exact comparison, so the tests can show
it without a signature. The event that started the run is not pinned: a manual run of
images.yml on the same tag runs the same workflow file at the same commit, and the
merge job publishes from it exactly as from the tag's push.

    python3 scripts/ci/release_signing.py verify --image ghcr.io/o/automatos-api@sha256:… \\
        --repository "$GITHUB_REPOSITORY" --ref "$GITHUB_REF" --sha "$GITHUB_SHA"
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

GITHUB = "https://github.com"
ISSUER = "https://token.actions.githubusercontent.com"
WORKFLOW_PATH = ".github/workflows/images.yml"
WORKFLOW_NAME = "images"
_BY_DIGEST = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class Signer:
    """What a Fulcio certificate for one GitHub Actions run says about that run."""

    repository: str
    workflow: str
    ref: str
    sha: str
    workflow_name: str
    issuer: str = ISSUER

    @property
    def identity(self) -> str:
        """The certificate's subject: the workflow file at the ref that ran."""
        return f"{GITHUB}/{self.repository}/{self.workflow}@{self.ref}"


# Each flag of the check, and the part of the certificate it must equal exactly.
CLAIM_FLAGS: Mapping[str, str] = {
    "--certificate-identity": "identity",
    "--certificate-oidc-issuer": "issuer",
    "--certificate-github-workflow-repository": "repository",
    "--certificate-github-workflow-ref": "ref",
    "--certificate-github-workflow-sha": "sha",
    "--certificate-github-workflow-name": "workflow_name",
}


def release_signer(repository: str, ref: str, sha: str) -> Signer:
    """The only signer a release accepts: this repository's image workflow, run at ``ref``."""
    for name, value in (("repository", repository), ("ref", ref), ("sha", sha)):
        if not value or any(char.isspace() for char in value):
            raise ValueError(f"the {name} is empty or holds whitespace: {value!r}")
    return Signer(repository=repository, workflow=WORKFLOW_PATH, ref=ref, sha=sha, workflow_name=WORKFLOW_NAME)


def verify_args(reference: str, signer: Signer) -> List[str]:
    """The ``cosign verify`` command for ``reference`` (``repository@sha256:…``)."""
    if not _BY_DIGEST.match(reference):
        raise ValueError(f"verify a digest, not a tag: {reference!r}")
    args = ["cosign", "verify"]
    for flag, part in CLAIM_FLAGS.items():
        args += [flag, getattr(signer, part)]
    return args + [reference]


def _pinned(args: Sequence[str]) -> Dict[str, str]:
    pinned = {}
    for position, arg in enumerate(args):
        if arg.endswith("-regexp"):
            raise ValueError(f"{arg} matches a pattern: a release pins the identity exactly")
        if arg in CLAIM_FLAGS:
            pinned[arg] = args[position + 1]
    return pinned


def accepts(args: Sequence[str], certificate: Signer) -> bool:
    """Whether a certificate passes the check ``args`` runs: each pinned claim equal, exactly."""
    pinned = _pinned(args)
    if set(pinned) != set(CLAIM_FLAGS):
        raise ValueError(f"the check leaves claims unpinned: {sorted(set(CLAIM_FLAGS) - set(pinned))}")
    return all(getattr(certificate, CLAIM_FLAGS[flag]) == value for flag, value in pinned.items())


def verify(args: argparse.Namespace, run=subprocess.run) -> int:
    """Run cosign's check on one signed digest; its exit status is the step's."""
    try:
        signer = release_signer(args.repository, args.ref, args.sha)
        command = verify_args(args.image, signer)
    except ValueError as error:
        print(f"::error title=Signature check::{error}")
        return 1
    print("+ " + " ".join(command), flush=True)
    result = run(command, check=False)
    if result.returncode != 0:
        print(f"::error title=Signature check::{args.image} is not signed by {signer.identity}")
    return result.returncode


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("verify", help="check one signed digest against the release's identity")
    one.add_argument("--image", required=True, help="repository@sha256:<digest>")
    one.add_argument("--repository", required=True, help="owner/repo ($GITHUB_REPOSITORY)")
    one.add_argument("--ref", required=True, help="the ref that ran ($GITHUB_REF)")
    one.add_argument("--sha", required=True, help="the commit that ran ($GITHUB_SHA)")
    args = parser.parse_args(argv)
    return verify(args)


if __name__ == "__main__":
    sys.exit(main())
