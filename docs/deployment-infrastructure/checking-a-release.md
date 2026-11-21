# Checking a release

Every release of Automatos is a set of images and one Helm chart, each signed by the
release workflow of this repository when it ran on the release's tag. This page gives the
commands to check that yourself before you install, says what each part of a release
carries besides the software, and says what a passing check proves and what it does not.
It is the record for [PRD-279](../PRDS/PRD-279-VERSIONED-RELEASES.md) US-302. How a release
is cut is on [Releases](releases.md).

Each release page on GitHub lists the chart's and every image's digest under "What to
install", and repeats the commands below with that release's tag and digests filled in.

## What signed it

Signing uses [cosign](https://docs.sigstore.dev/) with no key: when the workflow runs, it
exchanges its GitHub Actions identity token for a short-lived certificate, and that
certificate names the workflow run. A release's signature must name all of these:

| Part of the certificate | What a release's says |
|---|---|
| Identity (the certificate's subject) | `https://github.com/AutomatosAI/automatos-ai/.github/workflows/images.yml@refs/tags/v<version>` |
| Issuer | `https://token.actions.githubusercontent.com` |
| Repository | `AutomatosAI/automatos-ai` |
| Workflow | `images.yml`, whose name is `images` |
| Ref | `refs/tags/v<version>`: the tag, with its leading `v` (`refs/tags/v0.12.0`, `refs/tags/v0.12.0-rc.1`) |
| Commit | the full commit id the tag points at |

A signature made anywhere else names something else, and the commands below refuse it: a
pull request build (`…@refs/pull/<n>/merge`), a push to a branch (`…@refs/heads/main`),
another tag, another workflow file, or the same workflow in a fork (another repository).

## Checking an image

The images are `ghcr.io/automatosai/automatos-<name>`: `api`, `frontend` (the web app,
local edition), `enterprise-frontend` (the web app, enterprise edition) and
`workspace-worker`. Check an image by its digest, never by its tag, because a tag is a name
that points at a digest and the digest is what you install. A version's digest is on its
release page, or:

```bash
docker buildx imagetools inspect ghcr.io/automatosai/automatos-api:0.12.0
```

(the `Digest:` line at the top is the multi-architecture index, the one that is signed).
Then, with cosign 2.x or 3.x:

```bash
cosign verify \
  --certificate-identity "https://github.com/AutomatosAI/automatos-ai/.github/workflows/images.yml@refs/tags/v0.12.0" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  --certificate-github-workflow-repository "AutomatosAI/automatos-ai" \
  --certificate-github-workflow-ref "refs/tags/v0.12.0" \
  --certificate-github-workflow-name "images" \
  ghcr.io/automatosai/automatos-api@sha256:<digest>
```

Add `--certificate-github-workflow-sha "<commit id>"` to pin the commit too: the release
page gives it. This is the command the workflow runs on its own output before it publishes
a version (`scripts/ci/release_signing.py`), where every one of those claims is compared
exactly, never by pattern. The first two flags alone are the minimum; the others narrow it
further.

A passing check exits zero and prints the checks it made and the matching signatures as
JSON. A failing one exits non-zero and says why; do not install that image.

## Checking the chart

The chart is the OCI artifact `oci://ghcr.io/automatosai/charts/automatos`, under the
release's version. Its digest is on the release page, and `helm pull` prints it:

```bash
helm pull oci://ghcr.io/automatosai/charts/automatos --version 0.12.0
# Pulled: ghcr.io/automatosai/charts/automatos:0.12.0
# Digest: sha256:<digest>
```

Check it with the same command as an image, naming the chart by that digest:

```bash
cosign verify \
  --certificate-identity "https://github.com/AutomatosAI/automatos-ai/.github/workflows/images.yml@refs/tags/v0.12.0" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  --certificate-github-workflow-repository "AutomatosAI/automatos-ai" \
  --certificate-github-workflow-ref "refs/tags/v0.12.0" \
  --certificate-github-workflow-name "images" \
  ghcr.io/automatosai/charts/automatos@sha256:<digest>
```

Then install the chart by that same version. Its default values name every image by the
digest the release published, so a checked chart installs exactly the checked images:

```bash
helm install automatos oci://ghcr.io/automatosai/charts/automatos --version 0.12.0 \
  --set existingSecret=<your Secret>
helm template automatos oci://ghcr.io/automatosai/charts/automatos --version 0.12.0 \
  --set existingSecret=<your Secret> | grep 'image:'
```

The second command lists the images the chart runs: each must be `…@sha256:<digest>`, with
the digests on the release page.

## The contents list and the build record

Each image carries two records beside it, made by the build itself (BuildKit's
attestations: the build runs with `sbom: true` and `provenance: mode=max`). They are
attestation manifests in the image's index, one per architecture, each referring to that
architecture's image, so the signature on the index covers them too: the index lists each
one by its digest.

- **The contents list** (a software bill of materials, in SPDX JSON): every package the
  build found in the image, operating system packages and Python and Node packages, with
  their versions. Use it to answer "does this release contain X at version Y" without
  pulling the image.

  ```bash
  docker buildx imagetools inspect ghcr.io/automatosai/automatos-api@sha256:<digest> \
    --format '{{ json (index .SBOM "linux/amd64").SPDX }}'
  docker buildx imagetools inspect ghcr.io/automatosai/automatos-api@sha256:<digest> \
    --format '{{ range (index .SBOM "linux/amd64").SPDX.packages }}{{ .name }}@{{ .versionInfo }}{{ println }}{{ end }}'
  ```

  Use `linux/arm64` for the other architecture. A scanner reads the same list: Grype, which
  the release workflow runs on every image, reads the image itself.

- **The build record** (SLSA provenance, `mode=max`): what built the image. It names the
  source repository and commit, the Dockerfile, the build arguments, every base image by
  digest, and the build's steps.

  ```bash
  docker buildx imagetools inspect ghcr.io/automatosai/automatos-api@sha256:<digest> \
    --format '{{ json (index .Provenance "linux/amd64").SLSA }}'
  ```

  Read in it: the commit the source came from (it must be the tag's), the build arguments
  (the API's `AUTOMATOS_VERSION` is the release's version), and the base images'
  digests.

The chart has no contents list or build record of its own: what it installs is the images,
each named by its digest.

## What a passing check proves, and what it does not

A passing check proves:

- the image or chart, at exactly that digest, was signed by this repository's `images.yml`
  workflow, running on GitHub Actions for that tag and, with the commit flag, at that
  commit;
- nothing about it changed after it was signed: any change gives another digest, which has
  no such signature.

A passing check does not prove:

- **that the code is free of faults or vulnerabilities.** The release workflow refuses an
  image with a fixable critical finding, or one on CISA's Known Exploited Vulnerabilities
  list, that its allow-list does not cover with a reason and an expiry; findings published
  after the release are not in that scan. Scan the image yourself before you install it,
  and again over time.
- **who wrote or reviewed the code.** It proves which workflow built it, from which commit.
  What reached that commit is the repository's history and its review rules.
- **that the tag was pushed by a maintainer.** For that, a repository admin must have set the
  `Version tags` ruleset, which lets only maintainers create a `v*` tag, and, for a patch,
  the `Release branch creation` ruleset, which lets only maintainers create a `release/*`
  branch.
  Both settings are recorded on [Releases](releases.md#who-can-create-a-version-tag) and
  [Patching an older release](patching-a-release.md#the-branch-is-protected-as-main-is).
- **that GitHub Actions, Sigstore's certificate authority and its transparency log worked
  as they say.** The check trusts them: a certificate is only as good as the issuer that
  gave it.
- **anything about a tag.** It checks one digest. A tag can be moved to another digest;
  checking by digest, and installing by the chart's pinned digests, is what keeps you on
  what you checked.
- **anything about your install's settings.** A checked chart rendered with your values runs
  the checked images; what your values turn on is yours.

## When a check fails

Do not install it. A check fails when the artifact was signed by another workflow, ref or
repository, when its digest is not the one that was signed (a rebuilt or re-pushed image),
or when it was never signed. If the digest came from a release page of this repository and
the check still fails, report it as [SECURITY.md](../../SECURITY.md) describes.
