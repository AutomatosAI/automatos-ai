# Releases

A release is a version you can name: the images and one Helm chart, built from one
commit and published under one version number. It is cut by pushing a tag
`vMAJOR.MINOR.PATCH` to this repository. This page covers what the tag does, the
repository setting that limits who can push one, and the check a release runs on its
own signatures, and the release page each version gets. It is the record for
[PRD-279](../PRDS/PRD-279-VERSIONED-RELEASES.md) Waves 1 to 3.

## What a release is

- **The images.** `api`, `frontend` (the web app, local edition), `enterprise-frontend`
  (the web app, enterprise edition: PRD-277 US-505) and `workspace-worker`, as
  `ghcr.io/automatosai/automatos-<name>`, each tagged with the version (`0.7.0`)
  and, for a full release, its release line (`0.7`). Each one is signed by digest.
- **The chart.** `oci://ghcr.io/automatosai/charts/automatos`, under the same version.
  Its `version` and `appVersion` are the release's version, and its default values
  name every image by the digest the release published. It is signed too.

`edge` is not part of a release. It keeps following main.

## When a version is released

A version is released when its release run (the `images` workflow on the tag) has
finished green. Before that, treat anything under the version as not released. Before
you trust an image or the chart, check its signature with the command under
[Checking a signature](#checking-a-signature).

## Cutting one

Releases are cut from main, or for a patch to an older release from
`release/MAJOR.MINOR` (PRD-279 §5.5; the branch, the cherry-pick and the patch tag are on
[Patching an older release](patching-a-release.md)). A tag's version is never below the
chart's `version` at the tagged commit (0.11.0 today), so the examples below stand for
whatever version is next. Push a release candidate first:

```bash
git tag -a v0.7.0-rc.1 -m "v0.7.0-rc.1" <commit on main>
git push origin v0.7.0-rc.1
```

A tag with a suffix (`-rc.1`) is a release candidate. It runs every step of a release,
publishes the images under `0.7.0-rc.1` and the chart under `0.7.0-rc.1`, and does not
move the `0.7` release line. Once a candidate's run is green, push `v0.7.0` on the same
commit.

Before anything builds, the run stops when:

- the tag is not `vMAJOR.MINOR.PATCH` with an optional suffix, or its version is below
  the chart's `version` in [`charts/automatos/Chart.yaml`](../../charts/automatos/Chart.yaml);
- any published image already has a tag with that version (a version is never replaced);
- the tagged commit is in neither the default branch nor `release/MAJOR.MINOR` for the
  tag's own line.

### The first release candidate

The tag path has never run before the first candidate. Check, on its run:

- each step marked `(v* tags only)` ran, and the step summaries list what was published;
- `0.7.0-rc.1` is on all four images and on the chart, and `0.7` and `edge` did not move;
- the command under [Checking a signature](#checking-a-signature) passes for an image
  and for the chart;
- pushing the same chart version again is refused (re-run the chart job once).
- If the chart job's "Refuse a version that is already published" step stops the first
  candidate because the registry does not answer "not found" for a chart that has never
  been pushed, that is the step to adjust: its log shows the answer it got.

## Who can create a version tag

Only maintainers may create, move or delete a `v*` tag. That is a **repository
ruleset**, which a repository admin sets by hand. CI cannot check it: a ruleset is an
admin setting of the repository, not a file in it, and the workflows' tokens have no
admin rights. Look at it in the repository's settings after any change to its rules.

The setting, under **Settings → Rules → Rulesets → New ruleset → New tag ruleset**:

| Field | Value |
|---|---|
| Ruleset name | `Version tags` |
| Enforcement status | Active |
| Bypass list | the **Repository admin** and **Maintain** roles, always allowed |
| Target tags | include by pattern: `v*` |
| Rules | Restrict creations, Restrict updates, Restrict deletions, Block force pushes |

No workflow in this repository creates a tag, so the ruleset blocks nothing CI does.

## What a `v*` tag run does

All of it is in [`.github/workflows/images.yml`](../../.github/workflows/images.yml).
Every step and job that only a `v*` tag runs says so in its name. A push to main, a
pull request, the weekly rebuild and a manual run on a branch behave as before.
Nothing carries a version tag until it has been scanned, signed and its signature
checked.

1. **The version and the preflight** (`changes` job). `scripts/ci/release_version.py`
   reads the version off the tag, and `scripts/ci/release_preflight.py` refuses the
   run if a published image already has that version or the commit is on no release
   line. A registry or API that does not answer stops the run too.
2. **The scan fails, not warns** (`build` jobs). On a `v*` tag the scan policy
   (`scripts/ci/image_scan_policy.py`) runs in `fail` mode whatever `POLICY_MODE` is
   for branches. A fixable Critical, or a match on CISA's Known Exploited
   Vulnerabilities list, that `.grype.yaml` does not allow with a reason and an expiry
   fails that image's build job, and nothing after it runs. The allow-list's rules are
   the same as on any branch.
3. **Staged, signed and checked** (`merge` jobs, one per image). Each image's
   multi-arch index is published under `release-staging-<run id>-<attempt>`, not under
   a version, and not as `sha-<7>` (the commit's main push published it). Its digest is
   signed, and the signature checked with the identity of this workflow at the tag
   (the command below). A failed check fails the run, and no version tag is applied.
4. **The version tags, all at once** (`release-images` job, after every image is
   checked). `scripts/ci/release_tags.py` reads every image's tags first, then adds
   `X.Y.Z`, and `X.Y` for a full release, to each image's signed digest. `edge` does
   not move: smoke-images and its promote-edge job never run on a tag.
5. **The chart** (`release-chart` job, after the version tags). `scripts/ci/release_chart.py`
   copies `charts/automatos`, writes each image's repository and signed digest into the
   copy's `values.yaml` (only those lines change), and `helm package` stamps it with
   the version. The web app has one slot per published edition (`frontend.image` for
   local, `frontend.enterpriseImage` for enterprise), and each is pinned; a release whose
   two web slots would hold the same digest publishes nothing. The chart in
   the repository keeps `appVersion: "edge"`.
6. **The chart names no tag, in any edition.** The packaged chart is rendered once per
   published edition, each with `existingSecret` and the least that edition renders
   with (made-up values, written by `release_chart.py prepare`), and the job fails if
   an edition was not rendered, if any image is named by a tag, if an image that
   edition runs is missing or another one appears, or if the labels do not carry the
   version.
7. **The chart is published, signed and checked.** It is pushed to
   `oci://ghcr.io/automatosai/charts/automatos` under the version, signed by digest,
   and its signature checked with the same command as the images.

The staging tags stay on the images after the run. They are not versions, and nothing
installs from them.

## When a run fails part-way

- **Before the version tags** (the preflight, a scan, a build, signing or a signature
  check): no image carries the version, so the same tag can run again ("Re-run all
  jobs", or fix the cause and push the tag again after deleting it).
- **During the version tags:** the step reads every tag before it writes any, so it
  stops before tagging when one cannot be read or a version points elsewhere. If the
  registry fails between two images, "Re-run failed jobs" finishes the set from the
  same digests. A full re-run is refused once any image has the version: cut the next
  version.
- **The chart:** `helm push` tags the chart with its version before it can be signed,
  so the chart is visible under the version for the minutes between the push and the
  check. If its digest cannot be read, or signing or the check fails, the job deletes
  the version it pushed (GitHub's packages API, with the built-in token, which may
  delete a version of a package this repository's workflow published) and fails;
  "Re-run failed jobs" then pushes it again. If GitHub refuses the delete, the job's
  error names the chart version a maintainer must delete by hand under the
  organisation's packages. Deleting package versions from a workflow is in public
  preview on GitHub.

## A version is never replaced

The registry does not protect a version: GHCR moves a tag that is pushed again. What
does:

- **The images:** the preflight refuses a run whose version any published image already
  has, the tag step refuses a version that points at another digest, and the tag
  ruleset limits who can push a `v*` tag at all.
- **The chart:** the release job reads the registry before it pushes, and stops if that
  version is there, or if the registry cannot be read. Runs for one tag never overlap:
  the workflow's concurrency group is per ref.

## The first publish

GHCR makes a new package private. After the first release candidate publishes the
chart, an organisation owner makes `charts/automatos` public once, under the
organisation's packages, so `helm install` works with no login. The `api`, `frontend`
and `workspace-worker` image packages are public already.

`automatos-enterprise-frontend` is new: the first push to main that builds it creates
the package, private. An organisation owner makes it public once, as for the chart,
before the next release candidate. Until then an install cannot pull it without a
login, and the preflight, which reads the images anonymously, may not get the answer
"not found" for its version tag and would then stop the run: its log shows the answer.

## Checking a signature

Each image and the chart are signed by cosign with no key, through GitHub's OIDC
token. The certificate names the workflow file and the ref that ran. To check a
release yourself (cosign 2.x or 3.x), name the version's tag in the identity and the
artifact by digest:

```bash
cosign verify \
  --certificate-identity "https://github.com/AutomatosAI/automatos-ai/.github/workflows/images.yml@refs/tags/v0.7.0" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  ghcr.io/automatosai/automatos-api@sha256:<digest>
```

The same for `automatos-frontend`, `automatos-enterprise-frontend`, `automatos-workspace-worker` and
`ghcr.io/automatosai/charts/automatos`. A tag's digest:
`docker buildx imagetools inspect ghcr.io/automatosai/automatos-api:0.7.0`.

[Checking a release](checking-a-release.md) gives the full command for an image and for
the chart, says how to read each image's contents list and build record, and says what a
passing check does and does not prove. Each release page links to it.

The workflow's own check ([`scripts/ci/release_signing.py`](../../scripts/ci/release_signing.py))
pins more than that, each claim exactly and never by pattern: the identity, the issuer,
the repository, the ref, the commit and the workflow's name. A signature made by a pull
request build (`…@refs/pull/<n>/merge`), by a branch (`…@refs/heads/main`), by another
tag, by another workflow or in a fork does not pass it.
