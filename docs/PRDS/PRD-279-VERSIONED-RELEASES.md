# PRD-279: Versioned releases — a version you can name, a chart you can pull, and a signature you can check

**Status:** Draft for Dave's review. Written 20 Nov 2026. No code has been written.
**Owner:** Gerard · **Reviewer:** Dave
**Grounded on:** `automatos-ai` `main @ 97c727136`, read on 20 Nov 2026.
**Labels:** **[T]** read in code at that commit; **[I]** inferred; **[G]** Gerard's decision; **[?]** not known.
**Relates to:** PRD-262 (the image pipeline; its last wave turns the vulnerability scan from a warning into a failure). PRD-277 (its US-505 publishes an `enterprise` web image). PRD-278 and PRD-280 (what a release contains).

**Decisions already taken [G] (Gerard, 19 Nov 2026):**

| | Decision |
|---|---|
| G1 | An organisation's install is delivered by naming a version, never by following a moving tag. |
| G2 | The first install is a plain `helm install` of the chart in this repository. |

---

## 1. Introduction

Today there is nothing to install by name. The images are published under a commit id, a weekly date and a tag called `edge` that moves. The chart lives only in the repository, at a version that points at `edge`. No version tag has ever been cut.

That is fine for someone trying Automatos on a laptop. It does not work for an organisation that has to say which version it runs, prove where it came from, hold it still, and go back to the one before.

Most of the machinery exists. The image workflow already reacts to a `v*` tag, already signs each image, and already attaches a contents list and a build record. What is missing is the release itself: a tag, a published chart that names its images exactly, notes that say what changed, and a signature anyone can check against a named identity.

## 2. What exists today

| Area | Today | Where | What it means |
|---|---|---|---|
| Triggers | The image workflow runs on pushes to main, on `v*` tags, on pull requests, weekly and by hand. Only main and `v*` tags publish. | `.github/workflows/images.yml:107-146`, header `:6` **[T]** | The tag path exists. |
| Tags on an image | `sha-<7>`, `weekly-<date>`, and on a version tag `<version>` and `<major>.<minor>`. `latest` is never set. | `images.yml:876-881` **[T]** | A version tag would already produce version tags on the images. |
| `edge` | Moved to a commit's images after they pass a boot test. | `.github/workflows/smoke-images.yml:269-310` **[T]** | The moving tag people use today. |
| Version tags cut | None. The repository's only tags are five archive tags. | `git tag -l` **[T]** | The version path has never run. |
| What is published | `api`, `frontend` and `workspace-worker`, as `ghcr.io/<owner>/automatos-<name>`, for two processor types. | `images.yml:817-826`, `:839` **[T]** | Three images make a release. |
| Contents list and build record | Attached by the build: `provenance: mode=max`, `sbom: true`. | `images.yml:406-407` **[T]** | Present on every image. |
| Scan | Grype, on every image. The policy step runs with `POLICY_MODE: report`: it warns and does not fail. A comment says the last commit of PRD-262's first wave sets `fail`. | `images.yml:497-564`, `:551` **[T]** | A release could be cut today from an image with a fixable critical finding. |
| Signing | Each published image is signed by digest, with the workflow's identity and no key. Nothing checks the signature afterwards, and no document says which identity to expect. | `images.yml:907-912` **[T]** | A signature exists. Nobody is told how to trust it. |
| Chart | Version 0.6.0, `appVersion: "edge"`. Each image can be named by digest. With no digest and no tag it uses `appVersion`. | `charts/automatos/Chart.yaml`; `templates/_helpers.tpl:44-55` **[T]** | The chart supports exact images. Its default is the moving one. |
| Chart in CI | Linted, rendered and checked against one Kubernetes version, and unit-tested. | `.github/workflows/chart.yml:66-76` **[T]** | Checked on every change. |
| Chart publishing | None. No packaging, no registry push, no signature. | not found in `.github/workflows/` **[T]** | The chart can only be used from a checkout. |
| Release record | No release workflow, no change log, no version file. | not found **[T]** | Nothing says what a version contains. |
| What the API says it is | `"version": "1.0.0"`, written in the code. | `orchestrator/main.py:1342`, `:1468` **[T]** | A running install cannot say what it is. |
| Web image | One edition is published: `local`, with its API address, WebSocket address and media origins fixed when the image is built. | `images.yml:254-257`; `frontend/Dockerfile:73-101` **[T]** | An image built for one address cannot serve another. |
| Media origins | Read from `NEXT_PUBLIC_MEDIA_ORIGINS` into the content security policy in `next.config.js`. | `frontend/next.config.js:22` **[T]** | Fixed at build time **[T]**: the policy is a `headers()` rule, which `next build` evaluates (see US-402's first line). An install whose files sit on its own storage host needs its own value. |

## 3. Goals

- A maintainer cuts a release by pushing one tag, and everything else follows from it.
- A release is the images and one chart, cut from one commit, under one version number.
- The chart is published to the registry, and by default it names each image by its exact digest.
- Anyone can check, with one documented command, that an image or the chart was built by this repository's release workflow at that tag.
- A release cannot be published from an image the scan would fail.
- Each release says what changed, whether it changes the database, and which earlier version it can be upgraded from.
- A running install can say which version it is.
- The published web image for an organisation's install needs no rebuild for each address.

## 4. Non-goals

- **Turning the scan's default from warning to failing.** That is PRD-262's. This PRD makes a release fail on it regardless.
- **The `enterprise` web image itself.** PRD-277, US-505. This PRD removes the per-install settings that image would otherwise have fixed at build.
- **What the chart installs.** PRD-280.
- **A promise of how fast a fix is released.** That is an operating commitment, not a workflow.
- **Automatic upgrades.** A release is something an operator chooses to install.

## 5. Design

### 5.1 What a release is

One version number, in the form `MAJOR.MINOR.PATCH`, names:

- the images, each tagged with the version and with `MAJOR.MINOR`: `api`, `workspace-worker`, and a web image for each edition that is published (`local` today, and `enterprise` once PRD-277's US-505 lands);
- the chart, whose own version and `appVersion` are both that number;
- a release page with the notes.

All of it comes from the commit the tag points at.

`edge` stays for people who follow main. A released chart never refers to it.

### 5.2 Cutting a release

A maintainer pushes a tag `vMAJOR.MINOR.PATCH`. Only maintainers can create a tag of that form, by a ruleset on the repository.

1. The image workflow builds, scans, publishes and signs the images, as it does today.
2. On a version tag the scan policy runs in `fail` mode whatever the default is. A failing image publishes nothing, and the release stops.
3. The workflow checks each signature it just made, against its own identity at that tag.
4. A release job packages the chart with the version, writes each image's digest into the chart's default values (for the web app, one digest for each published edition, and the chart picks the one for `edition.authEdition`), pushes the chart to the registry as an OCI artifact and signs it.
5. The same job renders the packaged chart and fails if any image in the output is named by a tag.
6. It creates the release page, with the notes in section 5.4.

A tag with a suffix, such as `v0.8.0-rc.1`, is a release candidate. It does every step and is marked as a pre-release. Its images do not move the `MAJOR.MINOR` tag.

### 5.3 A signature that can be checked

Signing stays as it is: by the workflow's own identity, with no key to hold. Two things are added.

- **The identity is written down.** A page in `docs/` gives the exact command to check an image and the chart: the workflow file, the repository and the tag form to expect. A signature from a branch build, a fork or another workflow does not pass that check.
- **The workflow runs that command on its own output.** A release whose signature does not check is not published.

### 5.4 Release notes

Written by the release job, from the commits between the previous release and this one.

| Section | Holds | From |
|---|---|---|
| What changed | The merged pull requests, grouped by their type | Commit messages, which already follow `type(scope): description` |
| Database | "This release changes the database" or "does not", and the migrations added | A comparison of the Alembic revisions at the two tags |
| Upgrading | The earliest version this one can be upgraded from in one step | A value kept in the repository and changed on purpose |
| What to install | The chart's address and version, and each image's digest | The release job |
| Checking it | The command from section 5.3 | The same page |

The "Database" line matters most to an operator. A release that changes the database can only be undone by restoring a backup, because migrations go forward only.

### 5.5 A patch to an older release

Releases are cut from main. When a fix is needed for a release and main has moved too far to ship as a patch, a branch `release/MAJOR.MINOR` is made from that release's tag, the fix is cherry-picked to it by pull request, and the patch tag is cut there. The image workflow already runs on any `v*` tag.

### 5.6 An install that knows its version

The image is built with its version, and the API reports it wherever it reports `"version"` today. A build from main reports the commit id. The web app shows the same value where it shows the edition.

### 5.7 A web image that needs no rebuild

PRD-277 makes the `enterprise` web image call the API by a relative address, which removes the API address from the build. Two per-install settings remain.

- **The WebSocket address** is worked out from the page's own address when none is set and the API address is relative. Before Wave 4 the helper took it from its own setting, or worked it out from an absolute API address, and otherwise had none (`frontend/lib/api-config.ts` `getWebSocketUrl`, now `frontend/lib/websocket-url.ts` **[T]**). Correction: no module imports `lib/api-config.ts`, so that address reached no socket the web app opens; the only WebSocket the web app opens is the canvas terminal's, whose address the API hands it **[T]**. Wave 4 still gives the helper its third case, so any socket that reads it gets the right address.
- **Media origins** are read when the server starts, not when the image is built. The content security policy is then built from the running server's settings.

The `local` and `saas` images keep their build-time values and behave as today.

## 6. Waves and user stories

Each wave is one pull request or a short series. CI is the only gate.

### Wave 1: A release can be cut

#### US-101: The first version tag · S
**Description:** As a maintainer, I want the version path proven end to end on a release candidate, so that the first real release is not the first time it runs.

**Acceptance Criteria:**
- [ ] A ruleset lets only maintainers create a `v*` tag. A page records the setting. It is a repository setting: a person sets it, and CI cannot check it. *(Page written: `docs/deployment-infrastructure/releases.md`. Needs a repository setting: an admin creates the ruleset.)*
- [ ] A release-candidate tag publishes every image with the version tag, and does not move `MAJOR.MINOR` or `edge`. *(Built: the merge job publishes each image only under a staging tag, and one job adds `X.Y.Z` (and `X.Y` for a full release only) after every image is signed and its signature checked. No `sha-<7>` on a tag; smoke-images never runs on a tag. Needs a real tag to show.)*
- [ ] The workflow run is linked from the pull request that follows it, with what was published. *(Needs a real tag.)*
- [ ] Anything on the tag path that did not work is fixed in this wave. *(Found by reading and fixed: a tag run re-pushed `sha-<7>`, and applied version tags before the signature was made or checked. The rest needs a real tag.)*

#### US-102: A release fails on a failing scan · S
**Description:** As an operator, I want no release published from an image with a fixable critical finding, so that a version number means the scan passed.

**Acceptance Criteria:**
- [x] On a `v*` tag the policy step runs in `fail` mode, whatever `POLICY_MODE` is for branches.
- [x] A failing image is neither tagged with the version nor signed. *(The merge job needs every build job, and the version tags come only after every merge leg; `test_prd279_release_workflow.py`.)*
- [x] The allow-list's rules are unchanged: an entry still needs a reason and an expiry.
- [x] The scan policy's own tests cover the tag case.

#### US-103: The workflow checks its own signatures · S
**Description:** As an operator, I want a published image to have a signature that was checked before I ever see it, so that "signed" means something.

**Acceptance Criteria:**
- [ ] After signing, the workflow verifies each image against this repository's image workflow at the tag that ran. *(Built: on the staging digest, before any version tag exists. Needs a real tag to run.)*
- [ ] A failed check fails the run. *(Built: no image gets the version, and the chart is not published. Needs a real tag to run.)*
- [x] A signature made by a pull request build, or from a branch, does not pass the same check. A test of the command's arguments shows it.

### Wave 2: The chart is published

#### US-201: Packaging the chart with exact images · M
**Description:** As an operator, I want a chart that installs exactly the images it was released with, so that the version I name is the version I get.

**Acceptance Criteria:**
- [ ] On a version tag, the chart is packaged with its version and `appVersion` set to the tag's version. *(Built. Needs a real tag to run.)*
- [ ] The packaged chart's default values name every image by digest, with one web image digest for each published edition. *(Built for both published editions: `frontend.image` holds the local web image's digest and `frontend.enterpriseImage` the enterprise one's (PRD-277 US-505), and the templates pick the slot by `edition.authEdition`; the packaging step no longer refuses a second edition (`scripts/ci/release_chart.py`, `tests/test_prd279_release_tools.py`). Needs a real tag to run.)*
- [ ] Rendering the packaged chart with only `existingSecret` set, once for each published edition, shows no image named by a tag. The job fails if one is. *(Built, once per edition. Correction: an edition other than the chart's default also needs its own values to render, at least `edition.authEdition`, and enterprise its identity provider's settings (PRD-277 US-701). Each edition's render is given `existingSecret` and a values file `release_chart.py prepare` writes for it, of made-up values, and `check-rendered` fails when an edition was not rendered or runs any image but its own. Needs a real tag to run.)*
- [x] The chart in the repository keeps `appVersion: "edge"`, and its tests pass unchanged.

#### US-202: Publishing and signing the chart · M
**Description:** As an operator, I want to pull the chart from the registry and check where it came from, so that I do not need a checkout to install.

**Acceptance Criteria:**
- [ ] The packaged chart is pushed to the registry as an OCI artifact, under the release's version. *(Built. Needs a real tag to run.)*
- [ ] It is signed with the workflow's identity, and the signature is checked in the same run. *(Built. `helm push` tags the chart before it can be signed; a chart not proven is deleted again through GitHub's packages API with the built-in token (public preview), or the job names what a maintainer must delete. Needs a real tag to run.)*
- [ ] A version that already exists is never overwritten: a second push of the same version fails. Shown once, on a release candidate, and recorded in the pull request. *(GHCR overwrites a tag, so the job reads the registry first and refuses an existing version, or a registry it cannot read. Needs a real tag to show.)*
- [x] The chart's README gives the command to install a named version from the registry.

#### US-203: The chart checked against each supported Kubernetes version · S
**Description:** As an operator, I want the chart checked against the Kubernetes versions I might run, so that "supported" is tested.

**Acceptance Criteria:**
- [x] The chart workflow renders and checks the chart against the oldest supported version and the newest, not one.
- [x] The README states both versions, and a test fails if the README and the workflow disagree.

### Wave 3: A release says what it is

#### US-301: Release notes · M
**Description:** As an operator, I want each release to tell me what changed and whether my database will change, so that I can plan the upgrade and the way back.

**Acceptance Criteria:**
- [ ] The release job creates a release page with the five sections in section 5.4. *(Built: the `release-notes` job in `images.yml`, after the chart is published and checked, runs `scripts/ci/release_page.py publish`; it holds `contents: write` and nothing else, and never replaces a page that exists. Before any image builds, the `changes` job runs `release_page.py check` from a full checkout with a read-only token, and refuses there everything the page would be refused for that needs no digest (`tests/test_prd279_release_check.py`). The five sections, in order, and the job's shape are tested (`tests/test_prd279_release_notes.py`). Needs a real tag to create a page.)*
- [x] The "Database" section is worked out from the Alembic revisions at the two tags, and a test covers a release with a new migration and one without. *(`release_history.py` reads each `versions/*.py` at both tags with `git ls-tree` and `git show`, never importing it. Tested: a migration added, none added, a shipped one removed or edited (which reads as a change, never "does not change"), one added and another removed, a merge revision, and the first release.)*
- [x] The earliest version that can be upgraded from is read from one file, and the job fails if the file is missing. *(`scripts/ci/release_upgrade_from.txt`, whose format is written in it: one line, `MAJOR.MINOR.PATCH` of a tagged full release below the version, or `none`. The job fails, and creates no page, when it is missing, malformed, or names a version that is not tagged or not below this one. It says `none` until the first release.)*
- [x] A release candidate is marked as a pre-release. *(The request the job sends: `prerelease: true`, and never marked the latest. A full release is the latest only when no full release has a higher version, so a patch to an older line is not.)*

#### US-302: A page on checking a release · S
**Description:** As a security reviewer, I want the exact command and the identity to expect, so that I can check a release myself.

**Acceptance Criteria:**
- [x] A page in `docs/` gives the command for an image and for the chart, with the workflow, repository and tag form to expect. *(`docs/deployment-infrastructure/checking-a-release.md`; a test holds every claim the workflow's own check pins (`release_signing.CLAIM_FLAGS`) on the page.)*
- [x] It says what the contents list and the build record are, and how to read them. *(The SBOM and provenance attestations the build makes, read with `docker buildx imagetools inspect --format`.)*
- [x] It says plainly what a passing check does and does not prove.
- [x] The release notes link to it. *(At the release's own tag; tested in `tests/test_prd279_release_notes.py`.)*

#### US-303: An install reports its version · S
**Description:** As an operator, I want a running install to say which version it is, so that I never have to work it out from an image id.

**Acceptance Criteria:**
- [x] The API image is built with its version: the tag's version on a release, the commit id otherwise. *(images.yml passes `AUTOMATOS_VERSION` to the API alone: the tag's version, else `github.sha`, which on a pull request is its merge commit. The Dockerfile declares it after the last filesystem layer, so no layer is rebuilt for a new value, and the cache keys and `tree-<12>` tags are unchanged; a `tree-<12>` tag keeps the first image built from that tree, which reports that commit. `config.AUTOMATOS_VERSION` reads it; a build given none (compose, `make build`, Railway, the test suite) reports `dev`. `tests/test_prd279_install_version.py`.)*
- [x] Both places in `main.py` that return `"version": "1.0.0"` return that value. *(Correction: there were three places that said `1.0.0`: the app's own version (OpenAPI), `GET /`, `GET /health`, and also `GET /api/system/health` in `api/system.py`. All four return it.)*
- [x] The web app shows it where it shows the edition. *(On the profile page, beside the edition's name, in every edition: "Enterprise edition · version 0.12.0". Read through `apiClient` from the existing, authenticated `GET /api/system/health`, because in enterprise the chart exposes neither `/` nor `/health` to the browser. `frontend/app/settings/profile/install-version.tsx` and its test.)*
- [x] A test pins that the value is not the literal `1.0.0`.

#### US-304: Patching an older release · S
**Description:** As a maintainer, I want a written way to fix a release that main has moved past, so that a customer on last month's version can get a security fix alone.

**Acceptance Criteria:**
- [x] A page describes the `release/MAJOR.MINOR` branch, the cherry-pick by pull request, and the patch tag. *(`docs/deployment-infrastructure/patching-a-release.md`. A patch carries no migration. Another repository that imports a release may require the release's commit to be on main: the page says to check that before the first patch.)*
- [ ] The branch is protected as main is. A repository setting, recorded on the page. *(Page written: a branch ruleset on `release/*` with main's rules. Needs a repository setting: an admin creates the ruleset. images.yml now builds and scans a pull request into `release/**` as one into main, so main's required checks can run there; a merge into a release branch publishes nothing.)*
- [ ] The image workflow publishes a patch tag cut from such a branch, shown once with a release candidate. *(Built: the preflight already accepts a commit on `release/MAJOR.MINOR` for the tag's own `MAJOR.MINOR` only. Tested through its entry point: a tag on main, a patch tag or candidate on its own release branch, a tag on neither, and a tag on another line's branch (`tests/test_prd279_patch_releases.py`). Needs a real patch tag to show.)*

### Wave 4: A web image that needs no rebuild

#### US-401: The WebSocket address from the page · S
**Description:** As an operator, I want the web app to find its WebSocket endpoint from its own address, so that one image serves any install.

**Acceptance Criteria:**
- [x] With no WebSocket address set and no absolute API address, the web app uses the page's own host and scheme. In the `enterprise` edition: `wss://<host>/ws` on an https page, `ws://<host>/ws` on an http one. `local` and `saas` still answer none, as today (§5.7: they keep their build-time values). The address is worked out when it is read, so on the server, with no page, it is none.
- [x] With either set, behaviour is as today. In every edition, including the old helper's answer for an https API address (`ws://`, not `wss://`).
- [x] The one helper that works the address out (`getWebSocketUrl`) gains the third case, and a test covers all three. `frontend/lib/websocket-url.ts`, read by `WEBSOCKET_CONFIG.URL`; `frontend/lib/__tests__/websocket-url.test.ts` runs each case under each edition.

#### US-402: Media origins read at start-up · M
**Description:** As an operator, I want to give the web app my storage host when it starts, so that images and files stored on my own storage are not blocked.

**Acceptance Criteria:**
- [x] The pull request states whether `next.config.js` fixes the content security policy at build time, with evidence. It does. The policy is a `next.config.js` `headers()` rule whose value is computed when the config file is loaded, from `NEXT_PUBLIC_MEDIA_ORIGINS`, and that load happens in `next build` (`frontend/Dockerfile:110`). The image then runs `node server.js` from the build's `.next/standalone` output (`frontend/Dockerfile:159-162`, `:183`), whose `server.js` Next writes with the configuration the build resolved; the image workflow lists `next.config.js` and `csp-sources.js` among the frontend image's build inputs, and `csp-sources.js` already said the headers are "baked into the standalone build". Next's content security policy guide shows the run-time way: the middleware sets the header on each response. *Read from the code and Next's documentation, not from a built image: no image was built for this pull request.*
- [x] In the `enterprise` image, the media origins come from a setting read when the server starts. `MEDIA_ORIGINS` (the web container's environment, not a `NEXT_PUBLIC_*` variable): checked when the Node server starts (`instrumentation.ts` → `lib/enterprise/start-up.ts`; a bad value stops the server with one line naming the setting, never the value) and read by the middleware for each response. Each origin must be `https://host[:port]`, `https://*.<domain>.<suffix>`, or `http://localhost` / `http://127.0.0.1` with a port; nothing else gets in (`lib/enterprise/media-origins.ts`, `lib/enterprise/__tests__/media-origins.test.ts`, `start-up.test.ts`).
- [x] A response's content security policy contains the origins the running server was given. In enterprise the middleware sets the policy on every answer it gives, through or redirected, built by the same function as the build's policy (`frontend/csp-policy.mjs`); the build keeps its policy only on the paths the middleware never runs for (`_next` and files with an extension), so no answer carries two. With no media origin it is exactly the build's enterprise policy. `lib/enterprise/__tests__/runtime-csp.test.ts`.
- [ ] The `local` and `saas` images behave as today, and the existing policy tests pass unchanged. *None of the existing policy tests is changed; CI proves they pass. `runtime-csp.test.ts` also checks that the saas config and the local config (the published image's build arguments) give one rule and the same policy string as before, written out in full.*
- [ ] The chart passes the setting to the web app. *Built: with `edition.authEdition=enterprise` the web container runs the enterprise image and gets `MEDIA_ORIGINS` from `frontend.mediaOrigins` (empty by default) and `BACKEND_INTERNAL_URL`, the API's Service; rendering stops on a value the server would refuse, by the server's rules, and on any value in another edition (`charts/automatos/tests/enterprise_web_test.yaml`, `media_origins_refusals_test.yaml`). Left unticked until CI runs those suites, and until an install lane starts the enterprise image with them.*

## 7. Functional requirements

- FR-1: A `v*` tag must produce every published image and one chart under one version number, from one commit.
- FR-2: Only maintainers may create a `v*` tag.
- FR-3: On a `v*` tag the scan policy must fail the release for any finding it would fail in `fail` mode.
- FR-4: Every published image and chart must be signed, and the signature must be checked by the workflow before the release is complete.
- FR-5: A released chart's default values must name every image by digest.
- FR-6: A published version must never be overwritten. *(The registry does not enforce it. Images: a preflight before any build refuses a version any published image has, and the tag step refuses a version on another digest. The chart: the release job refuses an existing version. Both rest on the `v*` tag ruleset too. Needs a real tag to show.)*
- FR-7: Every release must have notes that say whether it changes the database.
- FR-8: The API must report the version it was built as.
- FR-9: A release candidate must not move the `MAJOR.MINOR` tag or `edge`.
- FR-10: The `enterprise` web image must take its WebSocket address and media origins without a rebuild.

## 8. Decisions for Gerard

Each has a recommendation, which stands unless he changes it.

| # | Decision | Recommendation | Why |
|---|---|---|---|
| D-1 | The first version number | `0.7.0`, for the chart and the application together | The chart is at 0.6.0 and must not go backwards. One number for both is simpler than two. *Correction (Wave 3): the chart is now at 0.11.0, and the workflow refuses a tag below the chart's version, so the first version is 0.11.0 or later.* |
| D-2 | Where the chart is published | The same registry as the images, as an OCI artifact | One place, one signing method, no separate chart repository to run |
| D-3 | Release notes | Generated from commit messages, with the database line worked out | The messages already follow a type and scope. A hand-written section can be added on top. |
| D-4 | Whether a release fails on the scan before PRD-262 flips the default | Yes | A version number an organisation installs should not be the lenient case |
| D-5 | Where a release is cut from | Main, with a `release/MAJOR.MINOR` branch only when a patch needs one | No standing branches to keep in step |

## 9. Technical considerations

- **House rules that apply:** every action pinned to a commit (the `action-pins` workflow checks it); signed-off commits; pull requests under about 400 lines.
- **Ask first.** US-402 changes how the content security policy is built, and that policy is a guardrail. Wave 4 opens with an issue.
- **Workflow permissions.** The release job needs to write packages, write a release and ask for an identity token, and nothing else.
- **PRD-262.** Its last wave sets `POLICY_MODE` to `fail` for every build. US-102 here is then the same behaviour, and its special case can be removed in the same pull request that flips the default.
- **PRD-277, US-505.** The `enterprise` web image is published by that story. Wave 4 here changes what that image fixes at build, so the two should land in that order or together.
- **The moving tag.** `smoke-images.yml` keeps moving `edge` after a boot test. A release candidate is boot-tested the same way before the release is marked complete **[I]**.
- **Tests that pin today's shape:** the chart's `image_test.yaml`, and `frontend/lib/__tests__/csp-media-sources.test.ts`.

## 10. Success metrics

- A person with no checkout installs a named version from the registry with one command.
- The command on the checking page passes for every image and chart of a release, and fails for a branch build.
- The release notes of two consecutive releases say correctly which one changed the database.
- A running install answers "which version are you" with the tag it was built from.

## 11. Risks

| Risk | What limits it |
|---|---|
| The version path has never run, and fails on the first real release | US-101 runs it on a release candidate first |
| A release is blocked by a scan finding with no fix available | The policy fails only on fixable critical findings and known-exploited ones, and the allow-list takes a reasoned, expiring entry |
| The chart's digests and the images drift apart | The digests are written by the same run that built the images, and the render check fails on any tag |
| Reading media origins at start-up changes the policy for the hosted service | Only the `enterprise` image changes. The existing policy tests pin the other two. |
| Signing with a workflow's identity depends on a public service being up at release time | A release that cannot be signed is not published, and is cut again |

## 12. Open questions

1. **[answered, US-402] Is `next.config.js` evaluated at build for the standalone server?** Yes: its `headers()` rules are fixed by `next build` (evidence in US-402's first line). The enterprise policy is therefore set by the middleware on each response.
2. **Should a release also carry the `media-assets` image?** It is published by a separate job today (`images.yml:962`) **[T]**.
3. **How long is a `MAJOR.MINOR` line patched?** That is a support promise, not part of this PRD, and the patch page should state it once it is decided.
4. **[open, not built] A tighter policy for the `enterprise` image.** Its content security policy (US-402) is exactly as strict as the build-time one it replaces: the same directive list, without the Clerk Frontend API host, the API origin, payment sources and the build's media origins. It still names the hosted product's origins (`https://*.automatos.app`, Automatos Cloud's catalogue files route) and third parties' (Clerk's development and protection hosts, Cloudflare's challenge host, the jsDelivr CDN, Google's and Composio's image hosts), and keeps `'unsafe-inline'` and `'unsafe-eval'` in `script-src`. Dropping any of them needs a browser lane that runs this edition's pages first, to prove none breaks.
5. **[open, Wave 3] May a line's first release (`X.Y.0`) be cut on `release/X.Y`?** The preflight accepts any version of a line from that line's branch, `X.Y.0` included. The patch page cuts only patches there; refusing `X.Y.0` off main would be one more check in the preflight.
6. **[open, Wave 3] The version a Railway build reports.** Railway builds the API without the workflow's build argument, so it reports `dev`. Passing Railway's commit id to the build would make the hosted API report its commit, as main's images do.
