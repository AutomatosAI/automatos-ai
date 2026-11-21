# Patching an older release

Releases are cut from main. When a release needs a fix, a security fix above all, and main
has moved too far past it to ship main as the next patch, the fix is released on a branch of
its own: `release/MAJOR.MINOR`, made from that release's tag. An organisation on last month's
version then gets the fix alone, as a patch version, without everything main has gained since.
It is the record for [PRD-279](../PRDS/PRD-279-VERSIONED-RELEASES.md) §5.5 and US-304. How a
release is cut, and what its tag runs, is on [Releases](releases.md).

How long a `MAJOR.MINOR` line is patched is a support promise, not part of the workflow. It is
not decided yet (PRD-279 §12, question 3); this page will state it once it is.

## The release branch

One branch per release line, `release/MAJOR.MINOR`, made once, from the tag of that line's
newest release, the first time a patch is needed:

```bash
git fetch origin --tags
git switch -c release/0.11 v0.11.2
git push -u origin release/0.11
```

Nothing else is ever pushed to it directly: every change reaches it by pull request. It never
merges back into main, and main never merges into it. The branch stays for as long as the line
is patched.

Pushing to the branch, and merging into it, publishes nothing: the image workflow publishes
from main and from `v*` tags only. A pull request into the branch is built and scanned by the
image workflow as one into main is, and runs the same tests.

## The fix, by pull request

The fix lands on main first, by its own pull request, unless main no longer has the code it
fixes. Then it is cherry-picked onto a branch made from the release branch, and that branch is
merged into the release branch by a pull request:

```bash
git fetch origin
git switch -c fix/0.11-<what it fixes> origin/release/0.11
git cherry-pick -x <the fix's commit on main>
git push -u origin fix/0.11-<what it fixes>
gh pr create --base release/0.11 --title "fix(<scope>): <what it fixes> (0.11)"
```

`-x` records the commit it was picked from in the message, so the patch's history names the fix
on main. When the change does not apply cleanly, resolve it in that branch, and say in the pull
request what differs from main's. A fix main does not have (the code is gone there) is written
on the branch made from the release branch, and the pull request says why it is not on main.

**A patch changes no database.** A release line's migrations are those of its first release;
a migration on the release branch would be a revision main does not have, and an install that
upgrades from the patch to the next release from main would stop on it. A fix that needs a
migration is released from main, as the next `MINOR`. The release page's "Database" section
says whether a patch changes the database, as for every release, and a patch that does is
called out there as breaking this rule.

## The patch tag

Once the fix is merged, cut a release candidate on the release branch's commit, then the patch
itself on the same commit once the candidate's run is green, as for any release:

```bash
git fetch origin --tags
git tag -a v0.11.3-rc.1 -m "v0.11.3-rc.1" origin/release/0.11
git push origin v0.11.3-rc.1
# once its run is green:
git tag -a v0.11.3 -m "v0.11.3" origin/release/0.11
git push origin v0.11.3
```

The image workflow runs on any `v*` tag, wherever it was cut. Its preflight lets a version be
cut only from a commit on a release line: one that main holds, or that `release/MAJOR.MINOR`
holds for the tag's own `MAJOR.MINOR` (`scripts/ci/release_preflight.py`). So:

| The tag | Its commit is on | The run |
|---|---|---|
| `v0.12.0` | main | builds |
| `v0.11.3` or `v0.11.3-rc.1` | `release/0.11` only | builds |
| `v0.11.3` | neither main nor `release/0.11` | stops before anything builds |
| `v0.12.1` | `release/0.11` only | stops: `release/0.11` is not its line, and it is never asked |

A patch is then a release like any other: its images are scanned, signed and checked, tagged
`0.11.3`, and, for the patch itself (not its candidate), the `0.11` line moves to it. `0.12`
and `edge` do not move. The chart is published as `0.11.3`. Its release page compares it with
the release before it on the same branch (`v0.11.2`), not with the newest release on main,
and it is not marked as the repository's latest release while a higher version exists.

A tag's version must not be below the chart's `version` in `charts/automatos/Chart.yaml` at
the tagged commit, so the release branch keeps the chart version of its line.

**Another repository that imports a release** may require the release's commit to be on main.
Whether it accepts a patch cut on a release branch is an open question for that repository,
not something this workflow decides: check it before the first patch.

## The branch is protected as main is

A release branch is made only by a maintainer, takes changes by pull request only, with the
same checks as main, and cannot be force-pushed or deleted. That is two **repository
rulesets**, which a repository admin sets by hand once, for every release branch at once. CI
cannot check them: a ruleset is an admin setting of the repository, not a file in it, and the
workflows' tokens have no admin rights. Look at them in the repository's settings after any
change to main's rules, and keep the protection the same as main's.

Both are under **Settings → Rules → Rulesets → New ruleset → New branch ruleset**, each with
Enforcement status **Active** and Target branches **include by pattern: `release/*`**:

| Ruleset name | Bypass list | Rules |
|---|---|---|
| `Release branch creation` | the **Repository admin** and **Maintain** roles, Always allow | Restrict creations |
| `Release branches` | the same as main's ruleset | Restrict deletions; Require a pull request before merging, with the same approvals as main; Require status checks to pass, with the same required checks as main; Block force pushes |

They are two rulesets because a bypass covers every rule in its ruleset: maintainers bypass
the creation rule so they can make the branch, and must not bypass the pull request rule with
it.

`release/*` matches `release/0.11` and not a deeper name such as `release/0.11/x`, which a
release branch never has.

**Restrict creations** lets only those on its bypass list create a matching branch. It
matters here because the image workflow publishes a version for a tag whose commit is on
`release/MAJOR.MINOR`: with the rule set, anyone else's push of a new `release/…` branch is
refused, and the tag ruleset on [Releases](releases.md#who-can-create-a-version-tag) then
limits who can cut a version on it. Set all three before the first release branch is made.
