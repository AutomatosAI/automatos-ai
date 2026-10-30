# Managing themes

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [frontend/components/settings/themes/installed-row.tsx](../../frontend/components/settings/themes/installed-row.tsx)
- [frontend/components/settings/themes/update-compare.tsx](../../frontend/components/settings/themes/update-compare.tsx)
- [frontend/components/settings/themes/theme-settings.tsx](../../frontend/components/settings/themes/theme-settings.tsx)
- [frontend/components/settings/themes/theme-pins.tsx](../../frontend/components/settings/themes/theme-pins.tsx)
- [frontend/components/settings/themes/theme-labels.ts](../../frontend/components/settings/themes/theme-labels.ts)
- [frontend/components/settings/themes/scan-report.tsx](../../frontend/components/settings/themes/scan-report.tsx)
- [frontend/components/settings/themes/install-theme.tsx](../../frontend/components/settings/themes/install-theme.tsx)
- [frontend/lib/theme-host/manifest.ts](../../frontend/lib/theme-host/manifest.ts)
- [frontend/components/theme-host/theme-host-provider.tsx](../../frontend/components/theme-host/theme-host-provider.tsx)
- [frontend/components/theme-host/page-boundary.tsx](../../frontend/components/theme-host/page-boundary.tsx)
- [frontend/lib/theme-host/failure-ledger.ts](../../frontend/lib/theme-host/failure-ledger.ts)
- [frontend/lib/theme-host/safe-mode.ts](../../frontend/lib/theme-host/safe-mode.ts)
- [frontend/lib/theme-host/constants.ts](../../frontend/lib/theme-host/constants.ts)
- [frontend/lib/theme-host/choice-sync.ts](../../frontend/lib/theme-host/choice-sync.ts)
- [frontend/app/safe-mode/page.tsx](../../frontend/app/safe-mode/page.tsx)
- [frontend/hooks/use-installed-themes.ts](../../frontend/hooks/use-installed-themes.ts)
- [orchestrator/api/themes_install.py](../../orchestrator/api/themes_install.py)
- [orchestrator/api/themes_settings.py](../../orchestrator/api/themes_settings.py)
- [orchestrator/services/themes/updates.py](../../orchestrator/services/themes/updates.py)
- [orchestrator/services/themes/listing_updates.py](../../orchestrator/services/themes/listing_updates.py)
- [orchestrator/services/themes/auto_update.py](../../orchestrator/services/themes/auto_update.py)
- [orchestrator/services/themes/listings.py](../../orchestrator/services/themes/listings.py)
- [orchestrator/services/themes/records.py](../../orchestrator/services/themes/records.py)
- [orchestrator/services/themes/settings.py](../../orchestrator/services/themes/settings.py)
- [orchestrator/services/themes/defaults.py](../../orchestrator/services/themes/defaults.py)
- [orchestrator/services/themes/choices.py](../../orchestrator/services/themes/choices.py)
- [orchestrator/services/themes/drift.py](../../orchestrator/services/themes/drift.py)
- [orchestrator/services/themes/review_stage.py](../../orchestrator/services/themes/review_stage.py)
- [orchestrator/services/themes/revocations.py](../../orchestrator/services/themes/revocations.py)
- [orchestrator/services/themes/signature_state.py](../../orchestrator/services/themes/signature_state.py)
- [orchestrator/services/themes/revocation_pull.py](../../orchestrator/services/themes/revocation_pull.py)
- [orchestrator/config.py](../../orchestrator/config.py)

</details>

This page is for workspace owners and admins. Everything here is in **Settings › Appearance › Themes**: the installed
list, with each theme's actions, and the workspace's theme settings under it. The Marketplace's **Themes** tab shows the
same installed list under **Installed in this workspace**.

Who can do what:

| Action | Who | Confirm it's you (hosted edition) |
|---|---|---|
| Install, update, roll back, pin | Owners and admins, as **Who may install themes** allows | Yes |
| Make or clear the default | Owners and admins | Yes |
| Change the theme settings | Owners and admins | Yes |
| Uninstall | Owners and admins | No |

---

## Updates

**A theme you added from the Marketplace updates itself.** When a newer version of it is approved, the install moves
to that version, keeping its looks, its members' choices and its place as the default. Its row shows **Updates itself;
pin to hold**. To keep the version you have, [pin it](#pin-a-version). If a version you moved to is taken off the
Marketplace, the install goes back to the newest listed version below it (see [Withdrawn themes](#withdrawn-themes)).

Everything else waits as an update until an owner or admin applies it:

- **an upload** of a higher version of a theme you uploaded before (an upload never moves by itself);
- **a Marketplace version** for a pinned install, for an install added at an older version than the newest, or for one
  the server could not move.

The theme's row then shows **Update available** and the new version. To apply it:

1. Click **Review update**. The dialog **Update** *name* says: "Members opt in again to a new private version before
   it runs for them."
2. It compares the two versions:
   - **Version** *old* → *new*, with "not reviewed" when nobody at Automatos reviewed the new version. Every uploaded
     version shows it, a signed one included.
   - The files **Added**, **Changed** and **Removed** (or "No file changed.").
   - **Scan verdict:** *old* → *new*, in the server's own words (such as `pass` or `review_required`; "not run" when a
     version has none), then "changed:" and the parts of the report that differ (such as "checks, verdict"). It
     doesn't say which checks changed.
3. Click **Apply update**. You see "Theme updated."

After an update:

- the old version becomes the one **Roll back** returns to;
- every member keeps their looks, which are kept per theme, not per version;
- if the new version is private, each member opts in to it again. Until then they see Studio, with "You have not opted
  in to its private code."

The server refuses an update when:

| The server says | Why |
|---|---|
| This install is pinned to version …. Unpin it in the theme settings first. | The install is [pinned](#pin-a-version). |
| An unreviewed theme can't be the workspace default: members would run its code without saying yes. | The theme is the workspace default and the new version is private. Clear the default first. |
| That theme version was withdrawn, so it can't be used. | The new version is [withdrawn](#withdrawn-themes). |
| That version is not newer than the one installed. Use rollback to go back to the previous version. | Updates only go forward. |
| That Marketplace version isn't approved yet. / That version was rejected or withdrawn. | The Marketplace version is not on offer. |

If the theme moved on while the dialog was open, the dialog shows the server's sentence and closes, and the list
refreshes. Open **Review update** again to compare what is there now.

When a version is taken off the Marketplace, its offer is replaced by the newest listed version newer than yours, or
it goes. An install that runs the version taken off goes back to the newest listed version below it, unless it is
pinned.

## Pin a version

**Pinned versions**, under the settings, lists each install with **Not pinned** and the versions it can be pinned to:
its current one, its waiting update and its previous one. "Marketplace themes update themselves to each new version.
A pinned theme stays on that version until it is unpinned."

- A pinned install shows **Pinned to** *version* on its row, in place of **Updates itself; pin to hold**.
- A new Marketplace version waits as **Update available**; the server never moves a pinned install by itself.
- It refuses any update or roll back to another version.
- Pinning follows **Who may install themes**.

## Roll back

**Roll back** shows once an install has a previous version. It swaps the current and the previous version, so rolling
back twice brings you back. You see "Theme rolled back."

- Looks stay as they were saved.
- Opt-ins are per version, so members who opted in to the previous version before don't need to again.
- The same refusals as an update apply (a pin, the default, a withdrawn version), and "There is no earlier version to
  go back to." when there is none.

## Uninstall

Click **Uninstall**, then **Confirm uninstall**. You see "Theme uninstalled."

- Members who chose the theme now see the workspace default, or Studio. Their choice of it is dropped.
- It stops being the workspace default.
- Everyone's looks for it stay on the server. Install the same theme again and they come back.
- A workspace has one install per theme. To swap your own upload of a theme for its Marketplace version, uninstall the
  upload first: the Marketplace says "This workspace runs its own upload of this theme. Uninstall it below to add the
  Marketplace version."

## Needs update

Some themes copy one of Automatos's own pages and change it (a cloned page). When a new release of Automatos changes
that page, the copy is behind. The theme's row then shows **Needs update**. Hover over the badge for its hint: "A core
page this theme copies changed in this release. It may look out of date until its author updates it."

The theme still runs. The fix is a new version from the theme's author.

The badge is checked against the Automatos release you run now, each time the list is read. The scan report is made
once, at upload, and you see it only in the result shown right after the upload. So its **Cloned core pages** check
warns only about a copy that was already behind then, for example "… copies …, which this release has changed
since." or "…, which this release no longer ships." After you move to a newer Automatos release, the badge is the only
sign.

## Withdrawn themes

A theme version, or a range of versions, can be withdrawn. Two levels:

| Level | What members see |
|---|---|
| **Warning** | The theme still runs, with the notice "This theme is still running, but its maintainer withdrew this version." and the reason. The reason also shows on its row and on its Marketplace card. |
| **Turned off** | The theme doesn't run. Members on it see its base (Studio or Classic) and the notice "This theme was withdrawn, so it is turned off for now." with the reason. Its **Use this theme** is greyed, the Appearance menu and ⌘K leave it out, it can't be the default, no update or roll back can move to it, and the Marketplace stops listing that version. |

When a Marketplace version is taken off the Marketplace (deactivated), an install of it that is not pinned goes back to
the newest listed version below it that your Automatos runs, and keeps working. A pinned install, or any install when no
earlier version is listed, stays on it, turned off as above.

- A version whose stored signature no longer checks out is turned off the same way, with the reason "This theme's
  signature doesn't check out." It comes back once the version is signed again.
- The server's administrators withdraw versions. A local edition also takes the versions Automatos withdraws: it reads
  Automatos's public list every hour by default. Its operator can change or turn off that pull: see
  [The revocation feed and the local pull](../theme-host/operations.md#the-revocation-feed-and-the-local-pull).

## When a theme fails

If a theme can't start, Studio shows in its place, with the notice "The theme could not start, so Studio is showing
instead." and one of these reasons:

| Reason | Counts toward the three failures |
|---|---|
| It failed while drawing the page. | Yes |
| Its code failed to load. | Yes |
| It failed while starting. | Yes |
| Its styles failed to load. | Yes |
| It was built for a different theme SDK. | Yes |
| Its theme.json could not be read. | No |
| Your session was refused. Sign in again to load it. | No |
| This version of it was withdrawn. | No |
| You have not opted in to its private code. | No |
| Its files could not be reached just now. Reload to try again. | No |

A theme built on Classic (its `base`) always fails with "Its theme.json could not be read.": this release runs only
Studio-based themes.

**Three failures in ten minutes, in one browser, send you back to Studio.** Your choice is set to Studio on the server,
and the notice says "This theme failed three times in ten minutes, so you are back on Studio." and "You can choose it
again in Settings once it is fixed." The browser keeps the failure times in its storage; without storage each failure
counts alone, so the theme only falls back to Studio for that load.

A page that fails inside a theme is the page's failure, not the theme's: you see "This page could not be shown" and
"Something on this page failed. The rest of the app still works.", with a retry. The theme keeps running, and it
doesn't count.

## Safe mode

Safe mode is the way back in when a theme stops the app from working.

- Open `/safe-mode`, or add `?safe_mode=themes` to any address.
- No theme code runs in that tab, and Studio shows. While your style is an installed theme, the notice says "Safe
  mode: themes are off in this tab."
- It lasts until you close the tab. Other tabs are not changed.
- The `/safe-mode` page draws none of the app around it and calls nothing on the server. Its **Back to the app**
  button opens chat, still in safe mode.
- To leave a theme for good, choose Studio or Classic in the Appearance menu (the sun, moon or screen button at the
  top of the app).

---

## The workspace's theme settings

Owners and admins see these under the installed list. Each change saves at once ("Theme settings saved.").

| Setting | Choices | Default | What it does |
|---|---|---|---|
| **Private theme uploads** | On, off | Local edition: on. Hosted edition: on in a workspace of one person, off otherwise. | Lets owners and admins upload theme code nobody reviewed. The hint says what applies now and whether it comes from the default or from this workspace. |
| **Members choose their own theme** | On, off | On | Off: everyone sees the workspace default, and nobody can save a choice. |
| **Who may install themes** | Owners and admins, the owner only | Owners and admins | Who may install (from a file or the Marketplace), update, roll back and pin. |
| **Pinned versions** | Not pinned, or a version the install holds | Not pinned | See [Pin a version](#pin-a-version). |

- **The upload default follows the workspace's size.** While a hosted workspace has no setting of its own, it counts
  its people (its active members plus its owner): with a second person, uploads turn off. Set the switch to keep a
  choice.
- **Only the owner changes who may install.** Changing it needs both the old and the new permission, so an admin can
  neither make installs owner-only nor undo the owner's choice: "Only the workspace owner can change who may install
  themes here."
- **The platform can stop uploads.** When the server's switch is off, the switch is greyed: "Switched off for the whole
  platform; a workspace can’t turn it on."

### For the server's administrator

Two settings in the server's environment sit above every workspace:

| Setting | Values | Default | Effect |
|---|---|---|---|
| `THEMES_PRIVATE_UPLOAD` | `auto`, `on`, `off`, `admins` | `auto` | `auto`: each workspace decides, with the defaults above. `on`: each workspace decides, on by default. `off`: no private uploads anywhere. `admins`: only the platform's administrators upload. Any other value reads as `off`. |
| `THEMES_MARKETPLACE` | `on`, `off` | `on` | `off` hides the Marketplace's theme listings and its **Publish theme**, and nothing can be added from it. Private uploads and the installed list stay. Any other value reads as `off`. |
