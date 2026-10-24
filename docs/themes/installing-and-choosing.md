# Installing and choosing a theme

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [frontend/components/settings/themes/appearance-tab.tsx](../../frontend/components/settings/themes/appearance-tab.tsx)
- [frontend/components/settings/themes/install-theme.tsx](../../frontend/components/settings/themes/install-theme.tsx)
- [frontend/components/settings/themes/scan-report.tsx](../../frontend/components/settings/themes/scan-report.tsx)
- [frontend/components/settings/themes/installed-list.tsx](../../frontend/components/settings/themes/installed-list.tsx)
- [frontend/components/settings/themes/installed-row.tsx](../../frontend/components/settings/themes/installed-row.tsx)
- [frontend/components/settings/themes/opt-in-dialog.tsx](../../frontend/components/settings/themes/opt-in-dialog.tsx)
- [frontend/components/settings/themes/theme-labels.ts](../../frontend/components/settings/themes/theme-labels.ts)
- [frontend/components/ui/theme-toggle.tsx](../../frontend/components/ui/theme-toggle.tsx)
- [frontend/components/shared/global-search.tsx](../../frontend/components/shared/global-search.tsx)
- [frontend/components/theme-host/theme-host-provider.tsx](../../frontend/components/theme-host/theme-host-provider.tsx)
- [frontend/hooks/use-installed-themes.ts](../../frontend/hooks/use-installed-themes.ts)
- [frontend/hooks/use-global-search.ts](../../frontend/hooks/use-global-search.ts)
- [frontend/lib/theme-host/choice-sync.ts](../../frontend/lib/theme-host/choice-sync.ts)
- [frontend/lib/theme-host/sources.ts](../../frontend/lib/theme-host/sources.ts)
- [frontend/lib/theme-host/api.ts](../../frontend/lib/theme-host/api.ts)
- [orchestrator/api/themes_import.py](../../orchestrator/api/themes_import.py)
- [orchestrator/api/themes_install.py](../../orchestrator/api/themes_install.py)
- [orchestrator/services/themes/records.py](../../orchestrator/services/themes/records.py)
- [orchestrator/services/themes/settings.py](../../orchestrator/services/themes/settings.py)
- [orchestrator/services/themes/review_stage.py](../../orchestrator/services/themes/review_stage.py)
- [orchestrator/services/themes/scan.py](../../orchestrator/services/themes/scan.py)
- [orchestrator/services/themes/sdk_uses.py](../../orchestrator/services/themes/sdk_uses.py)
- [orchestrator/services/themes/drift.py](../../orchestrator/services/themes/drift.py)
- [orchestrator/services/themes/review_limits.py](../../orchestrator/services/themes/review_limits.py)
- [orchestrator/services/themes/signed_upload.py](../../orchestrator/services/themes/signed_upload.py)
- [orchestrator/services/themes/signature_state.py](../../orchestrator/services/themes/signature_state.py)
- [orchestrator/services/themes/optin.py](../../orchestrator/services/themes/optin.py)
- [orchestrator/services/themes/choices.py](../../orchestrator/services/themes/choices.py)
- [orchestrator/services/themes/defaults.py](../../orchestrator/services/themes/defaults.py)
- [orchestrator/services/themes/contract.py](../../orchestrator/services/themes/contract.py)

</details>

Owners and admins add themes to a workspace. Every member then chooses for themselves. This page covers both halves,
in the order you meet them.

---

## Who can add a theme

- **Owners and admins** of the workspace. Every other member sees the installed list and **Use this theme**, and the
  line "A workspace owner or admin installs, updates and removes themes."
- The workspace setting **Who may install themes** can narrow this to the owner alone. An admin then sees **Install
  theme** greyed, with "In this workspace only the workspace owner can install themes."
- **Uploading a file** also needs private theme uploads to be on. There are two switches: the server's own, and the
  workspace's **Private theme uploads**. While uploads are off, **Install theme** is greyed with "Private theme uploads
  are off in this workspace." The server gives its reason in one of these sentences:

| The server says | Why |
|---|---|
| Private theme uploads are turned off on this platform. | The server's switch is off. No workspace can turn uploads on. |
| On this platform only an Automatos admin can upload a private theme. | The server's switch leaves uploads to the platform's own administrators. The tab then notes "On this platform only its administrators upload private themes." |
| Private theme uploads are off for this workspace. A workspace admin can turn them on in Settings, Appearance, Themes. | The workspace switched them off. |
| Private theme uploads are off by default in a workspace with more than one member. A workspace admin can turn them on in Settings, Appearance, Themes. | The hosted edition's default. The local edition has them on. |

Adding a theme from the Marketplace needs no upload switch: see
[Add a theme from the Marketplace](#add-a-theme-from-the-marketplace).

## Install a theme from a file

A theme comes as one `.autotheme` file. A signed release also carries a small `.sig.json` signature file next to it.

1. Open **Settings › Appearance › Themes**.
2. If you have the theme's signature file, click **Add its signature file** first and pick the `.sig.json`. The page
   says: "Optional: the .sig.json released with the theme. Automatos checks it before anything is installed."
3. Click **Install theme** and pick the `.autotheme` file. While the server checks the package and runs its automated
   checks, the button reads "Checking the package…".
4. On the hosted edition you may be asked to confirm it's you.

The result shows the theme's name and version, its label (**Private · not reviewed**, or **Signed by Automatos**) and
its [scan report](#the-scan-report), under one of these lines:

| Line | What it means |
|---|---|
| Installed in this workspace. | The theme is new here. It is now in the list below, where each member opts in and chooses it. |
| Stored as an update to the installed version: review it in the list below. | A higher version of a theme you already have. Its row shows **Update available**. Nothing switches until an owner or admin applies it. |
| Stored. It is not installed. | The version is kept, but it is neither the version the workspace runs nor its waiting update (for example, a newer update already waits). |
| … These exact files were uploaded before. | The same file again. You get the earlier upload back, and nothing new is stored. If the theme had been uninstalled, it is installed again. |

### When an upload is refused

Nothing is stored when an upload is refused. The common sentences:

| The server says | What to do |
|---|---|
| This theme can't be installed. (then a list of problems, each with its code) | The package failed a check: the file itself, a size limit, `theme.json`, the style sheet or the looks. Fix the theme and pack it again with the [Design Kit](https://github.com/AutomatosAI/automatos-design-kit). |
| This workspace already has … with different files. Raise the version in theme.json and upload it again. | Two different files can't share one version number. |
| This workspace has … installed. Upload a higher version to offer it as an update. | An upload must be newer than the installed version. To go back a version, use **Roll back** ([Managing themes](managing-themes.md#roll-back)). |
| Only an Automatos admin can upload a theme under the "automatos" publisher. | `automatos/…` ids, such as Play's and HUD's, are reserved. Upload the theme with its signature file, or add it from **Marketplace › Themes**. |
| This workspace has uploaded 20 themes in the last 24 hours, the most allowed. Try again tomorrow. | 20 is the server's default; its administrator can change it. |
| Theme reviews are busy right now. Try again in a minute. | Too many automated checks are running on the server at once. |
| This theme is being uploaded already. Try again in a moment. | The same file is still going up, from another tab or a double click. |
| The theme's files couldn't be stored. Try again. | The server could not write the files. Nothing was kept. |

The size limits, by default (the server's administrator can change them):

| What | Limit |
|---|---|
| The whole package | 10 MB |
| Its code (JavaScript and CSS together) | 3,072 KB |
| Its entry module | 300 KB |
| Each mascot clip | 1,024 KB |

### The signature file

A signature file tells the server that Automatos vouches for exactly these files.

- It is at most 4 KB. The server checks it against the package's own files before it stores anything.
- When it is taken, the theme shows **Signed by Automatos** in place of **Private · not reviewed**. A signed upload
  counts as reviewed: members don't opt in to it, it can be the workspace default, and it may carry an `automatos/…`
  id.
- Uploading a package you already have, this time with its signature file, adds the signature to the copy you have.
- If the signature later stops checking out (for example, its key is revoked), that version is turned off with the
  reason "This theme's signature doesn't check out."

| The server says | Why |
|---|---|
| This signature is for another package: its … isn't the uploaded theme's. Upload the .sig.json made for this .autotheme. | The file was made for another theme, version or build. |
| No Automatos signing key verifies this signature for these files, so nothing was installed. | The signature doesn't match these files. |
| This signature was made by an Automatos signing key that is retired or revoked, so nothing was installed. | The key that signed it no longer counts. |
| Theme signatures can't be checked right now, so nothing was installed. Try again later. | The server could not load its signing keys. |

---

## The scan report

The server checks every upload automatically before it stores it. The report shows a **Verdict**, then one row per
check, each with a badge.

### The badges

| Badge | Meaning |
|---|---|
| **Passed** | The check ran and found nothing. |
| **Warning** | Something to look at. |
| **Error** | A real problem. A Marketplace submission with an error can't be approved. |
| **Not measured** | The server can't make this check for this package yet. It counts neither way. |

An upload stored before the automated checks existed shows older badges: **Passed**, **No style sheet** and **Not
run**. A check that did not run is never shown as passed: its row gives the server's reason.

### The checks

| Check | What it looks at |
|---|---|
| **Sizes** | The package, its code, its entry module and its largest mascot clip, against the limits. The row shows the package, code and entry sizes. A package over a limit is refused before any report. |
| **Style sheet scope** | That the theme's style sheet only styles the app under the theme's own class. "No style sheet" when the package has none. |
| **Looks** | That every preset fits the theme's look rules. The row lists the presets. |
| **Contrast (AA)** | For a theme with a phone block: its colour pairs, text on its ground at 4.5:1 and the primary and focus-ring colours at 3:1. An Error names the pair under its minimum. A theme with no phone block is **Not measured**: it doesn't say which of its colours is text on which ground, so the server has nothing to measure. |
| **SDK areas** | The parts of the theme SDK the code uses, against the areas `theme.json` declares. An Error means the code asks for a name this Automatos doesn't have, so the theme can never load. A Warning means an area is used but not declared, a declared area doesn't exist, or the imports couldn't be checked one by one. |
| **Cloned core pages** | Each page the theme copied from one of Automatos's own pages, against this release's version of that page. A Warning means the page changed since the copy was made, so the copy may miss fixes. |
| **Automated review** | Only on uploads stored before the automated checks existed: **Not run**. |

### The verdict

| Verdict | When |
|---|---|
| **Passed** | No check found a warning or an error. |
| **Needs a reviewer** | At least one warning. |
| **Blocked** | At least one error. |
| **Not run** | The upload was stored before the automated checks existed. |

The server also has an AI read the theme's code. Its score and findings are for the Automatos reviewer only, so the
report leaves that check out. When the verdict is worse than every check shown, one line under it says why. For a
verdict of **Needs a reviewer**:

> Every check shown here passed or was not measured. The verdict comes from the AI code read, which this report leaves
> out: it could not read all of the code (for example, this install has no Anthropic API key), or it asked for a person
> to look. That review is only for a Marketplace listing: a private upload installs here whatever the verdict.

For **Blocked**: "No check shown here found an error. The verdict comes from the AI code read, which this report leaves
out: it rated this code high risk. A private upload installs here whatever the verdict."

### Why it says "not reviewed", and installs anyway

The scan is automatic. A person at Automatos reviews only themes submitted to the Marketplace, never a private upload.
So an upload keeps the label **Private · not reviewed** whatever its verdict, and it installs whatever its verdict.
When a check explains a **Needs a reviewer** or **Blocked** verdict, the report adds: "A private upload is not reviewed
by Automatos: it installs here whatever the verdict. The checks above say what to look at before you use it."

The safeguard for private code is the opt-in: it runs for a member only once that member has said yes to that version.

---

## Add a theme from the Marketplace

**Marketplace › Themes** lists themes a person at Automatos reviewed. Open a card and click **Add to my workspace**
(owners and admins). It adds the newest version your Automatos can run. Members don't opt in to a Marketplace theme.
See [Marketplace › Themes](../community-marketplace/themes-tab.md).

---

## Opting in to a private theme

When you click **Use this theme** on a private version you haven't opted in to, a dialog asks first:

- Its title: "Run *name* *version*?"
- The server's sentence: "This theme was not reviewed. Its code runs with your access to all your workspaces."
- Then: "You opt in to this version only. A new version asks again. Other members decide for themselves."
- Two buttons: **Cancel** and **Opt in and use**.

**Opt in and use** records your opt-in and makes the theme your choice. On the hosted edition you may be asked to
confirm it's you first.

- The opt-in is yours, for one workspace and one version. An update to a new private version asks again.
- Marketplace themes and signed uploads need no opt-in.
- If the theme moved to another version while the dialog was open, the server refuses with "That version is no longer
  this theme's current one. Reload to see the version installed now." The dialog closes and the list refreshes: click
  **Use this theme** again to see the version installed now.
- A theme you chose but haven't opted in to doesn't run. Studio shows instead, with the notice "The theme could not
  start, so Studio is showing instead." and the reason "You have not opted in to its private code."

---

## Choosing a theme

| Where | How |
|---|---|
| **Settings › Appearance › Themes** | **Use this theme** on its row. The theme you are on shows **In use**. |
| The **Appearance** menu at the top of the app | Under **Style**: Classic, Studio, then the workspace's installed themes. |
| ⌘K (Ctrl+K) | The **Themes** group. With nothing typed, or "theme", "style" or "appearance", it lists every style. |

What happens when you choose:

- Your choice is kept on the server, for you, in this workspace. A new tab or another browser opens on it.
- Classic and Studio switch at once. An installed theme is saved first, then the page reloads into it.
- If the server refuses, the tab stays on the style it had (a Classic or Studio pick switches back) and you see the
  server's reason. From the menu or ⌘K it comes under "Your theme choice was not saved."
- The choice is per workspace: another workspace can show another theme.
- The Appearance menu and ⌘K note "Private · not reviewed" beside an uploaded theme. A withdrawn theme is not offered.
- The menu's **Tone** (Light, Dark, System) is yours to set, unless the theme sets it. Then the tone items rest, with
  "Set by your look" (the look decides) or "Set by your theme" (the theme always uses one tone).
- If the workspace has switched off **Members choose their own theme**, the server refuses a choice: "This
  workspace's admins choose its theme; the workspace default applies."

### Which theme you see

1. Your own choice in this workspace, while the workspace lets members choose.
2. Otherwise the workspace default.
3. Otherwise Studio.

A choice of a theme that was uninstalled falls through to the default. If your theme's version is withdrawn, you see
the theme's base (Studio or Classic) and a notice. See
[Managing themes](managing-themes.md#withdrawn-themes).

---

## The workspace default

An owner or admin can make one installed theme the **workspace default**: **Make default** on its row. The row then
shows **Workspace default**, and **Clear default** removes it. On the hosted edition, both may ask you to confirm it's
you.

Members see the default when they haven't chosen a theme, when the theme they chose was uninstalled, and always while
members may not choose.

**Only a theme that isn't private can be the default**: a Marketplace theme or a signed upload. A default runs for
members who never said yes to its code. On a private upload, **Make default** is greyed, with this beside it: "A theme
that was not reviewed can’t be the workspace default." The server holds the same rule: "An unreviewed theme can't be
the workspace default: members would run its code without saying yes."

A withdrawn version can't be the default either: "That theme version was withdrawn, so it can't be used."
