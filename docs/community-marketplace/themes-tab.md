# Marketplace › Themes

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [frontend/components/marketplace/marketplace-homepage.tsx](../../frontend/components/marketplace/marketplace-homepage.tsx)
- [frontend/components/marketplace/themes/tab.tsx](../../frontend/components/marketplace/themes/tab.tsx)
- [frontend/components/marketplace/themes/listing-format.ts](../../frontend/components/marketplace/themes/listing-format.ts)
- [frontend/components/marketplace/themes/listing-grid.tsx](../../frontend/components/marketplace/themes/listing-grid.tsx)
- [frontend/components/marketplace/themes/listing-card.tsx](../../frontend/components/marketplace/themes/listing-card.tsx)
- [frontend/components/marketplace/themes/detail-dialog.tsx](../../frontend/components/marketplace/themes/detail-dialog.tsx)
- [frontend/components/marketplace/themes/detail-sections.tsx](../../frontend/components/marketplace/themes/detail-sections.tsx)
- [frontend/components/marketplace/themes/install-listing.tsx](../../frontend/components/marketplace/themes/install-listing.tsx)
- [frontend/components/marketplace/themes/install-use.tsx](../../frontend/components/marketplace/themes/install-use.tsx)
- [frontend/components/marketplace/themes/listing-hooks.ts](../../frontend/components/marketplace/themes/listing-hooks.ts)
- [frontend/components/marketplace/themes/publish-theme.tsx](../../frontend/components/marketplace/themes/publish-theme.tsx)
- [frontend/components/marketplace/themes/publish-result.tsx](../../frontend/components/marketplace/themes/publish-result.tsx)
- [frontend/components/settings/themes/theme-labels.ts](../../frontend/components/settings/themes/theme-labels.ts)
- [orchestrator/api/marketplace_themes.py](../../orchestrator/api/marketplace_themes.py)
- [orchestrator/api/marketplace_themes_submit.py](../../orchestrator/api/marketplace_themes_submit.py)
- [orchestrator/services/themes/listings.py](../../orchestrator/services/themes/listings.py)
- [orchestrator/services/themes/listing_updates.py](../../orchestrator/services/themes/listing_updates.py)
- [orchestrator/services/themes/submissions.py](../../orchestrator/services/themes/submissions.py)
- [orchestrator/services/themes/review_limits.py](../../orchestrator/services/themes/review_limits.py)
- [orchestrator/services/themes/optin.py](../../orchestrator/services/themes/optin.py)
- [orchestrator/services/themes/data/signing-keys.json](../../orchestrator/services/themes/data/signing-keys.json)
- [frontend/lib/theme-host/manifest.ts](../../frontend/lib/theme-host/manifest.ts)

</details>

The Marketplace's **Themes** tab lists the themes you can add to a workspace, and is where you submit your own. The tab
is called Themes. Its lede reads: "Themes change how Automatos looks for you. A person at Automatos reviewed every
version listed here."

For what a theme is, and how members choose one, see [Themes](../themes/_index.md).

---

## What is listed

- Only versions a person approved after review. A version waiting for review, rejected or taken off is never listed.
- A version that was withdrawn and turned off is not listed. One withdrawn with a warning is listed, with the reason.
- A workspace's own uploads are never listed.

## The cards

Each card shows the theme's first preview as its cover, its name and version, its publisher, its description, up to
three of the SDK areas it uses, and these badges:

| Badge | Meaning |
|---|---|
| **Official** | The publisher is the reserved `automatos` publisher **and** the version shown carries a signature an active Automatos key verifies. A reserved name alone is never Official. No card shows it yet: this release's signing keys file holds no Automatos key. |
| **In this workspace** | The workspace has this theme installed. |
| **Update available** | The workspace runs an older Marketplace version of it. |
| **Needs a newer Automatos** | No version of this theme runs on this Automatos yet. |

The Marketplace's search box filters the cards: a card stays when its name, id, description, publisher or SDK areas
hold every word you type. With nothing listed you see "No themes are listed in the Marketplace yet."; with no match,
"No theme matches your search."

## A theme's page

Opening a card shows:

- its name, version, description and badges, and the reason of a warning;
- **Add to my workspace**, or **Use this theme** once it is in the workspace;
- its previews, grouped by preset, with the theme's default look marked "(default look)";
- the facts: **Publisher**, **Theme id**, **Signature** ("Signed by Automatos", "Signed, but the signature doesn't
  check out" or "Not signed"), **Pages** it replaces, **Its own pages**, **Built on** (Studio or Classic; this release
  runs only Studio-based themes, so a Classic-based one installs but shows Studio), **Light or dark**, **Minimum
  width** and **Uses**;
- **Versions**: every listed version, the one marked **Newest for this Automatos**, any marked **Needs a newer
  Automatos**, and any withdrawal reason;
- **Read me** and **Changelog**, when the theme ships them.

## Adding a theme to the workspace

Owners and admins add themes, as the workspace's **Who may install themes** allows. Every other member sees "A
workspace owner or admin adds themes to this workspace." In a workspace where only the owner installs, an admin sees
"In this workspace only the workspace owner can install themes." in place of the button.

1. Open the theme and click **Add to my workspace**. It adds the newest version your Automatos can run. The button is
   greyed when there is none: "No version of this theme runs on this Automatos yet."
2. On the hosted edition you may be asked to confirm it's you.
3. You see "Added to this workspace. Each member chooses it for themselves." (or "This version is already in this
   workspace.").

Then each member clicks **Use this theme**, here or in **Settings › Appearance › Themes**. A Marketplace theme needs no
opt-in, and it can be the workspace default.

- When the workspace already runs the theme, the page shows **In this workspace** with its version. When a newer
  version is listed: "A newer version is listed: review it under "Installed in this workspace"."
- When the workspace runs its own upload of the theme: "This workspace runs its own upload of this theme. Uninstall it
  below to add the Marketplace version."
- A version approved later is offered to the workspace as **Update available**. Nothing switches until an owner or admin
  applies it. See [Managing themes](../themes/managing-themes.md#updates).

Below the listings, the tab repeats **Install theme** (for a private upload) and the workspace's installed list, under
**Installed in this workspace**, with the same actions as **Settings › Appearance › Themes**.

---

## Publish theme

Any member can submit a theme for review: "Submit your theme package for review. Once a person at Automatos approves
it, every workspace can install it from the Marketplace."

1. Choose the `.autotheme` file and click **Submit for review**. On the hosted edition you may be asked to confirm it's
   you.
2. The server checks the package as it checks an upload, then runs its automated checks before it answers ("Checking
   and reviewing the package…").
3. The submission shows **Pending review** with the automated verdict: "Passed the automated checks", "Needs a closer
   look by the reviewer" or "Blocked by the automated checks". A blocked version can't be approved: "Fix what they
   found and submit a new version."

The server's rules for a submission:

- The package must include its source (`source/`), so the reviewer can rebuild what runs.
- The `automatos` publisher is reserved for the platform's administrators. Any other publisher name belongs to the
  first person who submits under it.
- One open submission per person at a time, and 10 submissions per person in 24 hours (the server's defaults).
- A version must be higher than the theme's highest submitted version.

To build and pack a theme, use the [Automatos Design Kit](https://github.com/AutomatosAI/automatos-design-kit) and
its [guides](https://github.com/AutomatosAI/automatos-design-kit/tree/main/guides).

## When the theme Marketplace is off

A server can turn the theme Marketplace off. The tab then says: "The theme Marketplace is turned off on this platform,
so nothing is listed or published here. Private uploads and the themes in this workspace are below."
