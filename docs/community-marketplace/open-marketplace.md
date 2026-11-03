# The open Marketplace

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [frontend/components/marketplace/public/marketplace-page.tsx](../../frontend/components/marketplace/public/marketplace-page.tsx)
- [frontend/components/marketplace/marketplace-homepage.tsx](../../frontend/components/marketplace/marketplace-homepage.tsx)
- [frontend/components/marketplace/public/visitor-header.tsx](../../frontend/components/marketplace/public/visitor-header.tsx)
- [frontend/components/marketplace/public/visitor-themes.tsx](../../frontend/components/marketplace/public/visitor-themes.tsx)
- [frontend/components/marketplace/public/visitor-add.tsx](../../frontend/components/marketplace/public/visitor-add.tsx)
- [frontend/components/marketplace/public/copy.ts](../../frontend/components/marketplace/public/copy.ts)
- [frontend/components/auth/resume-after-sign-in.tsx](../../frontend/components/auth/resume-after-sign-in.tsx)
- [frontend/lib/marketplace/return-after-sign-in.ts](../../frontend/lib/marketplace/return-after-sign-in.ts)
- [frontend/lib/marketplace/share-card.ts](../../frontend/lib/marketplace/share-card.ts)
- [frontend/components/marketplace/themes/add-link.tsx](../../frontend/components/marketplace/themes/add-link.tsx)
- [frontend/components/marketplace/cloud/sync-button.tsx](../../frontend/components/marketplace/cloud/sync-button.tsx)
- [frontend/components/marketplace/cloud/copy.ts](../../frontend/components/marketplace/cloud/copy.ts)
- [frontend/components/marketplace/cloud/item-copy.ts](../../frontend/components/marketplace/cloud/item-copy.ts)
- [frontend/components/marketplace/cloud/cloud-item-badges.tsx](../../frontend/components/marketplace/cloud/cloud-item-badges.tsx)
- [frontend/components/marketplace/cloud/disclosure-dialog.tsx](../../frontend/components/marketplace/cloud/disclosure-dialog.tsx)
- [frontend/components/marketplace/cloud/update-notices.tsx](../../frontend/components/marketplace/cloud/update-notices.tsx)
- [frontend/components/settings/automatos-cloud-panel.tsx](../../frontend/components/settings/automatos-cloud-panel.tsx)
- [frontend/components/marketplace/themes/listing-format.ts](../../frontend/components/marketplace/themes/listing-format.ts)
- [orchestrator/services/marketplace_cloud/pull.py](../../orchestrator/services/marketplace_cloud/pull.py)
- [orchestrator/services/marketplace_cloud/package.py](../../orchestrator/services/marketplace_cloud/package.py)
- [orchestrator/services/marketplace_cloud/updates.py](../../orchestrator/services/marketplace_cloud/updates.py)
- [orchestrator/services/marketplace_bundles/sync_items.py](../../orchestrator/services/marketplace_bundles/sync_items.py)
- [orchestrator/services/marketplace_bundles/plugin_import.py](../../orchestrator/services/marketplace_bundles/plugin_import.py)

</details>

The Marketplace is open in two ways:

- **On the hosted edition**, anyone can browse `/marketplace` without an account, and sign in only to add something.
- **On a local install**, an owner or admin can press **Sync with Automatos Cloud** to bring what Automatos Cloud
  lists into the install's own Marketplace: its official themes, and the agents, Playbooks, packages and plugins its
  review approved.

This page says what you can do and where. For how it works (the routes, the checks, the settings and the limits), see
[Sync with Automatos Cloud](sync-with-automatos-cloud.md), the technical page.

---

## On the hosted edition: browse signed out

Open `/marketplace` on the hosted app. Signed out, you get the same Marketplace page members see, under a slim
header: the Automatos logo and **Studio** (both take you back to the Studio site), **Sign in** and **Join the
waitlist**. It reads: "Themes, agents, Playbooks, packages and tools for Automatos. Browse all of it here. To add any
of it to a workspace, sign in. Hosted beta opens November 2026 · Join the waitlist."

Until the hosted beta opens, new accounts are by waitlist: **Join the waitlist** goes to the waitlist on the Studio
site, and nothing on the page offers sign-up.

**What you can see:**

- **The page's top:** how many items are listed and how many times they were installed, the **Featured** items, and
  a **Popular** strip (the most installed agents and Playbooks) where members see "Recommended for You".
- **Themes**: the same storefront members see, with each theme's web and phone screenshots, its looks and screens,
  and its **Details**. See [Marketplace › Themes](themes-tab.md).
- **Every other tab**, read-only: Packages, Applications, Agents, Playbooks, LLMs, and Capabilities (plugins and
  skills). Each item opens its details.
- **Search** across what is listed.

**What you don't see:** anything that belongs to a workspace. There is no **In this workspace** mark or update badge,
no publish form, and no admin approve or reject. **Make your own theme** offers the Design Kit and says "Sign in to
submit it for review".

Signed in, `/marketplace` is the same page inside the app, with your workspace's marks and "Recommended for You".

### Add something, signed out

1. Click **Add to my workspace** on a theme, or the add button on any other item (**Add to Workspace**, **Install**,
   **Enable for Workspace**). It opens sign-in.
2. Sign in. Until the hosted beta opens, there is no sign-up here: join the waitlist instead.
3. Once you are in, the app opens the Marketplace on that item, with its add dialog ready.
4. Click it. **Nothing is installed until you do.** The add is the normal one: only workspace owners and admins add
   items, and you may be asked to confirm it's you.

The way back is kept in that browser tab for 30 minutes. If you sign in in another tab, or later than that, you land
where sign-in normally takes you: open the Marketplace again and add the item from there. **Sign in** in the header
remembers the page you were on the same way. **Join the waitlist** leaves for the Studio site and remembers nothing.

### Share an item

Each item has its own link, `/marketplace?tab=<tab>&item=<item>`. Posted in a chat or on social media, it shows the
item's name, description and picture (a theme's cover, else the Automatos logo). Search engines may list
`/marketplace`, and nothing else of the app.

---

## On a local install: Sync with Automatos Cloud

A local install's Marketplace lists only what was approved on that server and what it ships with. **Sync with
Automatos Cloud** adds what Automatos Cloud lists: its official themes, such as Play and HUD, and the agents,
Playbooks, packages and plugins Automatos Cloud's review approved. You add them like any other Marketplace item.

- **Who:** workspace owners and admins. The local edition's operator is the workspace owner. Other members don't see
  the controls.
- **When:** only when someone presses the button. Nothing syncs on a schedule.
- **What it sends:** one anonymous request for Automatos Cloud's public catalogue, then more of the same kind for
  each new or changed item it brings. It sends nothing about you, your workspace or what you have installed. It never publishes
  anything.

### Where to press it

- **The Marketplace header:** **Sync with Automatos Cloud**, with a line beside it: "Last synced 2m ago · 3 themes ·
  2 agents · 1 Playbook" (a type you hold none of is left out), "Not synced yet", or the last error in plain words.
  While it runs it shows "Syncing with Automatos Cloud…". Then a message says what it brought, themes and items
  together: "2 new, 1 updated from Automatos Cloud", or "Nothing new from Automatos Cloud".
- **Settings › Automatos Cloud:** the address it syncs from, the last success, the last check, the last error, the
  last Sync's counts ("2 new, 1 updated, 5 unchanged, 0 refused"), the themes and the items (agents, Playbooks,
  packages, plugins) held here from Automatos Cloud, **Sync now**, and how to turn it off.

If the button and the Settings tab are missing, the Sync is off on this install: see [Turning it off](#turning-it-off).

### What you get

**Themes** show on **Marketplace › Themes** with the badge **From Automatos Cloud**, and an official theme's card
also shows **Official**. Every one was reviewed and signed on Automatos Cloud, and your install checks each signature
with its own copy of Automatos's keys before it lists the theme. Their screenshots load in your browser straight from
Automatos Cloud.

**Agents, Playbooks, packages and plugins** show in the tab of their type (a plugin under Capabilities) with two
badges: **From Automatos Cloud** and **Reviewed on Automatos Cloud**. A person on Automatos Cloud approved each one,
and a plugin also passed Automatos Cloud's security scan. They are never **Official**: only themes are signed by
Automatos. A plugin arrives approved, after your install's own scan of its files; each workspace still chooses to
enable it. One Sync downloads at most five new or changed plugins by default; the rest come with the next Sync.

A Sync never changes your own uploads, what your own server made or ships with, or a copy a workspace already
added. A theme or item that clashes with one your own server lists (the same theme id, agent name or slug) is left
out of the Sync.

### Add a theme from Automatos Cloud

Click **Add to my workspace**, as for any Marketplace theme ([Adding a theme to the workspace](themes-tab.md#adding-a-theme-to-the-workspace)).
The first time, your install downloads that version's files from Automatos Cloud and checks them against the
signature before it installs anything. Members don't opt in to it, and it can be the workspace default.

If the download or the check fails, nothing is installed, and you see why, for example "… couldn't be fetched from
Automatos Cloud, so nothing was installed." Try again in a moment.

### Add an agent, Playbook, package or plugin from Automatos Cloud

Click **Add to my workspace** (for a plugin, enable it). First you see what it will use, read from your install,
with nothing asked of Automatos Cloud:

- **An agent:** its prompt as plain text, its model and whether your install has it, the apps it uses and which
  **needs a connection**, and its skills and plugins, each **on this install** or **not on this install**.
- **A Playbook:** each step's helper and prompt (a step whose helper isn't here **needs a helper**), and its apps.
- **A package:** its members and whether each is here, and the connections its setup asks for.
- **A plugin:** Automatos Cloud's scan report, your install's own check, what it contains, and "Updates apply to
  every workspace that enabled it".

**Add to my workspace** adds it, with the usual permissions and warnings. **Close** adds nothing ("Nothing was added:
you closed the review."). When you ask Auto to add a Cloud agent, package or plugin in chat, Auto shows the same
review on an approval card first, and adds nothing until you approve.

### Updates, and holding a version

**Themes.** After each Sync that brings a newer version of a theme you have installed:

- an install that is **not pinned** moves to the new version by itself, keeping its looks, its members' choices and
  its place as the workspace default;
- a **pinned** install stays on its version and shows **Update available** until an owner or admin applies it.

Between Syncs nothing moves. If a new version can't be downloaded, its installs stay as they were, the message says
how many themes could not be updated ("Their installs stay as they were; the next Sync tries again."), and the next
Sync tries again. Pins and **Roll back** work as for any theme: see
[Managing themes](../themes/managing-themes.md#updates).

**Agents and Playbooks.** Your workspace's copy never changes by itself. When a Sync brings a newer version, a
notice above the Marketplace's tabs says "… has a newer version on Automatos Cloud. Your copy stays as it is."
**Review the new version** shows what it will use, and **Add to my workspace** adds a new copy beside yours. Keep or
delete the old one.

**Plugins.** A plugin updates in place: every workspace that enabled it runs the new version from then on, and sees
"… was updated to … on Automatos Cloud; this workspace now runs it." **Review** shows it; **Dismiss** hides the
notice in this browser.

### Turning it off

The Sync is on by default. The server's operator turns it off by setting `AUTOMATOS_CLOUD_URL=` (empty) in
`envs/api.local` and restarting the backend. Then:

- the button and Settings › Automatos Cloud disappear;
- the themes and items a Sync already listed stay listed, installed themes keep running from their stored files, and
  synced agents, Playbooks, packages and plugins can still be added, after their review;
- adding a Cloud theme whose files aren't stored here is refused: "… comes from Automatos Cloud, and this install
  doesn't sync with it now (AUTOMATOS_CLOUD_URL), so its files can't be fetched."

See [Settings](sync-with-automatos-cloud.md#settings) on the technical page.

---

Related:

- [Marketplace › Themes](themes-tab.md)
- [Themes](../themes/_index.md)
- [Sync with Automatos Cloud](sync-with-automatos-cloud.md) (technical)
