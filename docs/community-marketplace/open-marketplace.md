# The open Marketplace

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [frontend/components/marketplace/public/marketplace-page.tsx](../../frontend/components/marketplace/public/marketplace-page.tsx)
- [frontend/components/marketplace/public/visitor-marketplace.tsx](../../frontend/components/marketplace/public/visitor-marketplace.tsx)
- [frontend/components/marketplace/public/visitor-themes.tsx](../../frontend/components/marketplace/public/visitor-themes.tsx)
- [frontend/components/marketplace/public/visitor-add.tsx](../../frontend/components/marketplace/public/visitor-add.tsx)
- [frontend/components/marketplace/public/copy.ts](../../frontend/components/marketplace/public/copy.ts)
- [frontend/components/auth/resume-after-sign-in.tsx](../../frontend/components/auth/resume-after-sign-in.tsx)
- [frontend/lib/marketplace/return-after-sign-in.ts](../../frontend/lib/marketplace/return-after-sign-in.ts)
- [frontend/lib/marketplace/share-card.ts](../../frontend/lib/marketplace/share-card.ts)
- [frontend/components/marketplace/themes/add-link.tsx](../../frontend/components/marketplace/themes/add-link.tsx)
- [frontend/components/marketplace/cloud/sync-button.tsx](../../frontend/components/marketplace/cloud/sync-button.tsx)
- [frontend/components/marketplace/cloud/copy.ts](../../frontend/components/marketplace/cloud/copy.ts)
- [frontend/components/settings/automatos-cloud-panel.tsx](../../frontend/components/settings/automatos-cloud-panel.tsx)
- [frontend/components/marketplace/themes/listing-format.ts](../../frontend/components/marketplace/themes/listing-format.ts)
- [orchestrator/services/marketplace_cloud/pull.py](../../orchestrator/services/marketplace_cloud/pull.py)
- [orchestrator/services/marketplace_cloud/package.py](../../orchestrator/services/marketplace_cloud/package.py)
- [orchestrator/services/marketplace_cloud/updates.py](../../orchestrator/services/marketplace_cloud/updates.py)

</details>

The Marketplace is open in two ways:

- **On the hosted edition**, anyone can browse `/marketplace` without an account, and sign in only to add something.
- **On a local install**, an owner or admin can press **Sync with Automatos Cloud** to bring the official themes
  Automatos Cloud lists into the install's own Marketplace.

This page says what you can do and where. For how it works (the routes, the checks, the settings and the limits), see
[Sync with Automatos Cloud](sync-with-automatos-cloud.md), the technical page.

---

## On the hosted edition: browse signed out

Open `/marketplace` on the hosted app. Signed out, you get the Marketplace on its own, with a slim header: the
Automatos logo, **Sign in** and **Start free**. It reads: "Themes, agents, Playbooks, packages and tools for
Automatos. Browse everything here; sign in to add any of it to your workspace."

**What you can see:**

- **Themes**: the same storefront members see, with each theme's web and phone screenshots, its looks and screens,
  and its **Details**. See [Marketplace › Themes](themes-tab.md).
- **Every other tab**, read-only: Packages, Applications, Agents, Playbooks, LLMs, and Capabilities (plugins and
  skills). Each item opens its details.
- **Search** across what is listed.

**What you don't see:** anything that belongs to a workspace. There is no installed list, no **In this workspace** or
update badge and no publish form. **Make your own theme** offers the Design Kit and says "Sign in to submit it for
review".

Signed in, `/marketplace` is the Marketplace you know, inside the app.

### Add something, signed out

1. Click **Add to my workspace** on a theme or an item. It opens sign-in.
2. Sign in, or create an account (**Start free**). A new account gets its first workspace as usual.
3. Once you are in, the app opens the Marketplace on that item, with its add dialog open and **Add to my workspace**
   ready.
4. Click it. **Nothing is installed until you do.** The add is the normal one: only workspace owners and admins add
   items, and you may be asked to confirm it's you.

The way back is kept in that browser tab for 30 minutes. If you sign in in another tab, or later than that, you land
where sign-in normally takes you: open the Marketplace again and add the item from there. **Sign in** and **Start
free** in the header remember the page you were on the same way.

### Share an item

Each item has its own link, `/marketplace?tab=<tab>&item=<item>`. Posted in a chat or on social media, it shows the
item's name, description and picture (a theme's cover, else the Automatos logo). Search engines may list
`/marketplace`, and nothing else of the app.

---

## On a local install: Sync with Automatos Cloud

A local install's Marketplace lists only what was approved on that server. **Sync with Automatos Cloud** adds the
official themes Automatos Cloud lists, such as Play and HUD, so you can add them like any other Marketplace theme.
Themes are the only type a Sync brings.

- **Who:** workspace owners and admins. The local edition's operator is the workspace owner. Other members don't see
  the controls.
- **When:** only when someone presses the button. Nothing syncs on a schedule.
- **What it sends:** one anonymous request for Automatos Cloud's public theme list. It sends nothing about you, your
  workspace or what you have installed. It never publishes anything.

### Where to press it

- **The Marketplace header:** **Sync with Automatos Cloud**, with a line beside it: "Last synced 2m ago · 3 themes",
  "Not synced yet", or the last error in plain words. While it runs it shows "Syncing with Automatos Cloud…". Then a
  message says what it brought: "2 new, 1 updated from Automatos Cloud", or "Nothing new from Automatos Cloud".
- **Settings › Automatos Cloud:** the address it syncs from, the last success, the last check, the last error, the
  last Sync's counts ("2 new, 1 updated, 5 unchanged, 0 refused"), how many themes are held here from Automatos Cloud,
  **Sync now**, and how to turn it off.

If the button and the Settings tab are missing, the Sync is off on this install: see [Turning it off](#turning-it-off).

### What you get

The themes a Sync brings show on **Marketplace › Themes** with the badge **From Automatos Cloud**, and an official
theme's card shows **Official**. Every one was reviewed and signed on Automatos Cloud, and your install checks each
signature with its own copy of Automatos's keys before it lists the theme. Their screenshots load in your browser
straight from Automatos Cloud.

A Sync never changes your own uploads or the themes your own server lists. A theme your own server lists is left out
of the Sync.

### Add a theme from Automatos Cloud

Click **Add to my workspace**, as for any Marketplace theme ([Adding a theme to the workspace](themes-tab.md#adding-a-theme-to-the-workspace)).
The first time, your install downloads that version's files from Automatos Cloud and checks them against the
signature before it installs anything. Members don't opt in to it, and it can be the workspace default.

If the download or the check fails, nothing is installed, and you see why, for example "… couldn't be fetched from
Automatos Cloud, so nothing was installed." Try again in a moment.

### Updates, and holding a version

After each Sync that brings a newer version of a theme you have installed:

- an install that is **not pinned** moves to the new version by itself, keeping its looks, its members' choices and
  its place as the workspace default;
- a **pinned** install stays on its version and shows **Update available** until an owner or admin applies it.

Between Syncs nothing moves. If a new version can't be downloaded, its installs stay as they were, the message says
how many themes could not be updated ("Their installs stay as they were; the next Sync tries again."), and the next
Sync tries again. Pins and **Roll back** work as for any theme: see
[Managing themes](../themes/managing-themes.md#updates).

### Turning it off

The Sync is on by default. The server's operator turns it off by setting `AUTOMATOS_CLOUD_URL=` (empty) in
`envs/api.local` and restarting the backend. Then:

- the button and Settings › Automatos Cloud disappear;
- the themes a Sync already listed stay listed, and installed themes keep running from their stored files;
- adding a Cloud theme whose files aren't stored here is refused: "… comes from Automatos Cloud, and this install
  doesn't sync with it now (AUTOMATOS_CLOUD_URL), so its files can't be fetched."

See [Settings](sync-with-automatos-cloud.md#settings) on the technical page.

---

Related:

- [Marketplace › Themes](themes-tab.md)
- [Themes](../themes/_index.md)
- [Sync with Automatos Cloud](sync-with-automatos-cloud.md) (technical)
