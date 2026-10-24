# Themes FAQ

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [frontend/components/settings/themes/theme-labels.ts](../../frontend/components/settings/themes/theme-labels.ts)
- [frontend/components/settings/themes/scan-report.tsx](../../frontend/components/settings/themes/scan-report.tsx)
- [frontend/components/settings/themes/installed-row.tsx](../../frontend/components/settings/themes/installed-row.tsx)
- [frontend/components/settings/themes/install-theme.tsx](../../frontend/components/settings/themes/install-theme.tsx)
- [frontend/components/settings/themes/opt-in-dialog.tsx](../../frontend/components/settings/themes/opt-in-dialog.tsx)
- [frontend/components/ui/theme-toggle.tsx](../../frontend/components/ui/theme-toggle.tsx)
- [frontend/components/theme-host/theme-host-provider.tsx](../../frontend/components/theme-host/theme-host-provider.tsx)
- [frontend/components/marketplace/themes/listing-format.ts](../../frontend/components/marketplace/themes/listing-format.ts)
- [frontend/lib/theme-host/choice-sync.ts](../../frontend/lib/theme-host/choice-sync.ts)
- [frontend/lib/theme-host/failure-ledger.ts](../../frontend/lib/theme-host/failure-ledger.ts)
- [frontend/lib/theme-host/installed-entry.ts](../../frontend/lib/theme-host/installed-entry.ts)
- [frontend/app/safe-mode/page.tsx](../../frontend/app/safe-mode/page.tsx)
- [orchestrator/services/themes/review_stage.py](../../orchestrator/services/themes/review_stage.py)
- [orchestrator/services/themes/records.py](../../orchestrator/services/themes/records.py)
- [orchestrator/services/themes/defaults.py](../../orchestrator/services/themes/defaults.py)
- [orchestrator/services/themes/choices.py](../../orchestrator/services/themes/choices.py)
- [orchestrator/services/themes/optin.py](../../orchestrator/services/themes/optin.py)
- [orchestrator/services/themes/looks.py](../../orchestrator/services/themes/looks.py)
- [orchestrator/services/themes/mobile.py](../../orchestrator/services/themes/mobile.py)
- [orchestrator/services/mobile/discovery.py](../../orchestrator/services/mobile/discovery.py)

</details>

The questions people ask first, with short answers and a link to the full story.

---

### The scan badges say "Passed". Is the theme paused?

No. The badge reads **Passed**: that check ran and found nothing. The scan never pauses or stops a theme. The other
badges are **Warning**, **Error** and **Not measured**. See [the scan report](installing-and-choosing.md#the-scan-report).

### Every check passed, so why does the verdict say "Needs a reviewer"?

The verdict also counts the AI read of the theme's code, which the report leaves out. When the AI read could not read
all of the code, for example on a server with no Anthropic API key, or when it asked for a person to look, the verdict
is **Needs a reviewer**, and a line under it says so. That review matters only for a Marketplace listing: a private
upload installs whatever the verdict.

### Why does my theme say "Private · not reviewed"?

Because nobody at Automatos reviewed it. The scan is automatic, and a person reviews only themes submitted to the
Marketplace. An uploaded theme keeps the label whatever its verdict, and each member opts in to it before it runs for
them.

A theme shows **Signed by Automatos** instead when you upload it with its signature file (`.sig.json`) and an active
Automatos key verifies it. A theme added from **Marketplace › Themes** never carries it: a person reviewed every version
listed there. See [Who vouches for a theme](_index.md#who-vouches-for-a-theme).

### Why is "Make default" greyed?

Only a theme that isn't private can be the workspace default, because a default runs for members who never said yes
to its code. The reason is written beside the button: "A theme that was not reviewed can’t be the workspace default."
Use the theme's Marketplace version, or upload it again with its signature file. See
[The workspace default](installing-and-choosing.md#the-workspace-default).

### Why is "Install theme" greyed?

The reason is written beside it. Either the workspace lets only its owner install themes, or private uploads are off
for the workspace or the whole server. See [Who can add a theme](installing-and-choosing.md#who-can-add-a-theme).

### I uploaded Play (or HUD) and got "Only an Automatos admin can upload a theme under the "automatos" publisher."

`automatos/…` ids are reserved. Upload the `.autotheme` again with its `.sig.json` from the same
[release](https://github.com/AutomatosAI/automatos-design-kit/releases), or add the theme from
**Marketplace › Themes** if your server lists it.

### Is there a "Skins" tab?

The Marketplace tab for themes is called **Themes**: **Marketplace › Themes**. See
[Marketplace › Themes](../community-marketplace/themes-tab.md).

### I opted in last week. Why does it ask again?

An opt-in covers one version. An update to a new private version asks every member again. Other members' opt-ins
are theirs: yours covers only you.

### My theme went back to Studio. Why?

| What you saw | Why | What to do |
|---|---|---|
| "This theme failed three times in ten minutes, so you are back on Studio." | The theme kept failing in this browser, so your choice was set to Studio. | Choose it again once it is fixed. |
| "The theme could not start, so Studio is showing instead." | It failed once; the line under it says why. | Reload. If it says "You have not opted in to its private code.", open **Settings › Appearance › Themes** and click **Use this theme** to opt in to the new version. |
| "This theme was withdrawn, so it is turned off for now." | Its version was withdrawn. You see the theme's base, Studio or Classic. | Wait for a new version, or choose another theme. |
| "Safe mode: themes are off in this tab." | This tab is in safe mode. | Close the tab and open a new one. |
| No notice | The theme was uninstalled (you now see the workspace default), or the workspace no longer lets members choose. | Ask an owner or admin. |
| Studio's layout in a narrow window, a phone browser included | The app draws a theme's frame and pages only in a window 1024 pixels wide or more. | Widen the window. |

See [When a theme fails](managing-themes.md#when-a-theme-fails) and [Safe mode](managing-themes.md#safe-mode).

### A theme broke the app. How do I get back in?

Open `/safe-mode`, or add `?safe_mode=themes` to the address. No theme code runs in that tab. Then choose Studio or
Classic in the Appearance menu to leave the theme for good.

### Where is "Customise look…"?

In the Appearance menu at the top of the app, but only while a theme is your style and its layout is showing (a window
1024 pixels wide or more). Classic and Studio have no looks. See [Looks](looks.md).

### Will an update, a roll back or an uninstall lose my looks?

No. Your looks are kept on the server per theme, not per version. If a new version changes its settings, your looks
are carried over to them each time they are read. See [After a theme update](looks.md#after-a-theme-update).

### Do other members see my theme or my looks?

No. Your choice and your looks are yours, in each workspace. Owners and admins can set a workspace default, which you
see until you choose a theme yourself, or always if the workspace doesn't let members choose.

### How do I get my phone to change?

1. Choose the theme on the web, in the workspace your phone uses. A paired phone uses the workspace it was paired in.
2. If it is a private theme, opt in to its version on the web first.
3. For a theme other than Play or HUD, the theme needs a phone block (its `mobile` block). Without one, nothing of it
   reaches the phone.
4. The server must offer phone themes: it lists `mobile_theme` in `/.well-known/automatos.json`. This release does.

See [On your phone](on-your-phone.md).
