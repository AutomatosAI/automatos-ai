# Arcade

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [Arcade's source in the design kit (`themes/arcade`)](https://github.com/AutomatosAI/automatos-design-kit/tree/main/themes/arcade)
- [orchestrator/api/themes_looks.py](../../orchestrator/api/themes_looks.py)
- [orchestrator/services/themes/looks.py](../../orchestrator/services/themes/looks.py)
- [orchestrator/api/themes_mobile.py](../../orchestrator/api/themes_mobile.py)
- [frontend/lib/theme-host/looks/provider.tsx](../../frontend/lib/theme-host/looks/provider.tsx)

</details>

Arcade is one of Automatos's own themes: Automatos as a 16-bit role-playing game.
- Your agents are your party.
- Auto is the guide in a dialogue box, drawn as a small pixel robot.
- Your Deliverables fill an Inventory.
- Your Missions are quests on a map.

Every number and name it shows is your workspace's own data. Arcade is `automatos/arcade` in the Marketplace.

To install it, see [Installing and choosing a theme](installing-and-choosing.md#get-play-or-hud). Once it's active, it
draws every page you see. Pages it doesn't redraw, such as the Command Centre, Agents and Settings, are the usual pages
inside Arcade's frame, in its colours.

---

## What you see

**The status bar** runs across the top on a wide screen. It holds:
- Auto's mark and the ARCADE wordmark;
- three counters from today's activity: **XP** (tasks done today), **!** (what needs you) and **QUEST** (work in
  progress);
- a pause button, search, and the usual help, notifications and account controls.

**The pause menu** on the left lists Arcade's pages, every other page, and settings, with a ▶ cursor beside the one
you're on. The arrow keys move through it.

**Chat** has three windows:
- **The story:** your messages (YOU), Auto's (AUTO), a ★ line for each thing a turn did, and an ITEM RECEIVED window
  when Auto makes a Deliverable. Approvals appear under "! A CHOICE AWAITS".
- **The dialogue box:** Auto's portrait and what he's doing (thinking, working, searching…). His newest reply types
  out at your text speed, and the message box and voice button sit under it.
- **The party:** your agents, each with a word for its state, such as NEEDS YOU or WORKING.

**Quests** shows your Missions as markers on an overworld map, in three regions: Greenwood holds the active ones, Storm
Peaks the ones that need you, Hearth Village the finished ones. Choose a marker to open its quest window: the
Mission's steps as a checklist, its progress, and a link to the Mission's own page.

**Inventory** shows your Deliverables as item slots, newest first, with tabs for All, Reports, Documents and Images,
and Posts if your workspace has Socials. Choose a slot to see what it is, and open it to read it.

**Options** is Arcade's look page (below). **The Sprite gallery** shows Auto's twelve states, moving and still.

On a screen narrower than 1024 px, Arcade becomes a handheld console:
- the page sits in a screen;
- a D-pad with four named buttons (Quests, Chat, Inventory, Menu) moves you around, and Menu holds everything else;
- the voice button sits in a round surround.

---

## Auto

Auto is drawn as a 32-pixel robot with twelve states: hello, idle, listening, thinking, working, searching, reading,
celebrate, oops, sleeping, lost and loading. His state follows what he's doing in the chat.

**PAUSE** in the status bar stops every moving thing: Auto, the ▶ cursor and the typing. Press it again to start
them. If your device asks for reduced motion, nothing moves and replies show whole.

---

## Looks

Arcade's look has five settings:

| Setting | Choices |
|---|---|
| Palette | Overworld, Dungeon, Neon Cabinet, Night City, Handheld |
| Scanlines | Off, Soft, Strong |
| Pixel scale | 2×, 3× |
| Window | Classic double, Thin |
| Text speed | Instant, Normal, Slow |

Each palette has a ready-made look to start from. In Options you can change the settings, see a live preview, save a
look, reset to the palette as it ships, and export or import a look as a file. Your looks are kept for
you in this workspace. [Looks](looks.md) has how looks work in every theme.

---

## On your phone

Automatos Go has an Arcade style of its own, built into the app:
- **Choosing it:** pick it in **Settings › Style and look**. If the phone is set to **Same as Studio**, installing
  Arcade in Studio switches the phone to it.
- **Looks:** the same palettes and settings as the web. They stay in step with the web while Arcade is installed in
  the workspace; if it isn't, they stay on the phone, and Options says so.
- **The home is a handheld:**
  - a status bar;
  - your party;
  - the story and the dialogue box, with the message box and hold to talk;
  - a D-pad: **▲ Quests** (the map, with your Board's tickets as quests), **▶ Inventory** (your Deliverables),
    **◀ Chat**, and **▼ Menu**.
- **Elsewhere:** the Command Centre, sheets and settings appear in Arcade's square windows and pixel type. Auto is
  the same pixel robot, with PAUSE.
- **Older app versions:** a version of Automatos Go from before the Arcade style paints Arcade's colours on the
  app's usual layout.

[On your phone](on-your-phone.md) covers the phone and themes in general.

---

## The brief, ideas and approvals on the phone

Where your workspace offers them, these three appear in Arcade in Automatos Go:
- **The day's brief:** when it's the newest message, the dialogue box says it, with **OPEN BRIEF** to see the whole
  brief.
- **Auto's ideas:** an **IDEAS** count and a **NEXT** line under the status bar, with Start and Hide.
- **Approvals:** a line under the dialogue box when something waits for your answer.

A status word is coloured by what it means:

| Status | Colour |
|---|---|
| needs you | yellow |
| working | blue |
| done | plain text |
| failed | red |

The word always says it too.

---

## What Arcade doesn't show

Arcade draws only what your workspace has. It has no health, mana or level, because Automatos keeps no such numbers.
XP is just today's count of finished tasks, and is never stored.
