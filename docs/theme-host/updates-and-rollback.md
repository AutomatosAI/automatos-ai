# Updates and rollback: how an install moves between versions

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/core/models/themes.py](../../orchestrator/core/models/themes.py)
- [orchestrator/services/themes/updates.py](../../orchestrator/services/themes/updates.py)
- [orchestrator/services/themes/auto_update.py](../../orchestrator/services/themes/auto_update.py)
- [orchestrator/services/themes/listing_updates.py](../../orchestrator/services/themes/listing_updates.py)
- [orchestrator/services/themes/listings.py](../../orchestrator/services/themes/listings.py)
- [orchestrator/services/themes/review.py](../../orchestrator/services/themes/review.py)
- [orchestrator/services/themes/publish.py](../../orchestrator/services/themes/publish.py)
- [orchestrator/services/themes/settings.py](../../orchestrator/services/themes/settings.py)
- [orchestrator/services/marketplace_cloud/updates.py](../../orchestrator/services/marketplace_cloud/updates.py)
- [orchestrator/services/marketplace_cloud/package.py](../../orchestrator/services/marketplace_cloud/package.py)
- [orchestrator/api/themes_install.py](../../orchestrator/api/themes_install.py)
- [orchestrator/api/marketplace_cloud_sync.py](../../orchestrator/api/marketplace_cloud_sync.py)
- [frontend/components/settings/themes/theme-labels.ts](../../frontend/components/settings/themes/theme-labels.ts)
- [frontend/components/settings/themes/theme-pins.tsx](../../frontend/components/settings/themes/theme-pins.tsx)
- [frontend/components/admin/themes/use-theme-review.ts](../../frontend/components/admin/themes/use-theme-review.ts)
- [frontend/components/admin/themes/deactivate-form.tsx](../../frontend/components/admin/themes/deactivate-form.tsx)
- [orchestrator/tests/test_prd261_theme_auto_update.py](../../orchestrator/tests/test_prd261_theme_auto_update.py)
- [orchestrator/tests/test_prd270w1_cloud_auto_update.py](../../orchestrator/tests/test_prd270w1_cloud_auto_update.py)
- [orchestrator/tests/test_prd270w1_cloud_deactivation.py](../../orchestrator/tests/test_prd270w1_cloud_deactivation.py)

</details>

An install runs one package version of a theme. A person moves it (**Apply update**, **Roll back**, a pin), and the
platform moves Marketplace installs on its own: forward when a newer version is listed, back when the version it runs
is taken down. This page is how those moves work and fail. For what a workspace admin sees, see
[Managing themes](../themes/managing-themes.md); for the routes, the [API reference](api-reference.md).

## The install row

`theme_installations` holds one active row per workspace and theme (`uq_theme_installations_active`, over rows with
no `uninstalled_at`) and at most one active default per workspace (`uq_theme_installations_default`).

| Column | What it holds |
|---|---|
| `id` | The install's id. It never changes on a move: members' choices, the workspace default and Automatos Go's phone theme are keyed by it |
| `theme_id` | The theme the install runs (`publisher/slug`) |
| `package_id` | The package version it runs now |
| `previous_package_id` | Where **Roll back** goes (`NULL` before any move) |
| `pending_package_id` | The update on offer, shown as **Update available** |
| `pinned_version` | A version the install is held on, or `NULL` |
| `is_default` | The workspace default |
| `uninstalled_at` | Set by a (soft) uninstall |

Looks are keyed by theme, not package (`theme_looks`), and opt-ins by package. So every move keeps looks, and a move
to a private package asks members to opt in again.

## The moves

All four are in `services/themes/updates.py`. Each target must pass the tenancy check (a Marketplace package, or this
workspace's own; 404 otherwise) and the same switch rules (`_refuse_unusable`, `_check_switch`): no rejected,
withdrawn or unlisted Marketplace version, no other version for a pinned install ("This install is pinned to version
…. Unpin it in the theme settings first."), no private code for the workspace default, and nothing a `disable`
revocation covers.

| Move | Called by | What it does to the row |
|---|---|---|
| `apply_update(package)` | **Apply update** (`POST /api/themes/installs/{id}/update`), and the platform's auto-update | Target must be newer. `previous_package_id` ← the old package; `package_id` ← the target; a pending offer of the target is cleared |
| `rollback()` | **Roll back** (`POST /api/themes/installs/{id}/rollback`) | Swaps `package_id` and `previous_package_id`. 409 "There is no earlier version to go back to." without one |
| `move_back(package)` | The platform only, when a version is taken down | Target must be older. `package_id` ← the target. The version it leaves never becomes the rollback target; `previous_package_id` is cleared only when it is the target. The waiting offer stays |
| `uninstall()` | **Uninstall** (`DELETE /api/themes/installs/{id}`) | Sets `uninstalled_at`, clears `is_default`, and drops members' choices of it. Looks stay |

A pin is set through the workspace theme settings (`PUT /api/themes/settings` with `pins: [{installId, version}]`,
`services/themes/settings.py`). It may name only a version the install holds: its current, pending or previous one
(`pinnable_versions`); `null` unpins.

## Offers

`pending_package_id` is set in three places:

- **An upload** of a higher version of a theme the workspace uploaded before (`records._offer_update`). A workspace's
  own upload never moves by itself.
- **A listing install** of an older version than the newest this host can run: the newest is set as its offer
  (`listings.install_listing`).
- **An approval**: `listing_updates.offer_listed_update(package)` points every active listing install of an older
  version of the theme at the new one, pinned installs included. An install that already waits on a version at least
  as new keeps that one.

`is_listed` decides what counts as listed: approved, reviewed, run by this host (`engines`) and covered by no
`disable`. A version this host can't run is never offered, so an install stays on the newest one it can run.

## Auto-update: forward on approval

`review.approve` runs, in the approval's transaction, `offer_listed_update` and then
`auto_update.apply_listed_update(package)`. That moves every active install that runs a Marketplace version of the
theme, has no `pinned_version` and was just offered the package, through `apply_update`. The approval's answer and
its `approved` event (`theme_review_events.detail`) carry `updatesOffered` and `updatesApplied`. The admin queue says
"Version approved. Unpinned installs of an older version move to it. Its signature request is open below."

Three callers reach it:

| Caller | Edition | Where |
|---|---|---|
| An admin's **Approve** in `/admin/themes` | both | `POST /api/admin/themes/{packageId}/approve` |
| The Design Kit's publish job | hosted | `POST /api/marketplace/themes/publish` approves, then signs, in one transaction ([The publish endpoint](publish-endpoint.md)) |
| A Sync with Automatos Cloud | local | `POST /api/marketplace/cloud-sync`, after the pull commits (below) |

Only installs with `owner_type` `marketplace` packages move. A workspace's own upload of the theme is not a listing
install and is never offered a Marketplace version.

## Rollback: back on deactivation

`review.deactivate` sets the version `deactivated`, writes a `disable` revocation of its digest, withdraws the offers
of it (`listing_updates.withdraw_offers`: each install offered it is offered the newest listed version newer than the
one it runs, or nothing) and then runs `auto_update.roll_back_installs(package)`:

1. It finds every active, unpinned listing install that runs the version.
2. It picks the target once (`previous_listed`): the newest listed version below it that this host runs and whose
   files this install holds. A local-origin row always counts as held. A Cloud row (`source` `cloud`) counts only
   once its `theme.json` is stored and hashes to its file list, because a Sync stores no file
   (`_first_held`). After one storage failure the later Cloud rows are skipped unread.
3. Each install moves there through `move_back`.

With no such version, nothing moves (logged at INFO), and the `disable` sends those members to the theme's base on
their next load. A pinned install stays on the version and is disabled the same way. The answer and the `deactivated`
event carry `installsRolledBack`. The admin queue says "Version deactivated. Unpinned installs on it go back to the
previous listed version, where there is one; the rest fall back to their base on the next load."

## After a Sync with Automatos Cloud (local edition)

A Sync writes Cloud rows as approved, reviewed Marketplace versions, but never calls `review.approve`. Instead
`marketplace_cloud/updates.py` `apply_cloud_updates` runs after every successful Sync:

- For each theme with a Cloud row, an active listing install and no local-origin Marketplace row, it takes the newest
  listed version this host runs (`listings.newest_compatible`). When that is a Cloud row and an install runs an older
  version, it goes on.
- If an install is unpinned, `marketplace_cloud/package.py` `ensure_stored` first fetches the version's package from
  `<AUTOMATOS_CLOUD_URL>/api/marketplace/catalog/files/<digest>/package.autotheme`, checks its digest, file list and
  signature, and stores the files. No transaction is open while Automatos Cloud answers. A failed download changes no
  install and no offer; the theme counts as failed and the next Sync tries again.
- Then `offer_listed_update` (every older install, pinned ones too) and, only when the files were stored,
  `apply_listed_update`. No event row is written, and no signature request is opened.
- Each theme is committed on its own. One pass runs at a time per process; a second caller waits at most
  `package.wait_seconds`, then moves nothing and logs it.

`POST /api/marketplace/cloud-sync` answers `updatesApplied` (themes moved) and `updatesFailed` (themes whose newest
version could not be applied). The pass is state-based: whatever an earlier Sync left undone, the next one applies.

A person's **Apply update** to a Cloud version goes through the same download first
(`api/themes_install.py` `update_theme_install`, `ensure_stored_for_update`).

## Failure modes

Every platform move runs in its own savepoint (`auto_update._moved`). A refusal or a database error leaves that
install as it was, never stops the others, and never fails the approval, the deactivation or the Sync. They are logged:

| Log line (WARNING unless noted) | Meaning |
|---|---|
| `[Themes] update of install <id> (workspace <id>) to <theme> <version> (package <id>) refused with <status>: <message>` | `apply_update` refused this install, for example a workspace default that would move to private code. It keeps the offer |
| `[Themes] roll back of install <id> … refused with <status>: <message>` | `move_back` refused it. It stays on the deactivated version, disabled |
| `[Themes] <action> of install <id> (workspace <id>) to package <id> failed in the database; it stays as it was` (ERROR, with the traceback) | The savepoint rolled back |
| `[Themes] roll-back skips <theme> <version> (package <id>, from Automatos Cloud): theme.json is not stored here or does not match its file list` | A Cloud version below was never fetched here, so it can't be a target |
| `[Themes] roll-back skips … : storage could not be read` | Storage failed; the later Cloud rows are skipped unread |
| `[Themes] <theme> <version> was taken down with no listed version below it to move to: <n> install(s) stay on it, disabled` (INFO) | Nothing to move back to |

To see why an install didn't move, read its row (`GET /api/themes/installed`: `pinnedVersion`, `pendingPackageId`,
`previousPackageId`) and search the backend log for its install id.

## Tests

`test_prd261_theme_auto_update.py` holds approval and deactivation to these rules (unpinned moves, pinned keeps the
offer, a refused move is logged and leaves the approval standing, no previous version leaves installs disabled).
`test_prd270w1_cloud_auto_update.py` and `test_prd270w1_cloud_deactivation.py` hold the Sync side (download first, no
move on a failed download, a never-fetched Cloud row is skipped as a rollback target).

Related:

- [The publish endpoint](publish-endpoint.md)
- [Operations](operations.md#revoke-a-theme-and-lift-a-revocation)
- [Sync with Automatos Cloud](../community-marketplace/sync-with-automatos-cloud.md)
