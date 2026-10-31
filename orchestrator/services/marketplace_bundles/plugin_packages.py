"""PRD-270 W2 P270W2-RVW-4 (§5.8, K12) — the public plugin package route's zips, each built once, a few at a time.

``GET /api/marketplace/catalog/plugins/{slug}/{version}/package.zip`` (``api/marketplace_catalog_items.py``)
is public on the hosted edition, and a zip (``plugin_files.built_package``) lists and reads up to
``PLUGIN_MAX_UPLOAD_SIZE_MB`` from storage and deflates it. So the route asks ``kept_package``:

* a built zip is kept (``BoundedCache``, ``PLUGIN_PACKAGES_KEPT``, the least recently served going first)
  under the whole :class:`plugin_files.StoredPlugin`: the slug, version and prefix, the export's revision of
  the plugin's bundle and the scan its row names, so a changed export, another version or another scan builds
  anew. A repeat request, and a matching ``If-None-Match``, read nothing from storage, and take no slot;
* a refusal (files that can't be served, logged) is kept for ``REFUSAL_KEPT_SECONDS``, so a broken version
  costs one build a minute however often it is asked for, and files put right in storage are served within
  that time;
* one request builds a version at a time (:class:`SoleBuilds`): another request for it waits for that build
  and serves its outcome, never building it too;
* this process runs at most ``PLUGIN_PACKAGE_BUILDS`` builds at once, as W1's catalogue files do
  (``catalog_files._rebuild_slot``; these slots are their own).

A request that waits longer than ``BUILD_SLOT_WAIT_SECONDS`` for either is :class:`PluginBuildsBusy` (the
route's 503 with ``Retry-After``), and storage is never read for it. A kept zip whose stored files are
rewritten without its row changing (``harvest_plugins --repair-s3``; a rescan names another scan) is served
until it is dropped or the process restarts. Blocking (S3): plain-``def`` routes only.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Hashable, Iterator, Optional, Set, Tuple

from services.marketplace_bundles.plugin_files import PluginPackage, StoredPlugin, built_package
from services.themes.bounded_cache import BoundedCache

BUILDS_BUSY = "Every plugin package build slot is taken."
# Package builds this process runs at once, how long a request waits for a slot or for another request's
# build of the same version, and the 503's Retry-After.
PLUGIN_PACKAGE_BUILDS = 2
BUILD_SLOT_WAIT_SECONDS = 1.0
BUILD_RETRY_SECONDS = 5
BUILD_SLOTS = threading.BoundedSemaphore(PLUGIN_PACKAGE_BUILDS)
# How many built zips stay in memory. A zip is a plugin's stored files deflated: at most about
# PLUGIN_MAX_UPLOAD_SIZE_MB, most a few tens of KB of markdown.
PLUGIN_PACKAGES_KEPT = 16
# How long a refusal is kept, and how many (each is a timestamp).
REFUSAL_KEPT_SECONDS = 60.0
REFUSALS_KEPT = 256


class PluginBuildsBusy(RuntimeError):
    """No build slot came free, or another request's build of the version outlasted, ``BUILD_SLOT_WAIT_SECONDS``
    (the route's 503 with Retry-After); storage was not read."""


class SoleBuilds:
    """The versions a request is building now: a second request for one waits for the first to finish."""

    def __init__(self) -> None:
        self._building: Set[Hashable] = set()
        self._changed = threading.Condition()

    @contextmanager
    def of(self, key: Hashable, wait_seconds: float) -> Iterator[None]:
        """Hold ``key`` while the block runs, once no other request holds it; PluginBuildsBusy when another
        still holds it after ``wait_seconds``."""
        with self._changed:
            if not self._changed.wait_for(lambda: key not in self._building, timeout=wait_seconds):
                raise PluginBuildsBusy(BUILDS_BUSY)
            self._building.add(key)
        try:
            yield
        finally:
            with self._changed:
                self._building.discard(key)
                self._changed.notify_all()


_packages: BoundedCache[PluginPackage] = BoundedCache(PLUGIN_PACKAGES_KEPT)
_refusals: BoundedCache[float] = BoundedCache(REFUSALS_KEPT)
_sole_builds = SoleBuilds()


def kept_package(plugin: StoredPlugin) -> Optional[PluginPackage]:
    """The zip ``built_package`` answers for ``plugin``, built once and kept; None when its files can't be
    served (kept ``REFUSAL_KEPT_SECONDS``). Raises PluginBuildsBusy when the wait for another request's build
    or for a slot runs out, PluginFilesUnavailable when storage fails (nothing kept)."""
    known, package = _kept(plugin)
    if known:
        return package
    with _sole_builds.of(plugin, BUILD_SLOT_WAIT_SECONDS):
        known, package = _kept(plugin)  # the outcome of the build this request waited for
        if known:
            return package
        with _build_slot():
            return _built_and_kept(plugin)


def clear_kept() -> None:
    """Drop every kept zip and refusal (tests; a process restart does the same)."""
    _packages.clear()
    _refusals.clear()


def _kept(plugin: StoredPlugin) -> Tuple[bool, Optional[PluginPackage]]:
    """(True, the zip) when one is kept, (True, None) for a refusal kept less than ``REFUSAL_KEPT_SECONDS``
    ago, (False, None) when neither."""
    package = _packages.get(plugin)
    if package is not None:
        return True, package
    refused_at = _refusals.get(plugin)
    return refused_at is not None and time.monotonic() - refused_at < REFUSAL_KEPT_SECONDS, None


def _built_and_kept(plugin: StoredPlugin) -> Optional[PluginPackage]:
    package = built_package(plugin)
    if package is None:
        _refusals.put(plugin, time.monotonic())
    else:
        _packages.put(plugin, package)
    return package


@contextmanager
def _build_slot() -> Iterator[None]:
    """One of this process's ``PLUGIN_PACKAGE_BUILDS`` slots, given back on the way out; PluginBuildsBusy when
    none comes free within ``BUILD_SLOT_WAIT_SECONDS``."""
    if not BUILD_SLOTS.acquire(timeout=BUILD_SLOT_WAIT_SECONDS):
        raise PluginBuildsBusy(BUILDS_BUSY)
    try:
        yield
    finally:
        BUILD_SLOTS.release()


__all__ = [
    "BUILD_RETRY_SECONDS", "PLUGIN_PACKAGE_BUILDS", "PluginBuildsBusy", "SoleBuilds", "clear_kept", "kept_package",
]
