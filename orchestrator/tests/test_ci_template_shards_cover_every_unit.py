"""6 Oct ("why is CI taking 45 mins"): the seeded templates' renders run as shards.

``scripts/ci/template_shards.py`` gives every render unit to exactly one shard, so
the shards together run every check the single step ran: none is skipped, none
runs twice. The driver's own loops take their units through it, and ``0/1`` (the
default) is every unit, as before.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shards = _load("template_shards")


def _walk(shard, families):
    """The driver's walk: several loops, one after another, through one shard."""
    return [unit for family in families for unit in shard.mine(family)]


FAMILIES = [
    [f"video-{i}" for i in range(4)],
    [f"image-{i}" for i in range(15)],
    [f"infographic-{i}" for i in range(3)],
    [f"kit-{i}" for i in range(78)],
]


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_every_unit_lands_in_exactly_one_shard(count):
    taken = [_walk(shards.Shard(index, count), FAMILIES) for index in range(count)]
    every = [unit for family in FAMILIES for unit in family]
    assert sorted(unit for share in taken for unit in share) == sorted(every)
    assert max(map(len, taken)) - min(map(len, taken)) <= 1  # an even split


def test_a_single_take_counts_in_the_same_walk():
    shard = shards.Shard(1, 2)
    assert [shard.take(), shard.take(), shard.take()] == [False, True, False]
    assert shard.label() == "shard 2 of 2: 1 of 3 render units"


@pytest.mark.parametrize("text", ["4/4", "-1/4", "1/0", "a/b", "3"])
def test_a_bad_shard_is_refused(text):
    with pytest.raises(ValueError):
        shards.parse(text)


def test_the_driver_takes_every_render_through_the_shard():
    driver = (_ROOT / "scripts" / "ci" / "social_template_previews.py").read_text()
    kits = (_ROOT / "scripts" / "ci" / "social_template_kits.py").read_text()
    assert "for starter in SHARD.mine(videos):" in driver
    assert "for starter in SHARD.mine(images):" in driver
    assert "in SHARD.mine(INFOGRAPHIC_RENDERS):" in driver
    assert "if not SHARD.take():\n        return []" in driver  # the heading-font card
    assert "in driver.SHARD.mine(night_bundles(driver, name, kit)):" in kits
    assert '"--shard", default="0/1"' in driver  # by default, every unit
