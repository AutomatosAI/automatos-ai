"""Shared by the AI-mark suites (PRD-264 EUAI-06a): read a file's mark back with the image's
exiftool, and hold its XMP packet to the allowlist.

The allowlist is the whole packet: the XMP wrapper, its ``rdf:Description`` blocks, the one
``Iptc4xmpExt:DigitalSourceType`` and the one ``xmpMM:InstanceID`` (FR-16: a random UUID,
not personal data). Any other element or attribute fails, so no user, workspace, prompt
or file path can ride along in a mark (Guidelines ¶94).
"""

from __future__ import annotations

import subprocess
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional

from media_render import provenance

X = "adobe:ns:meta/"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
IPTC_EXT = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
XMP_MM = "http://ns.adobe.com/xap/1.0/mm/"
SOURCE_TYPE = f"{{{IPTC_EXT}}}DigitalSourceType"
INSTANCE_ID = f"{{{XMP_MM}}}InstanceID"
MARK_FIELDS = (SOURCE_TYPE, INSTANCE_ID)
ALLOWED_ELEMENTS = frozenset({f"{{{X}}}xmpmeta", f"{{{RDF}}}RDF", f"{{{RDF}}}Description", *MARK_FIELDS})
# x:xmptk names the toolkit (exiftool and its version); rdf:about is empty. The mark and
# the instance id may be written as attributes of their description, too.
ALLOWED_ATTRIBUTES = frozenset({f"{{{X}}}xmptk", f"{{{RDF}}}about", *MARK_FIELDS})
TOOLKIT_PREFIX = "Image::ExifTool"
TIMEOUT_SECONDS = 60


def source_type_of(settings, path: Path) -> Optional[str]:
    """The file's DigitalSourceType as the image's exiftool reads it: the writer reads it back."""
    return provenance.read_source_type(path, exiftool_bin=settings.exiftool_bin, timeout_seconds=TIMEOUT_SECONDS)


def packet_of(settings, path: Path) -> bytes:
    return provenance.read_packet(path, exiftool_bin=settings.exiftool_bin, timeout_seconds=TIMEOUT_SECONDS)


def _values(root: ET.Element, field: str) -> List[str]:
    """Every value of ``field`` in the packet, written as an element or as an attribute."""
    found: List[str] = []
    for element in root.iter():
        if element.tag == field:
            found.append((element.text or "").strip())
        if field in element.attrib:
            found.append(element.attrib[field].strip())
    return found


def instance_id_of(packet: bytes) -> uuid.UUID:
    """The packet's one ``xmpMM:InstanceID`` (FR-16), which must be ``xmp.iid:`` and a UUID."""
    (value,) = _values(ET.fromstring(packet), INSTANCE_ID)
    assert value.startswith(provenance.INSTANCE_PREFIX), f"the InstanceID is not an xmp.iid: {value!r}"
    return uuid.UUID(value[len(provenance.INSTANCE_PREFIX):])


def assert_only_the_mark(packet: bytes, expected: str, *, absent: Iterable[str] = ()) -> None:
    """The packet holds the mark ``expected``, one random instance id, and nothing outside
    the allowlist, and none of the ``absent`` strings (the job's workspace, reference, text
    and paths)."""
    assert packet.strip(), "the file carries no XMP packet"
    root = ET.fromstring(packet)
    for element in root.iter():
        assert element.tag in ALLOWED_ELEMENTS, f"an XMP field outside the allowlist: {element.tag}"
        for name, value in element.attrib.items():
            assert name in ALLOWED_ATTRIBUTES, f"an XMP attribute outside the allowlist: {name}={value!r}"
            if name == f"{{{RDF}}}about":
                assert value == "", f"rdf:about names something: {value!r}"
            if name == f"{{{X}}}xmptk":
                assert value.startswith(TOOLKIT_PREFIX), f"x:xmptk is not exiftool's: {value!r}"
    marks = _values(root, SOURCE_TYPE)
    assert marks == [expected], marks
    assert instance_id_of(packet).version == 4, "the InstanceID is a random UUID"
    text = packet.decode("utf-8", "replace")
    for secret in absent:
        assert secret not in text, f"the packet carries {secret!r}"


def ffmpeg(settings, *argv: str) -> None:
    """Make a fixture file with the image's own ffmpeg."""
    subprocess.run([settings.ffmpeg_bin, "-v", "error", "-y", *argv], check=True, capture_output=True, timeout=120)


def seconds_of(settings, path: Path) -> float:
    """How long a media file plays, by the image's ffprobe: a marked file must still decode."""
    out = subprocess.run(
        [settings.ffprobe_bin, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout
    return float(out.strip())
