"""Machine-readable AI marks on generated documents and images (PRD-264 EUAI-09, Art. 50(2)).

Article 50(2) of the AI Act asks that generated content be marked in a machine-readable
format. The mark here is the IPTC Digital Source Type, the term detectors, platforms and
the C2PA tools read, written where each format keeps its metadata:

* **PDF:** an XMP packet in the document catalogue's ``/Metadata`` stream (pypdf).
* **DOCX and XLSX:** the custom document properties ``AIGenerated`` (the IPTC term) and
  ``AIInstanceID`` (``docProps/custom.xml``), written into the package with the standard
  library's zip and lxml (python-docx's own XML library), so every other part stays
  byte for byte.
* **PNG:** an XMP packet in an ``iTXt`` chunk keyed ``XML:com.adobe.xmp``, after ``IHDR``.
* **JPEG:** an XMP packet in an ``APP1`` segment, after the leading ``APP0``/``APP1``.

PNG and JPEG are marked chunk by chunk, never re-encoded, so the pixels are untouched.
An image that already carries a C2PA manifest (a PNG ``caBX`` chunk, a JPEG ``APP11``
JUMBF box labelled ``c2pa``) is returned as it came: the manifest states its source and
is kept whole. An XMP packet that already states a digital source type keeps it (an
instance id is added when it has none). Other image types (WebP, GIF) are returned
unmarked: no pinned library writes their XMP without re-encoding the picture.

Every mark also carries a random per-file instance id (FR-16): ``xmpMM:InstanceID``
(``xmp.iid:<uuid4>``) in the XMP of a PDF, PNG or JPEG, and the custom property
``AIInstanceID`` in a DOCX or XLSX. Two files made from identical input never share a
SHA-256, so a predictable document (a templated letter) cannot be confirmed by hashing
guesses against the public provenance route. An XMP packet that already has an
instance id keeps it.

The mark holds the IPTC term and the random id only: no user, workspace, prompt or file
name (Guidelines para. 94). A file that cannot be read as its format raises ValueError
(or the reader's own error), so the caller can refuse to ship it unmarked.
"""
from __future__ import annotations

import hashlib
import io
import os
import stat
import struct
import tempfile
import uuid
import zipfile
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from lxml import etree

# --------------------------------------------------------------------------- the IPTC terms

IPTC_DIGITAL_SOURCE_TYPES = "http://cv.iptc.org/newscodes/digitalsourcetype/"
# Every element is AI-made (an image model's picture).
TRAINED_ALGORITHMIC_MEDIA = f"{IPTC_DIGITAL_SOURCE_TYPES}trainedAlgorithmicMedia"
# AI-made content in non-AI parts (an agent's text in the workspace's template and letterhead).
COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA = f"{IPTC_DIGITAL_SOURCE_TYPES}compositeWithTrainedAlgorithmicMedia"
AI_SOURCE_TYPES = frozenset({TRAINED_ALGORITHMIC_MEDIA, COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA})
IPTC_EXT_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
AI_PROPERTY_NAME = "AIGenerated"
AI_INSTANCE_PROPERTY_NAME = "AIInstanceID"
_MARK_PROPERTIES = (AI_PROPERTY_NAME, AI_INSTANCE_PROPERTY_NAME)

# --------------------------------------------------------------------------- XMP

_XMP_HEAD = b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n<x:xmpmeta xmlns:x="adobe:ns:meta/">'
_RDF_OPEN = b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
_RDF_CLOSE = b"</rdf:RDF>"
_XMP_TAIL = b'</x:xmpmeta>\n<?xpacket end="w"?>'
_SOURCE_TYPE_PROPERTY = b"DigitalSourceType"
XMP_MM_NS = "http://ns.adobe.com/xap/1.0/mm/"
_INSTANCE_ID_PROPERTY = b"InstanceID"  # xmpMM:InstanceID (stEvt/stRef's instanceID is lower-case)
INSTANCE_ID_PREFIX = "xmp.iid:"


def new_instance_id() -> str:
    """A random per-file instance id, ``xmp.iid:<uuid4>`` (FR-16): nothing in it is derived
    from the file, its maker or its workspace."""
    return f"{INSTANCE_ID_PREFIX}{uuid.uuid4()}"


def _checked_term(term: str) -> str:
    """``term`` when it is one of the IPTC AI terms; ValueError otherwise (nothing else is written)."""
    if term not in AI_SOURCE_TYPES:
        raise ValueError(f"not an IPTC AI digital source type: {term!r}")
    return term


def _description(term: Optional[str], instance_id: Optional[str]) -> bytes:
    """An ``rdf:Description`` with the source type and the instance id (each when given)."""
    source = ("" if term is None else
              f"<Iptc4xmpExt:DigitalSourceType>{_checked_term(term)}</Iptc4xmpExt:DigitalSourceType>")
    instance = "" if instance_id is None else f"<xmpMM:InstanceID>{instance_id}</xmpMM:InstanceID>"
    return (f'<rdf:Description rdf:about="" xmlns:Iptc4xmpExt="{IPTC_EXT_NS}" xmlns:xmpMM="{XMP_MM_NS}">'
            f"{source}{instance}</rdf:Description>").encode("utf-8")


def xmp_packet(term: str, existing: Optional[bytes] = None, instance_id: Optional[str] = None) -> Optional[bytes]:
    """The XMP packet that marks a file: a new one, or ``existing`` with the mark added.

    The mark is the IPTC ``term`` and an instance id (``instance_id``, a new random one
    when None). An ``existing`` packet keeps what it already states: a digital source
    type its maker wrote, an instance id it already has; only what is missing is added.
    None when it already states both. ValueError when ``existing`` has no ``rdf:RDF``.
    """
    _checked_term(term)
    instance_id = instance_id or new_instance_id()
    if existing is None:
        return _XMP_HEAD + _RDF_OPEN + _description(term, instance_id) + _RDF_CLOSE + _XMP_TAIL
    missing_term = None if _SOURCE_TYPE_PROPERTY in existing else term
    missing_id = None if _INSTANCE_ID_PROPERTY in existing else instance_id
    if missing_term is None and missing_id is None:
        return None
    if _RDF_CLOSE not in existing:
        raise ValueError("the file's XMP packet has no rdf:RDF element to add the mark to")
    at = existing.rindex(_RDF_CLOSE)
    return existing[:at] + _description(missing_term, missing_id) + existing[at:]


# --------------------------------------------------------------------------- files


def sha256_hex(data: bytes) -> str:
    """The SHA-256 of ``data`` as 64 lower-case hex characters (what the provenance row keeps)."""
    return hashlib.sha256(data).hexdigest()


def _replace_file(path: str, data: bytes) -> None:
    """Write ``data`` over ``path`` atomically (a sibling temp file, then a rename), keeping its mode."""
    target = Path(path)
    mode = stat.S_IMODE(target.stat().st_mode)
    handle, temp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".marking")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.chmod(temp, mode)
        os.replace(temp, target)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------- PDF

_PDF_METADATA = "/Metadata"


def mark_pdf(path: str, term: str) -> None:
    """Put the XMP mark in the PDF at ``path`` (the catalogue's ``/Metadata`` stream), in place.

    The document is cloned whole (pages, fonts, its ``/Info`` with the F331 author), and an
    XMP packet it already has keeps its content with the mark added.
    """
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DecodedStreamObject, NameObject

    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(Path(path).read_bytes())))
    catalogue = writer._root_object  # the document catalogue (pypdf 4 to 6 all name it so)
    current = catalogue.get(_PDF_METADATA)
    packet = xmp_packet(term, current.get_object().get_data() if current is not None else None)
    if packet is None:
        return
    stream = DecodedStreamObject()
    stream.set_data(packet)
    stream[NameObject("/Type")] = NameObject(_PDF_METADATA)
    stream[NameObject("/Subtype")] = NameObject("/XML")
    catalogue[NameObject(_PDF_METADATA)] = writer._add_object(stream)
    out = io.BytesIO()
    writer.write(out)
    _replace_file(path, out.getvalue())


# --------------------------------------------------------------------------- DOCX and XLSX

CONTENT_TYPES_PART = "[Content_Types].xml"
PACKAGE_RELS_PART = "_rels/.rels"
CUSTOM_PROPERTIES_PART = "docProps/custom.xml"
_CUSTOM_PART_NAME = f"/{CUSTOM_PROPERTIES_PART}"
_CUSTOM_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.custom-properties+xml"
_CUSTOM_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/custom-properties"
_CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CUSTOM_NS = "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
VT_NS = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"
# The format id every user-defined custom property carries (ECMA-376 Part 1, 22.3.2.2).
_CUSTOM_FMTID = "{D5CDD505-2E9C-101B-9397-08002B2CF9AE}"
_FIRST_CUSTOM_PID = 2
_REL_ID_PREFIX = "rIdAiMark"
# No entity expansion, no network: a workspace's uploaded template is parsed here too.
_XML_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)


def _xml(data: bytes) -> etree._Element:
    return etree.fromstring(data, _XML_PARSER)


def _xml_bytes(root: etree._Element) -> bytes:
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def _with_custom_override(data: bytes) -> bytes:
    """``[Content_Types].xml`` that names the custom-properties part (unchanged when it does)."""
    root = _xml(data)
    if any(o.get("PartName") == _CUSTOM_PART_NAME for o in root.iter(f"{{{_CT_NS}}}Override")):
        return data
    etree.SubElement(root, f"{{{_CT_NS}}}Override", PartName=_CUSTOM_PART_NAME, ContentType=_CUSTOM_CONTENT_TYPE)
    return _xml_bytes(root)


def _free_rel_id(taken: Sequence[Optional[str]]) -> str:
    number = 1
    while f"{_REL_ID_PREFIX}{number}" in taken:
        number += 1
    return f"{_REL_ID_PREFIX}{number}"


def _with_custom_relationship(data: bytes) -> bytes:
    """``_rels/.rels`` that relates the package to ``docProps/custom.xml`` (unchanged when it does).

    ValueError when the package keeps its custom properties in another part: the mark
    would land where no reader looks.
    """
    root = _xml(data)
    relationships = list(root.iter(f"{{{_REL_NS}}}Relationship"))
    for relationship in relationships:
        if relationship.get("Type") == _CUSTOM_REL_TYPE:
            if (relationship.get("Target") or "").lstrip("/") != CUSTOM_PROPERTIES_PART:
                raise ValueError("the package keeps its custom properties outside docProps/custom.xml")
            return data
    etree.SubElement(root, f"{{{_REL_NS}}}Relationship", Id=_free_rel_id([r.get("Id") for r in relationships]),
                     Type=_CUSTOM_REL_TYPE, Target=CUSTOM_PROPERTIES_PART)
    return _xml_bytes(root)


def _custom_properties(data: Optional[bytes], term: str, instance_id: str) -> bytes:
    """``docProps/custom.xml`` with ``AIGenerated`` = ``term`` and ``AIInstanceID`` = ``instance_id``:
    the package's own properties kept, earlier marks replaced, the new ones numbered after
    the highest ``pid``. (The tree is parsed here from the part's bytes, so it is our own.)"""
    root = _xml(data) if data else etree.Element(f"{{{CUSTOM_NS}}}Properties", nsmap={None: CUSTOM_NS, "vt": VT_NS})
    pids = [_FIRST_CUSTOM_PID - 1]
    for prop in list(root.iter(f"{{{CUSTOM_NS}}}property")):
        if prop.get("name") in _MARK_PROPERTIES:
            prop.getparent().remove(prop)
        else:
            pids.append(int(prop.get("pid") or _FIRST_CUSTOM_PID))
    for offset, (name, value) in enumerate(((AI_PROPERTY_NAME, _checked_term(term)),
                                            (AI_INSTANCE_PROPERTY_NAME, instance_id)), start=1):
        prop = etree.SubElement(root, f"{{{CUSTOM_NS}}}property", fmtid=_CUSTOM_FMTID,
                                pid=str(max(pids) + offset), name=name)
        etree.SubElement(prop, f"{{{VT_NS}}}lpwstr").text = value
    return _xml_bytes(root)


def mark_ooxml(path: str, term: str) -> None:
    """Set the ``AIGenerated`` and ``AIInstanceID`` custom properties of the DOCX or XLSX at
    ``path``, in place (a new random instance id each time, FR-16).

    Only ``[Content_Types].xml``, ``_rels/.rels`` and ``docProps/custom.xml`` change; every
    other part is copied as it was. ValueError when the file is not an OOXML package.
    """
    _checked_term(term)
    instance_id = new_instance_id()
    with zipfile.ZipFile(io.BytesIO(Path(path).read_bytes())) as source:
        names = set(source.namelist())
        if not {CONTENT_TYPES_PART, PACKAGE_RELS_PART} <= names:
            raise ValueError(f"{Path(path).name} is not an OOXML package")
        rewrites: Dict[str, Callable[[bytes], bytes]] = {
            CONTENT_TYPES_PART: _with_custom_override,
            PACKAGE_RELS_PART: _with_custom_relationship,
            CUSTOM_PROPERTIES_PART: lambda part: _custom_properties(part, term, instance_id),
        }
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target:
            for info in source.infolist():
                part = source.read(info)
                target.writestr(info, rewrites[info.filename](part) if info.filename in rewrites else part)
            if CUSTOM_PROPERTIES_PART not in names:
                target.writestr(CUSTOM_PROPERTIES_PART, _custom_properties(None, term, instance_id))
    _replace_file(path, out.getvalue())


# --------------------------------------------------------------------------- PNG

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_XMP_KEYWORD = b"XML:com.adobe.xmp"
_PNG_C2PA_CHUNK = b"caBX"
_PNG_ITXT = b"iTXt"
_PNG_HEADER_CHUNK = b"IHDR"
_PNG_END_CHUNK = b"IEND"
_PNG_CHUNK_FRAME = 12  # length, type and CRC around each chunk's body

Chunk = Tuple[bytes, bytes]


def _png_chunks(data: bytes) -> List[Chunk]:
    """The PNG's chunks as (type, body), up to and including IEND; ValueError when it is not a PNG."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG")
    chunks: List[Chunk] = []
    at = len(PNG_SIGNATURE)
    while at < len(data) and (not chunks or chunks[-1][0] != _PNG_END_CHUNK):
        if at + 8 > len(data):
            raise ValueError("the PNG ends inside a chunk")
        length, kind = struct.unpack(">I4s", data[at:at + 8])
        if at + _PNG_CHUNK_FRAME + length > len(data):
            raise ValueError("the PNG ends inside a chunk")
        chunks.append((kind, data[at + 8:at + 8 + length]))
        at += _PNG_CHUNK_FRAME + length
    if not chunks or chunks[0][0] != _PNG_HEADER_CHUNK:
        raise ValueError("the PNG has no IHDR chunk")
    return chunks


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def _is_xmp_itxt(chunk: Chunk) -> bool:
    return chunk[0] == _PNG_ITXT and chunk[1].startswith(PNG_XMP_KEYWORD + b"\x00")


def _itxt_text(body: bytes) -> bytes:
    """The text of an iTXt chunk (keyword, flags, language and translated keyword skipped)."""
    rest = body[len(PNG_XMP_KEYWORD) + 1:]
    compressed, rest = rest[0], rest[2:]
    _language, _translated, text = rest.split(b"\x00", 2)
    return zlib.decompress(text) if compressed else text


def _marked_png(data: bytes, term: str) -> bytes:
    chunks = _png_chunks(data)
    if any(kind == _PNG_C2PA_CHUNK for kind, _ in chunks):
        return data
    at = next((i for i, chunk in enumerate(chunks) if _is_xmp_itxt(chunk)), None)
    packet = xmp_packet(term, _itxt_text(chunks[at][1]) if at is not None else None)
    if packet is None:
        return data
    itxt = (_PNG_ITXT, PNG_XMP_KEYWORD + b"\x00\x00\x00\x00\x00" + packet)  # uncompressed, no language
    marked = [chunks[0], itxt, *chunks[1:]] if at is None else [*chunks[:at], itxt, *chunks[at + 1:]]
    return PNG_SIGNATURE + b"".join(_png_chunk(kind, body) for kind, body in marked)


# --------------------------------------------------------------------------- JPEG

JPEG_SOI = b"\xff\xd8"
JPEG_XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
_JPEG_APP0, _JPEG_APP1, _JPEG_APP11, _JPEG_SOS = 0xE0, 0xE1, 0xEB, 0xDA
_JPEG_C2PA_LABEL = b"c2pa"
_JPEG_SEGMENT_MAX = 0xFFFF

Segment = Tuple[int, bytes]


def _jpeg_segments(data: bytes) -> Tuple[List[Segment], bytes]:
    """(the marker segments before the first scan as (marker, body), the bytes from SOS on)."""
    if not data.startswith(JPEG_SOI):
        raise ValueError("not a JPEG")
    segments: List[Segment] = []
    at = len(JPEG_SOI)
    while True:
        if at + 4 > len(data) or data[at] != 0xFF:
            raise ValueError("the JPEG has a malformed marker segment")
        marker = data[at + 1]
        if marker == _JPEG_SOS:
            return segments, data[at:]
        (length,) = struct.unpack(">H", data[at + 2:at + 4])
        if length < 2 or at + 2 + length > len(data):
            raise ValueError("the JPEG has a malformed marker segment")
        segments.append((marker, data[at + 4:at + 2 + length]))
        at += 2 + length


def _jpeg_segment(marker: int, body: bytes) -> bytes:
    if len(body) + 2 > _JPEG_SEGMENT_MAX:
        raise ValueError("the XMP packet is too large for one JPEG segment")
    return bytes((0xFF, marker)) + struct.pack(">H", len(body) + 2) + body


def _marked_jpeg(data: bytes, term: str) -> bytes:
    segments, scan = _jpeg_segments(data)
    if any(marker == _JPEG_APP11 and _JPEG_C2PA_LABEL in body for marker, body in segments):
        return data
    at = next((i for i, (marker, body) in enumerate(segments)
               if marker == _JPEG_APP1 and body.startswith(JPEG_XMP_HEADER)), None)
    packet = xmp_packet(term, segments[at][1][len(JPEG_XMP_HEADER):] if at is not None else None)
    if packet is None:
        return data
    xmp = (_JPEG_APP1, JPEG_XMP_HEADER + packet)
    if at is None:  # after JFIF's APP0 and an Exif APP1, which readers expect first
        lead = next((i for i, (marker, _) in enumerate(segments) if marker not in (_JPEG_APP0, _JPEG_APP1)),
                    len(segments))
        marked = [*segments[:lead], xmp, *segments[lead:]]
    else:
        marked = [*segments[:at], xmp, *segments[at + 1:]]
    return JPEG_SOI + b"".join(_jpeg_segment(marker, body) for marker, body in marked) + scan


# --------------------------------------------------------------------------- entry points

_IMAGE_MARKERS: Dict[str, Callable[[bytes, str], bytes]] = {
    "image/png": _marked_png,
    "image/jpeg": _marked_jpeg,
    "image/jpg": _marked_jpeg,
}
MARKED_IMAGE_TYPES = frozenset(_IMAGE_MARKERS)
_DOCUMENT_MARKERS: Dict[str, Callable[[str, str], None]] = {"pdf": mark_pdf, "docx": mark_ooxml, "xlsx": mark_ooxml}
MARKED_DOCUMENT_FORMATS = frozenset(_DOCUMENT_MARKERS)


def marked_image_bytes(data: bytes, mime: str, term: str = TRAINED_ALGORITHMIC_MEDIA) -> bytes:
    """``data`` with the XMP mark (PNG, JPEG: the IPTC term and an instance id); the same
    object when it is kept as it came (a C2PA manifest, an XMP packet that already states
    both, a type not in ``MARKED_IMAGE_TYPES``)."""
    _checked_term(term)
    marker = _IMAGE_MARKERS.get((mime or "").lower())
    if marker is None:
        return data
    try:
        return marker(data, term)
    except (IndexError, struct.error, zlib.error) as e:  # a chunk or segment cut short
        raise ValueError(f"the {mime} image could not be read to mark it: {e}") from e


def mark_document(path: str, fmt: str, term: str = COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA) -> str:
    """Mark the generated document at ``path`` (``fmt``: pdf, docx or xlsx) in place, and
    return the marked file's SHA-256. ValueError for any other format."""
    marker = _DOCUMENT_MARKERS.get(fmt)
    if marker is None:
        raise ValueError(f"no AI mark is written for {fmt!r} files")
    marker(path, _checked_term(term))
    return sha256_hex(Path(path).read_bytes())


__all__ = [
    "AI_INSTANCE_PROPERTY_NAME",
    "AI_PROPERTY_NAME",
    "COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA",
    "MARKED_DOCUMENT_FORMATS",
    "MARKED_IMAGE_TYPES",
    "TRAINED_ALGORITHMIC_MEDIA",
    "mark_document",
    "mark_ooxml",
    "mark_pdf",
    "marked_image_bytes",
    "new_instance_id",
    "sha256_hex",
    "xmp_packet",
]
