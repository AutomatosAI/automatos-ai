"""PRD-262 #47 (the 19 Oct breakdown): Auto's posters, recompressed without losing a pixel.

The 12 mascot stills in frontend/public/auto (lib/mascot/media.ts's posterUrl, the video
poster and the reduced-motion image) were 512 px RGBA PNGs of about 127 KB each, 1,526,290
bytes in all. Their image data was re-deflated with oxipng (level 6, zopfli, no colour or
alpha changes, every other chunk kept as it was), which made them 1,174,791 bytes: 0.35 MB
less in the image and in every first page load that shows Auto.

§9-Q12 allows no lossy change to the posters without design sign-off, so this pins the
SHA-256 of each poster's decoded RGBA pixels, taken from the files before the recompression:
a re-encode that changes any pixel fails here. It also holds each file to the recompressed
size, so a re-save at an editor's default compression is caught, and keeps the sRGB and EXIF
chunks the originals carry. No file a database row may name (public/images, favicon_io,
modal-fix.js) is touched.

File reads and Pillow's PNG decoder: no network.
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest
from PIL import Image

POSTERS = Path(__file__).resolve().parents[2] / "frontend" / "public" / "auto"
SIDE_PX = 512
# The largest recompressed poster is 99,246 bytes; the originals were 126,850 to 127,424.
MAX_BYTES = 100_000
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# SHA-256 of Image.convert("RGBA").tobytes(), from the posters as committed before PRD-262 #47.
PIXELS = {
    "celebrate.png": "b9bbdc6e81fe3c6a485ced03197224c86cb924b174f9cd331ec7129fe8b29918",
    "hello.png": "df388f698dbba4bb274a2153c8c7f9b2290cfbdccbfa17f42672b52fe3fd0c0d",
    "idle.png": "cde760d811fd05c8cc6fd82577c86570bf6849b26032a4ec89adf4512e8da3f1",
    "listening.png": "ad06772a05dc979d8ed95675b623ffaffbd482950ce8cdd6f8834f81b9d6377b",
    "loading.png": "67fd72d09ec04ece6af5793945cd3ff1059a60ad869434f994e33ae01eb02354",
    "lost.png": "655cb32451b5cbaa562e2bfda0259678eda5bc163b9c3a5329f4440e4523af9a",
    "oops.png": "ab158b8fb8d25ec0568446d317c3b2b787680601643dda8c2188c296b671bf6a",
    "reading.png": "5665c10cf0914672075e5a11871885b68bc56e8065d1548a500368dea5d7cee2",
    "searching.png": "ecd869898289d3b74c7451c8cf21d71e005d8b5612f7f99134af1185917d7bd1",
    "sleeping.png": "1486e58712dbbf238f34a54af08862d12c501d6ba1cbe66b1dfe23d21ac26502",
    "thinking.png": "4f9e9acb0e7908a7f26ab4747bc3e54e9bcabdd2311e80bc1971324f77292398",
    "working.png": "576eba8a0bc25577ec1a6850c55a6bea042dc2ce6d2469e837fbf4e23d895332",
}


def _chunk_types(data: bytes) -> list[str]:
    assert data.startswith(PNG_SIGNATURE)
    types, position = [], len(PNG_SIGNATURE)
    while position < len(data):
        (length,) = struct.unpack(">I", data[position:position + 4])
        types.append(data[position + 4:position + 8].decode("ascii"))
        position += 12 + length
    return types


def test_the_posters_are_the_twelve_states():
    assert sorted(path.name for path in POSTERS.glob("*.png")) == sorted(PIXELS)


@pytest.mark.parametrize("name", sorted(PIXELS))
def test_a_poster_keeps_every_pixel(name):
    with Image.open(POSTERS / name) as image:
        assert image.format == "PNG"
        assert image.size == (SIDE_PX, SIDE_PX)
        assert hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest() == PIXELS[name]


@pytest.mark.parametrize("name", sorted(PIXELS))
def test_a_poster_stays_recompressed_with_its_colour_and_exif_chunks(name):
    data = (POSTERS / name).read_bytes()
    assert len(data) <= MAX_BYTES, f"{name} is {len(data)} bytes: recompress it losslessly (oxipng)"
    types = _chunk_types(data)
    assert types[0] == "IHDR" and types[-1] == "IEND"
    assert {"sRGB", "eXIf"} <= set(types)
