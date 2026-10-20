"""ELF facts for the in-image suite: readelf is not in the runtime (binutils stays in
the builder stages), so the section names are read here (PRD-262 #18, #22, #38)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import List, Set

ELF_HEADER_BYTES = 64
# A 64-bit little-endian ELF file: both image arches, amd64 and arm64.
ELF64_LSB = b"\x7fELF\x02\x01"


def section_names(path: Path) -> Set[str]:
    """The section names of ``path``, read from its section header table."""
    with open(path, "rb") as elf:
        ident = elf.read(ELF_HEADER_BYTES)
        assert ident[:6] == ELF64_LSB, f"{path} is not a 64-bit little-endian ELF file"
        (table_at,) = struct.unpack_from("<Q", ident, 0x28)
        entry_size, count, names_index = struct.unpack_from("<HHH", ident, 0x3A)
        elf.seek(table_at)
        table = elf.read(entry_size * count)
        # Elf64_Shdr begins: name, type, flags, addr, offset, size.
        headers = [struct.unpack_from("<IIQQQQ", table, i * entry_size) for i in range(count)]
        elf.seek(headers[names_index][4])
        names = elf.read(headers[names_index][5])
    return {names[header[0] : names.index(b"\0", header[0])].decode() for header in headers}


def shared_libraries(root: Path) -> List[Path]:
    """Every regular ``*.so`` or ``*.so.*`` file under ``root`` (not the links to them)."""
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and (path.name.endswith(".so") or ".so." in path.name)
    )
