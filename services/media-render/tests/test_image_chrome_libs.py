"""PRD-262 #38: Chrome's own libraries carry no symbol table. Runs inside the image.

The chrome stage strips the arm64 headless shell's SwiftShader, ANGLE and Vulkan
libraries as it strips the shell; the amd64 build ships them stripped. The check
at build covers both arches, and this one reads the image.
"""

from __future__ import annotations

from pathlib import Path

from elf import section_names, shared_libraries
from media_render.config import load_settings


def test_the_headless_shell_and_its_libraries_carry_no_symbol_table():
    shell = Path(load_settings().browser_path).resolve()
    libraries = shared_libraries(shell.parent)
    assert libraries, f"no shared library beside {shell}"
    assert [str(path) for path in (shell, *libraries) if ".symtab" in section_names(path)] == []
