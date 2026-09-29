"""The public API docs (/docs, /openapi.json) state the project's real licence and name.

The FastAPI app once advertised the MIT licence and a misspelled product name
("Automotas") while the repository is Apache-2.0. These checks read main.py's
source, like test_f105, so they run without booting the app.
"""

import re
from pathlib import Path

MAIN_PY = Path(__file__).resolve().parents[1] / "main.py"


def _license_block(source: str) -> str:
    match = re.search(r"license_info=\{(.*?)\}", source, re.S)
    assert match, "main.py must declare license_info on the FastAPI app"
    return match.group(1)


def test_api_docs_declare_the_apache_licence():
    block = _license_block(MAIN_PY.read_text())

    assert '"Apache License 2.0"' in block
    assert "apache.org/licenses/LICENSE-2.0" in block
    assert "MIT" not in block


def test_api_docs_spell_the_product_name_correctly():
    assert "Automotas" not in MAIN_PY.read_text()
