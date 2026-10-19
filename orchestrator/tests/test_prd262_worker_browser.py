"""PRD-262 US-043: a browser only where one runs.

The workspace worker's Dockerfile installs Playwright and Chromium only when
INSTALL_BROWSER=true, the default Railway builds, from requirements-browser.lock
(US-044). The local edition and the GHCR image pass false and carry neither.
Where the browser is installed it is the headless shell alone (html_to_png launches headless with no channel), driven by the
image's own Node, with only the libraries the shell links and without Mesa's GPU
drivers. The step sits above the code, after the worker user exists, and the code
goes in last, worker-owned.

test.yml's prod-image-parity job builds the image and checks the result: the
directories, ldd, the purged packages, a render as uid 1000 with a CJK fixture,
and the INSTALL_BROWSER=false build. This suite runs in the required orchestrator
job and pins the definitions a later edit could quietly undo.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "services" / "workspace-worker"
DOCKERFILE = (WORKER / "Dockerfile").read_text()
OLD_CAP = (1, 53)  # playwright<1.53 held Chromium at 136 with no recorded reason
CODE_COPY = "COPY --chown=1000:1000 . ."
MESA = {"mesa-libgallium", "libllvm19", "libz3-4"}
KEPT_FOR_THE_SHELL = {"libgbm1", "libdrm2", "libexpat1"}
DROPPED_LIBS = {"libcups2t64", "libpango-1.0-0", "libcairo2", "libxshmfence1"}


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in re.sub(r"\\[ \t]*\n", " ", code).splitlines() if line.strip()]


INSTRUCTIONS = _instructions(DOCKERFILE)


def _index(predicate) -> int:
    found = [i for i, line in enumerate(INSTRUCTIONS) if predicate(line)]
    assert len(found) == 1, f"expected one matching instruction, found {[INSTRUCTIONS[i] for i in found]}"
    return found[0]


BROWSER_AT = _index(lambda line: line.startswith("RUN") and "playwright install" in line)
BROWSER_RUN = INSTRUCTIONS[BROWSER_AT]


def _words_after(command: str, text: str) -> list[str]:
    """The arguments of `command` in `text`, up to the next `;`, `&&` or redirection."""
    match = re.search(re.escape(command) + r"\s+([^;&|]*)", text)
    assert match, command
    return [word for word in match.group(1).split() if not word.startswith(("-", "2>"))]


def test_playwright_is_installed_only_by_the_browser_step_at_a_current_release():
    # US-044: requirements.txt became requirements.in, locked to requirements.lock.
    for name in ("requirements.in", "requirements.lock"):
        requirements = (WORKER / name).read_text()
        assert not re.search(r"^\s*playwright\b", requirements, re.M | re.I), f"playwright is back in {name}"
    browser_in = (WORKER / "requirements-browser.in").read_text()
    pins = re.findall(r"^playwright==(\d+)\.(\d+)\.(\d+)\s*$", browser_in, re.M)
    assert len(pins) == 1, "requirements-browser.in pins playwright exactly, once"
    assert tuple(map(int, pins[0][:2])) >= OLD_CAP, pins[0]
    locked = re.findall(r"^playwright==(\S+) \\$", (WORKER / "requirements-browser.lock").read_text(), re.M)
    assert locked == [".".join(pins[0])], "requirements-browser.lock pins the release requirements-browser.in names"
    installs = [line for line in INSTRUCTIONS if "requirements-browser" in line and "install" in line]
    assert installs == [BROWSER_RUN]
    gate = BROWSER_RUN.index('if [ "$INSTALL_BROWSER" != "true" ]; then')
    assert gate < BROWSER_RUN.index("exit 0") < BROWSER_RUN.index("apt-get update")


def test_the_headless_shell_runs_on_the_image_node():
    assert "playwright install --only-shell chromium" in BROWSER_RUN
    assert re.findall(r"playwright install\b[^;]*", "\n".join(INSTRUCTIONS)) == [
        "playwright install --only-shell chromium"
    ]
    # Playwright's bundled Node goes in the RUN that installed it, or the layer keeps it.
    pip = BROWSER_RUN.index("--require-hashes --no-deps -r requirements-browser.lock")
    assert pip < BROWSER_RUN.index("/driver/node") < BROWSER_RUN.index("playwright install")
    env = re.search(r"\bPLAYWRIGHT_NODEJS_PATH=(\S+)", DOCKERFILE)
    assert env and env.group(1) == "/usr/local/bin/node"
    assert _index(lambda line: line.startswith("ENV") and "PLAYWRIGHT_NODEJS_PATH" in line) < BROWSER_AT
    # PRD-262 (row 21): the binary is the node-stripped stage's (test_prd262_worker_diet.py).
    assert "COPY --link --from=node-stripped /node-layer/ /" in INSTRUCTIONS


def test_the_user_comes_first_the_browser_next_and_the_code_last_worker_owned():
    user_at = _index(lambda line: line.startswith("RUN") and "useradd" in line)
    assert "useradd -m -u 1000 worker" in INSTRUCTIONS[user_at]
    assert "install -d -o worker -g worker /app /workspaces /home/worker/.cache" in INSTRUCTIONS[user_at]
    code_at = _index(lambda line: line.startswith("COPY") and line.endswith(" . ."))
    assert INSTRUCTIONS[code_at] == CODE_COPY
    assert user_at < BROWSER_AT < code_at
    later = INSTRUCTIONS[code_at + 1:]
    assert not [line for line in later if line.split()[0] in {"RUN", "COPY", "ADD"}], later
    assert "chown -R" not in DOCKERFILE
    assert "mkdir -p /home/worker" not in BROWSER_RUN, "the browser step must not create HOME as root"


def test_the_browser_apt_list_is_what_the_headless_shell_links():
    packages = set(_words_after("apt-get install", BROWSER_RUN))
    assert {"fontconfig", "fonts-unifont", "libgbm1", "libnss3"} <= packages, packages
    assert not packages & DROPPED_LIBS, packages & DROPPED_LIBS
    assert [line for line in INSTRUCTIONS[BROWSER_AT + 1:] if "apt-get install" in line] == [], (
        "the Mesa purge leaves libgbm1's dependency unmet: no apt-get install may follow it"
    )


def test_mesa_is_purged_once_apt_shows_only_libgbm1_depends_on_it():
    listed = _words_after("dpkg-query -W --showformat='${db:Status-Abbrev} ${Package}\\n'", BROWSER_RUN)
    assert MESA <= set(listed), listed
    assert not KEPT_FOR_THE_SHELL & set(listed), listed
    check = BROWSER_RUN.index("apt-get -s purge $mesa")
    expected = BROWSER_RUN.index("printf 'Purg %s\\n' libgbm1 $mesa")
    purge = BROWSER_RUN.index("dpkg --purge --force-depends $mesa")
    ldd = BROWSER_RUN.index("grep 'not found'")
    assert check < expected < purge < ldd, "apt's plan is checked, then Mesa purged, then ldd run"
    assert 'ldd "$shell" "$(dirname "$shell")"/*.so*' in BROWSER_RUN


def test_the_parity_job_checks_the_built_worker():
    steps = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text())["jobs"]["prod-image-parity"]["steps"]
    script = "\n".join(step.get("run", "") for step in steps)
    for needle in (
        "docker build -t parity-wk services/workspace-worker/",
        "ls -d /opt/playwright-browsers/chromium-[0-9]*",
        "ls -d /opt/playwright-browsers/chromium_headless_shell-*",
        "grep 'not found' <<< \"$libs\"",
        "for pkg in mesa-libgallium libllvm19 libz3-4; do",
        '"driver", "node"',
        "docker run --rm -i --user 1000 --entrypoint python parity-wk -",
        "await page.set_content(html)",
        "await page.screenshot(type=\"png\")",
        "中文",
        '"fc-list", ":charset=4e2d"',
        "WorkspaceToolExecutor(ws).html_to_png(",
        "docker build --build-arg INSTALL_BROWSER=false -t parity-wk-local services/workspace-worker/",
        "docker run --rm --entrypoint python parity-wk-local -c 'import playwright'",
    ):
        assert needle in script, needle
    # The render uses the worker's own launch arguments.
    executor = (WORKER / "executor.py").read_text()
    for arg in ("--no-sandbox", "--disable-dev-shm-usage", "--allow-file-access-from-files", "--disable-web-security"):
        assert f'"{arg}"' in executor and f'"{arg}"' in script, arg
