"""Build helpers that do not need `make` (Windows, minimal Linux images).

`sentinel build-dashboard` runs `npm ci` + `npm run build` in the dashboard
sources and copies the result into the installed package, where `sentinel
serve` looks first. It is the same as `make dashboard`.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from modelsentinel.api import PACKAGE_WEB, ROOT


class BuildError(RuntimeError):
    pass


def find_npm() -> str | None:
    # On Windows npm is npm.cmd; shutil.which resolves it through PATHEXT.
    return shutil.which("npm") or shutil.which("npm.cmd")


def dashboard_sources(explicit: str | Path | None = None) -> Path:
    candidates = [Path(explicit)] if explicit else [ROOT / "dashboard", Path.cwd() / "dashboard"]
    for candidate in candidates:
        if (candidate / "package.json").is_file():
            return candidate.resolve()
    raise BuildError(
        "dashboard sources not found (looked for dashboard/package.json in "
        + ", ".join(str(c) for c in candidates)
        + "). Run this from a ModelSentinel checkout or pass --source."
    )


def build_dashboard(
    source: str | Path | None = None,
    dest: Path = PACKAGE_WEB,
    *,
    install: bool = True,
    npm: str | None = None,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    log: Callable[[str], None] = lambda message: print(message, flush=True),
) -> Path:
    """Build the dashboard and copy ``dist/`` to ``dest``. Returns ``dest``."""
    src = dashboard_sources(source)
    npm = npm or find_npm()
    if not npm:
        raise BuildError(
            "npm was not found. Install Node.js 20+ (https://nodejs.org, `winget install OpenJS.NodeJS.LTS`, "
            "`brew install node`, or your distro's nodejs package) and rerun `sentinel build-dashboard`."
        )
    steps = ([[npm, "ci", "--no-audit", "--no-fund"]] if install else []) + [[npm, "run", "build"]]
    for cmd in steps:
        log("$ " + " ".join(Path(cmd[0]).name if i == 0 else part for i, part in enumerate(cmd)))
        proc = run(cmd, cwd=str(src), check=False)
        if proc.returncode != 0:
            raise BuildError(f"`{' '.join(cmd[1:])}` failed with exit code {proc.returncode} in {src}")
    dist = src / "dist"
    if not (dist / "index.html").is_file():
        raise BuildError(f"build finished but {dist / 'index.html'} is missing")
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(dist, dest)
    log(f"dashboard copied to {dest}; `sentinel serve` will use it")
    return dest
