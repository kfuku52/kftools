"""Reproduce minimum, latest-quality, and installed-wheel checks in clean venvs."""

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*command: str, cwd: Path = ROOT) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def check(mode: str, interpreter: str, wheel: Path | None) -> None:
    required = {"minimum": "3.10", "latest": "3.14"}.get(mode)
    if required:
        version = subprocess.check_output(
            [interpreter, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], text=True
        ).strip()
        if version != required:
            raise SystemExit(f"{mode} checks require Python {required}, got {version}; set --python accordingly")
    with tempfile.TemporaryDirectory(prefix=f"kftools-{mode}-") as temporary:
        directory = Path(temporary)
        run(interpreter, "-m", "venv", str(directory / "venv"))
        bindir = "Scripts" if os.name == "nt" else "bin"
        python = str(directory / "venv" / bindir / ("python.exe" if os.name == "nt" else "python"))
        run(python, "-m", "pip", "install", "--upgrade", "pip")
        if mode == "wheel":
            candidates = [wheel] if wheel else sorted((ROOT / "dist").glob("*.whl"))
            if len(candidates) != 1:
                raise SystemExit("Build one wheel in dist/ or select it explicitly with --wheel PATH")
            run(python, "-m", "pip", "install", str(candidates[0].resolve()))
            run(python, str(ROOT / "scripts" / "wheel_smoke.py"), cwd=directory)
        elif mode == "minimum":
            run(
                python,
                "-m",
                "pip",
                "install",
                "-c",
                str(ROOT / "constraints/minimum-python310.txt"),
                "-e",
                f"{ROOT}[test]",
            )
            run(python, "-m", "pytest", "-q")
        else:
            run(python, "-m", "pip", "install", "-e", f"{ROOT}[test,quality]")
            run("make", "check", f"PYTHON={python}")
        run(python, "-m", "pip", "check")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["minimum", "latest", "wheel"])
    parser.add_argument("--python", default=sys.executable, help="interpreter used to create the isolated environment")
    parser.add_argument("--wheel", type=Path, help="wheel to smoke-test (default: the single wheel in dist/)")
    args = parser.parse_args()
    start = time.perf_counter()
    check(args.mode, args.python, args.wheel)
    print(f"{args.mode} checks passed in {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
