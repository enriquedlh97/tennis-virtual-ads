"""Smoke tests — verify that imports work and the CLI is wired up correctly."""

import subprocess
import sys
from pathlib import Path

# Ensure src/ is importable when running tests directly or via pytest
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = str(_PROJECT_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def test_core_imports():
    """VideoReader and VideoWriter can be imported without error."""
    from tennis_virtual_ads.io.video import VideoReader, VideoWriter

    assert VideoReader is not None
    assert VideoWriter is not None


def test_package_init_imports():
    """Top-level and sub-package __init__ modules import cleanly."""
    import tennis_virtual_ads
    import tennis_virtual_ads.io
    import tennis_virtual_ads.pipeline
    import tennis_virtual_ads.utils

    assert tennis_virtual_ads is not None


def test_cli_help_exits_zero():
    """``scripts/run_video.py --help`` exits with code 0 and shows usage."""
    script_path = str(_PROJECT_ROOT / "scripts" / "run_video.py")
    result = subprocess.run(
        [sys.executable, script_path, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI --help failed:\n{result.stderr}"
    assert "input" in result.stdout.lower()
    assert "output" in result.stdout.lower()
