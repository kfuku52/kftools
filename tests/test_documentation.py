"""Keep the runnable user guides executable without copying their examples."""

import re
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
GUIDES = ("examples.md", "file-formats.md")


@pytest.mark.parametrize("guide", GUIDES)
def test_documented_python_examples(guide, tmp_path, monkeypatch):
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / "docs" / guide).read_text(), re.DOTALL)
    assert blocks, f"No runnable examples found in {guide}"
    monkeypatch.chdir(tmp_path)
    for number, block in enumerate(blocks, 1):
        exec(compile(block, f"{guide}:block-{number}", "exec"), {})
    if guide == "examples.md":
        assert {path.name for path in tmp_path.iterdir()} == {"kftools-example.png"}
        pixels = mpimg.imread(tmp_path / "kftools-example.png")
        assert pixels.shape[:2] == (600, 1350)
        assert pixels[:, :, :3].min() < pixels[:, :, :3].max()
    else:
        assert not list(tmp_path.iterdir())
