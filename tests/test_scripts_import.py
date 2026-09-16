import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(rel: str) -> None:
    try:
        runpy.run_path(str(ROOT / rel), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise


class ScriptSmokeTests(unittest.TestCase):
    def test_attention_script(self) -> None:
        _run("examples/04_attention_replay.py")

    def test_reprint_script(self) -> None:
        _run("examples/07_reprint_scores.py")


if __name__ == "__main__":
    unittest.main()
