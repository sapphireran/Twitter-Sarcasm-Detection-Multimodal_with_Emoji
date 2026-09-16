import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScriptSmokeTests(unittest.TestCase):
    def test_attention_script(self) -> None:
        runpy.run_path(str(ROOT / "examples" / "04_attention_replay.py"), run_name="__main__")

    def test_reprint_script(self) -> None:
        runpy.run_path(str(ROOT / "examples" / "07_reprint_scores.py"), run_name="__main__")


if __name__ == "__main__":
    unittest.main()
