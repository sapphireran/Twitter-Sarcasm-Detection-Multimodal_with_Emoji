import unittest

from ccs2lab.recorded import RECORDED_ROWS, markdown_table, rows_for


class RecordedTests(unittest.TestCase):
    def test_has_bilstm_we_test(self) -> None:
        rows = rows_for(model="BiLSTM+Attn", split="test")
        we = next(row for row in rows if row.modality == "WE")
        self.assertAlmostEqual(we.accuracy, 0.8734999895095825)

    def test_markdown_contains_header(self) -> None:
        table = markdown_table()
        self.assertIn("| Model |", table)
        self.assertGreaterEqual(len(RECORDED_ROWS), 20)


if __name__ == "__main__":
    unittest.main()
