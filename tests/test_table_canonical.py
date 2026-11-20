import unittest

from app.evaluation.table_canonical import (
    canonical_metrics,
    canonicalize_cells,
    canonicalize_html,
    canonicalize_table_text,
)


class TableCanonicalTests(unittest.TestCase):
    def test_dot_leaders_and_spaces_are_normalized(self):
        self.assertEqual(
            canonicalize_table_text("  Revenue ....... 2024  "),
            "Revenue 2024",
        )

    def test_html_normalization_preserves_structure(self):
        normalized = canonicalize_html(
            '<table><tr><td colspan="2">Total....... 100</td></tr></table>'
        )

        self.assertIn('colspan="2"', normalized)
        self.assertIn(">Total 100<", normalized)

    def test_canonical_metrics_ignore_dot_leaders(self):
        truth = {
            "html": "<table><tr><td>Revenue.......</td><td>100</td></tr></table>",
            "cells": [
                {
                    "row": 0,
                    "col": 0,
                    "rowspan": 1,
                    "colspan": 1,
                    "bbox": [0, 0, 50, 20],
                    "text": "Revenue.......",
                },
                {
                    "row": 0,
                    "col": 1,
                    "rowspan": 1,
                    "colspan": 1,
                    "bbox": [50, 0, 100, 20],
                    "text": "100",
                },
            ],
        }
        prediction = canonicalize_cells(truth["cells"])
        prediction[0]["text"] = "Revenue"

        metrics = canonical_metrics(
            "<table><tr><td>Revenue</td><td>100</td></tr></table>",
            prediction,
            truth,
        )

        self.assertEqual(metrics["teds_canonical"], 1.0)
        self.assertEqual(metrics["text_recall_canonical"], 1.0)


if __name__ == "__main__":
    unittest.main()
