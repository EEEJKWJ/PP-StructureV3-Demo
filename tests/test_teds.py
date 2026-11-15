import unittest

from app.evaluation.teds import (
    calculate_html_metrics,
    normalize_text,
    parse_table_tree,
    tree_edit_distance,
)


class TEDSTests(unittest.TestCase):
    def test_identical_table_scores_one(self):
        html = (
            "<table><tr><td>A</td><td>B</td></tr>"
            "<tr><td>C</td><td>D</td></tr></table>"
        )

        metrics = calculate_html_metrics(html, html)

        self.assertEqual(metrics["teds"], 1)
        self.assertEqual(metrics["teds_s"], 1)
        self.assertEqual(metrics["status"], "available")

    def test_text_difference_only_reduces_teds(self):
        prediction = (
            "<table><tr><td>A</td><td>Wrong</td></tr></table>"
        )
        truth = "<table><tr><td>A</td><td>B</td></tr></table>"

        metrics = calculate_html_metrics(prediction, truth)

        self.assertLess(metrics["teds"], 1)
        self.assertEqual(metrics["teds_s"], 1)

    def test_colspan_difference_reduces_both_scores(self):
        prediction = (
            "<table><tr><td colspan='2'>A</td></tr></table>"
        )
        truth = (
            "<table><tr><td>A</td><td>B</td></tr></table>"
        )

        metrics = calculate_html_metrics(prediction, truth)

        self.assertLess(metrics["teds"], 1)
        self.assertLess(metrics["teds_s"], 1)

    def test_rowspan_difference_reduces_both_scores(self):
        prediction = (
            "<table><tr><td rowspan='2'>A</td><td>B</td></tr>"
            "<tr><td>C</td></tr></table>"
        )
        truth = (
            "<table><tr><td>A</td><td>B</td></tr>"
            "<tr><td>A</td><td>C</td></tr></table>"
        )

        metrics = calculate_html_metrics(prediction, truth)

        self.assertLess(metrics["teds"], 1)
        self.assertLess(metrics["teds_s"], 1)

    def test_over_segmentation_reduces_structure_score(self):
        prediction = (
            "<table><tr><td>A</td><td>B</td></tr></table>"
        )
        truth = "<table><tr><td colspan='2'>A B</td></tr></table>"

        metrics = calculate_html_metrics(prediction, truth)

        self.assertLess(metrics["teds_s"], 1)

    def test_row_order_difference_reduces_structure_score(self):
        prediction = (
            "<table><tr><td>C</td></tr>"
            "<tr><td>A</td><td>B</td></tr></table>"
        )
        truth = (
            "<table><tr><td>A</td><td>B</td></tr>"
            "<tr><td>C</td></tr></table>"
        )

        metrics = calculate_html_metrics(prediction, truth)

        self.assertLess(metrics["teds_s"], 1)

    def test_optional_wrappers_and_formatting_are_normalized(self):
        prediction = (
            "<html><body><table><thead><tr><th><b>A</b></th>"
            "<th>B</th></tr></thead></table></body></html>"
        )
        truth = "<table><tr><td>A</td><td>B</td></tr></table>"

        metrics = calculate_html_metrics(prediction, truth)

        self.assertEqual(metrics["teds"], 1)
        self.assertEqual(metrics["teds_s"], 1)

    def test_text_normalization_handles_case_unicode_and_whitespace(self):
        prediction = "<table><tr><td>  TABLE  </td></tr></table>"
        truth = "<table><tr><td>table</td></tr></table>"

        self.assertEqual(normalize_text("  ＴＡＢＬＥ  "), "table")
        self.assertEqual(
            calculate_html_metrics(prediction, truth)["teds"],
            1,
        )

    def test_missing_truth_and_prediction_have_explicit_status(self):
        missing_truth = calculate_html_metrics(
            "<table><tr><td>A</td></tr></table>",
            "",
        )
        missing_prediction = calculate_html_metrics(
            "",
            "<table><tr><td>A</td></tr></table>",
        )

        self.assertEqual(
            missing_truth["status"],
            "not_available_no_html_truth",
        )
        self.assertEqual(
            missing_prediction["status"],
            "missing_prediction_html",
        )

    def test_invalid_and_empty_tables_return_parse_error(self):
        no_table = calculate_html_metrics(
            "<div>no table</div>",
            "<table><tr><td>A</td></tr></table>",
        )
        empty_table = calculate_html_metrics(
            "<table></table>",
            "<table><tr><td>A</td></tr></table>",
        )

        self.assertEqual(no_table["status"], "parse_error")
        self.assertEqual(empty_table["status"], "parse_error")
        self.assertIsNone(no_table["teds"])

    def test_tree_edit_distance_is_symmetric_for_labels(self):
        first = parse_table_tree(
            "<table><tr><td>A</td><td>B</td></tr></table>"
        )
        second = parse_table_tree(
            "<table><tr><td colspan='2'>A</td></tr></table>"
        )

        self.assertEqual(
            tree_edit_distance(first, second, False),
            tree_edit_distance(second, first, False),
        )

    def test_br_is_normalized_to_space(self):
        prediction = (
            "<table><tr><td>A<br>B</td></tr></table>"
        )
        truth = "<table><tr><td>A B</td></tr></table>"

        self.assertEqual(
            calculate_html_metrics(prediction, truth)["teds"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
