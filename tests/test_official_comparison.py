import unittest

from scripts.compare_official_table_engines import (
    aggregate_engine,
    block_metrics,
    paired_summary,
    select_table_block,
)


class OfficialTableComparisonTests(unittest.TestCase):
    def test_select_table_block_uses_best_bbox_iou(self):
        page = {
            "blocks": [
                {
                    "label": "table",
                    "bbox": [0, 0, 10, 10],
                },
                {
                    "label": "table",
                    "bbox": [20, 20, 80, 80],
                },
                {
                    "label": "text",
                    "bbox": [20, 20, 80, 80],
                },
            ],
        }

        block = select_table_block(page, [20, 20, 80, 80])

        self.assertEqual(block["bbox"], [20, 20, 80, 80])

    def test_block_metrics_handles_missing_table(self):
        metrics = block_metrics(None, {"cells": [], "html": ""})

        self.assertEqual(metrics["status"], "no_table_block")
        self.assertIsNone(metrics["teds"])
        self.assertIsNone(metrics["text_recall"])

    def test_paired_summary_and_aggregate(self):
        results = [
            {
                "official": {
                    "status": "success",
                    "teds": 0.8,
                    "teds_s": 0.9,
                    "text_recall": 0.4,
                    "mean_best_iou": 0.5,
                    "match_rate_iou_0_5": 0.5,
                    "cer": 0.2,
                    "wer": 0.2,
                    "elapsed_ms": 100,
                    "fallback_detection": False,
                },
                "custom": {
                    "status": "success",
                    "teds": 0.7,
                    "teds_s": 0.8,
                    "text_recall": 0.6,
                    "mean_best_iou": 0.7,
                    "match_rate_iou_0_5": 0.7,
                    "cer": 0.1,
                    "wer": 0.1,
                    "injection_ms": 50,
                    "fallback_detection": False,
                },
            },
            {
                "official": {
                    "status": "no_table_block",
                    "teds": None,
                    "teds_s": None,
                    "text_recall": None,
                    "mean_best_iou": None,
                    "match_rate_iou_0_5": None,
                    "cer": None,
                    "wer": None,
                    "elapsed_ms": 200,
                    "fallback_detection": False,
                },
                "custom": {
                    "status": "success",
                    "teds": 0.5,
                    "teds_s": 0.6,
                    "text_recall": 0.3,
                    "mean_best_iou": 0.4,
                    "match_rate_iou_0_5": 0.4,
                    "cer": 0.3,
                    "wer": 0.3,
                    "injection_ms": 60,
                    "fallback_detection": True,
                },
            },
        ]

        paired = paired_summary(results)
        official = aggregate_engine(results, "official")
        custom = aggregate_engine(results, "custom")

        self.assertEqual(paired["both_success"], 1)
        self.assertEqual(paired["custom_only_success"], 1)
        self.assertEqual(official["success_rate"], 0.5)
        self.assertEqual(custom["success_rate"], 1.0)
        self.assertEqual(custom["fallback_count"], 1)
        self.assertEqual(custom["injection_p50_ms"], 50)


if __name__ == "__main__":
    unittest.main()
