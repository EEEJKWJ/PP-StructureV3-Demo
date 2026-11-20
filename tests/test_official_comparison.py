import unittest

from scripts.compare_official_table_engines import (
    aggregate_engine,
    block_metrics,
    paired_summary,
    select_table_block,
)
from scripts.run_table_preprocess_experiment import (
    _aggregate_variant,
    _attach_table_statistics,
    _identity_transform,
    _map_cells_to_page,
    _resolve_table_bbox,
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

    def test_preprocess_variant_cell_mapping_and_aggregate(self):
        crop_info = {"crop_bbox": [10, 20, 110, 120]}
        transform = _identity_transform(100, 100)
        cells = _map_cells_to_page(
            [{
                "row": 0,
                "col": 0,
                "text": "A",
                "bbox": [0, 0, 20, 10],
            }],
            crop_info,
            transform,
        )
        self.assertEqual(cells[0]["bbox"], [10, 20, 30, 30])

        records = [{
            "variants": {
                "A": {
                    "status": "success",
                    "teds": 0.5,
                    "teds_s": 0.6,
                    "text_recall": 0.4,
                    "mean_best_iou": 0.7,
                    "match_rate_iou_0_5": 0.8,
                    "cer": 0.2,
                    "wer": 0.3,
                    "elapsed_ms": 100,
                },
            },
        }]
        aggregate = _aggregate_variant(records, "A")
        self.assertEqual(aggregate["success_count"], 1)
        self.assertEqual(aggregate["teds"], 0.5)
        self.assertEqual(aggregate["elapsed_p50_ms"], 100)

    def test_table_bbox_prefers_official_block_near_evaluation_truth(self):
        page = {
            "blocks": [
                {
                    "label": "table",
                    "bbox": [0, 0, 20, 20],
                },
                {
                    "label": "table",
                    "bbox": [40, 30, 90, 80],
                },
            ],
        }
        selected = _resolve_table_bbox(
            page,
            [42, 32, 88, 78],
            "unused.jpg",
            lambda _: [],
            False,
        )

        self.assertEqual(selected, [40, 30, 90, 80])

    def test_merged_cell_structure_is_preserved_after_text_update(self):
        official = [
            {
                "row": 0,
                "col": 0,
                "rowspan": 1,
                "colspan": 2,
                "bbox": [0, 0, 100, 20],
                "text": "old",
            },
            {
                "row": 1,
                "col": 0,
                "rowspan": 1,
                "colspan": 1,
                "bbox": [0, 20, 50, 40],
                "text": "A",
            },
        ]
        fused = [dict(cell, text="new") for cell in official]
        result = _attach_table_statistics(
            {
                "status": "success",
                "cells": fused,
                "teds": 0.8,
            },
            {"cells": [{**official[0], "text": "truth"}]},
            official_cells=official,
        )

        self.assertTrue(result["structure_preserved"])
        self.assertEqual(result["merged_cell_count"], 1)
        self.assertEqual(result["over_segmentation_rate"], 1.0)
        self.assertEqual(result["teds_raw"], 0.8)


if __name__ == "__main__":
    unittest.main()
