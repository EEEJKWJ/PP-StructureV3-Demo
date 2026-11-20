import logging
import unittest

import numpy as np

from app.services.ocr import OCRService
from app.services.table_preprocess import (
    crop_table_from_image,
    enhance_table_crop,
    page_bbox_to_processed,
    processed_bbox_to_page,
)
from app.services.table_text_fusion import (
    build_fused_table,
    fuse_table_text,
)


class TablePreprocessTests(unittest.TestCase):
    def test_crop_and_affine_coordinates_are_reversible(self):
        image = np.full((80, 100, 3), 255, dtype=np.uint8)
        crop, crop_info = crop_table_from_image(
            image,
            [20, 10, 80, 60],
            padding=5,
        )
        enhanced, transform = enhance_table_crop(
            crop,
            target_long_side=120,
            deskew=False,
        )
        processed = page_bbox_to_processed(
            [20, 10, 80, 60],
            crop_info,
            transform,
        )
        restored = processed_bbox_to_page(
            processed,
            crop_info,
            transform,
        )

        self.assertEqual(crop_info["crop_bbox"], [15, 5, 85, 65])
        self.assertGreater(enhanced.shape[1], crop.shape[1])
        for actual, expected in zip(restored, [20, 10, 80, 60]):
            self.assertLessEqual(abs(actual - expected), 1)

    def test_fusion_preserves_official_cells_and_spans(self):
        block = {
            "cells": [
                {
                    "row": 0,
                    "col": 0,
                    "rowspan": 1,
                    "colspan": 2,
                    "bbox": [20, 10, 80, 30],
                    "text": "old",
                },
                {
                    "row": 1,
                    "col": 0,
                    "rowspan": 1,
                    "colspan": 1,
                    "bbox": [20, 30, 50, 50],
                    "text": "old",
                },
                {
                    "row": 1,
                    "col": 1,
                    "rowspan": 1,
                    "colspan": 1,
                    "bbox": [50, 30, 80, 50],
                    "text": "old",
                },
            ],
            "table_html": "<table>old</table>",
            "table_model": {"id": "official"},
            "cell_bbox_available": True,
        }
        crop_info = {
            "crop_bbox": [15, 5, 85, 65],
        }
        transform = {
            "matrix": [[2, 0, 0], [0, 2, 0]],
            "inverse_matrix": [[0.5, 0, 0], [0, 0.5, 0]],
        }
        ocr_lines = [
            {"text": "A", "bbox": [20, 20, 100, 40]},
            {"text": "B", "bbox": [20, 60, 60, 80]},
            {"text": "C", "bbox": [70, 60, 100, 80]},
        ]

        fused = fuse_table_text(
            block,
            ocr_lines,
            crop_info,
            transform,
        )

        cells, assignment = fused
        self.assertEqual(cells[0]["colspan"], 2)
        self.assertEqual(cells[0]["text"], "A")
        self.assertEqual(cells[1]["text"], "B")
        self.assertEqual(cells[2]["text"], "C")
        self.assertEqual(assignment["assigned_line_count"], 3)
        result = build_fused_table(
            block,
            ocr_lines,
            crop_info,
            transform,
        )
        self.assertIn('colspan="2"', result["html"])
        self.assertIn('<td colspan="2">A</td>', result["html"])

    def test_ocr_service_recognize_crop_returns_local_boxes(self):
        class Engine:
            results = [{
                "text": "value",
                "rec_pos": np.array([
                    [1, 2],
                    [30, 2],
                    [30, 12],
                    [1, 12],
                ]),
            }]

            def run(self, image):
                self.image = image

        service = OCRService(logging.getLogger("test-ocr"))
        service.engine = Engine()
        lines = service.recognize_crop(
            np.zeros((20, 40, 3), dtype=np.uint8)
        )

        self.assertEqual(lines[0]["text"], "value")
        self.assertEqual(lines[0]["bbox"], [1, 2, 30, 12])


if __name__ == "__main__":
    unittest.main()
