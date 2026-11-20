import unittest

import numpy as np

from app.services.page_alignment import (
    align_ground_truth,
    aligned_bbox_to_page,
    estimate_page_alignment,
    page_bbox_to_aligned,
)


class PageAlignmentTests(unittest.TestCase):
    def test_identical_pages_use_identity(self):
        image = np.full((80, 100, 3), 255, dtype=np.uint8)
        alignment = estimate_page_alignment(image, image.copy())

        self.assertEqual(alignment["status"], "available")
        self.assertEqual(alignment["method"], "identity")
        self.assertEqual(
            page_bbox_to_aligned([10, 20, 40, 50], alignment),
            [10, 20, 40, 50],
        )

    def test_ground_truth_mapping_round_trip(self):
        matrix = np.array(
            [[1.2, 0.0, 7.0], [0.0, 1.2, 4.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        alignment = {
            "status": "available",
            "matrix": matrix.tolist(),
            "inverse_matrix": np.linalg.inv(matrix).tolist(),
            "source_width": 100,
            "source_height": 80,
            "target_width": 127,
            "target_height": 100,
        }
        truth = {
            "table_bbox": [10, 10, 80, 60],
            "cells": [{
                "row": 0,
                "col": 0,
                "rowspan": 1,
                "colspan": 1,
                "bbox": [10, 10, 40, 30],
                "text": "A",
            }],
        }

        mapped = align_ground_truth(truth, alignment)
        restored = aligned_bbox_to_page(mapped["cells"][0]["bbox"], alignment)

        self.assertEqual(restored, [10, 10, 40, 30])
        self.assertEqual(mapped["cells"][0]["text"], "A")
        self.assertNotEqual(mapped["cells"][0]["bbox"], [10, 10, 40, 30])

    def test_failed_alignment_is_explicit_for_blank_different_pages(self):
        source = np.zeros((40, 50, 3), dtype=np.uint8)
        target = np.full((40, 50, 3), 255, dtype=np.uint8)

        alignment = estimate_page_alignment(source, target)

        self.assertEqual(alignment["status"], "failed")
        self.assertEqual(alignment["method"], "none")


if __name__ == "__main__":
    unittest.main()
