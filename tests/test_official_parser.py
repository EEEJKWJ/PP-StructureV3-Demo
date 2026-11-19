import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import server
from app.services.official_parser import (
    OfficialPPStructureParser,
    normalize_official_result,
)


def official_payload(with_boxes=True):
    return {
        "res": {
            "page_index": 0,
            "page_count": 2,
            "width": 100,
            "height": 80,
            "parsing_res_list": [
                {
                    "block_label": "doc_title",
                    "block_content": "年度报告",
                    "block_bbox": [5, 5, 60, 15],
                    "block_id": "title-1",
                    "block_order": 0,
                },
                {
                    "block_label": "text",
                    "block_content": "正文内容",
                    "block_bbox": [5, 18, 90, 28],
                    "block_id": "text-1",
                    "block_order": 1,
                },
                {
                    "block_label": "table",
                    "block_content": "<table><tr><td>A</td></tr></table>",
                    "block_bbox": [5, 30, 95, 65],
                    "block_id": "table-1",
                    "block_order": 2,
                },
                {
                    "block_label": "formula",
                    "block_content": "x = 1",
                    "block_bbox": [5, 68, 30, 75],
                    "block_id": "formula-1",
                    "block_order": 3,
                },
            ],
            "table_res_list": [{
                "pred_html": "<table><tr><td>A</td></tr></table>",
                "cell_box_list": (
                    [[8, 35, 40, 50]] if with_boxes else []
                ),
                "table_ocr_pred": {
                    "rec_texts": ["A"],
                    "rec_boxes": [[8, 35, 40, 50]],
                },
            }],
            "formula_res_list": [{
                "rec_formula": "x=1",
            }],
            "layout_det_res": {
                "boxes": [{}, {}, {}],
            },
            "overall_ocr_res": {
                "rec_texts": ["年度报告", "正文内容"],
            },
        },
    }


class OfficialResultNormalizationTests(unittest.TestCase):
    def test_normalizes_blocks_table_and_structured_content(self):
        result = normalize_official_result(
            official_payload(),
            page_num=1,
            is_pdf=True,
        )

        self.assertEqual(result["page"], 1)
        self.assertEqual(result["width"], 100)
        self.assertEqual(len(result["blocks"]), 4)
        table = result["blocks"][2]
        self.assertEqual(table["label"], "table")
        self.assertEqual(table["table_html"], (
            "<table><tr><td>A</td></tr></table>"
        ))
        self.assertTrue(table["cell_bbox_available"])
        self.assertEqual(table["cells"][0]["bbox"], [8, 35, 40, 50])
        self.assertEqual(
            [item["type"] for item in result["structured"]["content"]],
            ["title", "text", "table", "formula"],
        )
        self.assertEqual(
            result["official_context"],
            {
                "layout_block_count": 3,
                "ocr_text_count": 2,
                "table_count": 1,
                "formula_count": 1,
            },
        )
        self.assertIn("official_result", result)

    def test_missing_cell_boxes_are_not_fabricated(self):
        result = normalize_official_result(
            official_payload(with_boxes=False),
        )

        table = result["blocks"][2]
        self.assertFalse(table["cell_bbox_available"])
        self.assertNotIn("bbox", table["cells"][0])
        self.assertEqual(
            result["structured"]["content"][2]["capabilities"],
            {"cell_bbox": False},
        )

    def test_invalid_result_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_official_result([])


class OfficialParserServiceTests(unittest.TestCase):
    def test_engine_is_loaded_lazily_once(self):
        class Result:
            json = official_payload()

        class Engine:
            def __init__(self):
                self.calls = []

            def predict(self, image_path, **options):
                self.calls.append((image_path, options))
                return [Result()]

        engine = Engine()
        factory_calls = []

        def factory():
            factory_calls.append(True)
            return engine

        parser = OfficialPPStructureParser(
            logging.getLogger("test-official-parser"),
            engine_factory=factory,
        )
        first = parser.parse_page(
            "page.png",
            page_num=1,
            predict_options={"format_block_content": False},
        )
        second = parser.parse_page("page.png", page_num=1)

        self.assertEqual(first["page"], 1)
        self.assertEqual(second["page"], 1)
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(
            engine.calls[0][1],
            {"format_block_content": False},
        )


class OfficialParserApiTests(unittest.TestCase):
    def test_official_endpoint_returns_normalized_page(self):
        class FakeParser:
            def parse_page(self, image_path, page_num, is_pdf):
                self.args = (image_path, page_num, is_pdf)
                return normalize_official_result(
                    official_payload(),
                    page_num=page_num,
                    is_pdf=is_pdf,
                )

        parser = FakeParser()
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir)
            doc_dir = upload_dir / "doc"
            doc_dir.mkdir()
            Image.new("RGB", (100, 80), "white").save(
                doc_dir / "page_001.png"
            )
            (doc_dir / "source.pdf").write_bytes(b"pdf")
            with (
                patch.object(server, "UPLOAD_DIR", upload_dir),
                patch.object(server, "_official_parser", parser),
            ):
                response = TestClient(server.app).post(
                    "/api/official/detect/doc/1"
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["engine"]["id"], "official_v3")
        self.assertTrue(payload["is_pdf"])
        self.assertEqual(parser.args[1:], (1, True))


if __name__ == "__main__":
    unittest.main()
