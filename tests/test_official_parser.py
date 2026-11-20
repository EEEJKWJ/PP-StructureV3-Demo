import copy
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import server
from app.services.official_parser import (
    OfficialPPStructureParser,
    apply_custom_table_results,
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

    def test_custom_table_results_replace_official_table(self):
        page = normalize_official_result(official_payload())

        def Recognizer(image_path, bbox):
            self.assertEqual(image_path, "page.png")
            self.assertEqual(bbox, [5, 30, 95, 65])
            return {
                "cells": [{
                    "row": 0,
                    "col": 0,
                    "text": "custom",
                    "bbox": [8, 35, 40, 50],
                }],
                "html": "<table><tr><td>custom</td></tr></table>",
                "model": {"id": "custom-table"},
                "elapsed_ms": 12.5,
            }

        result = apply_custom_table_results(
            page,
            "page.png",
            Recognizer,
        )

        table = result["blocks"][2]
        self.assertEqual(table["table_source"], "custom")
        self.assertEqual(table["cells"][0]["text"], "custom")
        self.assertEqual(table["table_model"]["id"], "custom-table")
        self.assertEqual(
            result["structured"]["content"][2]["table_source"],
            "custom",
        )
        self.assertEqual(
            result["official_context"]["custom_table_count"],
            1,
        )

    def test_custom_table_uses_layout_fallback_when_official_misses(self):
        payload = copy.deepcopy(official_payload())
        payload["res"]["parsing_res_list"] = [
            block
            for block in payload["res"]["parsing_res_list"]
            if block["block_label"] != "table"
        ]
        payload["res"]["table_res_list"] = []
        page = normalize_official_result(payload)

        def Fallback(image_path):
            self.assertEqual(image_path, "page.png")
            return [{
                "label": "table",
                "bbox": [0, 0, 100, 80],
                "score": 0.88,
            }]

        def Recognizer(image_path, bbox):
            self.assertEqual(bbox, [0, 0, 100, 80])
            return {
                "cells": [{
                    "row": 0,
                    "col": 0,
                    "text": "fallback",
                    "bbox": [1, 1, 20, 20],
                }],
                "html": "<table><tr><td>fallback</td></tr></table>",
                "model": {"id": "custom"},
            }

        result = apply_custom_table_results(
            page,
            "page.png",
            Recognizer,
            Fallback,
        )

        table = next(
            block
            for block in result["blocks"]
            if block["label"] == "table"
        )
        self.assertTrue(table["table_fallback_detection"])
        self.assertEqual(table["table_source"], "custom")
        self.assertEqual(
            result["official_context"]["fallback_table_count"],
            1,
        )

    def test_full_image_table_fallback_is_explicit(self):
        payload = copy.deepcopy(official_payload())
        payload["res"]["parsing_res_list"] = [
            block
            for block in payload["res"]["parsing_res_list"]
            if block["block_label"] != "table"
        ]
        payload["res"]["table_res_list"] = []
        page = normalize_official_result(payload)

        def Recognizer(image_path, bbox):
            self.assertEqual(bbox, [0, 0, 100, 80])
            return {
                "cells": [{
                    "row": 0,
                    "col": 0,
                    "text": "full",
                    "bbox": [1, 1, 20, 20],
                }],
                "html": "<table><tr><td>full</td></tr></table>",
                "model": {"id": "custom"},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "table.png"
            Image.new("RGB", (100, 80), "white").save(image_path)
            result = apply_custom_table_results(
                page,
                str(image_path),
                Recognizer,
                lambda path: [],
                True,
            )

        table = next(
            block
            for block in result["blocks"]
            if block["label"] == "table"
        )
        self.assertTrue(table["full_image_fallback"])
        self.assertEqual(
            result["official_context"][
                "full_image_fallback_count"
            ],
            1,
        )


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

    def test_parse_pages_reuses_loaded_engine(self):
        class Result:
            json = official_payload()

        class Engine:
            def __init__(self):
                self.calls = []

            def predict(self, image_path, **options):
                self.calls.append(image_path)
                return [Result()]

        engine = Engine()
        parser = OfficialPPStructureParser(
            logging.getLogger("test-official-parser"),
            engine_factory=lambda: engine,
        )

        pages = parser.parse_pages(["page1.png", "page2.png"])

        self.assertEqual([page["page"] for page in pages], [1, 2])
        self.assertEqual(engine.calls, ["page1.png", "page2.png"])

    def test_parse_page_with_image_returns_corrected_page(self):
        corrected_image = object()

        class Result:
            json = official_payload()

            def get(self, key, default=None):
                if key == "doc_preprocessor_res":
                    return {"output_img": corrected_image}
                return default

        class Engine:
            def predict(self, image_path, **options):
                return [Result()]

        parser = OfficialPPStructureParser(
            logging.getLogger("test-official-parser"),
            engine_factory=lambda: Engine(),
        )

        page, image = parser.parse_page_with_image("page.png")

        self.assertEqual(page["page"], 1)
        self.assertIs(image, corrected_image)


class OfficialParserApiTests(unittest.TestCase):
    def test_official_endpoint_returns_normalized_page(self):
        class FakeParser:
            def parse_page(
                self,
                image_path,
                page_num,
                is_pdf,
                predict_options,
            ):
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

    def test_official_all_endpoint_returns_compatible_pages(self):
        class FakeParser:
            def parse_pages(
                self,
                image_paths,
                is_pdf,
                predict_options,
            ):
                self.args = (image_paths, is_pdf)
                return [
                    normalize_official_result(
                        official_payload(),
                        page_num=index,
                        is_pdf=is_pdf,
                    )
                    for index in range(1, len(image_paths) + 1)
                ]

        parser = FakeParser()
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir)
            doc_dir = upload_dir / "doc"
            doc_dir.mkdir()
            for page_num in (1, 2):
                Image.new("RGB", (100, 80), "white").save(
                    doc_dir / f"page_{page_num:03d}.png"
                )
            with (
                patch.object(server, "UPLOAD_DIR", upload_dir),
                patch.object(server, "_official_parser", parser),
            ):
                response = TestClient(server.app).post(
                    "/api/official/detect-all/doc"
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["pages"]), 2)
        self.assertEqual(payload["engine"]["id"], "official_v3")
        self.assertFalse(
            payload["engine"]["cross_page_table_merge"]
        )

    def test_detect_and_text_source_follow_engine_switch(self):
        class FakeParser:
            def parse_page(
                self,
                image_path,
                page_num,
                is_pdf,
                predict_options,
            ):
                return normalize_official_result(
                    official_payload(),
                    page_num=page_num,
                    is_pdf=is_pdf,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir)
            doc_dir = upload_dir / "doc"
            doc_dir.mkdir()
            Image.new("RGB", (100, 80), "white").save(
                doc_dir / "page_001.png"
            )
            with (
                patch.object(server, "UPLOAD_DIR", upload_dir),
                patch.object(server, "_official_parser", FakeParser()),
                patch.dict(
                    os.environ,
                    {"PP_DOCUMENT_ENGINE": "official_v3"},
                ),
            ):
                client = TestClient(server.app)
                source = client.get("/api/text-source/doc/1")
                detected = client.post("/api/detect/doc/1")
                status = client.get("/api/model-status")

        self.assertEqual(
            source.json()["text_source"],
            "official_v3",
        )
        self.assertEqual(detected.status_code, 200)
        self.assertEqual(detected.json()["engine"]["id"], "official_v3")
        self.assertEqual(
            status.json()["document_engine"],
            "official_v3",
        )

    def test_custom_table_engine_replaces_official_table(self):
        class FakeParser:
            def parse_page(
                self,
                image_path,
                page_num,
                is_pdf,
                predict_options,
            ):
                self.predict_options = predict_options
                return normalize_official_result(
                    official_payload(),
                    page_num=page_num,
                    is_pdf=is_pdf,
                )

        parser = FakeParser()
        table_result = {
            "cells": [{
                "row": 0,
                "col": 0,
                "text": "custom",
                "bbox": [8, 35, 40, 50],
            }],
            "html": "<table><tr><td>custom</td></tr></table>",
            "model": {"id": "pp-tablemagic-v2"},
            "elapsed_ms": 8.0,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir)
            doc_dir = upload_dir / "doc"
            doc_dir.mkdir()
            Image.new("RGB", (100, 80), "white").save(
                doc_dir / "page_001.png"
            )
            with (
                patch.object(server, "UPLOAD_DIR", upload_dir),
                patch.object(
                    server,
                    "_official_custom_parser",
                    parser,
                ),
                patch.object(
                    server._table_service,
                    "recognize",
                    return_value=table_result,
                ),
                patch.dict(
                    os.environ,
                    {
                        "PP_DOCUMENT_ENGINE": (
                            "official_v3_custom_table"
                        ),
                    },
                ),
            ):
                client = TestClient(server.app)
                response = client.post("/api/detect/doc/1")
                status = client.get("/api/model-status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(
            parser.predict_options["use_table_recognition"]
        )
        self.assertEqual(
            payload["blocks"][2]["table_source"],
            "custom",
        )
        self.assertEqual(
            status.json()["table_engine"],
            "custom",
        )
        self.assertEqual(
            status.json()["document_engine"],
            "official_v3_custom_table",
        )


if __name__ == "__main__":
    unittest.main()
