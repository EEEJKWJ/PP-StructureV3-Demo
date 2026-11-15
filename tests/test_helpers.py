import asyncio
import logging
import tempfile
import unittest
from pathlib import Path

import fitz
import server
from app.services.document import DocumentService
from app.services.layout import LayoutService
from app.services.ocr import OCRService
from app.services.table import TableService


class TableParserTests(unittest.TestCase):
    def test_parse_html_table_preserves_row_col_and_offset(self):
        html = (
            "<table>"
            "<tr><td>名称</td><td>数量</td></tr>"
            "<tr><td>苹果</td><td>2</td></tr>"
            "</table>"
        )
        cell_boxes = [[
            [0, 0, 10, 0, 10, 5, 0, 5],
            [10, 0, 20, 0, 20, 5, 10, 5],
            [0, 5, 10, 5, 10, 10, 0, 10],
            [10, 5, 20, 5, 20, 10, 10, 10],
        ]]

        cells = TableService.parse_html_table(html, cell_boxes, 100, 200)

        self.assertEqual([(c["row"], c["col"], c["text"]) for c in cells], [
            (0, 0, "名称"),
            (0, 1, "数量"),
            (1, 0, "苹果"),
            (1, 1, "2"),
        ])
        self.assertEqual(cells[0]["bbox"], [100, 200, 110, 205])
        self.assertEqual(cells[3]["bbox"], [110, 205, 120, 210])

    def test_table_to_grid(self):
        cells = [
            {"row": 0, "col": 0, "text": "A"},
            {"row": 0, "col": 1, "text": "B"},
            {"row": 1, "col": 0, "text": "1"},
            {"row": 1, "col": 1, "text": "2"},
        ]

        grid, max_row, max_col = server._table_to_grid(cells)

        self.assertEqual(grid, [["A", "B"], ["1", "2"]])
        self.assertEqual((max_row, max_col), (1, 1))

    def test_cells_to_html_escapes_text(self):
        cells = [
            {"row": 0, "col": 0, "text": "<名称>", "bbox": [0, 0, 1, 1]},
            {"row": 0, "col": 1, "text": "A&B", "bbox": [1, 0, 2, 1]},
        ]

        table_html = TableService.cells_to_html(cells)

        self.assertIn("&lt;名称&gt;", table_html)
        self.assertIn("A&amp;B", table_html)
        self.assertTrue(table_html.startswith("<table>"))


class TextAndTocTests(unittest.TestCase):
    def test_dedupe_repeated_chars(self):
        self.assertEqual(
            server._dedupe_repeated_chars("第二二节牙本本质"),
            "第二节牙本质",
        )

    def test_parse_toc_line_with_page(self):
        parsed = server._parse_toc_line("第一章 牙体组织............12")
        # 当前基线会清理标题中的空格，后续重构必须显式保持或升级该行为。
        self.assertEqual(parsed["title"], "第一章牙体组织")
        self.assertEqual(parsed["page"], 12)

    def test_is_toc_block(self):
        texts = [
            "第一章 牙体组织............1",
            "第二章 牙周组织............8",
            "第三章 口腔黏膜............20",
        ]
        self.assertTrue(server._is_toc_block(texts))

    def test_garbage_text_detection(self):
        self.assertTrue(server.is_garbage_text("\ufffd" * 20))
        self.assertFalse(server.is_garbage_text("第一章 牙体组织"))


class StructureTests(unittest.TestCase):
    def test_apply_table_result_adds_html(self):
        block = {}
        cells = [
            {"row": 0, "col": 0, "text": "名称", "bbox": [10, 10, 50, 30]},
            {"row": 0, "col": 1, "text": "数量", "bbox": [50, 10, 90, 30]},
        ]

        server._apply_table_result(block, cells)

        self.assertEqual(block["cells"], cells)
        self.assertIn("<td>名称</td>", block["table_html"])

    def test_build_structured_keeps_current_table_shape(self):
        block = {
            "label": "table",
            "cells": [
                {"row": 0, "col": 0, "text": "名称", "bbox": [10, 10, 50, 30]},
                {"row": 0, "col": 1, "text": "数量", "bbox": [50, 10, 90, 30]},
                {"row": 1, "col": 0, "text": "苹果", "bbox": [10, 30, 50, 50]},
                {"row": 1, "col": 1, "text": "2", "bbox": [50, 30, 90, 50]},
            ],
        }

        structured = server.build_structured([block], None, 1, "", False, {})

        self.assertEqual(structured["page"], 1)
        self.assertEqual(structured["content"][0]["type"], "table")
        self.assertEqual(structured["content"][0]["headers"], ["名称", "数量"])
        self.assertEqual(structured["content"][0]["rows"], [{"名称": "苹果", "数量": "2"}])
        self.assertIn("<td>苹果</td>", structured["content"][0]["html"])


class LayoutAndOcrServiceTests(unittest.TestCase):
    def test_reading_order_groups_nearby_rows_and_sorts_columns(self):
        blocks = [
            {"bbox": [210, 5, 300, 20], "label": "text"},
            {"bbox": [100, 100, 200, 120], "label": "table"},
            {"bbox": [10, 0, 90, 20], "label": "paragraph_title"},
        ]

        ordered = LayoutService.sort_reading_order(blocks, y_tolerance=30)

        self.assertEqual(
            [block["label"] for block in ordered],
            ["paragraph_title", "text", "table"],
        )

    def test_layout_iou_matches_existing_geometry(self):
        self.assertEqual(LayoutService.iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(LayoutService.iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)
        self.assertAlmostEqual(
            LayoutService.iou([0, 0, 10, 10], [5, 0, 15, 10]),
            1 / 3,
        )

    def test_ocr_nms_prefers_longer_overlapping_text(self):
        lines = [
            {"text": "第一章", "bbox": [0, 0, 100, 20]},
            {"text": "第一章牙体组织", "bbox": [0, 0, 100, 20]},
            {"text": "第二章", "bbox": [0, 40, 100, 60]},
        ]

        kept = OCRService.nms_lines(lines)

        self.assertEqual(
            [line["text"] for line in kept],
            ["第一章牙体组织", "第二章"],
        )

    def test_ocr_merge_removes_suffix_prefix_overlap(self):
        lines = [
            {"text": "牙体组织", "bbox": [0, 0, 100, 20]},
            {"text": "织的构成", "bbox": [90, 1, 190, 21]},
        ]

        merged = OCRService.merge_suffix_prefix_lines(lines)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "牙体组织的构成")
        self.assertEqual(merged[0]["bbox"], [0, 0, 190, 21])

    def test_ocr_status_reports_server_master_then_mobile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            service = OCRService(
                logging.getLogger("test-ocr"),
                model_dir=model_dir,
            )
            self.assertEqual(
                service.status(),
                {"available": False, "model": ""},
            )

            (model_dir / "ppocr5_mobile_det.onnx").touch()
            self.assertEqual(
                service.status(),
                {"available": True, "model": "mobile"},
            )

            (model_dir / "ppocr5_server_det.onnx").touch()
            self.assertEqual(
                service.status(),
                {"available": True, "model": "server"},
            )

    def test_model_status_keeps_existing_contract(self):
        status = asyncio.run(server.model_status())

        self.assertEqual(
            set(status),
            {
                "status",
                "error",
                "ocr_available",
                "ocr_model",
                "gpu",
                "gpu_name",
                "table_engine",
            },
        )
        self.assertIn(status["status"], {"loading", "ready", "error"})
        self.assertIsInstance(status["ocr_available"], bool)
        self.assertIsInstance(status["gpu"], bool)


class DocumentServiceTests(unittest.TestCase):
    def test_save_render_and_extract_text(self):
        service = DocumentService(
            logging.getLogger("test-document"),
            garbage_detector=lambda text: False,
        )
        document = fitz.open()
        page = document.new_page(width=200, height=200)
        page.insert_text((20, 40), "Hello table")
        pdf_bytes = document.tobytes()
        document.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            doc_dir = Path(temp_dir)
            page_count = service.save_pdf(pdf_bytes, doc_dir)
            image_path = service.ensure_page_image(doc_dir, 1)
            text = service.extract_text_from_pdf(
                str(doc_dir / "source.pdf"),
                1,
                [0, 0, 400, 400],
            )
            has_text = service.page_has_text(
                str(doc_dir / "source.pdf"),
                1,
            )

        self.assertEqual(page_count, 1)
        self.assertTrue(image_path.name == "page_001.png")
        self.assertIn("Hello table", text)
        self.assertTrue(has_text)


if __name__ == "__main__":
    unittest.main()
