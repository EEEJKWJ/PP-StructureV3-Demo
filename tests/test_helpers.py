import unittest

import server


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

        cells = server._parse_html_table(html, cell_boxes, 100, 200)

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


if __name__ == "__main__":
    unittest.main()
