"""目录、页面结构化与跨页表格合并服务。"""

import re


class StructureService:
    """把版面 block 转换为稳定 JSON，并合并跨页续表。"""

    _REPEATED_CHAR_WHITELIST = {
        "看看",
        "好好",
        "慢慢",
        "常常",
        "刚刚",
        "渐渐",
        "个个",
        "早早",
        "远远",
        "静静",
        "暗暗",
        "默默",
        "偷偷",
        "种种",
        "处处",
        "时时",
        "天天",
        "年年",
        "月月",
        "日日",
    }

    def __init__(self, logger, table_service, text_extractor=None):
        self.log = logger
        self.table_service = table_service
        self.text_extractor = text_extractor

    @classmethod
    def dedupe_repeated_chars(cls, text):
        """清理 OCR 目录文本中的连续重复中文字符。"""
        if len(text) < 2:
            return text

        output = [text[0]]
        index = 1
        while index < len(text):
            previous = output[-1]
            current = text[index]
            if (
                current == previous
                and "\u4e00" <= current <= "\u9fff"
                and previous + current not in cls._REPEATED_CHAR_WHITELIST
            ):
                index += 1
                continue
            output.append(current)
            index += 1
        return "".join(output)

    @classmethod
    def parse_toc_line(cls, line):
        """解析目录标题和页码。"""
        line = cls.dedupe_repeated_chars(line).strip()
        if not line or re.fullmatch(r"\d{1,4}", line):
            return None

        match = re.match(
            r"^(.+?)\s*[\.\.…·\-—─]{2,}\s*(\d+)\s*$",
            line,
        )
        if match:
            title = cls.clean_text(match.group(1))
            page = int(match.group(2))
            if title:
                return {"title": title, "page": page}

        match = re.match(r"^(.+?)[\s·\.…]+(\d{1,4})\s*$", line)
        if match:
            title = cls.clean_text(match.group(1))
            page = int(match.group(2))
            if title and len(title) > 1:
                return {"title": title, "page": page}

        if re.search(r"第.{1,3}[章节篇]", line):
            title = cls.clean_text(re.sub(r"[\.\.…·]+$", "", line))
            if title:
                return {"title": title, "page": None}
        return None

    @staticmethod
    def merge_toc_lines(ocr_items):
        """把被拆开的纯页码或引导线残行合并回标题行。"""
        merged = []
        for item in ocr_items:
            text = item["text"].strip()
            if not text:
                continue

            if re.fullmatch(r"[\d\s\.…·]+", text) and merged:
                merged[-1]["text"] = (
                    merged[-1]["text"].rstrip("·.… ")
                    + " "
                    + text.lstrip("·.… ")
                )
                continue

            if re.match(r"^[\.…·\-—]+\s*\d", text) and merged:
                merged[-1]["text"] = (
                    merged[-1]["text"].rstrip("·.… ") + text
                )
                continue

            merged.append(dict(item))
        return merged

    @classmethod
    def is_toc_block(cls, texts):
        """根据页码行比例和章节模式判断目录区域。"""
        if len(texts) < 3:
            return False

        parsed = sum(
            1 for text in texts if cls.parse_toc_line(text) is not None
        )
        if parsed / len(texts) > 0.3:
            return True

        chapter_patterns = sum(
            1
            for text in texts
            if re.search(r"第.{1,3}[章节篇]", text)
        )
        return chapter_patterns >= 3

    @staticmethod
    def clean_text(text):
        """移除中文字符之间的多余空格。"""
        text = re.sub(
            r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])",
            "",
            text,
        )
        return text.strip()

    @classmethod
    def split_value(cls, text):
        """把表格单元格中的多行值转换为字符串或字符串列表。"""
        text = cls.clean_text(text)
        raw_lines = [
            line.strip() for line in text.split("\n") if line.strip()
        ]
        lines = []
        for line in raw_lines:
            if re.fullmatch(r"[\W_]+", line) and lines:
                lines[-1] += line
            else:
                lines.append(line)
        if len(lines) > 1:
            return lines
        return lines[0] if lines else ""

    def build_structured(
        self,
        blocks,
        pdf_path,
        page_num,
        image_path,
        has_text_layer,
        ocr_by_block=None,
    ):
        """构建单页结构化内容，保持现有响应字段。"""
        content = []

        def get_text(block_index, bbox):
            if ocr_by_block and block_index in ocr_by_block:
                return "\n".join(
                    item["text"] for item in ocr_by_block[block_index]
                )
            if has_text_layer and pdf_path and self.text_extractor:
                return self.text_extractor(pdf_path, page_num, bbox)
            return ""

        for index, block in enumerate(blocks):
            label = block["label"]

            if label in ("paragraph_title", "document_title", "title"):
                text = get_text(index, block["bbox"])
                if text:
                    lines = [
                        self.clean_text(line)
                        for line in text.split("\n")
                        if line.strip()
                    ]
                    for line in lines:
                        duplicate = any(
                            item["type"] == "title"
                            and item["text"] == line
                            for item in content
                        )
                        if not duplicate:
                            content.append({"type": "title", "text": line})

            elif label == "table" and block.get("cells"):
                cells = block["cells"]
                headers, rows = self.table_structure(cells)
                table_html = (
                    block.get("table_html")
                    or self.table_service.cells_to_html(cells)
                )
                block["table_html"] = table_html
                content.append({
                    "type": "table",
                    "headers": headers,
                    "rows": rows,
                    "html": table_html,
                })

            elif label == "table" and block.get("cells") is None:
                text = get_text(index, block["bbox"])
                if text:
                    content.append({
                        "type": "table_text",
                        "text": self.clean_text(text),
                        "note": (
                            "扫描件表格，仅提取文字，无法解析单元格结构"
                        ),
                    })

            elif label in (
                "text",
                "content",
                "abstract",
                "reference",
                "header",
                "footer",
                "algorithm",
                "figure_title",
                "table_title",
            ):
                ocr_items = []
                if ocr_by_block and index in ocr_by_block:
                    ocr_items = ocr_by_block[index]
                elif (
                    has_text_layer
                    and pdf_path
                    and self.text_extractor
                ):
                    raw = self.text_extractor(
                        pdf_path,
                        page_num,
                        block["bbox"],
                    )
                    if raw:
                        ocr_items = [
                            {"text": line.strip(), "bbox": None}
                            for line in raw.split("\n")
                            if line.strip()
                        ]

                if not ocr_items:
                    continue

                ocr_items = self.merge_toc_lines(ocr_items)
                text_list = [item["text"] for item in ocr_items]

                if self.is_toc_block(text_list):
                    toc_items = []
                    for item in ocr_items:
                        parsed = self.parse_toc_line(item["text"])
                        if parsed:
                            parsed["bbox"] = item["bbox"]
                            toc_items.append(parsed)
                        else:
                            cleaned = self.clean_text(item["text"])
                            if cleaned and len(cleaned) > 1:
                                toc_items.append({
                                    "title": cleaned,
                                    "page": None,
                                    "bbox": item["bbox"],
                                })
                    if toc_items:
                        block["toc_items"] = toc_items
                        content.append({
                            "type": "toc",
                            "items": toc_items,
                        })
                else:
                    text = self.clean_text("\n".join(text_list))
                    if text:
                        content.append({"type": "text", "text": text})

        return {"page": page_num, "content": content}

    @staticmethod
    def table_to_grid(cells):
        """把 cells 列表转换为二维网格。"""
        if not cells:
            return [], -1, -1
        max_row = max(cell["row"] for cell in cells)
        max_col = max(cell["col"] for cell in cells)
        grid = [[""] * (max_col + 1) for _ in range(max_row + 1)]
        for cell in cells:
            grid[cell["row"]][cell["col"]] = cell.get("text") or ""
        return grid, max_row, max_col

    @classmethod
    def table_structure(cls, cells):
        """从单元格生成表头、行数据和向下继承的空值。"""
        grid, max_row, max_col = cls.table_to_grid(cells)
        headers = [
            cls.clean_text(grid[0][column] or f"列{column + 1}")
            for column in range(max_col + 1)
        ]

        rows = []
        previous = [""] * (max_col + 1)
        for row_index in range(1, max_row + 1):
            row_data = {}
            for column in range(max_col + 1):
                value = grid[row_index][column]
                if value is None or value == "":
                    value = previous[column]
                else:
                    previous[column] = value
                row_data[headers[column]] = cls.split_value(value)
            rows.append(row_data)
        return headers, rows

    @classmethod
    def row_similarity(cls, first, second):
        """计算两个表头行的文字相似度。"""
        if len(first) != len(second) or not first:
            return 0.0
        matches = sum(
            1
            for first_value, second_value in zip(first, second)
            if cls.clean_text(first_value) == cls.clean_text(second_value)
        )
        return matches / len(first)

    @staticmethod
    def col_widths_ratio(cells):
        """计算各列宽度在表格总宽度中的占比。"""
        if not cells:
            return []
        max_col = max(cell["col"] for cell in cells)
        column_bounds = {}
        for cell in cells:
            column = cell["col"]
            x1, _, x2, _ = cell["bbox"]
            if column not in column_bounds:
                column_bounds[column] = [x1, x2]
            else:
                column_bounds[column][0] = min(
                    column_bounds[column][0],
                    x1,
                )
                column_bounds[column][1] = max(
                    column_bounds[column][1],
                    x2,
                )

        widths = []
        for column in range(max_col + 1):
            if column in column_bounds:
                widths.append(
                    column_bounds[column][1] - column_bounds[column][0]
                )
            else:
                widths.append(0)
        total = sum(widths) or 1
        return [width / total for width in widths]

    @classmethod
    def col_structure_similar(cls, first, second, tolerance=0.15):
        """比较两个表格的列数和列宽比例。"""
        first_max_col = max(
            (cell["col"] for cell in first),
            default=-1,
        )
        second_max_col = max(
            (cell["col"] for cell in second),
            default=-1,
        )
        if first_max_col != second_max_col or first_max_col < 0:
            return False

        first_ratios = cls.col_widths_ratio(first)
        second_ratios = cls.col_widths_ratio(second)
        return all(
            abs(first_ratio - second_ratio) <= tolerance
            for first_ratio, second_ratio in zip(
                first_ratios,
                second_ratios,
            )
        )

    def merge_cross_page_tables(self, pages_result):
        """用滑动指针合并连续页表格，并重建受影响的结构化内容。"""
        merge_info = []
        open_page_index = None
        open_block = None

        for page_index, page in enumerate(pages_result):
            blocks = page.get("blocks", [])
            page_height = page.get("height", 0)

            first_table = None
            for block in blocks:
                if block["label"] == "table" and block.get("cells"):
                    first_table = block
                    break

            if open_block is not None and first_table is not None:
                open_cells = open_block["cells"]
                current_cells = first_table["cells"]
                current_top = min(
                    cell["bbox"][1] for cell in current_cells
                )
                position_ok = (
                    page_height <= 0
                    or current_top < page_height * 0.5
                )

                if (
                    position_ok
                    and self.col_structure_similar(
                        open_cells,
                        current_cells,
                    )
                ):
                    open_grid, open_max_row, _ = self.table_to_grid(
                        open_cells
                    )
                    current_grid, current_max_row, _ = self.table_to_grid(
                        current_cells
                    )

                    start_row = 0
                    if (
                        current_max_row >= 0
                        and self.row_similarity(
                            open_grid[0],
                            current_grid[0],
                        ) > 0.8
                    ):
                        start_row = 1
                        self.log.info(
                            "跨页表格合并: 第%s页续表表头与前页相同，跳过",
                            page["page"],
                        )

                    if start_row <= current_max_row:
                        row_offset = open_max_row + 1
                        for cell in current_cells:
                            if cell["row"] < start_row:
                                continue
                            open_block["cells"].append({
                                "bbox": cell["bbox"],
                                "row": (
                                    cell["row"]
                                    - start_row
                                    + row_offset
                                ),
                                "col": cell["col"],
                                "text": cell.get("text", ""),
                            })

                        open_page = pages_result[open_page_index]
                        open_block["merged_from_pages"] = open_block.get(
                            "merged_from_pages",
                            [open_page["page"]],
                        )
                        open_block["merged_from_pages"].append(page["page"])
                        first_table["_merged_into"] = (open_page_index,)

                        rows_added = current_max_row + 1 - start_row
                        merge_info.append({
                            "from_page": page["page"],
                            "into_page": open_page["page"],
                            "rows_added": rows_added,
                            "header_skipped": start_row > 0,
                        })
                        self.log.info(
                            "跨页表格合并: 第%s页续表 -> 第%s页 (+%s行%s)",
                            page["page"],
                            open_page["page"],
                            rows_added,
                            "，跳过重复表头" if start_row > 0 else "",
                        )

                        last_table = None
                        for block in reversed(blocks):
                            if (
                                block["label"] == "table"
                                and block.get("cells")
                                and not block.get("_merged_into")
                            ):
                                last_table = block
                                break

                        if (
                            last_table is not None
                            and last_table is not first_table
                        ):
                            bottom = max(
                                cell["bbox"][3]
                                for cell in last_table["cells"]
                            )
                            if (
                                page_height <= 0
                                or bottom >= page_height * 0.5
                            ):
                                open_page_index = page_index
                                open_block = last_table
                        continue

            open_page_index = None
            open_block = None
            for block in reversed(blocks):
                if (
                    block["label"] == "table"
                    and block.get("cells")
                    and not block.get("_merged_into")
                ):
                    bottom = max(
                        cell["bbox"][3] for cell in block["cells"]
                    )
                    if (
                        page_height <= 0
                        or bottom >= page_height * 0.5
                    ):
                        open_page_index = page_index
                        open_block = block
                    break

        if not merge_info:
            return merge_info

        for page_data in pages_result:
            blocks = page_data.get("blocks", [])
            for block in blocks:
                if (
                    block["label"] == "table"
                    and block.get("merged_from_pages")
                ):
                    cells = block["cells"]
                    headers, rows = self.table_structure(cells)
                    for item_index, item in enumerate(
                        page_data["structured"]["content"]
                    ):
                        if item["type"] == "table":
                            page_data["structured"]["content"][
                                item_index
                            ] = {
                                "type": "table",
                                "headers": headers,
                                "rows": rows,
                                "html": self.table_service.cells_to_html(
                                    cells
                                ),
                                "merged_from_pages": block[
                                    "merged_from_pages"
                                ],
                            }
                            break

            has_merged_away = any(
                block.get("_merged_into")
                for block in blocks
                if block["label"] == "table"
            )
            if has_merged_away:
                page_data["structured"]["content"] = [
                    item
                    for item in page_data["structured"]["content"]
                    if not (
                        item["type"] == "table"
                        and not item.get("merged_from_pages")
                        and item.get("headers")
                    )
                ]

        return merge_info
