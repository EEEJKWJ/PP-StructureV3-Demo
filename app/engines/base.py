"""表格模型适配器协议和通用输出工具。"""

import html as html_lib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from html.parser import HTMLParser

import cv2


@dataclass
class TableOutput:
    """适配器统一返回的表格识别结果。"""

    cells: list[dict]
    html: str
    raw: dict | None = None


class TableAdapter(ABC):
    """完整表格推理产线的轻量适配器。"""

    backend = "unknown"

    def __init__(self, config, logger, ocr_provider=None):
        self.config = dict(config)
        self.log = logger
        self.ocr_provider = ocr_provider
        self.engine = None
        self.loaded = False
        self.load_error = ""

    @abstractmethod
    def load(self):
        """加载模型及所需处理产线。失败时应抛出异常。"""

    @abstractmethod
    def predict(self, image_path, table_bbox, options=None):
        """识别单个表格，返回 TableOutput 或 None。"""

    def ensure_loaded(self):
        if not self.loaded:
            self.load()
        if not self.loaded:
            raise RuntimeError(f"模型 {self.model_id} 未成功加载")
        return self.engine

    def health_check(self):
        """返回不触发模型加载的健康状态。"""
        if self.loaded:
            status = "ready"
            error = ""
        elif self.load_error:
            status = "error"
            error = self.load_error
        else:
            status = "available"
            error = ""
        return {**self.model_info, "status": status, "error": error}

    def close(self):
        self.engine = None
        self.loaded = False

    @property
    def model_id(self):
        return self.config["id"]

    @property
    def model_info(self):
        return {
            "id": self.config["id"],
            "name": self.config["name"],
            "version": self.config["version"],
            "adapter": self.config["adapter"],
            "backend": self.config.get("backend", self.backend),
            "device": self.config.get("device", ""),
            "source": self.config.get("source", ""),
            "artifact_hash": self.config.get("artifact_hash"),
        }


class UnsupportedTableAdapter(TableAdapter):
    """对尚未实现的自定义适配器提供明确失败。"""

    backend = "unsupported"

    def load(self):
        adapter = self.config["adapter"]
        self.load_error = f"模型适配器尚未实现: {adapter}"
        raise NotImplementedError(self.load_error)

    def predict(self, image_path, table_bbox, options=None):
        self.ensure_loaded()


def crop_table(image_path, table_bbox, padding=5):
    """裁切表格区域并返回裁剪图与页面偏移。"""
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    x1, y1, x2, y2 = map(int, table_bbox)
    crop_x1 = max(0, x1 - padding)
    crop_y1 = max(0, y1 - padding)
    crop_x2 = min(image.shape[1], x2 + padding)
    crop_y2 = min(image.shape[0], y2 + padding)
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None, None
    return crop, (crop_x1, crop_y1)


def parse_html_table(table_html, cell_boxes_batch, offset_x, offset_y):
    """把模型 HTML 转换为带行列和页面坐标的单元格。"""

    class TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.cells = []
            self.row = -1
            self.col = 0
            self.in_cell = False
            self.current_text = ""
            self.occupied = {}
            self.colspan = 1
            self.rowspan = 1

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.row += 1
                self.col = 0
                while (self.row, self.col) in self.occupied:
                    self.col += 1
            elif tag == "td":
                self.in_cell = True
                self.current_text = ""
                while (self.row, self.col) in self.occupied:
                    self.col += 1
                attrs = dict(attrs)
                self.colspan = int(attrs.get("colspan", 1))
                self.rowspan = int(attrs.get("rowspan", 1))

        def handle_data(self, data):
            if self.in_cell:
                self.current_text += data

        def handle_endtag(self, tag):
            if tag != "td" or not self.in_cell:
                return
            self.in_cell = False
            self.cells.append({
                "row": self.row,
                "col": self.col,
                "text": self.current_text.strip(),
            })
            for row_delta in range(self.rowspan):
                for col_delta in range(self.colspan):
                    if row_delta or col_delta:
                        self.occupied[
                            (self.row + row_delta, self.col + col_delta)
                        ] = True
            self.col += self.colspan

    parser = TableParser()
    parser.feed(table_html)

    boxes = []
    if cell_boxes_batch:
        for box in cell_boxes_batch[0]:
            xs = [
                float(box[index])
                for index in range(0, len(box), 2)
            ]
            ys = [
                float(box[index])
                for index in range(1, len(box), 2)
            ]
            boxes.append([
                int(min(xs)) + offset_x,
                int(min(ys)) + offset_y,
                int(max(xs)) + offset_x,
                int(max(ys)) + offset_y,
            ])

    for index, cell in enumerate(parser.cells):
        cell["bbox"] = (
            boxes[index] if index < len(boxes) else [0, 0, 0, 0]
        )
    return parser.cells


def restore_cell_newlines(cells, cell_boxes, ocr_data):
    """根据 OCR 行坐标恢复单元格内换行。"""
    texts = (
        ocr_data.get("rec_texts", [])
        if isinstance(ocr_data, dict)
        else []
    )
    boxes = (
        ocr_data.get("rec_boxes", [])
        if isinstance(ocr_data, dict)
        else []
    )
    if not cell_boxes or not texts or not boxes:
        return

    text_by_cell = {}
    for index, text in enumerate(texts):
        if index >= len(boxes):
            break
        box = boxes[index]
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        best_cell = -1
        for cell_index, cell_box in enumerate(cell_boxes):
            if (
                cell_box[0] <= center_x <= cell_box[2]
                and cell_box[1] <= center_y <= cell_box[3]
            ):
                best_cell = cell_index
                break

        if best_cell == -1:
            min_distance = float("inf")
            for cell_index, cell_box in enumerate(cell_boxes):
                cell_x = (cell_box[0] + cell_box[2]) / 2
                cell_y = (cell_box[1] + cell_box[3]) / 2
                distance = (
                    (center_x - cell_x) ** 2
                    + (center_y - cell_y) ** 2
                )
                if distance < min_distance:
                    min_distance = distance
                    best_cell = cell_index

        if best_cell >= 0:
            text_by_cell.setdefault(best_cell, []).append(
                (center_y, text)
            )

    for cell_index, lines in text_by_cell.items():
        if cell_index < len(cells):
            lines.sort(key=lambda item: item[0])
            cells[cell_index]["text"] = "\n".join(
                text for _, text in lines
            )


def cells_to_html(cells):
    """从单元格生成无合并属性的基础 HTML 表格。"""
    if not cells:
        return ""
    max_row = max(cell["row"] for cell in cells)
    max_col = max(cell["col"] for cell in cells)
    grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for cell in cells:
        grid[cell["row"]][cell["col"]] = cell.get("text", "")

    rows = []
    for row in grid:
        columns = []
        for value in row:
            text = html_lib.escape(str(value)).replace("\n", "<br>")
            columns.append(f"<td>{text}</td>")
        rows.append("<tr>" + "".join(columns) + "</tr>")
    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def table_dimensions(cells):
    """返回表格行数和列数。"""
    if not cells:
        return 0, 0
    return (
        max(cell["row"] for cell in cells) + 1,
        max(cell["col"] for cell in cells) + 1,
    )
