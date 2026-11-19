"""上传文档与 PDF 读取服务。"""

from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image, ImageOps


class DocumentService:
    """封装 PDF 保存、页面渲染和文字/表格提取。"""

    def __init__(self, logger, garbage_detector):
        self.log = logger
        self.garbage_detector = garbage_detector

    def save_pdf(self, pdf_bytes, doc_dir):
        """保存 PDF 并返回页数，不在上传阶段渲染图片。"""
        doc_dir = Path(doc_dir)
        (doc_dir / "source.pdf").write_bytes(pdf_bytes)
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(document)
        document.close()
        return page_count

    def save_image(self, image_bytes, doc_dir, filename="page_001.png"):
        """校正图片方向并统一保存为 RGB PNG。"""
        doc_dir = Path(doc_dir)
        image_path = doc_dir / filename
        with Image.open(BytesIO(image_bytes)) as source:
            source.load()
            has_alpha = (
                source.mode in {"RGBA", "LA"}
                or "transparency" in source.info
            )
            dpi = source.info.get("dpi")
            image = ImageOps.exif_transpose(source)
            if has_alpha:
                rgba = image.convert("RGBA")
                flattened = Image.new("RGB", rgba.size, "white")
                flattened.paste(rgba, mask=rgba.getchannel("A"))
                image = flattened
            elif image.mode != "RGB":
                image = image.convert("RGB")
            save_kwargs = {"format": "PNG", "optimize": True}
            if dpi:
                save_kwargs["dpi"] = dpi
            image.save(image_path, **save_kwargs)
        return image_path

    def ensure_page_image(self, doc_dir, page_num):
        """按需把单页渲染为 PNG，并返回图片路径。"""
        doc_dir = Path(doc_dir)
        image_path = doc_dir / f"page_{page_num:03d}.png"
        if image_path.exists():
            return image_path

        pdf_path = doc_dir / "source.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"source.pdf not found in {doc_dir}")
        document = fitz.open(str(pdf_path))
        page = document[page_num - 1]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        pixmap.save(str(image_path))
        document.close()
        return image_path

    def page_has_text(self, pdf_path, page_num):
        """判断页面是否存在可用文字层。"""
        document = fitz.open(pdf_path)
        page = document[page_num - 1]
        text = page.get_text("text").strip()
        document.close()

        if not text or len(text) < 5:
            return False
        printable = sum(1 for char in text if char.isprintable() or char in "\n\r\t")
        if printable / len(text) <= 0.5:
            return False
        if self.garbage_detector(text):
            self.log.info("第 %s 页文字层检测为乱码，将回退 OCR", page_num)
            return False
        return True

    def extract_table_cells(self, pdf_path, page_num, table_bbox,
                            img_width, img_height):
        """从 PDF 文字层提取表格单元格。"""
        document = fitz.open(pdf_path)
        page = document[page_num - 1]

        scale = 2.0
        pad = 10
        page_rect = page.rect
        pdf_rect = fitz.Rect(
            max(0, table_bbox[0] / scale - pad),
            max(0, table_bbox[1] / scale - pad),
            min(page_rect.width, table_bbox[2] / scale + pad),
            min(page_rect.height, table_bbox[3] / scale + pad),
        )

        cells = []
        for table in page.find_tables(clip=pdf_rect):
            rows_text = table.extract()
            for row_index, row in enumerate(table.rows):
                for col_index, cell_rect in enumerate(row.cells):
                    if cell_rect is None:
                        continue
                    text = ""
                    if (
                        row_index < len(rows_text)
                        and col_index < len(rows_text[row_index])
                    ):
                        text = rows_text[row_index][col_index] or ""
                    cells.append({
                        "bbox": [
                            round(cell_rect[0] * scale),
                            round(cell_rect[1] * scale),
                            round(cell_rect[2] * scale),
                            round(cell_rect[3] * scale),
                        ],
                        "row": row_index,
                        "col": col_index,
                        "text": text,
                    })
        document.close()
        self.log.info("提取到 %s 个单元格", len(cells))
        return cells

    @staticmethod
    def extract_text_from_pdf(pdf_path, page_num, bbox):
        """按页面坐标提取文字层文本。"""
        document = fitz.open(pdf_path)
        page = document[page_num - 1]
        scale = 2.0
        rect = fitz.Rect(
            bbox[0] / scale,
            bbox[1] / scale,
            bbox[2] / scale,
            bbox[3] / scale,
        )
        text = page.get_text("text", clip=rect).strip()
        document.close()
        return text
