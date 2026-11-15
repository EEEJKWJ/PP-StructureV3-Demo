"""PP-StructureV3 版面结构化演示
- 版面检测：PP-DocLayoutV3 (HuggingFace transformers + PyTorch)
- 文本提取：PyMuPDF（文本型 PDF）/ PP-OCRv5 ONNX（扫描件，全页 OCR + 按区域归类）
- 表格提取：PyMuPDF（文本型 PDF）/ PP-TableMagic V2 完整产线（扫描件）
  · 表格分类（有线/无线自动区分）
  · 单元格检测 (RT-DETR-L)
  · 结构识别 (SLANeXt)
  · 文字检测+识别 (PP-OCRv4)
  · 降级方案：RapidTable (SLANet_plus ONNX)"""

import re
import uuid
import shutil
import logging
import threading
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse

from app.services.layout import LayoutService
from app.services.ocr import OCRService
from app.services.document import DocumentService
from app.services.table import TableService

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pp-struct")

app = FastAPI(title="PP-StructureV3 Demo")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

_layout_service = LayoutService(log)
_ocr_service = OCRService(log)


# ── 版面与 OCR 兼容入口 ────────────────────────────────────────────
def _load_model():
    """兼容入口：版面模型加载由 LayoutService 提供。"""
    return _layout_service.load()


def _ensure_model():
    """兼容入口：版面模型就绪检查由 LayoutService 提供。"""
    return _layout_service.ensure_ready()


def _sort_reading_order(blocks: list[dict], y_tolerance: int = 30) -> list[dict]:
    """兼容入口：阅读顺序排序由 LayoutService 提供。"""
    return _layout_service.sort_reading_order(blocks, y_tolerance)


def _iou(a, b):
    """兼容入口：版面框 IoU 由 LayoutService 提供。"""
    return _layout_service.iou(a, b)


def detect_layout(image_path: str) -> list[dict]:
    """兼容入口：版面检测由 LayoutService 提供。"""
    return _layout_service.detect(image_path)


def _download_ocr_models():
    """兼容入口：OCR 模型下载由 OCRService 提供。"""
    return _ocr_service.download_models()


def _get_ocr():
    """兼容入口：OCR 引擎由 OCRService 提供。"""
    return _ocr_service.get_engine()


def ocr_blocks(image_path: str, blocks: list[dict]) -> dict[int, list[dict]]:
    """兼容入口：区域 OCR 由 OCRService 提供。"""
    return _ocr_service.recognize_blocks(image_path, blocks)


def ocr_single_block(image_path: str, block: dict) -> list[dict]:
    """兼容入口：单区域 OCR 由 OCRService 提供。"""
    return _ocr_service.recognize_block(image_path, block)


def _nms_ocr_lines(
    lines: list[dict],
    iou_threshold: float = 0.5,
    contain_threshold: float = 0.7,
) -> list[dict]:
    """兼容入口：OCR 行去重由 OCRService 提供。"""
    return _ocr_service.nms_lines(
        lines,
        iou_threshold,
        contain_threshold,
    )


def _contain_ratio(a: list, b: list) -> float:
    """兼容入口：OCR 包含比例由 OCRService 提供。"""
    return _ocr_service.contain_ratio(a, b)


def _merge_suffix_prefix_lines(
    lines: list[dict],
    max_overlap: int = 6,
) -> list[dict]:
    """兼容入口：OCR 前后缀合并由 OCRService 提供。"""
    return _ocr_service.merge_suffix_prefix_lines(lines, max_overlap)


# ── 表格结构识别（扫描件）── PP-TableMagic V2 ────────────────────
# 优先用 PP-TableMagic V2 完整产线（表格分类+单元格检测+结构识别+OCR），
# 没有 PaddlePaddle 则降级为 RapidTable (SLANet_plus ONNX)
_table_engine = None
_table_engine_type = ""  # "paddle" | "rapid" | ""
_table_engine_lock = threading.Lock()
_table_service = TableService(log, _get_ocr)


def _get_table_engine():
    """兼容现有状态接口，表格引擎由 TableService 统一管理。"""
    global _table_engine, _table_engine_type
    _table_engine = _table_service.get_engine()
    _table_engine_type = _table_service.engine_type
    return _table_engine


def recognize_table_structure(image_path: str, table_bbox: list) -> dict | None:
    """识别表格并返回单元格和 HTML。"""
    return _table_service.recognize(image_path, table_bbox)


def _apply_table_result(block, cells, table_html=""):
    """统一写入表格单元格和 HTML，保持所有识别路径字段一致。"""
    block["cells"] = cells
    block["table_html"] = table_html or _table_service.cells_to_html(cells)


# ── 目录行解析 ────────────────────────────────────────────────────
def _dedupe_repeated_chars(text: str) -> str:
    """去除目录文本中连续重复的同一字符（OCR 常见错误）。
    例: "第二二节牙本质" → "第二节牙本质"
    例: "第一节节口腔黏膜的的基本结构" → "第一节口腔黏膜的基本结构"
    白名单：保留某些常见叠词（避免误删合法重叠）。
    """
    # 常见叠词白名单（在 TOC 中极少出现，但保留以防万一）
    WHITELIST = {"看看", "好好", "慢慢", "常常", "刚刚", "渐渐", "个个",
                 "早早", "远远", "静静", "暗暗", "默默", "偷偷", "种种",
                 "处处", "时时", "天天", "年年", "月月", "日日"}
    if len(text) < 2:
        return text
    out = [text[0]]
    i = 1
    while i < len(text):
        prev = out[-1]
        cur = text[i]
        # 仅对中文字符做去重（不处理英文、数字、标点、空格、引导线）
        if cur == prev and '\u4e00' <= cur <= '\u9fff':
            # 检查白名单
            pair = prev + cur
            if pair not in WHITELIST:
                # 跳过这个重复字符
                i += 1
                continue
        out.append(cur)
        i += 1
    return "".join(out)


def _parse_toc_line(line: str) -> dict | None:
    """解析目录行，拆分标题和页码。
    例: '第一章 牙体组织……1' → {"title": "第一章 牙体组织", "page": 1}
    """
    # 先做字符去重（修复 OCR 重复字符错误）
    line = _dedupe_repeated_chars(line)
    line = line.strip()
    if not line:
        return None

    # 纯页码行（如 "100"）跳过
    if re.fullmatch(r"\d{1,4}", line):
        return None

    # 匹配：标题 + 引导线 + 页码
    m = re.match(r"^(.+?)\s*[\.\.…·\-—─]{2,}\s*(\d+)\s*$", line)
    if m:
        title = _clean_text(m.group(1))
        page = int(m.group(2))
        if title:
            return {"title": title, "page": page}

    # 匹配：标题 + 空格/点 + 页码（短引导线）
    m = re.match(r"^(.+?)[\s·\.…]+(\d{1,4})\s*$", line)
    if m:
        title = _clean_text(m.group(1))
        page = int(m.group(2))
        if title and len(title) > 1:
            return {"title": title, "page": page}

    # 无页码的标题行（如"第一篇基础知识"）
    if re.search(r"第.{1,3}[章节篇]", line):
        title = _clean_text(re.sub(r"[\.\.…·]+$", "", line))
        if title:
            return {"title": title, "page": None}

    return None


def _merge_toc_lines(ocr_items: list[dict]) -> list[dict]:
    """将纯页码行合并到前一个标题行。
    如 ["第一章牙体组织", "1"] → ["第一章牙体组织 1"]
    同时将被引导线截断的行合并。
    """
    merged = []
    for item in ocr_items:
        text = item["text"].strip()
        if not text:
            continue
        # 纯页码行或纯引导线+页码，合并到前一行
        if re.fullmatch(r"[\d\s\.…·]+", text) and merged:
            merged[-1]["text"] = merged[-1]["text"].rstrip("·.… ") + " " + text.lstrip("·.… ")
            continue
        # 以引导线开头的残行（如"……60"），合并到前一行
        if re.match(r"^[\.…·\-—]+\s*\d", text) and merged:
            merged[-1]["text"] = merged[-1]["text"].rstrip("·.… ") + text
            continue
        merged.append(dict(item))
    return merged


def _is_toc_block(texts: list[str]) -> bool:
    """判断一组文本行是否像目录"""
    if len(texts) < 3:
        return False
    # 方式1：标题+页码行占比
    parsed = sum(1 for t in texts if _parse_toc_line(t) is not None)
    if parsed / len(texts) > 0.3:
        return True
    # 方式2：大量"第X章/节"模式
    chapter_pat = sum(1 for t in texts if re.search(r"第.{1,3}[章节篇]", t))
    if chapter_pat >= 3:
        return True
    return False


# ── 判断页面是否有文字层 ─────────────────────────────────────────
def is_garbage_text(text: str, threshold: float = 0.3) -> bool:
    """检测提取的文字是否为乱码（CID 字体解码失败、编码错误等）。
    threshold: 有意义字符占比低于此值判定为乱码。
    """
    if not text or len(text) < 3:
        return False

    n = len(text)

    # U+FFFD 替换字符过多
    if text.count('\ufffd') / n > 0.1:
        return True

    # Private Use Area 字符过多（CID 字体解码失败的典型特征）
    pua = sum(1 for c in text if '\ue000' <= c <= '\uf8ff')
    if pua / n > 0.15:
        return True

    # 有意义字符比例过低
    meaningful = sum(1 for c in text if (
        c.isalnum()
        or '\u4e00' <= c <= '\u9fff'   # CJK 基本区
        or '\u3400' <= c <= '\u4dbf'   # CJK 扩展 A
        or '\u3000' <= c <= '\u303f'   # CJK 符号
        or '\uff00' <= c <= '\uffef'   # 全角字符
        or c in ' \n\r\t.,;:!?()[]{}"\'-/\\·、。，；：！？（）【】《》""''—…'
    ))
    if n > 10 and meaningful / n < threshold:
        return True

    return False


_document_service = DocumentService(log, is_garbage_text)


def page_has_text(pdf_path: str, page_num: int) -> bool:
    """兼容入口：页面文字层检测由 DocumentService 提供。"""
    return _document_service.page_has_text(pdf_path, page_num)



@app.on_event("startup")
async def startup_event():
    threading.Thread(target=_load_model, daemon=True).start()
    threading.Thread(target=_download_ocr_models, daemon=True).start()
    threading.Thread(target=_get_table_engine, daemon=True).start()


@app.get("/api/model-status")
async def model_status():
    ocr_status = _ocr_service.status()
    gpu = _layout_service.gpu_info()
    return {
        "status": _layout_service.status,
        "error": _layout_service.error,
        "ocr_available": ocr_status["available"],
        "ocr_model": ocr_status["model"],
        "gpu": gpu["available"],
        "gpu_name": gpu["name"],
        "table_engine": _table_engine_type or "none",
    }


# ── PDF 保存 + 按需渲染 ──────────────────────────────────────────────
def pdf_save_and_count(pdf_bytes: bytes, doc_dir: Path) -> int:
    """兼容入口：PDF 保存由 DocumentService 提供。"""
    return _document_service.save_pdf(pdf_bytes, doc_dir)



def ensure_page_image(doc_dir: Path, page_num: int) -> Path:
    """兼容入口：页面渲染由 DocumentService 提供。"""
    return _document_service.ensure_page_image(doc_dir, page_num)



# ── 表格单元格提取（PyMuPDF）──────────────────────────────────────
def extract_table_cells(pdf_path: str, page_num: int, table_bbox: list,
                        img_width: int, img_height: int) -> list[dict]:
    """兼容入口：PDF 表格单元格提取由 DocumentService 提供。"""
    return _document_service.extract_table_cells(
        pdf_path, page_num, table_bbox, img_width, img_height
    )



# ── 文本提取（PDF 文字层）────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str, page_num: int, bbox: list) -> str:
    """兼容入口：PDF 文字层提取由 DocumentService 提供。"""
    return _document_service.extract_text_from_pdf(pdf_path, page_num, bbox)



# ── 结构化输出 ────────────────────────────────────────────────────
def _clean_text(text: str) -> str:
    text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
    return text.strip()


def _split_value(text: str):
    text = _clean_text(text)
    raw = [l.strip() for l in text.split("\n") if l.strip()]
    lines = []
    for l in raw:
        if re.fullmatch(r'[\W_]+', l) and lines:
            lines[-1] += l
        else:
            lines.append(l)
    return lines if len(lines) > 1 else (lines[0] if lines else "")


def build_structured(blocks: list, pdf_path: str | None, page_num: int,
                     image_path: str, has_text_layer: bool,
                     ocr_by_block: dict[int, list[str]] | None = None) -> dict:
    """构建结构化数据。
    ocr_by_block: 全页 OCR 结果按区域归类后的字典 {block_idx: [text_lines]}
    """
    content = []

    def get_text(block_idx: int, bbox: list) -> str:
        """获取区域文字：优先用 OCR 回退结果，否则有文字层用 PyMuPDF"""
        if ocr_by_block and block_idx in ocr_by_block:
            return "\n".join(it["text"] for it in ocr_by_block[block_idx])
        if has_text_layer and pdf_path:
            return extract_text_from_pdf(pdf_path, page_num, bbox)
        return ""

    for idx, block in enumerate(blocks):
        label = block["label"]

        # 标题类
        if label in ("paragraph_title", "document_title", "title"):
            text = get_text(idx, block["bbox"])
            if text:
                lines = [_clean_text(l) for l in text.split("\n") if l.strip()]
                for line in lines:
                    if not any(c["type"] == "title" and c["text"] == line for c in content):
                        content.append({"type": "title", "text": line})

        # 表格（有单元格）
        elif label == "table" and block.get("cells"):
            cells = block["cells"]
            max_row = max(c["row"] for c in cells)
            max_col = max(c["col"] for c in cells)

            grid = [[None] * (max_col + 1) for _ in range(max_row + 1)]
            for c in cells:
                grid[c["row"]][c["col"]] = c["text"] or ""

            headers = [_clean_text(grid[0][c] or f"列{c+1}") for c in range(max_col + 1)]

            rows = []
            prev = [""] * (max_col + 1)
            for r in range(1, max_row + 1):
                row_data = {}
                for c in range(max_col + 1):
                    val = grid[r][c]
                    if val is None or val == "":
                        val = prev[c]
                    else:
                        prev[c] = val
                    row_data[headers[c]] = _split_value(val)
                rows.append(row_data)

            table_html = block.get("table_html") or _table_service.cells_to_html(cells)
            block["table_html"] = table_html
            content.append({
                "type": "table",
                "headers": headers,
                "rows": rows,
                "html": table_html,
            })

        # 表格（扫描件，无结构识别）
        elif label == "table" and block.get("cells") is None:
            text = get_text(idx, block["bbox"])
            if text:
                content.append({"type": "table_text", "text": _clean_text(text),
                                "note": "扫描件表格，仅提取文字，无法解析单元格结构"})

        # 文本/内容类
        elif label in ("text", "content", "abstract", "reference",
                        "header", "footer", "algorithm",
                        "figure_title", "table_title"):
            ocr_items = []  # [{"text":..., "bbox":...}]
            if ocr_by_block and idx in ocr_by_block:
                ocr_items = ocr_by_block[idx]
            elif has_text_layer and pdf_path:
                raw = extract_text_from_pdf(pdf_path, page_num, block["bbox"])
                if raw:
                    ocr_items = [{"text": l.strip(), "bbox": None}
                                 for l in raw.split("\n") if l.strip()]

            if not ocr_items:
                continue

            # 合并被拆分的目录行（页码行合并回标题行）
            ocr_items = _merge_toc_lines(ocr_items)
            text_list = [it["text"] for it in ocr_items]

            # 判断是否为目录
            if _is_toc_block(text_list):
                toc_items = []
                for it in ocr_items:
                    parsed = _parse_toc_line(it["text"])
                    if parsed:
                        parsed["bbox"] = it["bbox"]
                        toc_items.append(parsed)
                    else:
                        cleaned = _clean_text(it["text"])
                        if cleaned and len(cleaned) > 1:
                            toc_items.append({"title": cleaned, "page": None, "bbox": it["bbox"]})
                if toc_items:
                    block["toc_items"] = toc_items
                    content.append({"type": "toc", "items": toc_items})
            else:
                text = _clean_text("\n".join(text_list))
                if text:
                    content.append({"type": "text", "text": text})

    return {"page": page_num, "content": content}



def _table_to_grid(cells: list[dict]) -> tuple[list[list[str]], int, int]:
    """将 cells 列表转为二维网格，返回 (grid, max_row, max_col)"""
    if not cells:
        return [], -1, -1
    max_row = max(c["row"] for c in cells)
    max_col = max(c["col"] for c in cells)
    grid = [[""] * (max_col + 1) for _ in range(max_row + 1)]
    for c in cells:
        grid[c["row"]][c["col"]] = c.get("text") or ""
    return grid, max_row, max_col


def _row_similarity(row_a: list[str], row_b: list[str]) -> float:
    """计算两行文字的相似度（0~1），用于判断续表表头是否重复"""
    if len(row_a) != len(row_b):
        return 0.0
    if not row_a:
        return 0.0
    matches = sum(1 for a, b in zip(row_a, row_b)
                  if _clean_text(a) == _clean_text(b))
    return matches / len(row_a)


def _col_widths_ratio(cells: list[dict]) -> list[float]:
    """计算每列宽度占比，用于判断两个表格列结构是否一致"""
    if not cells:
        return []
    max_col = max(c["col"] for c in cells)
    col_bounds = {}
    for c in cells:
        col = c["col"]
        x1, _, x2, _ = c["bbox"]
        if col not in col_bounds:
            col_bounds[col] = [x1, x2]
        else:
            col_bounds[col][0] = min(col_bounds[col][0], x1)
            col_bounds[col][1] = max(col_bounds[col][1], x2)

    widths = []
    for col in range(max_col + 1):
        if col in col_bounds:
            widths.append(col_bounds[col][1] - col_bounds[col][0])
        else:
            widths.append(0)
    total = sum(widths) or 1
    return [w / total for w in widths]


def _col_structure_similar(cells_a: list[dict], cells_b: list[dict],
                           tolerance: float = 0.15) -> bool:
    """判断两个表格的列结构是否相似（列数相同且列宽比例接近）"""
    max_col_a = max(c["col"] for c in cells_a) if cells_a else -1
    max_col_b = max(c["col"] for c in cells_b) if cells_b else -1
    if max_col_a != max_col_b or max_col_a < 0:
        return False

    ratios_a = _col_widths_ratio(cells_a)
    ratios_b = _col_widths_ratio(cells_b)
    for ra, rb in zip(ratios_a, ratios_b):
        if abs(ra - rb) > tolerance:
            return False
    return True


def merge_cross_page_tables(pages_result: list[dict]) -> list[dict]:
    """跨页表格合并：检测连续页的续表并合并到第一个表格所在页。
    使用滑动指针追踪待续表格，支持三页及以上的连续合并。
    """
    merge_info = []
    # 滑动指针：当前"待续"表格
    open_page_idx = None   # 待续表格所在页的 index
    open_block = None      # 待续表格的 block 引用

    for i in range(len(pages_result)):
        page = pages_result[i]
        blocks = page.get("blocks", [])
        page_h = page.get("height", 0)

        # 找本页第一个有 cells 的 table（续表候选）
        first_table = None
        for b in blocks:
            if b["label"] == "table" and b.get("cells"):
                first_table = b
                break

        # 尝试与 open_block 合并
        if open_block is not None and first_table is not None:
            cells_a = open_block["cells"]
            cells_b = first_table["cells"]

            # 位置：续表应在页面上半部
            table_b_top = min(c["bbox"][1] for c in cells_b)
            position_ok = page_h <= 0 or table_b_top < page_h * 0.5

            if position_ok and _col_structure_similar(cells_a, cells_b):
                grid_a, max_row_a, _ = _table_to_grid(cells_a)
                grid_b, max_row_b, _ = _table_to_grid(cells_b)

                # 检测并跳过重复表头
                start_row_b = 0
                if max_row_b >= 0 and _row_similarity(grid_a[0], grid_b[0]) > 0.8:
                    start_row_b = 1
                    log.info(f"跨页表格合并: 第{page['page']}页续表表头与前页相同，跳过")

                if start_row_b <= max_row_b:
                    # 执行合并
                    new_row_offset = max_row_a + 1
                    for c in cells_b:
                        if c["row"] < start_row_b:
                            continue
                        open_block["cells"].append({
                            "bbox": c["bbox"],
                            "row": c["row"] - start_row_b + new_row_offset,
                            "col": c["col"],
                            "text": c.get("text", ""),
                        })

                    open_page = pages_result[open_page_idx]
                    open_block["merged_from_pages"] = open_block.get(
                        "merged_from_pages", [open_page["page"]])
                    open_block["merged_from_pages"].append(page["page"])
                    first_table["_merged_into"] = (open_page_idx,)

                    rows_added = max_row_b + 1 - start_row_b
                    merge_info.append({
                        "from_page": page["page"],
                        "into_page": open_page["page"],
                        "rows_added": rows_added,
                        "header_skipped": start_row_b > 0,
                    })
                    log.info(
                        f"跨页表格合并: 第{page['page']}页续表 → 第{open_page['page']}页"
                        f" (+{rows_added}行"
                        f"{'，跳过重复表头' if start_row_b > 0 else ''})")

                    # 合并后 open_block 继续保持（支持三页+连续合并）
                    # 但需检查本页是否还有其他表格在底部
                    last_table_this_page = None
                    for b in reversed(blocks):
                        if b["label"] == "table" and b.get("cells") and not b.get("_merged_into"):
                            last_table_this_page = b
                            break
                    # 如果本页底部还有另一个表格（不是被合并的那个），更新 open
                    if last_table_this_page and last_table_this_page is not first_table:
                        tb = max(c["bbox"][3] for c in last_table_this_page["cells"])
                        if page_h <= 0 or tb >= page_h * 0.5:
                            open_page_idx = i
                            open_block = last_table_this_page
                    # 否则 open_block 保持不变（继续用合并后的表格）
                    continue  # 跳过下面的 open_block 更新

        # 更新 open_block：找本页最后一个在底部的表格
        open_page_idx = None
        open_block = None
        for b in reversed(blocks):
            if b["label"] == "table" and b.get("cells") and not b.get("_merged_into"):
                tb = max(c["bbox"][3] for c in b["cells"])
                if page_h <= 0 or tb >= page_h * 0.5:
                    open_page_idx = i
                    open_block = b
                break

    # 重建受影响页的 structured content
    if merge_info:
        for page_data in pages_result:
            blocks = page_data.get("blocks", [])

            # 重建合并源表格的 structured
            for block in blocks:
                if block["label"] == "table" and block.get("merged_from_pages"):
                    cells = block["cells"]
                    max_row = max(c["row"] for c in cells)
                    max_col = max(c["col"] for c in cells)
                    grid = [[""] * (max_col + 1) for _ in range(max_row + 1)]
                    for c in cells:
                        grid[c["row"]][c["col"]] = c.get("text") or ""

                    headers = [_clean_text(grid[0][c] or f"列{c+1}") for c in range(max_col + 1)]
                    rows = []
                    prev = [""] * (max_col + 1)
                    for r in range(1, max_row + 1):
                        row_data = {}
                        for ci in range(max_col + 1):
                            val = grid[r][ci]
                            if val is None or val == "":
                                val = prev[ci]
                            else:
                                prev[ci] = val
                            row_data[headers[ci]] = _split_value(val)
                        rows.append(row_data)

                    for idx, item in enumerate(page_data["structured"]["content"]):
                        if item["type"] == "table":
                            page_data["structured"]["content"][idx] = {
                                "type": "table", "headers": headers, "rows": rows,
                                "html": _table_service.cells_to_html(cells),
                                "merged_from_pages": block["merged_from_pages"],
                            }
                            break

            # 移除被合并走的表格
            has_merged_away = any(b.get("_merged_into") for b in blocks if b["label"] == "table")
            if has_merged_away:
                page_data["structured"]["content"] = [
                    item for item in page_data["structured"]["content"]
                    if not (item["type"] == "table" and
                            not item.get("merged_from_pages") and
                            item.get("headers"))
                ]

    return merge_info


# ── API 路由 ───────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("index.html")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    doc_id = uuid.uuid4().hex[:12]
    doc_dir = UPLOAD_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    is_pdf = file.filename.lower().endswith(".pdf")

    if is_pdf:
        page_count = pdf_save_and_count(content, doc_dir)
        filenames = [f"page_{i:03d}.png" for i in range(1, page_count + 1)]
    else:
        name = f"page_001{Path(file.filename).suffix}"
        (doc_dir / name).write_bytes(content)
        filenames = [name]

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "pages": filenames,
        "is_pdf": is_pdf,
    }


@app.get("/api/images/{doc_id}/{filename}")
async def get_image(doc_id: str, filename: str):
    doc_dir = UPLOAD_DIR / doc_id
    path = doc_dir / filename
    if not path.exists():
        if filename.startswith("page_") and filename.endswith(".png"):
            page_num = int(filename.split("_")[1].split(".")[0])
            try:
                ensure_page_image(doc_dir, page_num)
            except (FileNotFoundError, IndexError):
                return {"error": "not found"}
        else:
            return {"error": "not found"}
    return FileResponse(path)


@app.get("/api/text-source/{doc_id}/{page_num}")
async def text_source(doc_id: str, page_num: int):
    """快速判断某页文字来源（pdf/ocr），不做检测"""
    doc_dir = UPLOAD_DIR / doc_id
    pdf_path = doc_dir / "source.pdf"
    if pdf_path.exists():
        has_text = page_has_text(str(pdf_path), page_num)
        return {"text_source": "pdf" if has_text else "ocr"}
    return {"text_source": "ocr"}


@app.post("/api/detect/{doc_id}/{page_num}")
async def detect(doc_id: str, page_num: int):
    doc_dir = UPLOAD_DIR / doc_id
    try:
        img_path = ensure_page_image(doc_dir, page_num)
    except FileNotFoundError:
        return {"error": "page not found"}

    blocks = detect_layout(str(img_path))

    with Image.open(img_path) as img:
        w, h = img.size

    pdf_path = doc_dir / "source.pdf"
    is_pdf = pdf_path.exists()

    # 逐页判断是否有文字层
    has_text_layer = False
    if is_pdf:
        has_text_layer = page_has_text(str(pdf_path), page_num)

    text_source = "pdf" if has_text_layer else "ocr"
    log.info(f"第 {page_num} 页文字来源: {text_source}")

    # 扫描件：全页 OCR 一次，按区域归类
    ocr_by_block = None
    if not has_text_layer:
        ocr_by_block = ocr_blocks(str(img_path), blocks)

    for block in blocks:
        if block["label"] == "table":
            if has_text_layer:
                cells = extract_table_cells(
                    str(pdf_path), page_num, block["bbox"], w, h
                )
                garbage_cells = sum(1 for c in cells if is_garbage_text(c.get("text", "")))
                if cells and garbage_cells > len(cells) * 0.3:
                    log.info(f"表格单元格乱码 ({garbage_cells}/{len(cells)})，回退 PP-TableMagic V2")
                    table_result = recognize_table_structure(str(img_path), block["bbox"])
                    if table_result and table_result["cells"]:
                        _apply_table_result(
                            block,
                            table_result["cells"],
                            table_result.get("html", ""),
                        )
                    else:
                        _apply_table_result(block, cells)
                else:
                    _apply_table_result(block, cells)
            else:
                table_result = recognize_table_structure(str(img_path), block["bbox"])
                if table_result and table_result["cells"]:
                    _apply_table_result(
                        block,
                        table_result["cells"],
                        table_result.get("html", ""),
                    )
                else:
                    block["cells"] = None

    # 给每个 block 挂上提取到的文字（供前端可视化显示）
    # 文字层乱码时逐 block 回退 OCR
    if ocr_by_block is None:
        ocr_by_block = {}
    for idx, block in enumerate(blocks):
        if block.get("cells") or block.get("table_html"):
            continue
        if has_text_layer and is_pdf:
            text = extract_text_from_pdf(str(pdf_path), page_num, block["bbox"])
            if is_garbage_text(text):
                log.info(f"区域 #{idx}({block['label']}) 文字乱码，回退 OCR")
                ocr_result = ocr_single_block(str(img_path), block)
                if ocr_result:
                    ocr_by_block[idx] = ocr_result
                    block["ocr_text"] = "\n".join(it["text"] for it in ocr_result)
                    block["text_fallback"] = "ocr"
                else:
                    block["ocr_text"] = text
            else:
                block["ocr_text"] = text
        elif idx in ocr_by_block:
            block["ocr_text"] = "\n".join(it["text"] for it in ocr_by_block[idx])

    structured = build_structured(
        blocks, str(pdf_path) if is_pdf else None, page_num,
        str(img_path), has_text_layer, ocr_by_block
    )

    return {
        "page": page_num, "width": w, "height": h,
        "blocks": blocks, "is_pdf": is_pdf,
        "text_source": text_source,
        "structured": structured,
    }


@app.post("/api/detect-all/{doc_id}")
async def detect_all(doc_id: str):
    doc_dir = UPLOAD_DIR / doc_id
    if not doc_dir.exists():
        return {"error": "document not found"}

    pdf_path = doc_dir / "source.pdf"
    is_pdf = pdf_path.exists()

    if is_pdf:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        doc.close()
    else:
        total_pages = len(list(doc_dir.glob("page_*.*")))

    pages_result = []
    for page_num in range(1, total_pages + 1):
        try:
            img_path = ensure_page_image(doc_dir, page_num)
        except FileNotFoundError:
            continue
        blocks = detect_layout(str(img_path))
        with Image.open(img_path) as img:
            w, h = img.size

        has_text_layer = False
        if is_pdf:
            has_text_layer = page_has_text(str(pdf_path), page_num)

        ocr_by_block = None
        if not has_text_layer:
            ocr_by_block = ocr_blocks(str(img_path), blocks)

        for block in blocks:
            if block["label"] == "table":
                if has_text_layer:
                    cells = extract_table_cells(
                        str(pdf_path), page_num, block["bbox"], w, h
                    )
                    garbage_cells = sum(1 for c in cells if is_garbage_text(c.get("text", "")))
                    if cells and garbage_cells > len(cells) * 0.3:
                        log.info(f"[detect_all] 表格单元格乱码，回退 PP-TableMagic V2")
                        table_result = recognize_table_structure(str(img_path), block["bbox"])
                        if table_result and table_result["cells"]:
                            _apply_table_result(
                                block,
                                table_result["cells"],
                                table_result.get("html", ""),
                            )
                        else:
                            _apply_table_result(block, cells)
                    else:
                        _apply_table_result(block, cells)
                else:
                    table_result = recognize_table_structure(str(img_path), block["bbox"])
                    if table_result and table_result["cells"]:
                        _apply_table_result(
                            block,
                            table_result["cells"],
                            table_result.get("html", ""),
                        )
                    else:
                        block["cells"] = None

        # 逐 block 乱码回退
        if ocr_by_block is None:
            ocr_by_block = {}
        for idx, block in enumerate(blocks):
            if block.get("cells") or block.get("table_html"):
                continue
            if has_text_layer and is_pdf:
                text = extract_text_from_pdf(str(pdf_path), page_num, block["bbox"])
                if is_garbage_text(text):
                    log.info(f"[detect_all] 区域 #{idx}({block['label']}) 文字乱码，回退 OCR")
                    ocr_result = ocr_single_block(str(img_path), block)
                    if ocr_result:
                        ocr_by_block[idx] = ocr_result
                        block["text_fallback"] = "ocr"

        structured = build_structured(
            blocks, str(pdf_path) if is_pdf else None, page_num,
            str(img_path), has_text_layer, ocr_by_block
        )

        pages_result.append({
            "page": page_num, "width": w, "height": h,
            "blocks": blocks, "text_source": "pdf" if has_text_layer else "ocr",
            "structured": structured,
        })

    # 跨页表格合并
    merge_info = merge_cross_page_tables(pages_result)
    if merge_info:
        log.info(f"跨页表格合并完成: {len(merge_info)} 处")

    return {"doc_id": doc_id, "pages": pages_result, "is_pdf": is_pdf,
            "table_merges": merge_info if merge_info else None}


@app.delete("/api/documents/{doc_id}")
async def delete_doc(doc_id: str):
    doc_dir = UPLOAD_DIR / doc_id
    if doc_dir.exists():
        shutil.rmtree(doc_dir)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3003)
