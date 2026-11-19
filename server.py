"""PP-StructureV3 版面结构化演示
- 版面检测：PP-DocLayoutV3 (HuggingFace transformers + PyTorch)
- 文本提取：PyMuPDF（文本型 PDF）/ PP-OCRv5 ONNX（扫描件，全页 OCR + 按区域归类）
- 表格提取：PyMuPDF（文本型 PDF）/ PP-TableMagic V2 完整产线（扫描件）
  · 表格分类（有线/无线自动区分）
  · 单元格检测 (RT-DETR-L)
  · 结构识别 (SLANeXt)
  · 文字检测+识别 (PP-OCRv4)
  · 降级方案：RapidTable (SLANet_plus ONNX)"""

import uuid
import shutil
import logging
import threading
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.services.layout import LayoutService
from app.services.ocr import OCRService
from app.services.document import DocumentService
from app.services.table import TableService
from app.services.structure import StructureService
from app.services.official_parser import OfficialPPStructureParser
from app.evaluation.reports import EvaluationReports
from app.datasets.registry import DatasetRegistry
from app.evaluation.workbench import SampleWorkbenchService
from app.evaluation.comparison_reports import ComparisonReports

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pp-struct")

app = FastAPI(title="PP-StructureV3 Demo")
app.mount(
    "/frontend",
    StaticFiles(directory="frontend"),
    name="frontend",
)

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
_evaluation_reports = EvaluationReports(Path("experiments"))
_dataset_registry = DatasetRegistry()
_sample_workbench = SampleWorkbenchService(_dataset_registry, log)
_comparison_reports = ComparisonReports(Path("experiments/comparisons"))
_official_parser = OfficialPPStructureParser(log)


def _sync_table_engine_state():
    global _table_engine, _table_engine_type
    _table_engine = _table_service.engine
    _table_engine_type = _table_service.engine_type
    return _table_engine


def _get_table_engine():
    """兼容现有状态接口，表格引擎由 TableService 统一管理。"""
    _table_service.get_engine()
    return _sync_table_engine_state()


def _activate_table_model(model_id: str) -> dict:
    """兼容层：激活模型后同步旧状态变量。"""
    result = _table_service.activate_model(model_id)
    _sync_table_engine_state()
    return result


def recognize_table_structure(image_path: str, table_bbox: list) -> dict | None:
    """识别表格并返回单元格和 HTML。"""
    return _table_service.recognize(image_path, table_bbox)


def _apply_table_result(
    block,
    cells,
    table_html="",
    table_result=None,
):
    """统一写入表格单元格和 HTML，保持所有识别路径字段一致。"""
    block["cells"] = cells
    block["table_html"] = table_html or _table_service.cells_to_html(cells)
    if table_result:
        block["table_model"] = table_result.get("model")
        block["table_elapsed_ms"] = table_result.get("elapsed_ms")
        block["table_row_count"] = table_result.get("row_count")
        block["table_col_count"] = table_result.get("col_count")


# ── 目录行解析 ────────────────────────────────────────────────────
def _dedupe_repeated_chars(text: str) -> str:
    """兼容入口：目录字符去重由 StructureService 提供。"""
    return _structure_service.dedupe_repeated_chars(text)


def _parse_toc_line(line: str) -> dict | None:
    """兼容入口：目录行解析由 StructureService 提供。"""
    return _structure_service.parse_toc_line(line)


def _merge_toc_lines(ocr_items: list[dict]) -> list[dict]:
    """兼容入口：目录残行合并由 StructureService 提供。"""
    return _structure_service.merge_toc_lines(ocr_items)


def _is_toc_block(texts: list[str]) -> bool:
    """兼容入口：目录区域判断由 StructureService 提供。"""
    return _structure_service.is_toc_block(texts)


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
_structure_service = StructureService(
    log,
    _table_service,
    _document_service.extract_text_from_pdf,
)


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


@app.get("/api/models")
async def list_models():
    """返回已注册表格模型及当前激活状态。"""
    return _table_service.list_models()


@app.get("/api/models/health")
async def models_health():
    """返回注册清单和当前适配器健康状态。"""
    return _table_service.health()


@app.post("/api/models/reload")
async def reload_models():
    """重载模型清单，不隐式切换当前模型。"""
    try:
        return _table_service.reload_models()
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "active_model": _table_service.active_model_id,
            "config_error": _table_service.registry.last_reload_error,
        }


@app.post("/api/models/{model_id}/activate")
async def activate_model(model_id: str):
    """加载并激活指定模型；失败时保留或回退到可用模型。"""
    return _activate_table_model(model_id)


@app.get("/api/evaluations")
async def list_evaluations():
    """返回存在指标报告的批量实验摘要。"""
    return {
        "ok": True,
        "reports": _evaluation_reports.list_reports(),
    }


@app.get("/api/evaluations/latest/metrics")
async def latest_evaluation_metrics():
    """返回最新实验指标；没有实验时返回明确空状态。"""
    report = _evaluation_reports.latest_metrics()
    if report is None:
        return {
            "ok": False,
            "error": "暂无评测结果",
            "report": None,
        }
    return {"ok": True, "report": report}


@app.get("/api/evaluations/{experiment_id}/metrics")
async def evaluation_metrics(experiment_id: str):
    """返回指定实验的完整指标报告。"""
    try:
        return _evaluation_reports.get_metrics(experiment_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/datasets")
async def list_datasets():
    """返回可在单样本工作台浏览的数据集。"""
    return {
        "ok": True,
        "datasets": _sample_workbench.list_datasets(),
    }


@app.get("/api/datasets/{dataset_id}/samples")
async def list_dataset_samples(
    dataset_id: str,
    offset: int = 0,
    limit: int = 20,
):
    """分页返回数据集样本元数据。"""
    try:
        return _sample_workbench.list_samples(
            dataset_id,
            offset=offset,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/datasets/{dataset_id}/samples/{sample_id}")
async def dataset_sample(dataset_id: str, sample_id: str):
    """返回一个样本的元数据。"""
    try:
        sample = _sample_workbench.get_sample(dataset_id, sample_id)
        return {
            "ok": True,
            "sample": _sample_workbench.public_sample(sample),
        }
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/api/datasets/{dataset_id}/samples/{sample_id}/image"
)
async def dataset_sample_image(dataset_id: str, sample_id: str):
    """返回数据集中的原始样本图片。"""
    try:
        content = _sample_workbench.get_image(dataset_id, sample_id)
        return Response(content=content, media_type="image/jpeg")
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/api/datasets/{dataset_id}/samples/{sample_id}/run"
)
async def run_dataset_sample(dataset_id: str, sample_id: str):
    """使用当前模型重新识别一个数据集样本。"""
    try:
        return _sample_workbench.run_sample(
            dataset_id,
            sample_id,
            recognize_table_structure,
        )
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/comparisons")
async def list_comparisons():
    """列出已有模型对比报告。"""
    return {
        "ok": True,
        "reports": _comparison_reports.list_reports(),
    }


@app.get("/api/comparisons/{comparison_id}")
async def comparison_detail(comparison_id: str):
    """返回完整模型对比报告。"""
    try:
        return _comparison_reports.get_report(comparison_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/comparisons/{comparison_id}/export/{export_format}")
async def comparison_export(
    comparison_id: str,
    export_format: str,
):
    """下载 JSON、CSV 或 HTML 对比报告。"""
    try:
        path = _comparison_reports.export_path(
            comparison_id,
            export_format,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media_types = {
        "json": "application/json",
        "csv": "text/csv",
        "html": "text/html",
    }
    return FileResponse(
        path,
        media_type=media_types[export_format],
        filename=f"{comparison_id}-report.{export_format}",
    )


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
    """兼容入口：文本清理由 StructureService 提供。"""
    return _structure_service.clean_text(text)


def _split_value(text: str):
    """兼容入口：单元格多行值解析由 StructureService 提供。"""
    return _structure_service.split_value(text)


def build_structured(blocks: list, pdf_path: str | None, page_num: int,
                     image_path: str, has_text_layer: bool,
                     ocr_by_block: dict[int, list[dict]] | None = None) -> dict:
    """兼容入口：页面结构化由 StructureService 提供。"""
    return _structure_service.build_structured(
        blocks,
        pdf_path,
        page_num,
        image_path,
        has_text_layer,
        ocr_by_block,
    )

def _table_to_grid(cells: list[dict]) -> tuple[list[list[str]], int, int]:
    """兼容入口：表格网格转换由 StructureService 提供。"""
    return _structure_service.table_to_grid(cells)


def _row_similarity(row_a: list[str], row_b: list[str]) -> float:
    """兼容入口：表头相似度由 StructureService 提供。"""
    return _structure_service.row_similarity(row_a, row_b)


def _col_widths_ratio(cells: list[dict]) -> list[float]:
    """兼容入口：列宽比例由 StructureService 提供。"""
    return _structure_service.col_widths_ratio(cells)


def _col_structure_similar(cells_a: list[dict], cells_b: list[dict],
                           tolerance: float = 0.15) -> bool:
    """兼容入口：列结构比较由 StructureService 提供。"""
    return _structure_service.col_structure_similar(
        cells_a,
        cells_b,
        tolerance,
    )


def merge_cross_page_tables(pages_result: list[dict]) -> list[dict]:
    """兼容入口：跨页表格合并由 StructureService 提供。"""
    return _structure_service.merge_cross_page_tables(pages_result)


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
    filename = Path(file.filename or "upload").name
    is_pdf = filename.lower().endswith(".pdf")

    if is_pdf:
        page_count = pdf_save_and_count(content, doc_dir)
        filenames = [f"page_{i:03d}.png" for i in range(1, page_count + 1)]
    else:
        try:
            _document_service.save_image(content, doc_dir)
        except (OSError, ValueError) as exc:
            shutil.rmtree(doc_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail="无法读取上传图片，请使用 JPG、PNG、WEBP、BMP 或 TIFF 格式",
            ) from exc
        filenames = ["page_001.png"]

    return {
        "doc_id": doc_id,
        "filename": filename,
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
                            table_result,
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
                        table_result,
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


@app.post("/api/official/detect/{doc_id}/{page_num}")
async def official_detect(doc_id: str, page_num: int):
    """使用官方 PP-StructureV3 解析整页。"""
    doc_dir = UPLOAD_DIR / doc_id
    try:
        img_path = ensure_page_image(doc_dir, page_num)
    except (FileNotFoundError, IndexError) as exc:
        raise HTTPException(
            status_code=404,
            detail="page not found",
        ) from exc

    is_pdf = (doc_dir / "source.pdf").exists()
    try:
        return await run_in_threadpool(
            _official_parser.parse_page,
            str(img_path),
            page_num,
            is_pdf,
        )
    except Exception as exc:
        log.error("官方 PP-StructureV3 页面解析失败: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"官方 PP-StructureV3 解析失败: {exc}",
        ) from exc


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
                                table_result,
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
                            table_result,
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
