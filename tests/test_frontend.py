import re
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import server


PROJECT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"


class FrontendStructureTests(unittest.TestCase):
    def setUp(self):
        self.index = (PROJECT_DIR / "index.html").read_text(
            encoding="utf-8"
        )

    def test_index_uses_external_css_and_module(self):
        self.assertNotIn("<style", self.index)
        self.assertIn(
            '<link rel="stylesheet" href="/frontend/css/app.css"',
            self.index,
        )
        self.assertIn(
            '<script type="module" src="/frontend/js/app.js"></script>',
            self.index,
        )

    def test_all_frontend_assets_exist(self):
        assets = [
            "css/app.css",
            "js/app.js",
            "js/api.js",
            "js/comparison.js",
            "js/interactions.js",
            "js/overlay.js",
            "js/results.js",
            "js/state.js",
            "js/table-html.js",
            "js/theme.js",
            "js/workbench.js",
            "tests/sanitizer.html",
        ]
        for relative_path in assets:
            self.assertTrue(
                (FRONTEND_DIR / relative_path).is_file(),
                relative_path,
            )

    def test_module_imports_resolve_to_local_files(self):
        import_pattern = re.compile(
            r'from\s+["\'](\.[^"\']+)["\']'
        )
        for module_path in (FRONTEND_DIR / "js").glob("*.js"):
            source = module_path.read_text(encoding="utf-8")
            for imported in import_pattern.findall(source):
                target = (module_path.parent / imported).resolve()
                self.assertTrue(
                    target.is_file(),
                    f"{module_path.name} -> {imported}",
                )

    def test_app_exposes_inline_event_handlers(self):
        app_source = (
            FRONTEND_DIR / "js" / "app.js"
        ).read_text(encoding="utf-8")
        handlers = [
            "switchResultView",
            "selectPage",
            "onImageLoaded",
            "highlightBlock",
            "highlightCell",
            "highlightScanCell",
            "highlightTocItem",
            "toggleCells",
            "switchCellsMode",
            "switchTableHtmlMode",
            "toggleTreeNode",
        ]
        binding_start = app_source.index("Object.assign(window")
        binding_block = app_source[binding_start:]
        for handler in handlers:
            self.assertIn(handler, binding_block)
        self.assertIn(
            'dataset.frontendReady = "true"',
            app_source,
        )

    def test_css_keeps_three_column_core_styles(self):
        stylesheet = (
            FRONTEND_DIR / "css" / "app.css"
        ).read_text(encoding="utf-8")
        for selector in (
            ".container",
            ".sidebar",
            ".preview",
            ".result-panel",
            ".cells-table",
            ".bbox-overlay",
            ".json-view",
            ".metrics-panel",
            ".table-html-render",
            ".model-picker",
            ".workbench-view",
            ".workbench-layout",
            ".comparison-view",
            ".comparison-table",
        ):
            self.assertIn(selector, stylesheet)

    def test_page_contains_model_picker_and_metrics_placeholders(self):
        self.assertIn('id="modelSelect"', self.index)
        self.assertIn('id="metricsPanel"', self.index)
        self.assertIn('id="metricsContent"', self.index)
        self.assertIn('id="workbenchView"', self.index)
        self.assertIn('id="workbenchTab"', self.index)
        self.assertIn('id="comparisonView"', self.index)
        self.assertIn('id="comparisonTab"', self.index)

    def test_sanitizer_blocks_executable_content(self):
        source = (
            FRONTEND_DIR / "js" / "table-html.js"
        ).read_text(encoding="utf-8")
        for token in (
            "ALLOWED_TAGS",
            "ALLOWED_ATTRIBUTES",
            "BLOCKED_SUBTREES",
            '"script"',
            '"iframe"',
        ):
            self.assertIn(token, source)

    def test_comparison_view_and_cli_are_registered(self):
        comparison_source = (
            FRONTEND_DIR / "js" / "comparison.js"
        ).read_text(encoding="utf-8")

        self.assertTrue(
            (PROJECT_DIR / "scripts" / "compare_models.py").is_file()
        )
        self.assertIn("pp:comparisons-open", comparison_source)
        cli_source = (
            PROJECT_DIR / "scripts" / "compare_models.py"
        ).read_text(encoding="utf-8")
        self.assertIn("report.json", cli_source)
        self.assertIn("不可比较", comparison_source)

    def test_comparison_empty_state_does_not_fabricate_metrics(self):
        comparison_source = (
            FRONTEND_DIR / "js" / "comparison.js"
        ).read_text(encoding="utf-8")

        self.assertIn("暂无模型对比报告", comparison_source)
        self.assertIn("compare_models.py", comparison_source)

    def test_detection_response_is_validated_before_drawing(self):
        app_source = (
            FRONTEND_DIR / "js" / "app.js"
        ).read_text(encoding="utf-8")
        self.assertIn("function requirePageDetection", app_source)
        self.assertIn("缺少 blocks 数组", app_source)
        self.assertIn(
            "requirePageDetection(\n      await detectPage",
            app_source,
        )
        self.assertIn('text_source === "official_v3"', app_source)
        self.assertIn("官方 PP-StructureV3", app_source)
        self.assertIn("当前官方适配层尚未执行跨页表格合并", app_source)

    def test_no_frontend_build_manifest_is_required(self):
        self.assertFalse((PROJECT_DIR / "package.json").exists())
        self.assertFalse((PROJECT_DIR / "vite.config.js").exists())


class FrontendStaticRouteTests(unittest.TestCase):
    def test_fastapi_serves_page_css_and_modules(self):
        client = TestClient(server.app)

        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        for relative_path in (
            "css/app.css",
            "js/api.js",
            "js/app.js",
            "js/comparison.js",
            "js/interactions.js",
            "js/overlay.js",
            "js/results.js",
            "js/state.js",
            "js/table-html.js",
            "js/theme.js",
            "js/workbench.js",
        ):
            response = client.get(f"/frontend/{relative_path}")
            self.assertEqual(response.status_code, 200, relative_path)
            if relative_path.endswith(".css"):
                self.assertIn(
                    "text/css",
                    response.headers["content-type"],
                )
            else:
                self.assertIn(
                    "javascript",
                    response.headers["content-type"],
                )


class ImageUploadApiTests(unittest.TestCase):
    @staticmethod
    def jpeg_bytes():
        image = Image.new("RGB", (32, 20), "white")
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_jpeg_upload_is_saved_as_png(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir)
            with patch.object(server, "UPLOAD_DIR", upload_dir):
                client = TestClient(server.app)
                response = client.post(
                    "/api/upload",
                    files={
                        "file": (
                            "table.jpg",
                            self.jpeg_bytes(),
                            "image/jpeg",
                        ),
                    },
                )
                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertFalse(payload["is_pdf"])
                self.assertEqual(payload["pages"], ["page_001.png"])
                image_path = (
                    upload_dir
                    / payload["doc_id"]
                    / "page_001.png"
                )
                self.assertTrue(image_path.is_file())
                with Image.open(image_path) as image:
                    self.assertEqual(image.format, "PNG")

                image_response = client.get(
                    f"/api/images/{payload['doc_id']}/page_001.png"
                )
                self.assertEqual(image_response.status_code, 200)
                self.assertEqual(
                    image_response.headers["content-type"],
                    "image/png",
                )
                blocks = [{
                    "label": "text",
                    "bbox": [0, 0, 32, 20],
                    "order": 0,
                }]
                with (
                    patch.object(
                        server,
                        "detect_layout",
                        return_value=blocks,
                    ),
                    patch.object(
                        server,
                        "ocr_blocks",
                        return_value={0: [{
                            "text": "测试",
                            "bbox": [1, 1, 10, 10],
                        }]},
                    ),
                    patch.object(
                        server,
                        "build_structured",
                        return_value=[],
                    ),
                ):
                    detect_response = client.post(
                        f"/api/detect/{payload['doc_id']}/1"
                    )
                self.assertEqual(detect_response.status_code, 200)
                detected = detect_response.json()
                self.assertEqual(detected["text_source"], "ocr")
                self.assertEqual(len(detected["blocks"]), 1)
                self.assertEqual(
                    detected["blocks"][0]["ocr_text"],
                    "测试",
                )

    def test_invalid_image_upload_returns_400(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir)
            with patch.object(server, "UPLOAD_DIR", upload_dir):
                client = TestClient(server.app)
                response = client.post(
                    "/api/upload",
                    files={
                        "file": (
                            "broken.jpg",
                            b"not an image",
                            "image/jpeg",
                        ),
                    },
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(list(upload_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
