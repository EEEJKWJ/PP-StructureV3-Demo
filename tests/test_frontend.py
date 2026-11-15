import re
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

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
        ):
            self.assertIn(selector, stylesheet)

    def test_page_contains_model_picker_and_metrics_placeholders(self):
        self.assertIn('id="modelSelect"', self.index)
        self.assertIn('id="metricsPanel"', self.index)
        self.assertIn('id="metricsContent"', self.index)
        self.assertIn('id="workbenchView"', self.index)
        self.assertIn('id="workbenchTab"', self.index)

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


if __name__ == "__main__":
    unittest.main()
