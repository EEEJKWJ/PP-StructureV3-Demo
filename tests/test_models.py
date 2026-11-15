import json
import logging
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from fastapi.testclient import TestClient

import server
from app.engines.base import TableAdapter, TableOutput
from app.engines.http import HttpTableAdapter
from app.engines.registry import (
    DEFAULT_CONFIG_PATH,
    ModelRegistry,
    RegistryConfigError,
)
from app.services.table import TableService


class FakeTableAdapter(TableAdapter):
    backend = "fake"

    def load(self):
        if self.model_id.startswith("broken"):
            self.load_error = "模拟加载失败"
            raise RuntimeError(self.load_error)
        self.engine = {"model_id": self.model_id}
        self.loaded = True
        self.load_error = ""
        return self.engine

    def predict(self, image_path, table_bbox, options=None):
        self.ensure_loaded()
        cells = [
            {
                "row": 0,
                "col": 0,
                "text": self.model_id,
                "bbox": [0, 0, 10, 10],
            },
            {
                "row": 0,
                "col": 1,
                "text": "value",
                "bbox": [10, 0, 20, 10],
            },
        ]
        return TableOutput(
            cells=cells,
            html="<table><tr><td>fake</td><td>value</td></tr></table>",
        )


def fake_factory(config, logger, ocr_provider):
    return FakeTableAdapter(config, logger, ocr_provider)


class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test-model-registry")

    @staticmethod
    def _model(model_id, adapter="fake", fallback=None):
        return {
            "id": model_id,
            "name": model_id,
            "version": "test",
            "adapter": adapter,
            "backend": "fake",
            "device": "cpu",
            "source": "test",
            "fallback": fallback,
            "options": {},
        }

    def _write_config(self, directory, models, default_model=None):
        path = Path(directory) / "registry.json"
        payload = {
            "default_model": default_model or models[0]["id"],
            "models": models,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def test_default_registry_lists_builtin_models(self):
        registry = ModelRegistry(
            self.logger,
            config_path=DEFAULT_CONFIG_PATH,
        )

        models = registry.list_models()

        self.assertEqual(models["default_model"], "pp-tablemagic-v2")
        self.assertEqual(
            [item["id"] for item in models["models"]],
            ["pp-tablemagic-v2", "rapidtable-slanet-plus"],
        )
        self.assertTrue(
            all(item["status"] == "available" for item in models["models"])
        )

    def test_activation_switches_adapter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_config(
                temp_dir,
                [self._model("first"), self._model("second")],
            )
            registry = ModelRegistry(
                self.logger,
                config_path=path,
                adapter_factories={"fake": fake_factory},
            )

            first = registry.activate("first")
            second = registry.activate("second")

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertEqual(second["active_model"], "second")
        self.assertEqual(registry.active_adapter.model_id, "second")
        self.assertEqual(
            registry.health()["active_health"]["status"],
            "ready",
        )

    def test_failed_activation_keeps_current_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_config(
                temp_dir,
                [
                    self._model("working"),
                    self._model("broken-model"),
                ],
            )
            registry = ModelRegistry(
                self.logger,
                config_path=path,
                adapter_factories={"fake": fake_factory},
            )
            registry.activate("working")

            result = registry.activate("broken-model")

        self.assertFalse(result["ok"])
        self.assertEqual(result["active_model"], "working")
        self.assertTrue(result["kept_previous"])
        self.assertIn("模拟加载失败", result["error"])

    def test_initial_failure_uses_configured_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_config(
                temp_dir,
                [
                    self._model(
                        "broken-default",
                        fallback="working-fallback",
                    ),
                    self._model("working-fallback"),
                ],
                default_model="broken-default",
            )
            registry = ModelRegistry(
                self.logger,
                config_path=path,
                adapter_factories={"fake": fake_factory},
            )

            result = registry.activate("broken-default")

        self.assertFalse(result["ok"])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["active_model"], "working-fallback")
        self.assertIn("模拟加载失败", result["error"])

    def test_unknown_model_falls_back_to_default_when_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_config(
                temp_dir,
                [self._model("working")],
            )
            registry = ModelRegistry(
                self.logger,
                config_path=path,
                adapter_factories={"fake": fake_factory},
            )

            result = registry.activate("missing")

        self.assertFalse(result["ok"])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["active_model"], "working")

    def test_invalid_config_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_config(
                temp_dir,
                [self._model("duplicate"), self._model("duplicate")],
            )

            with self.assertRaises(RegistryConfigError):
                ModelRegistry(
                    self.logger,
                    config_path=path,
                    adapter_factories={"fake": fake_factory},
                )

    def test_reload_failure_preserves_active_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_config(
                temp_dir,
                [self._model("working")],
            )
            registry = ModelRegistry(
                self.logger,
                config_path=path,
                adapter_factories={"fake": fake_factory},
            )
            registry.activate("working")
            path.write_text("{invalid json", encoding="utf-8")

            with self.assertRaises(RegistryConfigError):
                registry.reload()

        self.assertEqual(registry.active_adapter.model_id, "working")
        self.assertIn("无法读取", registry.last_reload_error)

    def test_unsupported_adapter_reports_explicit_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_config(
                temp_dir,
                [self._model("custom-onnx", adapter="onnx")],
            )
            registry = ModelRegistry(
                self.logger,
                config_path=path,
            )

            result = registry.activate("custom-onnx")

        self.assertFalse(result["ok"])
        self.assertIsNone(result["active_model"])
        self.assertIn("适配器尚未实现", result["error"])

    def test_table_service_adds_standard_model_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_config(
                temp_dir,
                [self._model("fake-table")],
            )
            service = TableService(
                self.logger,
                model_config_path=path,
                adapter_factories={"fake": fake_factory},
            )

            result = service.recognize("unused.jpg", [0, 0, 20, 10])

        self.assertEqual(result["model"]["id"], "fake-table")
        self.assertEqual(result["row_count"], 1)
        self.assertEqual(result["col_count"], 2)
        self.assertGreaterEqual(result["elapsed_ms"], 0)
        self.assertIn("<table>", result["html"])

    def test_http_adapter_converts_relative_cell_coordinates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "table.jpg"
            Image.new("RGB", (100, 80), "white").save(image_path)
            adapter = HttpTableAdapter(
                {
                    "id": "http-table",
                    "name": "HTTP Table",
                    "version": "1",
                    "adapter": "http",
                    "backend": "http",
                    "device": "remote",
                    "source": "https://example.test/table",
                    "options": {
                        "cells_relative": True,
                    },
                    "artifact_hash": None,
                },
                self.logger,
            )
            adapter._post_json = lambda endpoint, payload: {
                "html": "<table><tr><td>A</td></tr></table>",
                "cells": [{
                    "row": 0,
                    "col": 0,
                    "text": "A",
                    "bbox": [0, 0, 10, 10],
                }],
            }

            output = adapter.predict(
                str(image_path),
                [20, 10, 80, 60],
            )

        self.assertIsNotNone(output)
        self.assertEqual(output.cells[0]["bbox"], [15, 5, 25, 15])

    def test_model_api_lists_registered_models(self):
        client = TestClient(server.app)

        response = client.get("/api/models")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["default_model"], "pp-tablemagic-v2")
        self.assertEqual(len(payload["models"]), 2)


if __name__ == "__main__":
    unittest.main()
