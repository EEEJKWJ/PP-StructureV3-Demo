"""表格模型注册与标准识别服务。"""

import threading
import time

from app.engines.base import (
    cells_to_html,
    parse_html_table,
    table_dimensions,
)
from app.engines.registry import ModelRegistry


class TableService:
    """通过注册表选择模型，并维持现有表格输出接口。"""

    def __init__(
        self,
        logger,
        ocr_provider=None,
        model_config_path=None,
        adapter_factories=None,
    ):
        self.log = logger
        self.ocr_provider = ocr_provider
        self.registry = ModelRegistry(
            logger,
            config_path=model_config_path,
            ocr_provider=ocr_provider,
            adapter_factories=adapter_factories,
        )
        self.engine = None
        self.engine_type = ""
        self._lock = threading.Lock()

    @property
    def active_model_id(self):
        adapter = self.registry.active_adapter
        return adapter.model_id if adapter is not None else None

    def _sync_active_engine(self):
        adapter = self.registry.active_adapter
        if adapter is None:
            self.engine = None
            self.engine_type = ""
        else:
            self.engine = adapter.engine
            self.engine_type = adapter.backend

    def get_engine(self):
        """初始化默认模型，并返回正在使用的底层引擎。"""
        with self._lock:
            self.registry.ensure_active()
            self._sync_active_engine()
            return self.engine

    def recognize(self, image_path, table_bbox, options=None):
        """使用当前模型识别一个表格区域。"""
        self.get_engine()
        adapter = self.registry.active_adapter
        if adapter is None:
            return None

        started_at = time.perf_counter()
        output = adapter.predict(image_path, table_bbox, options)
        if output is None:
            return None

        elapsed_ms = round(
            (time.perf_counter() - started_at) * 1000,
            2,
        )
        row_count, col_count = table_dimensions(output.cells)
        result = {
            "cells": output.cells,
            "html": output.html,
            "model": adapter.model_info,
            "elapsed_ms": elapsed_ms,
            "row_count": row_count,
            "col_count": col_count,
        }
        if output.raw is not None:
            result["raw"] = output.raw
        return result

    def activate_model(self, model_id):
        with self._lock:
            result = self.registry.activate(model_id)
            self._sync_active_engine()
            return result

    def reload_models(self):
        with self._lock:
            result = self.registry.reload()
            self._sync_active_engine()
            return result

    def list_models(self):
        return self.registry.list_models()

    def health(self):
        return self.registry.health()

    @staticmethod
    def parse_html_table(table_html, cell_boxes_batch, offset_x, offset_y):
        """兼容入口：HTML 表格解析由适配器基础层提供。"""
        return parse_html_table(
            table_html,
            cell_boxes_batch,
            offset_x,
            offset_y,
        )

    @staticmethod
    def cells_to_html(cells):
        """兼容入口：基础表格 HTML 生成由适配器基础层提供。"""
        return cells_to_html(cells)
