"""表格模型注册、加载、切换与回退。"""

import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from app.engines.base import UnsupportedTableAdapter


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "configs" / "models" / "registry.json"
MODEL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


class RegistryConfigError(ValueError):
    """模型注册清单无效。"""


@dataclass(frozen=True)
class ModelConfig:
    """一个已校验的模型配置。"""

    data: dict

    @property
    def model_id(self):
        return self.data["id"]

    @property
    def adapter(self):
        return self.data["adapter"]

    @property
    def fallback(self):
        return self.data.get("fallback")

    def public_dict(self):
        options = dict(self.data.get("options", {}))
        if "headers" in options:
            options["headers"] = {
                key: "***" for key in options["headers"]
            }
        for key in ("api_key", "token", "password", "secret"):
            if key in options:
                options[key] = "***"
        return {
            "id": self.data["id"],
            "name": self.data["name"],
            "version": self.data["version"],
            "adapter": self.data["adapter"],
            "backend": self.data["backend"],
            "device": self.data["device"],
            "source": self.data["source"],
            "fallback": self.data.get("fallback"),
            "artifact_hash": self.data.get("artifact_hash"),
            "options": options,
        }


class ModelRegistry:
    """维护模型清单，并以原子方式切换当前适配器。"""

    def __init__(
        self,
        logger,
        config_path=None,
        ocr_provider=None,
        adapter_factories=None,
    ):
        self.log = logger
        self.config_path = Path(
            config_path
            or os.environ.get(
                "PP_STRUCTURE_MODEL_CONFIG",
                str(DEFAULT_CONFIG_PATH),
            )
        )
        self.ocr_provider = ocr_provider
        self.adapter_factories = (
            adapter_factories or self._default_adapter_factories()
        )
        self.default_model_id = ""
        self.models = {}
        self.active_adapter = None
        self.last_reload_error = ""
        self._active_lock = threading.RLock()
        self.reload()

    @staticmethod
    def _default_adapter_factories():
        def tablemagic(config, logger, ocr_provider):
            from app.engines.tablemagic import TableMagicAdapter

            return TableMagicAdapter(config, logger, ocr_provider)

        def rapidtable(config, logger, ocr_provider):
            from app.engines.rapid_table import RapidTableAdapter

            return RapidTableAdapter(config, logger, ocr_provider)

        def http(config, logger, ocr_provider):
            from app.engines.http import HttpTableAdapter

            return HttpTableAdapter(config, logger, ocr_provider)

        def unsupported(config, logger, ocr_provider):
            return UnsupportedTableAdapter(config, logger, ocr_provider)

        return {
            "tablemagic": tablemagic,
            "rapidtable": rapidtable,
            "http": http,
            "onnx": unsupported,
            "pytorch": unsupported,
        }

    def _read_config(self):
        if not self.config_path.exists():
            raise RegistryConfigError(
                f"模型注册清单不存在: {self.config_path}"
            )
        try:
            data = json.loads(
                self.config_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryConfigError(
                f"模型注册清单无法读取: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise RegistryConfigError("模型注册清单根节点必须是对象")
        if not isinstance(data.get("models"), list):
            raise RegistryConfigError("模型注册清单缺少 models 数组")

        models = {}
        for raw_model in data["models"]:
            model = self._normalize_model(raw_model)
            if model.model_id in models:
                raise RegistryConfigError(
                    f"模型 ID 重复: {model.model_id}"
                )
            models[model.model_id] = model

        default_model = data.get("default_model")
        if default_model not in models:
            raise RegistryConfigError(
                f"default_model 不存在: {default_model}"
            )

        for model in models.values():
            fallback = model.fallback
            if fallback and fallback not in models:
                raise RegistryConfigError(
                    f"{model.model_id} 的 fallback 不存在: {fallback}"
                )
            if fallback == model.model_id:
                raise RegistryConfigError(
                    f"{model.model_id} 不能回退到自身"
                )
        return default_model, models

    @staticmethod
    def _normalize_model(raw_model):
        if not isinstance(raw_model, dict):
            raise RegistryConfigError("每个模型配置必须是对象")

        model_id = str(raw_model.get("id", "")).strip()
        if not MODEL_ID_PATTERN.fullmatch(model_id):
            raise RegistryConfigError(f"模型 ID 无效: {model_id!r}")

        adapter = str(raw_model.get("adapter", "")).strip().lower()
        if not adapter:
            raise RegistryConfigError(f"{model_id} 缺少 adapter")

        options = raw_model.get("options", {})
        if not isinstance(options, dict):
            raise RegistryConfigError(
                f"{model_id} 的 options 必须是对象"
            )

        config = {
            "id": model_id,
            "name": str(raw_model.get("name") or model_id),
            "version": str(raw_model.get("version") or "unknown"),
            "adapter": adapter,
            "backend": str(raw_model.get("backend") or adapter),
            "device": str(raw_model.get("device") or "auto"),
            "source": str(raw_model.get("source") or ""),
            "options": options,
            "fallback": raw_model.get("fallback"),
            "artifact_hash": raw_model.get("artifact_hash"),
        }
        return ModelConfig(config)

    def reload(self):
        """重新读取清单；失败时保留上一份有效注册信息。"""
        with self._active_lock:
            try:
                default_model, models = self._read_config()
            except Exception as exc:
                self.last_reload_error = str(exc)
                raise

            active_id = (
                self.active_adapter.model_id
                if self.active_adapter is not None
                else None
            )
            self.default_model_id = default_model
            self.models = models
            self.last_reload_error = ""
            warning = ""
            if active_id and active_id not in models:
                warning = (
                    f"当前模型 {active_id} 已不在新清单中，"
                    "保持运行但不再列为可激活模型"
                )
                self.log.warning(warning)
            return {
                "ok": True,
                "config_path": str(self.config_path),
                "default_model": self.default_model_id,
                "model_count": len(self.models),
                "active_model": active_id,
                "warning": warning,
            }

    def _create_adapter(self, config):
        factory = self.adapter_factories.get(config.adapter)
        if factory is None:
            factory = lambda cfg, logger, provider: (
                UnsupportedTableAdapter(cfg.data, logger, provider)
            )
        return factory(config.data, self.log, self.ocr_provider)

    def _load_model(self, model_id):
        config = self.models[model_id]
        adapter = self._create_adapter(config)
        adapter.ensure_loaded()
        return adapter

    def activate(self, model_id=None, use_fallback=True):
        """加载成功后切换模型；失败时保留当前可用模型。"""
        requested_id = model_id or self.default_model_id
        with self._active_lock:
            previous = self.active_adapter
            requested = self.models.get(requested_id)
            if requested is None:
                error = f"模型不存在: {requested_id}"
                fallback = self._try_fallback(
                    requested=None,
                    previous=previous,
                    use_fallback=use_fallback,
                )
                return self._activation_result(
                    False,
                    requested_id,
                    error,
                    fallback,
                )

            try:
                adapter = self._load_model(requested_id)
            except Exception as exc:
                error = str(exc) or exc.__class__.__name__
                fallback = self._try_fallback(
                    requested=requested,
                    previous=previous,
                    use_fallback=use_fallback,
                )
                return self._activation_result(
                    False,
                    requested_id,
                    error,
                    fallback,
                )

            self.active_adapter = adapter
            return self._activation_result(
                True,
                requested_id,
                "",
                None,
            )

    def _try_fallback(self, requested, previous, use_fallback):
        if not use_fallback:
            return None

        if previous is not None:
            return {
                "model_id": previous.model_id,
                "fallback_used": False,
                "kept_previous": True,
                "error": "",
            }

        candidates = []
        if requested is not None and requested.fallback:
            candidates.append(requested.fallback)
        if self.default_model_id not in candidates:
            candidates.append(self.default_model_id)

        errors = []
        for candidate_id in candidates:
            if candidate_id not in self.models:
                continue
            if (
                requested is not None
                and candidate_id == requested.model_id
            ):
                continue
            try:
                adapter = self._load_model(candidate_id)
            except Exception as exc:
                errors.append(f"{candidate_id}: {exc}")
                continue
            self.active_adapter = adapter
            return {
                "model_id": candidate_id,
                "fallback_used": True,
                "kept_previous": False,
                "error": "; ".join(errors),
            }
        return {
            "model_id": None,
            "fallback_used": True,
            "kept_previous": False,
            "error": "; ".join(errors),
        }

    def _activation_result(self, ok, requested_id, error, fallback):
        active = self.active_adapter
        result = {
            "ok": ok,
            "requested_model": requested_id,
            "active_model": (
                active.model_id if active is not None else None
            ),
            "error": error,
            "fallback_used": False,
            "kept_previous": False,
        }
        if fallback:
            result["active_model"] = fallback["model_id"]
            result["fallback_used"] = fallback["fallback_used"]
            result["kept_previous"] = fallback["kept_previous"]
            if fallback["error"]:
                result["fallback_error"] = fallback["error"]
        if active is not None:
            result["model"] = active.model_info
        return result

    def ensure_active(self):
        if self.active_adapter is None:
            result = self.activate(self.default_model_id)
            if not result["ok"] and self.active_adapter is None:
                raise RuntimeError(
                    result.get("fallback_error")
                    or result.get("error")
                    or "没有可用的表格模型"
                )
        return self.active_adapter

    def list_models(self):
        active_id = (
            self.active_adapter.model_id
            if self.active_adapter is not None
            else None
        )
        models = []
        for model in self.models.values():
            item = model.public_dict()
            if model.model_id == active_id:
                item.update(
                    self.active_adapter.health_check()
                )
                item["active"] = True
            else:
                item["status"] = "available"
                item["error"] = ""
                item["active"] = False
            models.append(item)
        return {
            "ok": True,
            "config_path": str(self.config_path),
            "default_model": self.default_model_id,
            "active_model": active_id,
            "models": models,
        }

    def health(self):
        active = self.active_adapter
        active_health = (
            active.health_check()
            if active is not None
            else {
                "status": "not_loaded",
                "error": "",
            }
        )
        ok = (
            self.last_reload_error == ""
            and active is not None
            and active_health["status"] == "ready"
        )
        return {
            "ok": ok,
            "config_path": str(self.config_path),
            "config_error": self.last_reload_error,
            "default_model": self.default_model_id,
            "active_model": (
                active.model_id if active is not None else None
            ),
            "active_health": active_health,
        }
