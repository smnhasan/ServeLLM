import logging
import time
from pathlib import Path

import yaml

from app.core.config import settings
from app.schemas.openai import ModelCard, ModelPermission

logger = logging.getLogger(__name__)


# ─── Default built-in models ──────────────────────────────────────────────────
DEFAULT_MODELS = [
    {
        "id": "gpt-oss-20b",
        "object": "model",
        "owned_by": "llm-serve",
        "created": 1730000000,
        "capabilities": ["chat", "completions"],
        "context_window": 10048,      # Adjust based on your actual model
        "max_tokens": 4096,
    },
    {
        "id": "intfloat/multilingual-e5-large",
        "object": "model",
        "owned_by": "llm-serve",
        "created": 1700000000,
        "capabilities": ["embeddings"],
        "context_window": 512,
        "dimensions": 1024,
    },
    {
        "id": "hkunlp/instructor-large",
        "object": "model",
        "owned_by": "llm-serve",
        "created": 1700000001,
        "capabilities": ["embeddings"],
        "context_window": 768,
        "dimensions": 768,
    }
]

class ModelManager:
    def __init__(self):
        self._models: dict[str, dict] = {}
        self._load_defaults()
        self._load_from_yaml()

    def _load_defaults(self):
        for m in DEFAULT_MODELS:
            self._models[m["id"]] = m

    def _load_from_yaml(self):
        path = Path(settings.MODELS_CONFIG_PATH)
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if data and "models" in data:
                for m in data["models"]:
                    if "id" in m:
                        self._models[m["id"]] = {**{"owned_by": "llm-serve", "created": int(time.time()), "capabilities": ["chat"]}, **m}
        except Exception as e:
            logger.warning("Could not load %s: %s", path, e)

    def list_models(self) -> list[ModelCard]:
        return [self._to_card(m) for m in self._models.values()]

    def get_model(self, model_id: str) -> ModelCard | None:
        m = self._models.get(model_id)
        if m is None:
            return None
        return self._to_card(m)

    def model_exists(self, model_id: str) -> bool:
        return model_id in self._models

    def get_model_info(self, model_id: str) -> dict | None:
        return self._models.get(model_id)

    def has_capability(self, model_id: str, capability: str) -> bool:
        m = self._models.get(model_id)
        if not m:
            return False
        return capability in m.get("capabilities", [])

    def get_context_window(self, model_id: str) -> int:
        m = self._models.get(model_id, {})
        return m.get("context_window", 4096)

    def get_max_tokens(self, model_id: str) -> int:
        m = self._models.get(model_id, {})
        return m.get("max_tokens", 4096)

    def get_embedding_dims(self, model_id: str) -> int | None:
        """Native output size of an embedding model, if known."""
        return self._models.get(model_id, {}).get("dimensions")

    def register_model(self, model_info: dict):
        """Dynamically register a new model."""
        model_id = model_info["id"]
        self._models[model_id] = model_info

    def unregister_model(self, model_id: str) -> bool:
        if model_id in self._models:
            del self._models[model_id]
            return True
        return False

    def _to_card(self, m: dict) -> ModelCard:
        return ModelCard(
            id=m["id"],
            created=m.get("created", int(time.time())),
            owned_by=m.get("owned_by", "llm-serve"),
            permission=[ModelPermission()],
            root=m.get("root", m["id"]),
        )


model_manager = ModelManager()
