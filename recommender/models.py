"""
In-memory model registry for the TrustScore Recommender.

Loads model definitions from models.yaml and provides fast lookup by
canonical name, alias, or display name. Also supports family resolution.
"""

import logging
from typing import Dict, Optional

from recommender.config import get_model_definitions
from recommender.schemas import ModelIdentity

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Singleton-style registry that indexes models by canonical name and aliases.
    Provides O(1) lookup for canonicalization and family resolution.
    """

    def __init__(self) -> None:
        self._by_canonical: Dict[str, ModelIdentity] = {}
        self._alias_to_canonical: Dict[str, str] = {}
        self._loaded = False

    def load(self) -> None:
        """Load (or reload) model definitions from config."""
        definitions = get_model_definitions()
        self._by_canonical.clear()
        self._alias_to_canonical.clear()

        for model in definitions:
            canon = model.canonical_name.lower()
            self._by_canonical[canon] = model

            # Index every alias (case-insensitive)
            for alias in model.aliases:
                key = alias.strip().lower()
                if key in self._alias_to_canonical and self._alias_to_canonical[key] != canon:
                    logger.warning(
                        "Alias '%s' maps to multiple canonical names: '%s' and '%s'. "
                        "Last-write wins.",
                        alias, self._alias_to_canonical[key], canon,
                    )
                self._alias_to_canonical[key] = canon

            # Also index display_name and canonical_name itself as aliases
            self._alias_to_canonical[model.display_name.strip().lower()] = canon
            self._alias_to_canonical[canon] = canon

        self._loaded = True
        logger.info(
            "ModelRegistry loaded: %d models, %d alias entries",
            len(self._by_canonical),
            len(self._alias_to_canonical),
        )

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def resolve(self, name: str) -> Optional[ModelIdentity]:
        """
        Look up a model by any name (canonical, alias, or display name).
        Returns the ModelIdentity or None if not found.
        """
        self._ensure_loaded()
        key = name.strip().lower()

        canon = self._alias_to_canonical.get(key)
        if canon is not None:
            return self._by_canonical.get(canon)

        return None

    def get_canonical_name(self, name: str) -> str:
        """
        Resolve a raw name to its canonical name.
        Returns the original name unchanged if not found in registry.
        """
        model = self.resolve(name)
        return model.canonical_name if model else name

    def get_family(self, name: str) -> str:
        """
        Resolve a raw name to its model family.
        Returns 'unknown' if not found in registry.
        """
        model = self.resolve(name)
        return model.family if model else "unknown"

    def get_display_name(self, name: str) -> str:
        """Resolve a raw name to its display name, falling back to the input."""
        model = self.resolve(name)
        return model.display_name if model else name

    def all_models(self) -> Dict[str, ModelIdentity]:
        """Return all models keyed by canonical name."""
        self._ensure_loaded()
        return dict(self._by_canonical)


# Module-level singleton for convenience
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Return the global ModelRegistry singleton (lazy-initialized)."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
