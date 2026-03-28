"""Plugin architecture for third-party extensions."""

import importlib
import sys
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class PluginInfo:
    """Metadata for a registered plugin."""
    name: str
    version: str
    plugin_type: str  # "encoder", "decoder", "metric", "domain"
    description: str = ""
    author: str = ""
    entry_point: str = ""


class PluginRegistry:
    """Registry for WMCP plugins.

    Plugins can be registered programmatically or discovered via
    setuptools entry_points (group: "wmcp.plugins").

    Usage:
        registry = PluginRegistry()
        registry.register("my_encoder", "1.0", "encoder", factory_fn)
        encoders = registry.list_plugins("encoder")
    """

    def __init__(self):
        self._plugins: Dict[str, Dict] = {}

    def register(self, name: str, version: str, plugin_type: str,
                 factory: Optional[Callable] = None,
                 description: str = "", author: str = ""):
        """Register a plugin.

        Args:
            name: Plugin name.
            version: Plugin version.
            plugin_type: One of "encoder", "decoder", "metric", "domain".
            factory: Callable that creates the plugin instance.
            description: Human-readable description.
            author: Plugin author.
        """
        self._plugins[name] = {
            "info": PluginInfo(name, version, plugin_type, description, author),
            "factory": factory,
        }

    def discover(self):
        """Discover plugins via setuptools entry_points."""
        try:
            if sys.version_info >= (3, 10):
                from importlib.metadata import entry_points
                eps = entry_points(group="wmcp.plugins")
            else:
                from importlib.metadata import entry_points
                all_eps = entry_points()
                eps = all_eps.get("wmcp.plugins", [])

            for ep in eps:
                try:
                    plugin_cls = ep.load()
                    if hasattr(plugin_cls, "wmcp_plugin_info"):
                        info = plugin_cls.wmcp_plugin_info()
                        self.register(
                            info.get("name", ep.name),
                            info.get("version", "0.0.0"),
                            info.get("type", "encoder"),
                            factory=plugin_cls,
                            description=info.get("description", ""),
                            author=info.get("author", ""),
                        )
                except Exception:
                    pass
        except Exception:
            pass

    def get(self, name: str) -> Optional[Dict]:
        """Get a registered plugin by name."""
        return self._plugins.get(name)

    def create(self, name: str, **kwargs):
        """Create a plugin instance."""
        plugin = self._plugins.get(name)
        if plugin is None:
            raise KeyError(f"Plugin '{name}' not found")
        if plugin["factory"] is None:
            raise ValueError(f"Plugin '{name}' has no factory function")
        return plugin["factory"](**kwargs)

    def list_plugins(self, plugin_type: Optional[str] = None) -> List[PluginInfo]:
        """List registered plugins, optionally filtered by type."""
        plugins = [p["info"] for p in self._plugins.values()]
        if plugin_type:
            plugins = [p for p in plugins if p.plugin_type == plugin_type]
        return plugins


# Global registry
_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _registry


# ═══ Example plugin ═══

class ExampleEncoderPlugin:
    """Example: register a custom encoder as a WMCP plugin."""

    @staticmethod
    def wmcp_plugin_info():
        return {
            "name": "example_encoder",
            "version": "0.1.0",
            "type": "encoder",
            "description": "Example encoder plugin for WMCP",
            "author": "WMCP Team",
        }

    def __init__(self, input_dim: int = 512):
        import torch.nn as nn
        self.projection = nn.Linear(input_dim, 128)

    def encode(self, features):
        return self.projection(features)
