"""Optional, opt-in anonymous usage telemetry."""

import os
import json
import time
import uuid
import hashlib
from pathlib import Path
from typing import Dict, Optional


_CONFIG_DIR = Path.home() / ".wmcp"
_CONFIG_FILE = _CONFIG_DIR / "config.json"
_TELEMETRY_ENABLED = None


def _load_config() -> Dict:
    if _CONFIG_FILE.exists():
        try:
            return json.loads(_CONFIG_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_config(config: Dict):
    _CONFIG_DIR.mkdir(exist_ok=True)
    _CONFIG_FILE.write_text(json.dumps(config, indent=2))


def is_enabled() -> bool:
    """Check if telemetry is enabled. Default: disabled."""
    global _TELEMETRY_ENABLED
    if _TELEMETRY_ENABLED is not None:
        return _TELEMETRY_ENABLED

    # Environment variable override
    env = os.environ.get("WMCP_TELEMETRY", "").lower()
    if env in ("1", "true", "yes"):
        _TELEMETRY_ENABLED = True
        return True
    if env in ("0", "false", "no"):
        _TELEMETRY_ENABLED = False
        return False

    # Config file
    config = _load_config()
    _TELEMETRY_ENABLED = config.get("telemetry_enabled", False)
    return _TELEMETRY_ENABLED


def enable():
    """Enable anonymous telemetry."""
    global _TELEMETRY_ENABLED
    _TELEMETRY_ENABLED = True
    config = _load_config()
    config["telemetry_enabled"] = True
    if "installation_id" not in config:
        config["installation_id"] = str(uuid.uuid4())[:8]
    _save_config(config)
    print("WMCP telemetry enabled. Anonymous usage data will be collected.")
    print("Disable anytime: wmcp telemetry --disable")


def disable():
    """Disable telemetry."""
    global _TELEMETRY_ENABLED
    _TELEMETRY_ENABLED = False
    config = _load_config()
    config["telemetry_enabled"] = False
    _save_config(config)
    print("WMCP telemetry disabled. No data will be collected.")


def record_event(event_type: str, properties: Optional[Dict] = None):
    """Record a telemetry event (only if enabled).

    Events are stored locally. No network calls in v0.1.

    Args:
        event_type: Event name (e.g., "protocol_trained", "onboard_complete").
        properties: Optional event properties.
    """
    if not is_enabled():
        return

    config = _load_config()
    install_id = config.get("installation_id", "unknown")

    event = {
        "timestamp": time.time(),
        "installation_id": install_id,
        "event": event_type,
        "wmcp_version": "0.1.0",
        "properties": properties or {},
    }

    # Append to local log (no network in v0.1)
    log_path = _CONFIG_DIR / "telemetry.jsonl"
    _CONFIG_DIR.mkdir(exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(event) + "\n")


def get_local_events(n: int = 100) -> list:
    """Read recent local telemetry events."""
    log_path = _CONFIG_DIR / "telemetry.jsonl"
    if not log_path.exists():
        return []
    lines = log_path.read_text().strip().split("\n")
    return [json.loads(l) for l in lines[-n:]]
