"""First-party extension framework.

Extensions are registered explicitly from this package — there is no dynamic
discovery or plugin scanning. ``register_builtin_extensions`` is called from
the app lifespan before settings are loaded so that contributed SettingDef
entries are present in services.settings when load_all_settings() runs.
"""

from financial_dashboard.extensions.base import (
    EXTENSION_CONTRACT_VERSION,
    Capability,
    ExtensionHealthMeta,
    ExtensionManifest,
    ExtensionNavItem,
    ExtensionRegistrationError,
    ExtensionRuntime,
)
from financial_dashboard.extensions.paisa import PAISA_EXTENSION
from financial_dashboard.extensions.registry import ExtensionRegistry
from financial_dashboard.services.settings import SETTINGS_REGISTRY, register_setting

BUILTIN_EXTENSIONS: tuple[ExtensionManifest, ...] = (PAISA_EXTENSION,)


def enabled_builtin_extensions(*, paisa_enabled: bool) -> tuple[ExtensionManifest, ...]:
    """Return the builtin manifests enabled for this deployment."""
    return BUILTIN_EXTENSIONS if paisa_enabled else ()


def register_builtin_extensions(
    registry: ExtensionRegistry, *, paisa_enabled: bool = True
) -> None:
    """Register enabled builtin manifests and their settings globally.

    Idempotent across app lifecycles within a single process: manifests register
    into the per-app registry, while contributed settings are reconciled against
    the process-wide SETTINGS_REGISTRY. Definitions belonging to a disabled
    builtin are removed without touching persisted values. An equal existing
    SettingDef is accepted; a different definition raises
    ExtensionRegistrationError rather than being silently overwritten.
    """
    enabled = enabled_builtin_extensions(paisa_enabled=paisa_enabled)
    enabled_ids = {manifest.id for manifest in enabled}
    for manifest in BUILTIN_EXTENSIONS:
        if manifest.id in enabled_ids:
            continue
        for key, defn in manifest.settings.items():
            if SETTINGS_REGISTRY.get(key) == defn:
                del SETTINGS_REGISTRY[key]

    for manifest in enabled:
        registry.register(manifest)
        for key, defn in manifest.settings.items():
            existing = SETTINGS_REGISTRY.get(key)
            if existing is not None:
                if existing != defn:
                    raise ExtensionRegistrationError(
                        f"Extension {manifest.id!r} contributed a conflicting "
                        f"definition for setting {key!r}"
                    )
                continue
            register_setting(key, defn)


__all__ = [
    "BUILTIN_EXTENSIONS",
    "EXTENSION_CONTRACT_VERSION",
    "Capability",
    "ExtensionHealthMeta",
    "ExtensionManifest",
    "ExtensionNavItem",
    "ExtensionRegistrationError",
    "ExtensionRegistry",
    "ExtensionRuntime",
    "PAISA_EXTENSION",
    "enabled_builtin_extensions",
    "register_builtin_extensions",
]
