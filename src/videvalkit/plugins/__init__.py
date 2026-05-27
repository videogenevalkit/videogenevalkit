"""Plugin discovery for videvalkit.

See ``loader`` module and ``docs/INTEGRATION_FRAMEWORK_DESIGN.md`` §4.
"""

from videvalkit.plugins.loader import (
    SUPPORTED_GROUPS,
    discover,
    discover_all,
    plugin_sources_report,
)

__all__ = [
    "SUPPORTED_GROUPS",
    "discover",
    "discover_all",
    "plugin_sources_report",
]
