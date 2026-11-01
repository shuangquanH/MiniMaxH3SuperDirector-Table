"""MiniMax H3 Timeline Director — standalone editable copy.

Registers exactly one node. The planner, plan encoder and Omni bridge of the
upstream package are not part of this copy, so it can coexist with
ComfyUI-MiniMaxH3-TimelineDirector without duplicating registrations.

Node id, display name and the HTTP routes carry a ``Custom`` suffix to avoid
clashing with the original package. Uploads still land in the original
subfolder so existing timelines keep resolving their media.
"""

from .director import MiniMaxH3TimelineDirector

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TimelineDirectorCustom": MiniMaxH3TimelineDirector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TimelineDirectorCustom": "MiniMaxH3导演台",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
