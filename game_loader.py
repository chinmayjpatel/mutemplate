from __future__ import annotations
from muforge.loader import registry
import game_schemas


def validate_all():
    """Run once on startup to make sure all TOML/loaded objects conform to our fields."""
    # nodes
    if hasattr(registry, "nodes"):
        for node_id, node in registry.nodes.items():
            try:
                game_schemas.validate_node(node)
            except Exception as e:
                # don't crash the whole game, just report
                print(f"[schema] Node '{node_id}' failed validation: {e}")

    # rooms
    if hasattr(registry, "rooms"):
        for room_id, room in registry.rooms.items():
            try:
                game_schemas.validate_room(room)
            except Exception as e:
                print(f"[schema] Room '{room_id}' failed validation: {e}")

    # attributes (if present)
    attrs = getattr(registry, "attributes", {})
    for attr_id, attr in attrs.items():
        try:
            game_schemas.validate_attr(attr)
        except Exception as e:
            print(f"[schema] Attr '{attr_id}' failed validation: {e}")

    print("✅ Game schema validation completed (with warnings above if any).")
