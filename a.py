"""JSON export for computed copybook layouts.

Builds JSON-compatible Python dicts from a LayoutTree. Output shape is
controlled by attributes on the JsonExporter instance — set them after
construction or override in a subclass.

Typical use::

    from copybook.exporter import JsonExporter

    exporter = JsonExporter(layout_tree)
    data = exporter.export()

    # Tweak shape on the instance
    exporter.include_picture = False
    exporter.array_format = "dict_keyed"
    data = exporter.export()
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from copybook.models.layout import LayoutNode, LayoutTree

logger = logging.getLogger(__name__)


FieldPredicate = Callable[[LayoutNode], bool]


class JsonExporter:
    """Build JSON-compatible dicts from a LayoutTree.

    All shape options are public attributes — tweak them directly on the
    instance. The exporter walks the tree once per ``export()`` call and
    does not cache output.

    Per-field attribute toggles:
        include_offset:     Emit ``start`` (1-based byte offset).
        include_length:     Emit ``length`` (one element's byte size).
        include_end:        Emit ``end`` (offset + length - 1, inclusive).
        include_picture:    Emit ``picture`` on elementary fields.
        include_usage:      Emit ``usage`` on elementary fields.
        include_level:      Emit ``level`` (COBOL level number).
        include_path:       Emit ``path`` (dotted ancestor chain).
        include_conditions: Emit ``conditions`` list for level-88s.
        include_redefines:  Emit ``redefines`` when target is set.

    Structural options:
        array_format:       ``"list"`` → JSON array of element dicts.
                            ``"dict_keyed"`` → object with ``NAME_n`` keys.
        array_index_base:   0 or 1. Used by ``dict_keyed`` keys.
        filler_strategy:    ``"enumerate"`` → FILLER_0, FILLER_1, …
                            ``"keep"`` → name stays ``"FILLER"`` (collisions
                            log a warning and overwrite).
                            ``"skip"`` → drop FILLER fields entirely.
        include_field:      Optional predicate; nodes returning False are
                            skipped along with their entire subtree.
    """

    def __init__(self, tree: LayoutTree) -> None:
        self.tree = tree

        # Per-field attributes
        self.include_offset: bool = True
        self.include_length: bool = True
        self.include_end: bool = False
        self.include_picture: bool = True
        self.include_usage: bool = True
        self.include_level: bool = False
        self.include_path: bool = False
        self.include_conditions: bool = True
        self.include_redefines: bool = True

        # Structural options
        self.array_format: str = "list"  # "list" | "dict_keyed"
        self.array_index_base: int = 0
        self.filler_strategy: str = "enumerate"  # "enumerate" | "keep" | "skip"

        # Field filtering
        self.include_field: FieldPredicate | None = None

    # --- Public API ---------------------------------------------------------

    def export(self) -> dict[str, Any]:
        """Export the entire tree as a JSON-compatible dict.

        Returns:
            Dict keyed by root record name. ``{}`` if the tree is empty
            or every root is filtered out.
        """
        out: dict[str, Any] = {}
        for root in self.tree.roots:
            if not self._included(root):
                continue
            out[root.name] = self._node_to_dict(root)
        return out

    def export_node(self, name: str) -> Any | None:
        """Export a single node by name (case-insensitive).

        Returns the JSON shape for the first matching node, or None if
        no node has that name.
        """
        matches = self.tree.find(name)
        if not matches:
            logger.warning(f"export_node: no node named {name!r}")
            return None
        return self._node_to_dict(matches[0])

    def to_json(self, indent: int | None = 2) -> str:
        """Convenience: export() and serialize with json.dumps."""
        return json.dumps(self.export(), indent=indent)

    # --- Walker -------------------------------------------------------------

    def _node_to_dict(self, node: LayoutNode) -> Any:
        """Dispatch to array or non-array handler."""
        if node.is_array:
            return self._build_array(node)
        return self._build_node(node)

    def _build_node(self, node: LayoutNode) -> dict[str, Any]:
        """Build dict for a non-array node — group or elementary."""
        out: dict[str, Any] = {}

        if self.include_offset:
            out["start"] = node.offset
        if self.include_length:
            out["length"] = node.length
        if self.include_end:
            out["end"] = node.offset + node.length - 1
        if self.include_level:
            out["level"] = node.level
        if self.include_path:
            out["path"] = node.path
        if self.include_redefines and node.node.field.redefines:
            out["redefines"] = node.node.field.redefines

        if node.children:
            out["fields"] = self._build_children(node.children)
        else:
            self._add_leaf_attrs(out, node)

        return out

    def _build_array(self, node: LayoutNode) -> Any:
        """Expand all occurrences of an array template."""
        elements = [self._build_node(node.at(i)) for i in range(node.occurs)]

        if self.array_format == "dict_keyed":
            base = self.array_index_base
            return {f"{node.name}_{i + base}": e for i, e in enumerate(elements)}
        if self.array_format == "list":
            return elements
        raise ValueError(
            f"array_format must be 'list' or 'dict_keyed', got {self.array_format!r}"
        )

    def _build_children(self, children: list[LayoutNode]) -> dict[str, Any]:
        """Build name-keyed dict, applying filtering and FILLER policy."""
        out: dict[str, Any] = {}
        filler_index = 0

        for child in children:
            if not self._included(child):
                continue

            if child.name == "FILLER":
                if self.filler_strategy == "skip":
                    continue
                if self.filler_strategy == "enumerate":
                    key = f"FILLER_{filler_index}"
                    filler_index += 1
                elif self.filler_strategy == "keep":
                    key = "FILLER"
                else:
                    raise ValueError(
                        f"filler_strategy must be one of 'enumerate', 'keep', "
                        f"'skip', got {self.filler_strategy!r}"
                    )
            else:
                key = child.name

            if key in out:
                logger.warning(
                    f"Duplicate JSON key {key!r} under parent — overwriting"
                )
            out[key] = self._node_to_dict(child)

        return out

    def _add_leaf_attrs(self, out: dict[str, Any], node: LayoutNode) -> None:
        """Attach picture, usage, and conditions to an elementary field dict."""
        f = node.node.field

        if self.include_picture and f.picture is not None:
            out["picture"] = f.picture
        if self.include_usage:
            out["usage"] = f.usage or "DISPLAY"
        if self.include_conditions and node.node.conditions:
            out["conditions"] = [
                {"name": c.field.name, "value": c.field.value}
                for c in node.node.conditions
            ]

    def _included(self, node: LayoutNode) -> bool:
        """Run the user predicate; True if no predicate is set."""
        if self.include_field is None:
            return True
        return self.include_field(node)
