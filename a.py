from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator

from copybook.models.fields import CopybookField
from copybook.models.tree import FieldNode, CopybookTree
from copybook.patterns import EXPAND_PIC_RE

from enum import Enum


class LayoutMode(Enum):
    INDEPENDENT = "independent"  # each root starts at offset 0
    SEQUENTIAL = "sequential"  # each root starts where the previous ended


@dataclass
class LayoutNode:
    """A FieldNode decorated with its resolved memory position.

    Produced by compute_layout(). Every node in the tree gets a LayoutNode —
    groups, elementaries, redefines, and OCCURS arrays alike.

    Attributes:
        node:           The source FieldNode this layout entry describes.
        offset:         Byte offset from the start of the root record.
        length:         Byte length of a single occurrence of this field.
        total_length:   Byte length of the full array (length × occurs).
                        Equal to length for non-array fields.
        children:       Direct subordinate LayoutNodes, in declaration order.
                        Mirrors FieldNode.children but with layout attached.

    Notes on redefines:
        A redefining field shares its offset with its target — both start at
        the same byte. The parent group's contribution to total size is the
        max of the target and all its redefiners, not their sum.

    Notes on OCCURS:
        length       — size of one element.
        total_length — size of the entire array (length * occurs).
        For OCCURS DEPENDING ON, occurs_max is used for layout purposes.
        Use at(i) to get a virtual LayoutNode for a single occurrence,
        with offsets translated to that element's position.

    Notes on groups:
        length = sum of direct children's total_lengths, excluding conditions.
        Groups have no PIC so pic_size() is never called for them.
    """

    node: FieldNode
    offset: int
    length: int
    occurs: int  # 1 for non-arrays, n for OCCURS
    total_length: int
    children: list[LayoutNode]
    index: int | None = None  # Set on virtual nodes returned by at(); None on templates.

    # --- Identity (delegated to FieldNode) ----------------------------------

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def path(self) -> str:
        return self.node.path

    @property
    def level(self) -> int:
        return self.node.level

    # --- Derived position info ----------------------------------------------

    @property
    def end_offset(self) -> int:
        """First byte after this field (offset + total_length)."""
        return self.offset + self.total_length

    # --- Traversal ----------------------------------------------------------

    def walk(self) -> Iterator[LayoutNode]:
        """Pre-order depth-first over this node and all descendants."""
        yield self
        for child in self.children:
            yield from child.walk()

    @property
    def leaves(self) -> list[LayoutNode]:
        """All descendant elementary nodes — the fields that hold actual data."""
        return [n for n in self.walk() if not n.children]

    @property
    def is_array(self) -> bool:
        """True if this is an unindexed array template (not a specific occurrence)."""
        return self.occurs > 1 and self.index is None

    @property
    def is_occurrence(self) -> bool:
        """True if this is a virtual view of one element produced by at()."""
        return self.index is not None

    @property
    def indexed_path(self) -> str:
        """Field path with [i] suffix when this node is an array occurrence."""
        return f"{self.path}[{self.index}]" if self.index is not None else self.path

    @property
    def occurence_offsets(self) -> list[int]:
        """Start offset of each occurrence — useful for data movement later."""
        return [self.offset + i * self.length for i in range(self.occurs)]

    # --- Indexing -----------------------------------------------------------

    def at(self, index: int) -> LayoutNode:
        """Return a virtual LayoutNode for one occurrence of this array.

        Offsets in the returned subtree are shifted by ``index * length``
        from the template, so children land on correct byte positions for
        the requested element. The returned node carries ``index=index``;
        its children retain ``index=None`` (translated, not themselves indexed).

        Only supports single-level OCCURS — nested arrays are not handled.

        Args:
            index: Zero-based occurrence index in range [0, occurs).

        Returns:
            A LayoutNode rooted at the indexed element's offset.

        Raises:
            ValueError: If this node is not an array template.
            IndexError: If index is out of bounds.
        """
        if not self.is_array:
            raise ValueError(f"at() called on non-array node '{self.name}'")
        if not 0 <= index < self.occurs:
            raise IndexError(
                f"index {index} out of range for OCCURS {self.occurs} on '{self.name}'"
            )

        delta = index * self.length
        return LayoutNode(
            node=self.node,
            offset=self.offset + delta,
            length=self.length,
            occurs=self.occurs,
            total_length=self.total_length,
            children=[_translate(c, delta) for c in self.children],
            index=index,
        )

    def occurrences(self) -> Iterator[LayoutNode]:
        """Iterate all occurrences as virtual LayoutNodes.

        For non-array nodes, yields self once. For arrays, yields at(0)
        through at(occurs - 1) in order.
        """
        if not self.is_array:
            yield self
            return
        for i in range(self.occurs):
            yield self.at(i)

    def walk_expanded(self) -> Iterator[LayoutNode]:
        """Walk this subtree, replacing array templates with their occurrences.

        Like walk(), but array templates are not yielded — instead each
        at(i) is yielded in turn and recursed into. Useful for iterating
        concrete byte positions in a buffer where every array element
        should produce its own entry with correctly translated offsets.

        Only single-level OCCURS is expanded.
        """
        if self.is_array:
            for occurrence in self.occurrences():
                yield from occurrence.walk_expanded()
        else:
            yield self
            for child in self.children:
                yield from child.walk_expanded()

    # --- Display ------------------------------------------------------------

    def __repr__(self) -> str:
        return f"LayoutNode({self.level:02} {self.name} offset={self.offset} length={self.length})"


# ---------------------------------------------------------------------------
# LayoutTree
# ---------------------------------------------------------------------------


@dataclass
class LayoutTree:
    """A fully laid-out forest of LayoutNodes.

    Mirrors the shape of CopybookTree but every node carries its resolved
    byte offset and length. This is the primary structure for flat maps,
    schema export, and data movement.

    Attributes:
        roots: Top-level LayoutNodes, one per root record. Each starts at
               offset 0 independently.
    """

    roots: list[LayoutNode]
    mode: LayoutMode = LayoutMode.SEQUENTIAL

    @property
    def nodes(self) -> list[LayoutNode]:
        """All nodes in pre-order depth-first source order."""
        return [node for root in self.roots for node in root.walk()]

    @property
    def leaves(self) -> list[LayoutNode]:
        """Elementary fields only — nodes that hold actual data."""
        return [node for node in self.nodes if not node.children]

    @property
    def expanded_nodes(self) -> list[LayoutNode]:
        """All nodes in pre-order, with array templates replaced by per-occurrence views.

        Where ``nodes`` yields an array template once, this yields one entry
        per array element with offsets translated to that element. Useful
        for iterating concrete byte positions in a record buffer.
        """
        return [n for root in self.roots for n in root.walk_expanded()]

    @property
    def expanded_leaves(self) -> list[LayoutNode]:
        """Elementary nodes only, with arrays expanded — one entry per byte slot."""
        return [n for n in self.expanded_nodes if not n.children]

    def find(self, name: str) -> list[LayoutNode]:
        """All nodes whose name matches (case-insensitive)."""
        name = name.upper()
        return [node for node in self.nodes if node.name == name]

    def flat_map(self) -> dict[str, tuple[int, int]]:
        """Flat mapping of field path → (offset, length) for every leaf occurrence.

        Array elements are expanded with bracket notation, so each occurrence
        gets its own entry::

            RECORD.ITEMS[0].QTY → (offset, length)
            RECORD.ITEMS[1].QTY → (offset, length)

        Non-array leaves keep their plain path. This is the primary output
        for consumers parsing a raw data buffer — slice
        ``bytes[offset:offset+length]`` to read a field.

        Only single-level OCCURS is expanded.
        """
        result: dict[str, tuple[int, int]] = {}
        for root in self.roots:
            _collect_leaves(root, "", result)
        return result

    def print(self) -> None:
        """Print an indented layout view to stdout."""
        for root in self.roots:
            for node in root.walk():
                depth = len(node.node.ancestors)
                indent = "  " * depth
                redefines = ""
                if node.node.is_redefine and node.node.redefines_target is not None:
                    redefines = f" → redefines {node.node.redefines_target.name}"
                print(
                    f"{indent}{node.level:02} {node.name}"
                    f"  offset={node.offset}"
                    f"  length={node.length}"
                    f"  total={node.total_length}"
                    f"{redefines}"
                )

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"LayoutTree({len(self.roots)} roots, {len(self)} nodes)"


# ---------------------------------------------------------------------------
# PIC helpers
# ---------------------------------------------------------------------------


def _expand_pic(pic: str) -> str:
    """Expand repetition notation: '9(7)' → '9999999'."""
    return EXPAND_PIC_RE.sub(lambda m: m.group(1) * int(m.group(2)), pic)


def _digit_count(pic: str) -> int:
    """Count 9 and P characters in a PIC string (used for COMP sizing)."""
    return sum(1 for c in _expand_pic(pic).upper() if c in "9P")


def _display_size(pic: str, sign_separate: bool) -> int:
    """DISPLAY storage size: every symbol except V and P = 1 byte."""
    expanded = _expand_pic(pic.upper())
    if expanded.startswith("S"):
        expanded = expanded[1:]
    size = sum(1 for c in expanded if c not in "VP")
    return size + (1 if sign_separate else 0)


def _comp3_size(pic: str) -> int:
    """COMP-3 size: ceil((digits + 1) / 2)."""
    return math.ceil((_digit_count(pic) + 1) / 2)


def _comp_binary_size(pic: str) -> int:
    """COMP/BINARY size: 2 bytes (≤4 digits), 4 bytes (≤9), 8 bytes (≤18)."""
    d = _digit_count(pic)
    if d <= 4:
        return 2
    if d <= 9:
        return 4
    return 8


def pic_size(node: FieldNode) -> int:
    """Compute storage size in bytes of one occurrence of an elementary field.

    For arrays (OCCURS), this is the size of a single element — multiply by
    node.field.occurs to get the total array size.

    Raises:
        ValueError: If called on a group node (no PIC clause).
    """
    field = node.field

    if field.picture is None:
        raise ValueError(
            f"pic_size called on group field '{field.name}' at {node.node_id} — no PIC clause."
        )

    usage = (field.usage or "DISPLAY").upper()

    if usage == "COMP-1":
        return 4
    if usage == "COMP-2":
        return 8
    if usage in ("COMP-3", "PACKED-DECIMAL"):
        return _comp3_size(field.picture)
    if usage in ("COMP", "COMP-4", "COMP-5", "COMP-X", "BINARY"):
        return _comp_binary_size(field.picture)
    return _display_size(field.picture, field.sign_separate)


# ---------------------------------------------------------------------------
# Layout computation
# ---------------------------------------------------------------------------


def _translate(node: LayoutNode, delta: int) -> LayoutNode:
    """Recursively shift offsets in a LayoutNode subtree by ``delta`` bytes.

    Used by LayoutNode.at() to project a template's children onto a specific
    array element's base offset. Structure and metadata are preserved; only
    offsets change.
    """
    return LayoutNode(
        node=node.node,
        offset=node.offset + delta,
        length=node.length,
        occurs=node.occurs,
        total_length=node.total_length,
        children=[_translate(c, delta) for c in node.children],
        index=node.index,
    )


def _collect_leaves(
    node: LayoutNode, path_prefix: str, out: dict[str, tuple[int, int]]
) -> None:
    """Walk a LayoutNode subtree, emitting (path, offset, length) per leaf.

    Array templates are expanded: each occurrence contributes its own entries
    with ``[i]`` bracket notation in the path. Non-array leaves are emitted
    with their plain dotted path.
    """
    own_path = f"{path_prefix}.{node.name}" if path_prefix else node.name

    if node.is_array:
        for i in range(node.occurs):
            indexed_path = f"{own_path}[{i}]"
            occurrence = node.at(i)
            if occurrence.children:
                for child in occurrence.children:
                    _collect_leaves(child, indexed_path, out)
            else:
                out[indexed_path] = (occurrence.offset, occurrence.length)
    elif not node.children:
        out[own_path] = (node.offset, node.length)
    else:
        for child in node.children:
            _collect_leaves(child, own_path, out)


def _layout_node(node: FieldNode, offset: int) -> LayoutNode:
    if node.is_group:
        children, length = _layout_children(node.children, offset)
    else:
        children = []
        length = pic_size(node)

    occurs = node.field.occurs_max or node.field.occurs or 1
    total_length = length * occurs

    return LayoutNode(
        node=node,
        offset=offset,
        length=length,
        occurs=occurs,
        total_length=total_length,
        children=children,
    )


def _layout_children(
    children: list[FieldNode], base_offset: int
) -> tuple[list[LayoutNode], int]:
    """Lay out a list of sibling nodes, handling redefines overlaps.

    Redefining fields start at the same offset as their target. The cursor
    advances by max(current, this alternative's end) so the parent group
    captures the largest alternative, not the sum.
    """
    result: list[LayoutNode] = []
    layout_map: dict[str, LayoutNode] = {}  # node_id → LayoutNode
    cursor = base_offset

    for child in children:
        if child.is_redefine and child.redefines_target is not None:
            target = layout_map[child.redefines_target.node_id]
            child_offset = target.offset
        else:
            child_offset = cursor

        child_layout = _layout_node(child, child_offset)
        layout_map[child.node_id] = child_layout
        result.append(child_layout)

        cursor = max(cursor, child_offset + child_layout.total_length)

    return result, cursor - base_offset


def compute_layout(
    tree: CopybookTree, mode: LayoutMode = LayoutMode.SEQUENTIAL
) -> LayoutTree:
    """Compute byte offsets and lengths for every node in the tree.

    Args:
        tree: A fully-wired CopybookTree as returned by build_copybook_tree().
        mode: INDEPENDENT — each root starts at offset 1 (default).
              SEQUENTIAL  — each root starts where the previous root ended.
                            Use when a copybook describes a single flat buffer
                            with multiple contiguous record sections.
    """
    roots = []
    root_map: dict[str, LayoutNode] = {}  # name → LayoutNode for redefines lookup
    cursor = 1

    for root in tree.roots:
        if root.is_redefine and root.redefines_target is not None:
            # Always overlays its target regardless of mode
            target = root_map[root.redefines_target.name]
            offset = target.offset
        elif mode == LayoutMode.SEQUENTIAL:
            offset = cursor
        else:
            offset = 1

        layout_root = _layout_node(root, offset)
        roots.append(layout_root)
        root_map[root.name] = layout_root

        # Only advance cursor for non-redefining roots
        if not root.is_redefine:
            cursor = max(cursor, layout_root.end_offset)

    return LayoutTree(roots=roots, mode=mode)
