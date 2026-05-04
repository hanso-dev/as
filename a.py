def info(self) -> None:
    """Print debug view: own properties + direct children."""
    from tabulate import tabulate

    redefines = (
        self.node.redefines_target.name
        if self.node.is_redefine and self.node.redefines_target
        else None
    )

    props = [
        ("name",          self.name),
        ("path",          self.indexed_path),
        ("level",         self.level),
        ("offset",        self.offset),
        ("length",        self.length),
        ("total_length",  self.total_length),
        ("end_offset",    self.end_offset),
        ("occurs",        self.occurs),
        ("index",         self.index),
        ("is_array",      self.is_array),
        ("is_occurrence", self.is_occurrence),
        ("is_group",      bool(self.children)),
        ("redefines",     redefines),
    ]
    print(tabulate(props, headers=["property", "value"], tablefmt="simple"))

    if not self.children:
        return

    rows = [
        (
            c.level,
            c.name,
            c.offset,
            c.length,
            c.total_length,
            c.occurs,
            "group" if c.children else "leaf",
        )
        for c in self.children
    ]
    print()
    print(tabulate(
        rows,
        headers=["lvl", "name", "offset", "length", "total", "occurs", "kind"],
        tablefmt="simple",
    ))


def info(self) -> None:
    from tabulate import tabulate
    rows = [
        (r.level, r.name, r.offset, r.length, r.total_length, r.occurs)
        for r in self.roots
    ]
    print(f"LayoutTree (mode={self.mode.value}, {len(self.roots)} roots, {len(self)} nodes)")
    print(tabulate(rows, headers=["lvl", "name", "offset", "length", "total", "occurs"]))
