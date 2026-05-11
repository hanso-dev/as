def to_dict(node, buffer):
    if node.is_array:
        return [to_dict(node.at(i), buffer) for i in range(node.occurs)]

    details = {"offset": node.offset, "length": node.length}

    if node.children:
        return {
            "Details": details,
            **{c.name: to_dict(c, buffer) for c in node.children},
        }

    details["value"] = buffer[node.offset : node.offset + node.length]
    return {"Details": details}
