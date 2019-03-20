def text_to_labels(text, align_map):
    ret = []
    for char in text:
        if char in align_map:
            ret.append(align_map[char])
        elif char == ' ':
            ret.append(0)
    return ret


def labels_to_text(labels, align_map):
    reverse_map = {v: k for k, v in align_map.items()}
    text = ''
    for c in labels:
        if c in reverse_map:
            text += reverse_map[c]
        elif c == 0:
            text += ' '
    return text
