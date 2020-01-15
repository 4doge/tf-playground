def get_label_index(label_or_index, labels, is_label=True):
    # lbs = ['dogs', 'cats', 'flowers']
    lbs = labels

    if is_label:
        idx = 4

        for index, label in enumerate(lbs):
            if label == label_or_index:
                idx = index
        return idx

    else:
        lb = 'hz'

        for index, label in enumerate(lbs):
            if index == label_or_index:
                lb = label
        return lb


def get_label_index2(label_or_index, labels, is_label=True):
    # lbs = ['dogs', 'cats', 'flowers']
    lbs = labels

    if is_label:
        idx = 4

        for index, label in enumerate(lbs):
            if label == label_or_index:
                idx = index
        return idx

    else:
        lb = 'hz'

        for index, label in enumerate(lbs):
            if index == label_or_index:
                lb = label
        return lb


def test(file_path, labels):
    t = 1
    for i, v in enumerate(labels):
        if v == file_path:
            t = i
        return t

    return t
