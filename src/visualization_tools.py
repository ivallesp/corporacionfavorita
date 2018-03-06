import matplotlib.pyplot as plt


def plot_by_ranges(*args):
    offset = 0
    for range_ in args:
        markers = cycle(".*ov^8sp")
        for line in range_:
            length = len(line)
            plt.scatter(list(range(offset, offset + length)), line, marker=next(markers))
        offset += length


def plot_sequence_prediction(pre, post, pred):
    assert (len(post) == len(pred))
    plot_by_ranges([pre], [post, pred])