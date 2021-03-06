flatten = lambda l: [item for sublist in l for item in sublist]


def batching(list_of_iterables, n=1, infinite=False, return_incomplete_batches=False):
    list_of_iterables = [list_of_iterables] if type(list_of_iterables) is not list else list_of_iterables
    assert(len({len(it) for it in list_of_iterables}) == 1)
    n_elements = len(list_of_iterables[0])
    while 1:
        for ndx in range(0, n_elements, n):
            if not return_incomplete_batches:
                if (ndx+n) > n_elements:
                    break
            yield [iterable[ndx:min(ndx + n, n_elements)] for iterable in list_of_iterables]

        if not infinite:
            break


def exponential_decay_generator(start, finish, decay=0.999):
    x = start
    while 1:
        x = x*decay + finish*(1-decay)
        yield x