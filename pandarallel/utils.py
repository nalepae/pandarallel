import itertools as _itertools
from pyarrow.lib import PlasmaStoreFull as _PlasmaStoreFull

def chunk(nb_item, nb_chunks, start_offset=0):
    """
    Return `nb_chunks` slices of approximatively `nb_item / nb_chunks` each.

    Parameters
    ----------
    nb_item : int
        Total number of items

    nb_chunks : int
        Number of chunks to return

    start_offset : int
        Shift start of slice by this amount

    Returns
    -------
    A list of slices


    Examples
    --------
    >>> chunks = _pandarallel._chunk(103, 4)
    >>> chunks
    [slice(0, 26, None), slice(26, 52, None), slice(52, 78, None),
     slice(78, 103, None)]
    """
    if nb_item <= nb_chunks:
        return [
            slice(max(0, idx - start_offset), idx + 1)
            for idx in range(nb_item)
        ]

    quotient = nb_item // nb_chunks
    remainder = nb_item % nb_chunks

    quotients = [quotient] * nb_chunks
    remainders = [1] * remainder + [0] * (nb_chunks - remainder)

    nb_elems_per_chunk = [
                            quotient + remainder for quotient, remainder
                            in zip(quotients, remainders)
                        ]

    accumulated = list(_itertools.accumulate(nb_elems_per_chunk))
    shifted_accumulated = accumulated.copy()
    shifted_accumulated.insert(0, 0)
    shifted_accumulated.pop()

    return [
            slice(max(0, begin - start_offset), end) for begin, end
            in zip(shifted_accumulated, accumulated)
        ]

def parallel(client):
    def decorator(func):
        def wrapper(*args, **kwargs):
            """Please see the docstring of this method without `parallel`"""
            try:
                return func(*args, **kwargs)

            except _PlasmaStoreFull:
                msg = "The pandarallel shared memory is too small to allow \
parallel computation. \
Just after pandarallel import, please write: \
pandarallel.initialize(<size of memory in MB>), and retry."

                raise Exception(msg)

            finally:
                client.delete(client.list().keys())

        return wrapper
    return decorator
