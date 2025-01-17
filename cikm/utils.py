import sys

import numpy as np
from logbook import Logger, StreamHandler

import cikm


class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls.instance = None

    def __call__(cls, *args, **kw):
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)

        return cls.instance


def rankdata(a, method='ordinal'):
    return len(a) + 1 - _rankdata(a, method=method)


def _rankdata(a, method='average'):
    """
    rankdata(a, method='average')

    Assign ranks to data, dealing with ties appropriately.

    Ranks begin at 1.  The `method` argument controls how ranks are assigned
    to equal values.  See [1]_ for further discussion of ranking methods.

    Parameters
    ----------
    a : array_like
        The array of values to be ranked.  The array is first flattened.
    method : str, optional
        The method used to assign ranks to tied elements.
        The options are 'average', 'min', 'max', 'dense' and 'ordinal'.

        'average':
            The average of the ranks that would have been assigned to
            all the tied values is assigned to each value.
        'min':
            The minimum of the ranks that would have been assigned to all
            the tied values is assigned to each value.  (This is also
            referred to as "competition" ranking.)
        'max':
            The maximum of the ranks that would have been assigned to all
            the tied values is assigned to each value.
        'dense':
            Like 'min', but the rank of the next highest element is assigned
            the rank immediately after those assigned to the tied elements.
        'ordinal':
            All values are given a distinct rank, corresponding to the order
            that the values occur in `a`.

        The default is 'average'.

    Returns
    -------
    ranks : ndarray
         An array of length equal to the size of `a`, containing rank
         scores.

    References
    ----------
    .. [1] "Ranking", http://en.wikipedia.org/wiki/Ranking

    Examples
    --------
    >>> from scipy.stats import rankdata
    >>> rankdata([0, 2, 3, 2])
    array([ 1. ,  2.5,  4. ,  2.5])
    >>> rankdata([0, 2, 3, 2], method='min')
    array([ 1,  2,  4,  2])
    >>> rankdata([0, 2, 3, 2], method='max')
    array([ 1,  3,  4,  3])
    >>> rankdata([0, 2, 3, 2], method='dense')
    array([ 1,  2,  3,  2])
    >>> rankdata([0, 2, 3, 2], method='ordinal')
    array([ 1,  2,  4,  3])
    """
    if method not in ('average', 'min', 'max', 'dense', 'ordinal'):
        raise ValueError('unknown method "{0}"'.format(method))

    arr = np.ravel(np.asarray(a))
    algo = 'mergesort' if method == 'ordinal' else 'quicksort'

    sorter = np.argsort(arr, kind=algo)

    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    if method == 'ordinal':
        return inv + 1

    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]

    if method == 'dense':
        return dense

    # cumulative counts of each unique value
    count = np.r_[np.nonzero(obs)[0], len(obs)]

    if method == 'max':
        return count[dense]

    if method == 'min':
        return count[dense - 1] + 1

    # average method
    return .5 * (count[dense] + count[dense - 1] + 1)


def get_logger(format_string=None):
    """Returns a singleton instance of a LogBook Logger

    Args:
        format_string: specifies how the log messages should be formatted

    Returns:
        A logbook Logger
    """
    if format_string is None:
        format_string = (
            u'[{record.time:%Y-%m-%d %H:%M:%S.%f} pid({record.process})] ' +
            u'{record.level_name}: {record.module}::{record.func_name}:{record.lineno} {record.message}'
        )
    # default_handler = StderrHandler(format_string=log_format)
    default_handler = StreamHandler(sys.stdout, format_string=format_string)
    default_handler.push_application()
    return LoggerSingle(__name__)


class LoggerSingle(Logger):
    __metaclass__ = Singleton


def compute_and_plot_ndcg(df_explanations, min_session_size=10, title=None):
    title = title or "NDCG"
    df_ndcg = cikm.compute_ndcg(df_explanations, min_session_size=min_session_size)
    df_ndcg_summary = cikm.summarize_ndcg(df_ndcg)

    # TODO: Save df_ndcg and df_ndcg_summary to disk.
    cikm.plot_ndcg(df_ndcg_summary, title=title)
