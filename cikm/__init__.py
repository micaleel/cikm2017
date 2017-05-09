from itertools import cycle

import numpy as np
import pandas as pd
from lightfm import LightFM
from matplotlib import pyplot as plt
from scipy import sparse as sp


def extract_ratings(df_explanations):
    """Extract columns needed to build interaction matrix."""
    cols = ['target_item_average_rating', 'target_item_id', 'user_id']
    df_ratings = df_explanations[cols].drop_duplicates()
    df_ratings.rename(columns={'target_item_average_rating': 'rating', 'target_item_id': 'item_id'}, inplace=True)
    print('Found {:,} user-item interactions'.format(len(df_ratings)))
    return df_ratings


def add_scores_to_explanations(df_explanations, df_scores):
    df_explanations['user_id_target_item_id'] = df_explanations.user_id + df_explanations.target_item_id
    df_scores['user_id_target_item_id'] = df_scores.user_id + df_scores.item_id
    score_map = df_scores[['score', 'user_id_target_item_id']].set_index('user_id_target_item_id').score.to_dict()
    df_explanations['score'] = df_explanations.user_id_target_item_id.map(score_map)
    df_explanations.rename(columns={'score': 'rank_mf'}, inplace=True)
    del df_explanations['user_id_target_item_id']
    del df_scores['user_id_target_item_id']
    return df_explanations


def fit(df_explanations, loss='warp', epochs=30, num_threads=4):
    """Learn rankings using Matrix Factorization."""
    df_ratings = extract_ratings(df_explanations)
    user_ids, item_ids, matrix = build_interaction_matrix(df=df_ratings)

    model = LightFM(loss=loss)
    model.fit(matrix, epochs=epochs, num_threads=num_threads)

    df_scores = get_scores(df_explanations, user_ids, item_ids, model)
    df_explanations = add_scores_to_explanations(df_explanations, df_scores)
    return df_explanations


def get_scores(df_explanations, user_ids, item_ids, model):
    def _get_scores():
        for _, row in df_explanations.iterrows():
            user_idx = user_ids.index(row.user_id)
            item_idx = item_ids.index(row.target_item_id)
            score = model.predict(user_idx, np.array([item_idx]))[0]
            yield dict(user_id=row.user_id, item_id=row.target_item_id, score=score)

    return pd.DataFrame(_get_scores())


def compute_ndcg(df_explanations):
    """Computes NDCG for sessions in a explanation DataFrame

    Args:
        df_explanations: Explanation DataFrame

    Returns:

    """
    rank_cols = ['rank_related_items_sims_np', 'rank_mf', 'rank_strength', 'rank_target_item_average_rating']

    def _compute(df, method=2, session_id=None):
        df_len = len(df)
        for col in rank_cols:
            df_sorted = df.sort_values(col)
            average_ratings = df_sorted.target_item_average_rating.values
            for idx in range(df_len):
                k = idx + 1
                yield dict(k=k, col=col, session_id=session_id, ndcg=ndcg_at_k(average_ratings, k=k, method=method))

    def _run():
        for idx, (session_id, df) in enumerate(df_explanations.groupby('session_id', sort=False)):
            yield from _compute(df=df, session_id=session_id)

    return pd.DataFrame.from_dict(_run())


def summarize_ndcg(df_ndcgs):
    exclude_cols = ['rank_target_item_average_rating']
    df_ndcgs_summary = df_ndcgs.query('col != @exclude_cols').groupby(['col', 'k']).ndcg.mean()
    df_ndcgs_summary = df_ndcgs_summary.reset_index()
    return df_ndcgs_summary


def build_interaction_matrix(df=None):
    """Create user-item interaction matrix.

    Args:
        df: DataFrame with columns `user_id`, `item_id` and `rating`

    Returns:
        Tuple of (user_ids, item_ids, matrix)
    """
    expected_cols = ['user_id', 'item_id', 'rating']
    common_cols = set(df.columns).intersection(expected_cols)
    assert len(common_cols) == 3

    def _build_matrix():
        """Build user-item matrix."""
        mat = sp.lil_matrix((len(_user_ids), len(_item_ids)), dtype=np.int32)
        for idx, row in df.iterrows():
            user_idx = _user_ids.index(row.user_id)
            item_idx = _item_ids.index(row.item_id)
            mat[user_idx, item_idx] = 1.0 if row.rating >= 2.5 else -1.0
        return mat.tocoo()

    _user_ids = df.user_id.unique().tolist()
    _item_ids = df.item_id.unique().tolist()
    _matrix = _build_matrix()

    return _user_ids, _item_ids, _matrix


def plot_ndcg(df_ndcgs_summary, markersize=12, figsize=(9, 7), title=None, ylim=None, ylabel=None,
              ax=None, title_fontsize='small'):
    markers = cycle(['s', 'v', '^', 'o', 'D', '<', '>', 'p', '*', '+', 'd', '1',
                     '2', '3', '4', '8'])
    colors = cycle(['r', 'b', 'g', 'y', 'c', 'm', 'k'])
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    for col in df_ndcgs_summary.col.unique():
        df = df_ndcgs_summary
        label = col
        c = next(colors)

        df.query('col == @col').set_index('k').rename(columns={'ndcg': label}).plot(
            kind='line',
            ax=ax,
            marker=next(markers),
            markersize=markersize,
            linewidth=1,
            alpha=0.9,
            label=label,
            color=c,
            linestyle='-',
            markerfacecolor='w',
            markeredgewidth=0.9,
            legend=False
        )

    ax.legend(scatterpoints=2, ncol=2, loc='best')
    ax.set_title(title, fontsize=title_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        elif method == 2:
            # return sum([(np.power(2.0, r[i]) - 1)/ (np.log2(i+2)) for i in range(k)])
            return np.sum((np.power(2.0, v) - 1) / (np.log2(i + 2)) for i, v in enumerate(r))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.0


def ndcg_at_k(r, k, method=2):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)

    if not dcg_max:
        print('Oops!!')
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
