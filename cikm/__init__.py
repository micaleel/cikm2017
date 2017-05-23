import logging
import os
from itertools import cycle

import numpy as np
import pandas as pd
import simplejson as json
from matplotlib import pyplot as plt
from scipy import sparse as sp

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Filter incomplete sessions.
def filter_incomplete_sessions(df_exp, session_size=10):
    """Remove sessions that have less than `min_session_size` recommendations.

    Args:
        df_exp: DataFrame of explanations.
        session_size (int): Expected minimum number of recommendations per session.

    Returns:
        Filtered DataFrame of explanations.
    """
    session_sizes = df_exp.groupby('session_id', sort=False).size()
    session_sizes = session_sizes[session_sizes == session_size]
    session_ids = session_sizes.index

    return df_exp[df_exp.session_id.isin(session_ids)]


def ratings_from_explanations(df_explanations):
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


def get_scores(df_explanations, user_ids, item_ids, model):
    def _get_scores():
        for _, row in df_explanations.iterrows():
            user_idx = user_ids.index(row.user_id)
            item_idx = item_ids.index(row.target_item_id)
            score = model.predict(user_idx, np.array([item_idx]))[0]
            yield dict(user_id=row.user_id, item_id=row.target_item_id, score=score)

    return pd.DataFrame(_get_scores())


def compute_ndcg(df_explanations: pd.DataFrame, min_session_size: int = 10, rank_cols=None):
    """Computes NDCG for sessions in a explanation DataFrame

    Args:
        df_explanations (DataFrame): Explanation DataFrame
        min_session_size (int): Minimum expected recommendations per session.
        rank_cols: Columns to rank sessions by.
    Returns:

    """
    # rank_cols = ['rank_related_items_sims_np', 'rank_strength', 'rank_target_item_average_rating']
    rank_cols = rank_cols or [c for c in df_explanations.columns if c.startswith('rank_')]

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
            if min_session_size:
                n_recommendations = len(df)
                if n_recommendations < min_session_size:
                    logger.warning('Skipping session: {} with {:,} recommendations. min_session_size={}'.format(
                        session_id, n_recommendations, min_session_size))
                    continue
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
            # mat[user_idx, item_idx] = 1.0 if row.rating >= 2.5 else -1.0
            mat[user_idx, item_idx] = row.rating
        return mat.tocoo()

    _user_ids = df.user_id.unique().tolist()
    _item_ids = df.item_id.unique().tolist()
    print('#users: {:,}\n#items: {:,}'.format(len(_user_ids), len(_item_ids)))
    _matrix = _build_matrix()

    return _user_ids, _item_ids, _matrix


def plot_ndcg(df_ndcgs_summary, markersize=12, figsize=(9, 7), title=None, ylim=None, ylabel=None,
              ax=None, title_fontsize='small', plot_kwds=None):
    markers = cycle(['s', 'v', '^', 'o', 'D', '<', '>', 'p', '*', '+', 'd', '1',
                     '2', '3', '4', '8'])
    colors = cycle(['r', 'b', 'g', 'y', 'c', 'm', 'k'])
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    for col in df_ndcgs_summary.col.unique():
        df = df_ndcgs_summary
        label = col
        c = next(colors)

        default_plot_kwds = dict(marker=next(markers),
                                 markersize=markersize,
                                 linewidth=1,
                                 alpha=0.9,
                                 label=label,
                                 color=c,
                                 linestyle='-',
                                 markerfacecolor='w',
                                 markeredgewidth=0.9,
                                 legend=False)
        plot_kwds = plot_kwds or default_plot_kwds

        df.query('col == @col').set_index('k').rename(columns={'ndcg': label}).plot(
            kind='line',
            ax=ax,

            **plot_kwds
        )

    # ax.legend(scatterpoints=2, ncol=2, loc='best')
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


def merge_predictions(extraction_csv_path, prediction_csv_path, algorithm, explanation_hdf5_path):
    """Merge rating predictions to explanation DataFrame
    
    Args:
        extraction_csv_path: Path to extractions file.
        prediction_csv_path: Path to rating predictions.
        algorithm: short name of librec recommendation algorithm (e.g. itemknn, biasedmf).
        explanation_hdf5_path: Path to explanation HDF5 file.
        
    Returns:
         DataFrame of explanations.
    """
    # Load extractions.
    result = ratings_from_extractions(ext_csv_path=extraction_csv_path)
    df_ratings, user_ids_map, item_ids_map = result
    print('len(df_ratings) ', len(df_ratings))

    # Load predictions.
    print(prediction_csv_path)
    df_predictions = pd.read_csv(prediction_csv_path, names=['user_id', 'item_id', 'rating'])

    # Merge predictions with user ratings from reviews.
    cols = ['user_id', 'item_id', 'rating']
    df_all_ratings = pd.concat([df_ratings[cols], df_predictions])
    del df_ratings, df_predictions

    # Map integer IDs to their oldinal versions, then
    # create column to merge all ratings to explanations DataFrame.
    df_all_ratings['user_id_old'] = df_all_ratings['user_id'].map(invert_dict(user_ids_map))
    df_all_ratings['item_id_old'] = df_all_ratings['item_id'].map(invert_dict(item_ids_map))
    df_all_ratings['user_id_old'] = df_all_ratings['user_id_old'].astype(str)
    df_all_ratings['item_id_old'] = df_all_ratings['item_id_old'].astype(str)
    df_all_ratings['user_id_target_item_id'] = df_all_ratings.user_id_old + '#' + df_all_ratings['item_id_old']

    # Load explanations.
    df_explanations = pd.read_hdf(explanation_hdf5_path)
    df_explanations['user_id'] = df_explanations.user_id.astype(str)
    df_explanations['target_item_id'] = df_explanations.target_item_id.astype(str)
    df_explanations['user_id_target_item_id'] = df_explanations.user_id + '#' + df_explanations.target_item_id
    n_explanations = len(df_explanations)
    print('Loaded {:,} explanations'.format(len(df_explanations)))

    # Add predicted ratings to explanations DataFrame.
    rating_cols = ['user_id_target_item_id', 'rating']
    rating_map = df_all_ratings[rating_cols].set_index('user_id_target_item_id').rating.to_dict()
    df_explanations['rating_{}'.format(algorithm)] = df_explanations.user_id_target_item_id.map(rating_map)
    del df_explanations['user_id_target_item_id']  # Remove the column after merge operation.
    assert len(df_explanations) == n_explanations

    # Rank explanations in each session.
    def _rank_explanations(df_session):
        return rank_explanations(df_session, rank_col='rating_{}'.format(algorithm),
                                 rank_col_suffix=algorithm)

    if 'rank_{}'.format(algorithm) in df_explanations.columns:
        df_explanations.drop('rank_{}'.format(algorithm), axis=1, inplace=True)

    df_exp_ranks = df_explanations.groupby('session_id').apply(_rank_explanations)
    df_explanations = df_explanations.merge(df_exp_ranks, on='explanation_id')
    assert len(df_explanations) == n_explanations
    return df_explanations


def invert_dict(d):
    return {v: k for k, v in d.items()}


def rank_explanations(df_session, rank_col: str, rank_col_suffix: str):
    """Rank explanations in each session.
    
    Args:
        df_session: DataFrame of explanations belonging to a single session.
        rank_col: Name of column to rank explanations by.
        rank_col_suffix: Suffix to add to resulting column from rank operation.
        
    Returns:
        DataFrame of explanation IDs and their rank.
    """
    col_name = 'rank_{}'.format(rank_col_suffix)
    df_session[col_name] = df_session[rank_col].rank(method='first', ascending=False)
    return df_session[['explanation_id', col_name]]


def ratings_from_extractions(ext_csv_path, save_dir=None):
    """Extract user, item and rating pairs from extractions file.
    
    Args:
        ext_csv_path: Path to extractions CSV file.
        save_dir: Directory to save ratings to, if necessary.
        
    Returns:
        A tuple like (df_ratings, user_ids_map, item_ids_map).
            df_ratings: DataFrame of ratings with columns user_id, item_id, rating
            user_ids_map: Dictionary of original user IDs maps to integer values.
            item_ids_map: Dictionary of original item IDs maps to integer values.
    """
    cols = ['rating', 'user_id', 'item_id', 'review_id']
    logger.info('Loading extractions from {}'.format(ext_csv_path))
    df_extractions = (pd.read_csv(ext_csv_path)
                      .rename(columns={'member_id': 'user_id',
                                       'hotel_id': 'item_id',
                                       'user_rating': 'rating'})[cols])
    df_extractions = df_extractions.drop_duplicates().copy()
    df_extractions.drop('review_id', axis=1)
    df_extractions['user_id'] = df_extractions.user_id.astype(str)
    df_extractions['item_id'] = df_extractions.item_id.astype(str)

    # Map user_ids and item_ids to integers
    _user_ids_map = _map_ids_to_ints(df_extractions.user_id)
    _item_ids_map = _map_ids_to_ints(df_extractions.item_id)

    # Create record of user ratings.
    _df_ratings = (df_extractions.rename(columns={'user_id': 'user_id_orig',
                                                  'item_id': 'item_id_orig'}))
    _df_ratings['user_id'] = _df_ratings.user_id_orig.map(_user_ids_map)
    _df_ratings['item_id'] = _df_ratings.item_id_orig.map(_item_ids_map)

    if save_dir:
        # Save map of user & item IDs to disk.
        logger.info('Saving ratings, user and item ID maps to {}'.format(save_dir))
        path_fmt = os.path.join(save_dir, '{}_ids_map.json')
        _save_ids_map(_user_ids_map, path_fmt.format('users'))
        _save_ids_map(_item_ids_map, path_fmt.format('items'))
        save_path = os.path.join(save_dir, 'ratings.csv')
        _df_ratings[['user_id', 'item_id', 'rating']].to_csv(save_path, index=False, header=False)

    return _df_ratings, _user_ids_map, _item_ids_map


def get_rating_density(extraction_path):
    aliases = {'hotel_id': 'item_id', 'member_id': 'user_id', 'user_rating': 'rating'}
    cols = ['user_id', 'item_id', 'rating', 'review_id']
    df_extractions = pd.read_csv(extraction_path).rename(columns=aliases)[cols].drop_duplicates()
    density = len(df_extractions) / (df_extractions.user_id.nunique() * df_extractions.item_id.nunique())
    return density


def create_train_and_test_data(ext_path, exp_path):
    """Create training and test data for librec
    
    Args:
        ext_path: Path to extractions DataFrame
        exp_path: Path to explanations DataFrame
        
    Returns:
        Tuple (df_ratings, user_ids_map, item_ids_map, df_test)
    """

    # Create directory to save librec data
    save_dir = os.path.join(ext_path[:ext_path.rfind('/') + 1], 'librec-data')
    os.makedirs(save_dir, exist_ok=True)

    result = ratings_from_extractions(ext_csv_path=ext_path, save_dir=save_dir)
    df_ratings, user_ids_map, item_ids_map = result

    df_test = extract_test_data(exp_path=exp_path,
                                df_ratings=df_ratings,
                                item_ids_map=item_ids_map,
                                user_ids_map=user_ids_map,
                                save_dir=save_dir)

    return df_ratings, user_ids_map, item_ids_map, df_test


def extract_test_data(exp_path, df_ratings, user_ids_map, item_ids_map, save_dir=None):
    cols = ['user_id', 'is_seed', 'target_item_id', 'seed_item_id',
            'target_item_average_rating']
    path = os.path.join(exp_path)
    df_explanations = pd.read_hdf(path)[cols]
    df_explanations.rename(columns={'user_id': 'user_id_orig',
                                    'target_item_average_rating': 'rating'},
                           inplace=True)
    df_explanations['user_id'] = df_explanations.user_id_orig.map(user_ids_map)
    df_explanations['item_id'] = df_explanations.target_item_id.map(item_ids_map)

    # Create test set -- pairs to generate personalized recommendations for.
    df_test = df_explanations.query('is_seed == False')[['user_id', 'item_id', 'rating']].copy()
    df_test.drop_duplicates(inplace=True)

    df_explanations['user_id'] = df_explanations.user_id.astype(str)
    df_explanations['target_item_id'] = df_explanations.target_item_id.astype(str)
    df_test['user_id'] = df_test.user_id.astype(str)
    df_test['item_id'] = df_test.item_id.astype(str)
    df_ratings['user_id'] = df_ratings.user_id.astype(str)
    df_ratings['item_id'] = df_ratings.item_id.astype(str)

    # Remove observations in test set that are already in training set.
    df_test['user_id_item_id'] = df_test.user_id + '#' + df_test.item_id
    df_ratings['user_id_item_id'] = df_ratings.user_id + '#' + df_ratings.item_id

    exclude = df_ratings.user_id_item_id.unique().tolist()
    df_test = df_test[~df_test.user_id_item_id.isin(exclude)]

    # Save test set.
    if save_dir:
        save_path = os.path.join(save_dir, 'test.csv')
        df_test[['user_id', 'item_id', 'rating']].to_csv(save_path, index=False, header=False)
    return df_test


def _save_ids_map(ids_dict, file_path):
    with open(file_path, 'w') as fp:
        json.dump(ids_dict, fp)


def _map_ids_to_ints(s: pd.Series):
    # display(s.unique())
    ids_list = sorted(s.unique())
    return dict(zip(ids_list, range(len(ids_list))))
