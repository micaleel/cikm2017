from lightfm import LightFM

from cikm import ratings_from_explanations, build_interaction_matrix, get_scores, add_scores_to_explanations


def fit(df_explanations, loss='warp', epochs=30, num_threads=4):
    """Learn rankings using Matrix Factorization."""
    df_ratings = ratings_from_explanations(df_explanations)
    user_ids, item_ids, matrix = build_interaction_matrix(df=df_ratings)

    # model = LightFM(loss=loss)
    model = LightFM()
    model.fit(matrix, epochs=epochs, num_threads=num_threads)

    df_scores = get_scores(df_explanations, user_ids, item_ids, model)
    df_explanations = add_scores_to_explanations(df_explanations, df_scores)
    return df_explanations