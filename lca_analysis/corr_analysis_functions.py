import numpy as np
import pandas as pd
import scipy.stats as stats


def get_models_set(task_dict):
    # get set of models for each task.
    model_dict = {}
    for task, model_scores in task_dict.items():
        model_set = {model.lower() for model in model_scores.keys()}
        model_dict[task] = model_set

    return model_dict


def model_task_count(tasks_dict):
    # Calculate number of tasks, in which models are used
    all_tasks = set(tasks_dict.keys())
    # all_tasks.remove("plcc_proj_lev_path_dist_em")
    model_analysis = {}

    for task, model_scores in tasks_dict.items():
        for model in model_scores.keys():
            model_lower = model.lower()
            if model_lower not in model_analysis:
                model_analysis[model_lower] = {
                    "num_tasks": 0,
                    "tasks": set(),
                    "missed_tasks": set(),
                }
            model_analysis[model_lower]["num_tasks"] += 1
            model_analysis[model_lower]["tasks"].add(task)

    for model, details in model_analysis.items():
        details["missed_tasks"] = list(all_tasks - details["tasks"])
        details["tasks"] = list(details["tasks"])

    sorted_model_analysis = dict(
        sorted(
            model_analysis.items(),
            key=lambda item: len(item[1]["missed_tasks"]),
            reverse=False,
        )
    )

    return sorted_model_analysis


def normalize_scores(task_scores: dict) -> dict:
    min_score = min(task_scores.values())
    max_score = max(task_scores.values())
    scaling_factor = max_score - min_score
    normalized_dict = {
        model: (score - min_score) / scaling_factor
        for model, score in task_scores.items()
    }

    return normalized_dict


def filter_common_models(task_results):

    model_count = model_task_count(task_results)
    common_models = set()
    for model, task_info in model_count.items():
        if len(task_info["missed_tasks"]) == 0:
            common_models.add(model)

    task_results_filtered = dict()
    for task, scores in task_results.items():
        filtered_scores = {
            model: score for model, score in scores.items() if model in common_models
        }
        if len(filtered_scores) != 0:
            task_results_filtered[task] = filtered_scores

    return task_results_filtered


def rank_column_with_margin(series, margin=0.05, ascending=False):
    """
    Rank values with a margin of equality.

    Parameters:
    -----------
    series : pandas.Series
        The series of values to rank
    margin : float
        The margin within which values are considered equal
    ascending : bool
        Whether smaller values should receive smaller ranks

    Returns:
    --------
    pandas.Series with ranks
    """
    # Sort values (descending by default for ranking)
    if ascending:
        sorted_vals = series.sort_values()
    else:
        sorted_vals = series.sort_values(ascending=False)

    current_rank = 1
    previous_val = sorted_vals.iloc[0]
    ranks = {sorted_vals.index[0]: current_rank}

    # Assign ranks with margin consideration
    for idx in range(1, len(sorted_vals)):
        current_val = sorted_vals.iloc[idx]
        if abs(current_val - previous_val) <= margin:
            # Same rank
            ranks[sorted_vals.index[idx]] = current_rank
        else:
            # New rank
            current_rank = idx + 1
            ranks[sorted_vals.index[idx]] = current_rank
            previous_val = current_val

    return pd.Series(ranks)


def rank_with_margin(data: pd.DataFrame, margin: float = 0.05):
    ranked_data = pd.DataFrame()
    for column in data.columns:
        ranked_data[column] = rank_column_with_margin(
            data[column], margin=margin, ascending=False
        )

    return ranked_data


def kendall_w(df):
    """
    Calculate Kendall's W (Coefficient of Concordance) for a DataFrame of rankings.

    Parameters:
    df (pandas.DataFrame): A DataFrame where rows - models,
                          columns - tasks, and values are the ranks.

    Returns:
    float: Kendall's W coefficient, ranging from 0 (no agreement) to 1 (complete agreement)
    float: Chi-square statistic
    float: p-value for the null hypothesis that rankings are independent
    """
    rankings = df.transpose().to_numpy()

    m = rankings.shape[0]  # Number of tasks
    n = rankings.shape[1]  # Number of models

    rank_sums = np.sum(rankings, axis=0)
    mean_rank_sum = np.mean(rank_sums)
    S = np.sum((rank_sums - mean_rank_sum) ** 2)
    W = 12 * S / (m**2 * (n**3 - n))
    # Calculate chi-square statistic
    chi2 = m * (n - 1) * W
    # Calculate p-value
    df_chi = n - 1
    p_value = 1 - stats.chi2.cdf(chi2, df_chi)

    return {"W": W, "chi2": chi2, "p_value": p_value}
