from math import floor
import numpy as np
import bittensor as bt


def get_alpha(
    self,  # Validator
    num_miner_groups: int,
    miner_group_index: int,
    override_min_moving_average_alpha: float | None = None,
):
    """
    Get's the alpha value for a specific group ("tiered" alpha) where the alpha is higher for miner groups that are worse (higher number)
    Ex:
        the first miner group as the "highest rank" and therefore the lowest alpha value
        the last miner group as the "lowest rank" and therefore the highest alpha

    This means that the scores are updated more slowly at higher ranks, because these miners should only be punished
    if they _consistently_ produce low quality responses. At lower ranks, the alpha value is higher, this allows for
    higher variability in the scores at lower ranks, allowing new miners with high quality responses to rise the ranks
    more quickly.

    Args:
        num_miner_groups (int): The number of miner groups.
        miner_group_index (int): The index of the miner group.
        override_min_moving_average_alpha (float | None): The alpha to use if the override is provided.

    Returns:
        float: The alpha value.
    """
    min_moving_average_alpha = (
        override_min_moving_average_alpha
        if override_min_moving_average_alpha
        else self.config.neuron.min_moving_average_alpha
    )
    alpha_adjustment = (1 - min_moving_average_alpha) / max((num_miner_groups - 1), 1)
    alpha = min_moving_average_alpha + alpha_adjustment * miner_group_index

    return alpha


def get_rank_value_to_count(ranks: np.ndarray[np.float64]) -> dict[float, int]:
    """
    Gets the count of each rank value.
    """
    rank_value_to_count: dict[float, int] = {}
    for rank in ranks:
        if rank not in rank_value_to_count:
            rank_value_to_count[rank] = 1
        else:
            rank_value_to_count[rank] += 1

    bt.logging.debug(f"rank_value_to_count: {rank_value_to_count}")

    return rank_value_to_count


def get_new_scores(
    scores: np.ndarray[np.float64],
    uids: np.ndarray[np.int64],
    alpha: float,
    group_best_possible_rank_value: float,
    rank_values: np.ndarray[np.float64],
    miner_group_index: int,
) -> np.ndarray[np.float64]:

    scores_copy = scores.copy()

    if np.isnan(rank_values).any():
        bt.logging.warning(f"NaN values detected in rank values: {rank_values}")
        # Replace any NaN values in rank values with inf.
        rank_values = np.nan_to_num(rank_values, nan=np.inf)

    group_alpha = alpha

    # for use later by tieing mechanism
    rank_value_to_count = get_rank_value_to_count(rank_values)

    for rank_value, uid in zip(rank_values, uids):
        # unranked
        if np.isinf(rank_value):
            continue

        did_win = rank_value == group_best_possible_rank_value

        bt.logging.debug(f"group alpha: {group_alpha}")

        alpha = group_alpha

        # adjust alpha if miner lost (did not get first place within group)
        if not did_win:
            alpha = alpha * (
                2 if miner_group_index == 0 else (1 + 0.25**miner_group_index)
            )
        bt.logging.debug(f"loss alpha: {alpha}")

        # adjust alpha based on how many other uids got the same score as this uid
        # if more people tie, affects score less
        alpha = alpha / max(rank_value_to_count[rank_value], 1)

        bt.logging.debug(f"tie alpha: {alpha}")

        bt.logging.debug(
            f"uid: {uid}, cur score: {scores_copy[uid]}, rank value: {rank_value} group best possible rank value: {group_best_possible_rank_value}. did win: {did_win}"
        )
        score_str = f"score: {scores_copy[uid]} -> "

        if did_win and scores_copy[uid] < rank_value:
            # miner should not be penalized (score increases) if:
            # 1. they did the best in their group
            # 2. the miner's score is already lower than the group's best possible rank value
            #
            # this is possible if the miner is in 2 groups (all miners except the first place miner)
            score_str += f"{scores_copy[uid]} (no change)"
            bt.logging.debug(score_str)
            continue

        # initialize score if it is np.inf
        if np.isinf(scores_copy[uid]):
            scores_copy[uid] = alpha * rank_value + (1 - alpha) * floor(
                np.sum(np.isfinite(scores_copy)) / 2
            )
        elif scores_copy[uid] < 0:
            scores_copy[uid] = np.inf
        else:
            scores_copy[uid] = alpha * rank_value + (1 - alpha) * scores_copy[uid]

        score_str += f"{scores_copy[uid]}"
        bt.logging.debug(score_str)

    return scores_copy
