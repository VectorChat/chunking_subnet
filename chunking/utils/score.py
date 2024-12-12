from math import floor
import numpy as np
import bittensor as bt

from chunking.validator.types import EndTournamentRoundInfo


#  rank_value_to_adjusted_alpha[rank] = alpha / (2 ** (count - 1))


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

        # adjust alpha if lost (did not tie)
        if not did_win:
            alpha = alpha * (
                2 if miner_group_index == 0 else (1 + 0.25**miner_group_index)
            )
        bt.logging.debug(f"loss alpha: {alpha}")

        # adjust alpha based on how many other uids got the same score as this uid
        # if more people tie, affects score less
        alpha = alpha / (2 ** (rank_value_to_count[rank_value] - 1))

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
