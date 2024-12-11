from math import floor
import numpy as np
import bittensor as bt

from chunking.validator.types import EndTournamentRoundInfo


def get_rank_value_to_adjusted_alpha(ranks: np.ndarray[np.float64], alpha: float):
    """
    Gets the adjusted alpha for each rank value. This adjust alpha according to those miners that tie.
    """
    rank_value_to_count = {}
    for rank in ranks:
        if rank not in rank_value_to_count:
            rank_value_to_count[rank] = 1
        else:
            rank_value_to_count[rank] += 1

    bt.logging.debug(f"rank_value_to_count: {rank_value_to_count}")

    rank_value_to_adjusted_alpha = {}

    for rank, count in rank_value_to_count.items():
        rank_value_to_adjusted_alpha[rank] = alpha / (2 ** (count - 1))

    bt.logging.debug(f"rank_value_to_adjusted_alpha: {rank_value_to_adjusted_alpha}")

    return rank_value_to_adjusted_alpha


def get_new_scores(
    scores: np.ndarray[np.float64],
    uids: np.ndarray[np.int64],
    alpha: float,
    group_best_possible_rank_value: float,
    rank_values: np.ndarray[np.float64],
) -> np.ndarray[np.float64]:

    scores_copy = scores.copy()

    if np.isnan(rank_values).any():
        bt.logging.warning(f"NaN values detected in rank values: {rank_values}")
        # Replace any NaN values in rank values with inf.
        rank_values = np.nan_to_num(rank_values, nan=np.inf)

    bt.logging.debug(f"group alpha: {alpha}")

    # adjust for tiebreaking
    rank_value_to_adjusted_alpha = get_rank_value_to_adjusted_alpha(rank_values, alpha)

    for rank_value, uid in zip(rank_values, uids):
        # unranked
        if np.isinf(rank_value):
            continue

        adjusted_alpha = rank_value_to_adjusted_alpha[rank_value]

        bt.logging.debug(
            f"uid: {uid}, cur score: {scores_copy[uid]}, rank value: {rank_value}, adjusted_alpha: {adjusted_alpha}. group best possible rank value: {group_best_possible_rank_value}."
        )
        score_str = f"score: {scores_copy[uid]} -> "

        if (
            rank_value == group_best_possible_rank_value
            and scores_copy[uid] < rank_value
        ):
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
            scores_copy[uid] = adjusted_alpha * rank_value + (
                1 - adjusted_alpha
            ) * floor(np.sum(np.isfinite(scores_copy)) / 2)
        elif scores_copy[uid] < 0:
            scores_copy[uid] = np.inf
        else:
            scores_copy[uid] = (
                adjusted_alpha * rank_value + (1 - adjusted_alpha) * scores_copy[uid]
            )

        score_str += f"{scores_copy[uid]}"
        bt.logging.debug(score_str)

    return scores_copy