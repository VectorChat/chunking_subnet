import numpy as np
import bittensor as bt


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
