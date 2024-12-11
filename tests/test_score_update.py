import asyncio
import logging
import bittensor as bt
import numpy as np

from chunking.utils.score import get_new_scores

logger = logging.getLogger(__name__)

NUM_GROUPS = 21


async def main():
    bt.debug()
    uids = np.arange(7)
    scores = np.array([0.3, 0.4, 0.9, 1.2, 1.5, 1.8, 2.1])

    base_alpha = 0.025
    alpha_inc = (1 - base_alpha) / NUM_GROUPS
    group_alphas = [base_alpha + i * alpha_inc for i in range(21)]
    logger.info(f"group_alphas: {group_alphas}")

    logger.info(f"uids: {uids}")
    logger.info(f"scores: {scores}")

    # group 1
    group_1_uids = uids[1:5]

    group_1_rank_values = np.array([0.5, 0.5, 1.5, 2.5])

    group_1_alpha = group_alphas[1]

    new_scores = get_new_scores(
        scores=scores,
        uids=group_1_uids,
        alpha=group_1_alpha,
        group_best_possible_rank_value=0.5,
        rank_values=group_1_rank_values,
    )

    logger.info(f"new_scores: {new_scores}")

    # uid 1 should not change because their score is already lower than the group's best possible rank value
    assert new_scores[1] == scores[1]
    logger.info("uid 1 group 1 score is right")

    # uid 2 should get lower score, alpha should have tiebreak
    assert new_scores[2] < scores[2]
    uid_2_alpha = group_1_alpha * 0.5
    assert new_scores[2] == (uid_2_alpha * group_1_rank_values[1]) + (
        (1 - uid_2_alpha) * scores[2]
    )
    logger.info("uid 2 group 1 score is right")

    assert new_scores[3] > 0
    assert new_scores[3] == (group_1_alpha * group_1_rank_values[2]) + (
        (1 - group_1_alpha) * scores[3]
    )
    logger.info("uid 3 group 1 score is right")

    assert new_scores[4] > 0
    assert new_scores[4] == (
        group_1_alpha * group_1_rank_values[3] + ((1 - group_1_alpha) * scores[4])
    )
    logger.info("uid 4 group 1 score is right")

    # increase uid 1 score > best possible rank value
    new_scores[1] = 0.6

    scores = new_scores

    new_scores = get_new_scores(
        scores=new_scores,
        uids=group_1_uids,
        alpha=group_1_alpha,
        group_best_possible_rank_value=0.5,
        rank_values=group_1_rank_values,
    )
    logger.info(f"new_scores: {new_scores}")

    # uid 1 should change because their score is not lower than the group's best possible rank value
    assert new_scores[1] < scores[1]
    # apply tie mechanism
    uid_1_alpha = group_1_alpha * 0.5
    assert new_scores[1] == (uid_1_alpha * group_1_rank_values[0]) + (
        (1 - uid_1_alpha) * scores[1]
    )
    logger.info("uid 1 group 1 score is right")

    # uid 2 should get lower score, alpha should have tiebreak
    assert new_scores[2] < scores[2]
    # apply tie mechanism
    uid_2_alpha = group_1_alpha * 0.5
    assert new_scores[2] == (uid_2_alpha * group_1_rank_values[1]) + (
        (1 - uid_2_alpha) * scores[2]
    )
    logger.info("uid 2 group 1 score is right")

    assert new_scores[3] > 0
    assert new_scores[3] == (group_1_alpha * group_1_rank_values[2]) + (
        (1 - group_1_alpha) * scores[3]
    )
    logger.info("uid 3 group 1 score is right")

    assert new_scores[4] > 0
    assert new_scores[4] == (
        group_1_alpha * group_1_rank_values[3] + ((1 - group_1_alpha) * scores[4])
    )
    logger.info("uid 4 group 1 score is right")


def test_score_update():
    asyncio.run(main())
