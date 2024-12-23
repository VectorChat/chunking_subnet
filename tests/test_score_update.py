import asyncio
import logging
import bittensor as bt
import numpy as np

from chunking.utils.score import get_new_scores, get_rank_value_to_count

logger = logging.getLogger(__name__)

NUM_GROUPS = 21


async def main():
    bt.debug()
    uids = np.arange(7)
    scores = np.array([0.3, 0.4, 0.9, 1.2, 1.5, 1.8, 2.1])

    base_alpha = 0.025
    alpha_inc = (1 - base_alpha) / NUM_GROUPS
    group_alphas = [base_alpha + i * alpha_inc for i in range(21)]
    loss_alpha_mults = [2 if i == 0 else 1 + 0.25**i for i in range(21)]
    logger.info(f"group_alphas: {group_alphas}")

    logger.info(f"uids: {uids}")
    logger.info(f"scores: {scores}")

    # group 1
    group_1_uids = uids[1:5]

    group_1_rank_values = np.array([0.5, 0.5, 1.5, 2.5])
    logger.info("-- testing group 1 tie, loss alpha, no score change --")

    group_1_alpha = group_alphas[1]

    new_scores = get_new_scores(
        scores=scores,
        uids=group_1_uids,
        alpha=group_1_alpha,
        group_best_possible_rank_value=0.5,
        rank_values=group_1_rank_values,
        miner_group_index=1,
    )

    logger.info(f"new_scores: {new_scores}")

    # uid 1 should not change because their score is already lower than the group's best possible rank value
    assert new_scores[1] == scores[1]
    logger.info("uid 1 group 1 score is right")

    # uid 2 should get lower score, alpha should have tiebreak
    assert new_scores[2] < scores[2]
    uid_2_alpha = group_1_alpha / 2
    assert new_scores[2] == (uid_2_alpha * group_1_rank_values[1]) + (
        (1 - uid_2_alpha) * scores[2]
    )
    logger.info("uid 2 group 1 score is right")

    # uid 3 should get higher score, bc rank_value > cur_score
    assert new_scores[3] > scores[3]
    # factor in loss alpha
    uid_3_alpha = group_1_alpha * 1.25
    assert new_scores[3] == (uid_3_alpha * group_1_rank_values[2]) + (
        (1 - uid_3_alpha) * scores[3]
    )
    logger.info("uid 3 group 1 score is right")

    assert new_scores[4] > scores[4]
    # factor in loss alpha
    uid_4_alpha = group_1_alpha * 1.25
    assert new_scores[4] == (
        uid_4_alpha * group_1_rank_values[3] + ((1 - uid_4_alpha) * scores[4])
    )
    logger.info("uid 4 group 1 score is right")

    # increase uid 1 score > best possible rank value
    new_scores[1] = 0.6

    scores = new_scores

    # have uid 3 and 4 tie
    group_1_rank_values = np.array([0.5, 0.5, 2.5, 2.5])
    logger.info("-- testing uid 1 and 2 tie, uid 3 and 4 tie --")

    new_scores = get_new_scores(
        scores=new_scores,
        uids=group_1_uids,
        alpha=group_1_alpha,
        group_best_possible_rank_value=0.5,
        rank_values=group_1_rank_values,
        miner_group_index=1,
    )
    logger.info(f"new_scores: {new_scores}")

    rank_value_to_count = get_rank_value_to_count(group_1_rank_values)

    # uid 1 should change because their score is not lower than the group's best possible rank value
    assert new_scores[1] < scores[1]
    # apply tie mechanism
    tie_mult = 1 / 2 ** (rank_value_to_count[group_1_rank_values[0]] - 1)
    assert 0.5 == tie_mult
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

    assert new_scores[3] > scores[3]
    # apply loss alpha
    uid_3_alpha = group_1_alpha * loss_alpha_mults[1]
    # apply tie alpha
    uid_3_alpha = (
        uid_3_alpha * 1 / 2 ** (rank_value_to_count[group_1_rank_values[2]] - 1)
    )
    assert new_scores[3] == (uid_3_alpha * group_1_rank_values[2]) + (
        (1 - uid_3_alpha) * scores[3]
    )
    logger.info("uid 3 group 1 score is right")

    assert new_scores[4] > scores[4]
    uid_4_alpha = group_1_alpha * loss_alpha_mults[1]
    # apply tie alpha
    uid_4_alpha = uid_4_alpha * 0.5
    assert new_scores[4] == (
        uid_4_alpha * group_1_rank_values[3] + ((1 - uid_3_alpha) * scores[4])
    )
    logger.info("uid 4 group 1 score is right")

    # other scores should not change
    assert np.array_equal(scores[:1], new_scores[:1])
    assert np.array_equal(scores[5:], new_scores[5:])

    # 4-way tie
    group_1_rank_values = np.array([0.5, 0.5, 0.5, 0.5])
    logger.info("-- testing 4-way tie for first place --")

    scores = new_scores

    new_scores = get_new_scores(
        scores=new_scores,
        uids=group_1_uids,
        alpha=group_1_alpha,
        group_best_possible_rank_value=0.5,
        rank_values=group_1_rank_values,
        miner_group_index=1,
    )

    logger.info(f"new_scores: {new_scores}")

    # uid 1 should get lower score, alpha should have tiebreak
    assert new_scores[1] < scores[1]
    uid_1_alpha = group_1_alpha / 4
    assert new_scores[1] == (uid_1_alpha * group_1_rank_values[0]) + (
        (1 - uid_1_alpha) * scores[1]
    )
    logger.info("uid 1 group 1 score is right")

    # uid 2 should get lower score, alpha should have tiebreak
    assert new_scores[2] < scores[2]
    uid_2_alpha = group_1_alpha / 4
    assert new_scores[2] == (uid_2_alpha * group_1_rank_values[1]) + (
        (1 - uid_2_alpha) * scores[2]
    )
    logger.info("uid 2 group 1 score is right")

    assert new_scores[3] < scores[3]
    uid_3_alpha = group_1_alpha / 4
    assert new_scores[3] == (uid_3_alpha * group_1_rank_values[2]) + (
        (1 - uid_3_alpha) * scores[3]
    )
    logger.info("uid 3 group 1 score is right")

    assert new_scores[4] < scores[4]
    uid_4_alpha = group_1_alpha / 4
    assert new_scores[4] == (uid_4_alpha * group_1_rank_values[3]) + (
        (1 - uid_4_alpha) * scores[4]
    )
    logger.info("uid 4 group 1 score is right")

    # test group 0
    group_0_rank_values = [0.0, 1.0]
    group_0_uids = uids[:2]
    group_0_alpha = group_alphas[0]

    scores = new_scores

    new_scores = get_new_scores(
        scores=scores,
        uids=group_0_uids,
        alpha=group_0_alpha,
        group_best_possible_rank_value=0.0,
        miner_group_index=0,
        rank_values=group_0_rank_values,
    )

    # uid 0 score should go down
    assert new_scores[0] < scores[0]
    assert new_scores[0] == group_0_rank_values[0] * group_0_alpha + scores[0] * (
        1 - group_0_alpha
    )
    logger.info(f"group 0 uid 0 score zero right")

    # uid 1 score should go up
    assert new_scores[1] > scores[1]
    # should apply loss alpha
    uid_1_alpha = group_0_alpha * 2
    assert new_scores[1] == group_0_rank_values[1] * uid_1_alpha + scores[1] * (
        1 - uid_1_alpha
    )
    logger.info(f"group 0 uid 1 score is right")

    # no other scores should change
    assert np.array_equal(scores[2:], new_scores[2:])


def test_score_update():
    asyncio.run(main())
